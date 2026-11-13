"""Turn probe output into pass/warn/fail checks and a results document."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from bcp import __version__
from bcp.parsers import (
    DcgmResult,
    NcclResult,
    NvbwResult,
    parse_dcgm,
    parse_gpu_query,
    parse_nccl,
    parse_nvbandwidth,
)
from bcp.probes import NCCL_OPS, PROBE_BY_KEY, PROBES, base_params
from bcp.runner import ProbeOutput, Runner
from bcp.topology import Topology, analyze

STATUS_ORDER = {"info": 0, "skip": 0, "pass": 1, "warn": 2, "fail": 3}


@dataclass
class Check:
    name: str
    status: str  # pass | warn | fail | info | skip
    measured: str = ""
    expected: str = ""
    detail: str = ""


def load_thresholds(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"thresholds file not found: {path} (run from the repo root or pass --thresholds)"
        )
    return yaml.safe_load(path.read_text()) or {}


def _profile(thresholds: dict, model: str) -> dict:
    return (thresholds.get("profiles") or {}).get(model) or {}


def _fmt(value: float | None, unit: str = "GB/s") -> str:
    return "n/a" if value is None else f"{value:,.1f} {unit}"


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


# --------------------------------------------------------------------------- checks


def topology_checks(topo: Topology, profile: dict) -> list[Check]:
    p = topo.platform
    checks: list[Check] = []
    expected_gpus = profile.get("expected_gpus")
    if expected_gpus:
        checks.append(
            Check(
                "inventory.gpu_count",
                "pass" if p.gpu_count == expected_gpus else "fail",
                str(p.gpu_count),
                f"== {expected_gpus}",
            )
        )
    if p.nvlink_links is not None:
        if topo.weak_nvlink_gpus:
            weak = ", ".join(f"{g} NV{n}" for g, n in sorted(topo.weak_nvlink_gpus.items()))
            checks.append(
                Check(
                    "topology.nvlink_symmetry",
                    "fail",
                    weak,
                    f"all NV{p.nvlink_links}",
                    "a GPU with fewer active NVLinks than its peers",
                )
            )
        else:
            checks.append(
                Check("topology.nvlink_symmetry", "pass", f"all NV{p.nvlink_links}", "symmetric")
            )
    if p.superchip and p.gpu_model != "GB10":
        ok = (p.fabric_state or "").lower() == "completed"
        checks.append(
            Check(
                "topology.nvlink_fabric",
                "pass" if ok else "fail",
                p.fabric_state or "unknown",
                "Completed",
                "multi-node domain" if p.multi_node_nvlink else "single-node only",
            )
        )
    for warning in p.warnings:
        if "no NIC on its socket" in warning:
            checks.append(Check("topology.nic_locality", "warn", detail=warning))
    return checks


def dcgm_checks(result: DcgmResult | None, profile: dict, defaults: dict) -> list[Check]:
    if result is None:
        return [Check("dcgm.diag", "skip", detail="dcgmi not run")]
    if result.error:
        return [Check("dcgm.diag", "fail", detail=result.error)]
    allow_warn = (profile.get("dcgm") or defaults.get("dcgm") or {}).get("allow_warn", True)
    overall = result.overall
    status = {"pass": "pass", "warn": "warn" if allow_warn else "fail", "fail": "fail"}.get(
        overall, "skip"
    )
    failing = [n for n, s in result.tests.items() if s in ("warn", "fail")]
    return [
        Check(
            "dcgm.diag",
            status,
            overall,
            "pass",
            "; ".join(result.messages[:3]) or (", ".join(failing) if failing else ""),
        )
    ]


def nccl_checks(op: str, result: NcclResult | None, profile: dict) -> list[Check]:
    name = f"nccl.{op}"
    if result is None or not result.rows:
        return [Check(name, "skip", detail="nccl-tests output not available")]
    peak = result.peak
    measured = f"{_fmt(peak.best_busbw)} @ {_human_bytes(peak.size)}"
    checks = []
    if result.wrong_total:
        checks.append(
            Check(
                f"{name}.correctness",
                "fail",
                f"{result.wrong_total} wrong",
                "0",
                "nccl-tests found corrupted results: suspect GPU memory or NVLink errors",
            )
        )
    limit = ((profile.get("nccl") or {}).get(op) or {}).get("min_peak_busbw_gbps")
    if limit is None:
        checks.append(Check(f"{name}.busbw", "info", measured, "no threshold for this model"))
    else:
        checks.append(
            Check(
                f"{name}.busbw",
                "pass" if peak.best_busbw >= limit else "fail",
                measured,
                f">= {limit:,} GB/s",
            )
        )
    return checks


def nvbandwidth_checks(result: NvbwResult | None, profile: dict) -> list[Check]:
    if result is None or not result.tests:
        return [Check("nvbandwidth", "skip", detail="nvbandwidth output not available")]
    limits = profile.get("nvbandwidth") or {}
    checks = []
    for test_name, test in result.tests.items():
        name = f"nvbandwidth.{test_name}"
        avg = test.avg_gbps
        limit = (limits.get(test_name) or {}).get("min_avg_gbps")
        if limit is None:
            checks.append(Check(name, "info", _fmt(avg) + " avg", "no threshold for this model"))
        else:
            checks.append(
                Check(
                    name,
                    "pass" if (avg or 0) >= limit else "fail",
                    _fmt(avg) + " avg",
                    f">= {limit:,} GB/s",
                )
            )
        weakest, median = test.weakest, test.median_gbps
        if weakest and median and weakest[2] < 0.8 * median:
            row, col, value = weakest
            checks.append(
                Check(
                    f"{name}.uniformity",
                    "warn",
                    f"{row}->{col}: {_fmt(value)}",
                    f">= 80% of median ({_fmt(median)})",
                    "one path is much slower than the rest",
                )
            )
    return checks


# --------------------------------------------------------------------------- driver


def run_validation(
    runner: Runner,
    thresholds: dict,
    suite: str = "full",
    progress: Callable[[str], None] | None = None,
) -> dict:
    say = progress or (lambda msg: print(msg, file=sys.stderr))
    params = base_params(suite)
    outputs: dict[str, ProbeOutput] = {}

    first = runner.run(PROBE_BY_KEY["gpu_query"], params)
    outputs["gpu_query"] = first
    gpus = parse_gpu_query(first.text) if first.available else []
    params["ngpus"] = str(len(gpus) or 1)
    for probe in PROBES:
        if probe.key == "gpu_query":
            continue
        say(f"  {probe.key:<20} {probe.description}")
        outputs[probe.key] = runner.run(probe, params)

    def text(key: str) -> str | None:
        out = outputs.get(key)
        return out.text if out and out.available else None

    topo = analyze(text("gpu_query"), text("topo"), text("smi_q"), text("dgx_release"))
    profile = _profile(thresholds, topo.platform.gpu_model)
    defaults = thresholds.get("defaults") or {}

    dcgm = parse_dcgm(text("dcgm_diag")) if text("dcgm_diag") else None
    nccl = {op: parse_nccl(text(key)) if text(key) else None for key, op in NCCL_OPS.items()}
    nvbw = parse_nvbandwidth(text("nvbandwidth")) if text("nvbandwidth") else None

    checks: list[Check] = []
    checks += topology_checks(topo, profile)
    checks += dcgm_checks(dcgm, profile, defaults)
    for op, result in nccl.items():
        checks += nccl_checks(op, result, profile)
    checks += nvbandwidth_checks(nvbw, profile)

    verdict = max((c.status for c in checks), key=STATUS_ORDER.__getitem__, default="skip")
    if verdict in ("info", "skip"):
        verdict = "pass" if any(c.status == "pass" for c in checks) else "skip"

    nccl_version = next((r.nccl_version for r in nccl.values() if r and r.nccl_version), None)
    meta = runner.metadata()
    return {
        "schema_version": 1,
        "tool": {"name": "bcp", "version": __version__},
        "run": {**meta, "suite": suite, "profile": topo.platform.gpu_model if profile else None},
        "platform": asdict(topo.platform),
        "stack": {
            "driver": topo.platform.driver,
            "cuda": topo.platform.cuda,
            "nccl": nccl_version,
            "nvbandwidth": nvbw.version if nvbw else None,
            "dcgm": dcgm.version if dcgm else None,
        },
        "metrics": {
            "nccl": {
                op: {
                    "peak_busbw_gbps": r.peak.best_busbw,
                    "peak_at_bytes": r.peak.size,
                    "avg_busbw_gbps": r.avg_busbw,
                    "nranks": r.nranks,
                    "wrong": r.wrong_total,
                    "curve": [[row.size, row.oop_busbw] for row in r.rows],
                }
                for op, r in nccl.items()
                if r and r.rows
            },
            "nvbandwidth": {
                name: {
                    "sum_gbps": t.sum_gbps,
                    "avg_gbps": t.avg_gbps,
                    "min_gbps": t.weakest[2] if t.weakest else None,
                }
                for name, t in (nvbw.tests.items() if nvbw else [])
            },
            "dcgm": (
                {"overall": dcgm.overall, "tests": dcgm.tests, "messages": dcgm.messages}
                if dcgm
                else None
            ),
        },
        "checks": [asdict(c) for c in checks],
        "verdict": verdict,
        "probe_notes": {k: o.note for k, o in outputs.items() if o.note},
    }


def render_checks(doc: dict) -> str:
    run = doc["run"]
    tag = " · SYNTHETIC fixture, not a real measurement" if run.get("synthetic") else ""
    lines = [
        f"Verdict: {doc['verdict'].upper()}  ({run.get('host')}, {run.get('source')}){tag}",
        f"Platform: {doc['platform']['form_factor']} · {doc['platform']['gpu_count']} x "
        f"{doc['platform']['gpu_name']}",
        "",
        f"{'STATUS':<7} {'CHECK':<46} {'MEASURED':<26} EXPECTED",
    ]
    for c in doc["checks"]:
        lines.append(f"{c['status']:<7} {c['name']:<46} {c['measured']:<26} {c['expected']}")
        if c["detail"] and c["status"] in ("warn", "fail"):
            lines.append(f"{'':<7}   ↳ {c['detail']}")
    return "\n".join(lines)
