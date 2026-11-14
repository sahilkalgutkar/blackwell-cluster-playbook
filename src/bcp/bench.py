"""Turn a rental day's raw output into summary.md, and make a redacted copy to commit.

Reads what `bcp rental run` leaves in a run directory:

    validate/<suite>/*.json             bcp validate results
    inference/<id>/meta.json            what was served, on how many GPUs
    inference/<id>/c<N>/**/profile_export_aiperf.json
    training/<id>.json                  training/bench_train.py results
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
from pathlib import Path

from bcp import report

# ---------------------------------------------------------------------------------------
# Inference


def parse_aiperf(doc: dict, concurrency: int, gpus: int) -> dict:
    def stat(tag: str, key: str = "avg") -> float | None:
        value = (doc.get(tag) or {}).get(key)
        return float(value) if value is not None else None

    out_tps = stat("output_token_throughput")
    itl_p50 = stat("inter_token_latency", "p50")
    return {
        "concurrency": concurrency,
        "output_tps": out_tps,
        "output_tps_per_gpu": out_tps / gpus if out_tps is not None else None,
        "tps_per_user": 1000.0 / itl_p50 if itl_p50 else None,
        "request_tps": stat("request_throughput"),
        "ttft_p50_ms": stat("time_to_first_token", "p50"),
        "ttft_p99_ms": stat("time_to_first_token", "p99"),
        "itl_p50_ms": itl_p50,
        "itl_p99_ms": stat("inter_token_latency", "p99"),
        "requests": stat("request_count"),
        "errors": stat("error_request_count") or 0.0,
    }


def load_inference(run_dir: Path) -> list[dict]:
    configs = []
    for meta_path in sorted((run_dir / "inference").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        points = []
        for cdir in meta_path.parent.glob("c*"):
            if not cdir.is_dir() or not cdir.name[1:].isdigit():
                continue
            export = next(iter(sorted(cdir.rglob("profile_export_aiperf.json"))), None)
            if export:
                doc = json.loads(export.read_text())
                points.append(parse_aiperf(doc, int(cdir.name[1:]), int(meta["gpus"])))
        meta["points"] = sorted(points, key=lambda p: p["concurrency"])
        configs.append(meta)
    return sorted(configs, key=lambda c: (c.get("order", 10**6), c["id"]))


def peak(points: list[dict]) -> dict | None:
    ok = [p for p in points if p["output_tps_per_gpu"] is not None]
    return max(ok, key=lambda p: p["output_tps_per_gpu"], default=None)


def best_under_sla(points: list[dict], ttft_ms: float) -> dict | None:
    ok = [
        p
        for p in points
        if p["output_tps_per_gpu"] is not None
        and p["ttft_p99_ms"] is not None
        and p["ttft_p99_ms"] <= ttft_ms
    ]
    return max(ok, key=lambda p: p["output_tps_per_gpu"], default=None)


def _f(value: float | None, fmt: str = ",.0f") -> str:
    return "–" if value is None else format(value, fmt)


def _at(point: dict | None) -> str:
    return "–" if point is None else f"{_f(point['output_tps_per_gpu'])} @ c{point['concurrency']}"


def _ratio(a: dict | None, b: dict | None) -> str | None:
    if not a or not b or not a["output_tps_per_gpu"] or not b["output_tps_per_gpu"]:
        return None
    return f"{a['output_tps_per_gpu'] / b['output_tps_per_gpu']:.2f}x"


def inference_markdown(configs: list[dict], ttft_sla_ms: float) -> str:
    if not configs:
        return "_No inference runs._\n"
    workload = configs[0].get("workload") or {}
    sla_s = f"{ttft_sla_ms / 1000:g} s"
    lines = [
        f"Synthetic chat load: {workload.get('isl', '?')} tokens in, {workload.get('osl', '?')} "
        f"out (ignore_eos), same prompts for every server. Throughput is output tokens/s per "
        f"GPU.",
        "",
        f"| Config | Engine | Precision | GPUs | Peak tok/s/GPU | Best with p99 TTFT ≤ {sla_s} "
        "| TTFT p50 / p99 there | ITL p50 there | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in configs:
        best = best_under_sla(c["points"], ttft_sla_ms)
        layout = c.get("layout")
        engine = c["engine"] + (
            " (" + " + ".join(f"{len(v)}×{k}" for k, v in layout.items()) + ")" if layout else ""
        )
        lines.append(
            f"| `{c['id']}` | {engine} | {c['precision'].upper()} | {c['gpus']} "
            f"| {_at(peak(c['points']))} | {_at(best)} "
            f"| {_f(best and best['ttft_p50_ms'])} / {_f(best and best['ttft_p99_ms'])} ms "
            f"| {_f(best and best['itl_p50_ms'], '.1f')} ms | {c.get('status', '?')} |"
        )

    notes = comparisons(configs, ttft_sla_ms)
    if notes:
        lines += ["", "**Comparisons**", "", *[f"- {n}" for n in notes]]
    lines += ["", "<details><summary>Every sweep point</summary>", ""]
    for c in configs:
        lines += [
            f"**{c['id']}** · `{c.get('model', '?')}` · `{c.get('image', '?')}`",
            "",
            "| Concurrency | tok/s | tok/s/GPU | tok/s/user | TTFT p50 | TTFT p99 | ITL p50 "
            "| ITL p99 | Errors |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for p in c["points"]:
            lines.append(
                f"| {p['concurrency']} | {_f(p['output_tps'])} | {_f(p['output_tps_per_gpu'])} "
                f"| {_f(p['tps_per_user'], '.1f')} | {_f(p['ttft_p50_ms'])} "
                f"| {_f(p['ttft_p99_ms'])} | {_f(p['itl_p50_ms'], '.1f')} "
                f"| {_f(p['itl_p99_ms'], '.1f')} | {_f(p['errors'])} |"
            )
        lines.append("")
    lines.append("</details>")
    return "\n".join(lines) + "\n"


def comparisons(configs: list[dict], ttft_sla_ms: float) -> list[str]:
    notes = []
    by_engine: dict[str, dict[str, dict]] = {}
    for c in configs:
        if not c.get("layout"):
            by_engine.setdefault(c["engine"], {})[c["precision"]] = c
    for engine, precs in by_engine.items():
        for hi, lo in (("nvfp4", "fp8"), ("fp8", "bf16")):
            if hi in precs and lo in precs:
                r = _ratio(peak(precs[hi]["points"]), peak(precs[lo]["points"]))
                if r:
                    notes.append(
                        f"{engine}: {hi.upper()} vs {lo.upper()} peak throughput per GPU: **{r}**"
                    )
    agg = [c for c in configs if c.get("layout") and set(c["layout"]) == {"agg"}]
    for c in configs:
        layout = c.get("layout") or {}
        if "prefill" not in layout:
            continue
        for a in agg:
            if a["gpus"] != c["gpus"]:
                continue
            r = _ratio(
                best_under_sla(c["points"], ttft_sla_ms), best_under_sla(a["points"], ttft_sla_ms)
            )
            if r:
                notes.append(f"`{c['id']}` vs `{a['id']}` (best under the TTFT limit): **{r}**")
    return notes


def plot_inference(configs: list[dict], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in configs:
        pts = [p for p in c["points"] if p["tps_per_user"] and p["output_tps_per_gpu"]]
        if pts:
            ax.plot(
                [p["tps_per_user"] for p in pts],
                [p["output_tps_per_gpu"] for p in pts],
                marker="o",
                label=c["id"],
            )
    ax.set_xlabel("Interactivity: output tokens/s per user (1000 / ITL p50)")
    ax.set_ylabel("Throughput: output tokens/s per GPU")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    return path


# ---------------------------------------------------------------------------------------
# Training


def load_training(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((run_dir / "training").glob("*.json"))]


def training_markdown(runs: list[dict]) -> str:
    if not runs:
        return "_No training runs._\n"
    first = runs[0]
    lines = [
        f"Synthetic tokens, sequence length {first['seq_len']}, {first['steps']} measured steps "
        f"after {first['warmup_steps']} warmup. MFU is against the dense BF16 peak "
        "(see training/bench_train.py for the FLOP formula).",
        "",
        "| Run | Model | Precision | Parallel | GPUs | tokens/s/GPU | TFLOP/s/GPU | MFU "
        "| Scaling eff. | Peak mem |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    base: dict[tuple, dict] = {}
    for r in sorted(runs, key=lambda r: r["gpus"]):
        base.setdefault((r["preset"], r["precision"], r["micro_batch"]), r)
    for r in runs:
        b = base[(r["preset"], r["precision"], r["micro_batch"])]
        eff = "–"
        if b is not r and b["tokens_per_s_per_gpu"]:
            ratio = r["tokens_per_s_per_gpu"] / b["tokens_per_s_per_gpu"]
            eff = f"{ratio * 100:.0f}% of {b['gpus']} GPU"
        mfu = "–" if r.get("mfu") is None else f"{r['mfu'] * 100:.1f}%"
        lines.append(
            f"| `{r['id']}` | {r['preset']} ({r['model']['params'] / 1e9:.1f}B) "
            f"| {r['precision'].upper()} | {r['parallel']} | {r['gpus']} "
            f"| {_f(r['tokens_per_s_per_gpu'])} | {_f(r['tflops_per_gpu'], '.0f')} "
            f"| {mfu} | {eff} | {_f(r.get('peak_mem_gb'), '.0f')} GB |"
        )
    notes = []
    groups: dict[tuple, dict[str, dict]] = {}
    for r in runs:
        groups.setdefault((r["preset"], r["gpus"]), {})[r["precision"]] = r
    for (preset, gpus), precs in groups.items():
        bf16 = precs.get("bf16")
        for prec, r in precs.items():
            if prec != "bf16" and bf16 and bf16["tokens_per_s_per_gpu"]:
                notes.append(
                    f"{preset} on {gpus} GPUs: {prec.upper()} vs BF16 "
                    f"**{r['tokens_per_s_per_gpu'] / bf16['tokens_per_s_per_gpu']:.2f}x**"
                )
    if notes:
        lines += ["", "**Comparisons**", "", *[f"- {n}" for n in notes]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------
# Summary


def write_summary(run_dir: Path, out: Path | None = None, plot: bool = True) -> Path:
    run_dir = Path(run_dir)
    plan = _read_json(run_dir / "plan.json") or {}
    state = _read_json(run_dir / "state.json") or {}
    cfg = plan.get("config") or {}
    node = plan.get("node") or {}
    rehearse = cfg.get("rehearse")

    lines = [
        f"# {'Rehearsal' if rehearse else 'Rental day'}: {run_dir.name}",
        "",
        f"{node.get('gpu_count', '?')}x {node.get('gpu_name', '?')} · driver "
        f"{node.get('driver', '?')} · started {state.get('started_at', '?')}",
        "",
    ]
    if rehearse:
        lines += [
            "> Rehearsal on a small model and a cheap node: this checks the pipeline, "
            "the numbers mean nothing.",
            "",
        ]

    lines += ["## Validation", ""]
    docs = sorted((run_dir / "validate").glob("*/*.json"))
    if docs:
        lines.append(report.markdown_table(report.load_runs(docs)))
    else:
        lines.append("_No validation results._\n")

    configs = load_inference(run_dir)
    ttft = float((cfg.get("inference") or {}).get("ttft_sla_ms", 2000))
    lines += ["## Inference", "", inference_markdown(configs, ttft)]
    if plot and configs:
        try:
            img = plot_inference(configs, run_dir / "inference.png")
            lines += [f"![Throughput vs interactivity]({img.name})", ""]
        except ImportError:
            pass

    lines += ["## Training", "", training_markdown(load_training(run_dir))]

    steps = state.get("steps") or {}
    bad = {k: v for k, v in steps.items() if v.get("status") in ("failed", "skipped")}
    if bad:
        lines += ["## Skipped or failed steps", "", "| Step | Status | Detail |", "|---|---|---|"]
        for k, v in bad.items():
            detail = (v.get("detail") or "").splitlines()[0][:160] if v.get("detail") else ""
            lines.append(f"| `{k}` | {v['status']} | {detail} |")
        lines.append("")
    images = cfg.get("images") or {}
    if images:
        lines += ["## Software", "", *[f"- {k}: `{v}`" for k, v in images.items()], ""]

    out = out or run_dir / "summary.md"
    out.write_text("\n".join(lines))
    return out


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------
# Redaction and packaging

_GPU_UUID = re.compile(r"\b(GPU|MIG)-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b")
_UUID = re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b")
_SERIAL = re.compile(r"(?im)^(\s*(?:Serial Number|Board Serial|Serial)\s*:\s*)\S.*$")
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4 = re.compile(rf"(?<![\d.]){_OCTET}(?:\.{_OCTET}){{3}}(?![\d.])")
_MAC = re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b")
_KEEP_IPS = {"127.0.0.1", "0.0.0.0"}
TEXT_SUFFIXES = {".txt", ".json", ".log", ".csv", ".md", ".mmd", ".yaml", ".yml", ".sh", ""}


class Redactor:
    """Removes serials, UUIDs, IPs, MACs and hostnames, consistently across files.

    Non-zero UUIDs (an NVL72 tray's ClusterUUID, for example) map to stable fake values that
    are still non-zero, so `bcp validate --replay` on the redacted copy reaches the same
    verdict as on the original.
    """

    def __init__(self, hostnames: list[str]) -> None:
        names = {h for h in hostnames if h and h not in ("localhost",)}
        names |= {h.split(".")[0] for h in names if len(h.split(".")[0]) > 2}
        self.hosts = sorted(names, key=len, reverse=True)
        self.uuids: dict[str, str] = {}

    def _uuid(self, m: re.Match) -> str:
        u = m.group(0).lower()
        if set(u) <= {"0", "-"}:
            return m.group(0)
        if u not in self.uuids:
            self.uuids[u] = f"00000000-0000-0000-0000-{len(self.uuids) + 1:012d}"
        return self.uuids[u]

    def text(self, s: str) -> str:
        s = _GPU_UUID.sub(lambda m: f"{m.group(1)}-REDACTED", s)
        s = _UUID.sub(self._uuid, s)
        s = _SERIAL.sub(lambda m: m.group(1) + "REDACTED", s)
        s = _IPV4.sub(lambda m: m.group(0) if m.group(0) in _KEEP_IPS else "x.x.x.x", s)
        s = _MAC.sub("xx:xx:xx:xx:xx:xx", s)
        for h in self.hosts:
            s = re.sub(rf"(?<![\w-]){re.escape(h)}(?![\w-])", "node01", s)
        return s

    def name(self, name: str) -> str:
        for h in self.hosts:
            name = name.replace(h, "node01")
        return name


def package(run_dir: Path) -> tuple[Path, Path]:
    """Write <run>-share/ (redacted, safe to commit) and <run>.tgz (everything, private)."""
    run_dir = Path(run_dir)
    pre = _read_json(run_dir / "host" / "preflight.json") or {}
    hosts = [pre.get("host", "")]
    for doc in (run_dir / "validate").glob("*/*.json"):
        hosts.append((_read_json(doc) or {}).get("run", {}).get("host", ""))
    red = Redactor(hosts)

    share = run_dir.parent / f"{run_dir.name}-share"
    if share.exists():
        shutil.rmtree(share)
    for src in sorted(run_dir.rglob("*")):
        rel = Path(*[red.name(part) for part in src.relative_to(run_dir).parts])
        dst = share / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in TEXT_SUFFIXES:
            try:
                dst.write_text(red.text(src.read_text(errors="replace")))
                continue
            except OSError:
                pass
        shutil.copy2(src, dst)

    tarball = run_dir.parent / f"{run_dir.name}.tgz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)
    return share, tarball
