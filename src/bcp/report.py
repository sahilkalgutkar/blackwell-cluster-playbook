"""Compare results documents written by `bcp validate`."""

from __future__ import annotations

import json
from pathlib import Path

OPS = ("all_reduce", "all_gather", "reduce_scatter", "alltoall")


def load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for path in paths:
        doc = json.loads(path.read_text())
        doc["_path"] = str(path)
        runs.append(doc)
    return runs


def _num(value: float | None) -> str:
    return "–" if value is None else f"{value:,.0f}"


def markdown_table(runs: list[dict]) -> str:
    header = (
        "| Run | Platform | Driver / NCCL | "
        + " | ".join(f"{op} busbw" for op in OPS)
        + " | D2D read avg | H2D avg | DCGM | Verdict |"
    )
    sep = "|" + "---|" * (header.count("|") - 1)
    rows = [header, sep]
    for doc in runs:
        run, platform, stack = doc["run"], doc["platform"], doc["stack"]
        nccl = doc["metrics"].get("nccl") or {}
        nvbw = doc["metrics"].get("nvbandwidth") or {}
        dcgm = doc["metrics"].get("dcgm") or {}
        host = run.get("host", "?") + (" *(synthetic)*" if run.get("synthetic") else "")
        cells = [
            host,
            f"{platform['gpu_count']}x {platform['gpu_model']} · {platform['form_factor']}",
            f"{stack.get('driver') or '?'} / {stack.get('nccl') or '?'}",
            *[_num((nccl.get(op) or {}).get("peak_busbw_gbps")) for op in OPS],
            _num((nvbw.get("device_to_device_memcpy_read_ce") or {}).get("avg_gbps")),
            _num((nvbw.get("host_to_device_memcpy_ce") or {}).get("avg_gbps")),
            dcgm.get("overall", "–"),
            f"**{doc['verdict'].upper()}**",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    note = (
        "\nBandwidth columns are GB/s. nccl columns are peak bus bandwidth over all message sizes."
    )
    return "\n".join(rows) + "\n" + note + "\n"


def plot_busbw(runs: list[dict], out: Path, op: str = "all_reduce") -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("plotting needs matplotlib: pip install -e '.[plot]'") from exc

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for doc in runs:
        curve = ((doc["metrics"].get("nccl") or {}).get(op) or {}).get("curve") or []
        points = [(s, b) for s, b in curve if s and b is not None]
        if not points:
            continue
        label = doc["run"].get("host", "?") + (
            " (synthetic)" if doc["run"].get("synthetic") else ""
        )
        ax.plot([s for s, _ in points], [b for _, b in points], marker="o", ms=3, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("bus bandwidth (GB/s)")
    ax.set_title(f"nccl-tests {op}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    return out
