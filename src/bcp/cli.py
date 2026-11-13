"""bcp: blackwell-cluster-playbook command line."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from bcp import __version__, report, stack
from bcp.parsers import parse_gpu_query
from bcp.probes import PROBE_BY_KEY, PROBES, base_params
from bcp.runner import LiveRunner, ReplayRunner
from bcp.topology import analyze, render_json, render_mermaid, render_text
from bcp.validate import load_thresholds, render_checks, run_validation


def _runner(replay: str | None, save_dir: Path | None = None):
    return ReplayRunner(Path(replay)) if replay else LiveRunner(save_dir=save_dir)


def cmd_topology(args: argparse.Namespace) -> int:
    runner = _runner(args.replay)
    params = base_params("quick")
    texts = {}
    for key in ("gpu_query", "topo", "smi_q", "dgx_release"):
        out = runner.run(PROBE_BY_KEY[key], params)
        texts[key] = out.text if out.available else None
    if not any(texts.values()):
        print(
            "No GPU information found. Is nvidia-smi installed? Try --replay fixtures/dgx-b200",
            file=sys.stderr,
        )
        return 2
    topo = analyze(texts["gpu_query"], texts["topo"], texts["smi_q"], texts["dgx_release"])
    render = {"text": render_text, "json": render_json, "mermaid": render_mermaid}[args.format]
    print(render(topo))
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    out = Path(args.out)
    runner = LiveRunner(save_dir=out)
    params = base_params(args.suite)
    first = runner.run(PROBE_BY_KEY["gpu_query"], params)
    if first.available:
        params["ngpus"] = str(len(parse_gpu_query(first.text)) or 1)
    for probe in PROBES:
        if probe.key == "gpu_query":
            continue
        print(f"  {probe.key:<20} {probe.description}", file=sys.stderr)
        result = runner.run(probe, params)
        if result.note:
            print(f"  {'':<20} note: {result.note.splitlines()[0][:120]}", file=sys.stderr)
    print(f"Raw output saved to {out}. Analyse it with: bcp validate --replay {out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    thresholds = load_thresholds(Path(args.thresholds))
    out_dir = Path(args.out)
    stamp_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runner = _runner(args.replay, save_dir=None if args.replay else out_dir / "raw" / stamp_now)
    print(f"Running {args.suite} suite ({'replay' if args.replay else 'live'})", file=sys.stderr)
    doc = run_validation(runner, thresholds, suite=args.suite)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (doc["run"].get("collected_at") or "unknown").replace(":", "").replace("-", "")
    path = out_dir / f"{doc['run'].get('host', 'host')}-{stamp}.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(render_checks(doc))
    print(f"\nResults written to {path}")
    if doc["verdict"] == "skip":
        print("Nothing could be checked: no GPU tools or output found.", file=sys.stderr)
        return 2
    return 1 if doc["verdict"] == "fail" else 0


def cmd_report(args: argparse.Namespace) -> int:
    runs = report.load_runs([Path(p) for p in args.runs])
    table = report.markdown_table(runs)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(table)
    if args.plot:
        print(f"Wrote {report.plot_busbw(runs, Path(args.plot), op=args.plot_op)}", file=sys.stderr)
    return 0


def cmd_stack(args: argparse.Namespace) -> int:
    path = Path(args.file)
    statuses = stack.check(stack.load(path), stack.UrllibHttp())
    print(stack.render_table(statuses))
    if args.summary_file:
        Path(args.summary_file).write_text(stack.render_markdown(statuses))
    updates = {s.name: s.latest for s in statuses if s.state == "update" and s.latest}
    if args.update and updates:
        path.write_text(stack.rewrite_pinned(path.read_text(), updates))
        print(f"\nUpdated {len(updates)} pin(s) in {path}", file=sys.stderr)
    return 1 if args.fail_on_outdated and updates else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bcp", description="Validate and benchmark NVIDIA GPU nodes and clusters."
    )
    parser.add_argument("--version", action="version", version=f"bcp {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("topology", help="detect platform and show GPU/NVLink/NIC topology")
    p.add_argument("--replay", metavar="DIR", help="read saved output instead of running tools")
    p.add_argument("--format", choices=("text", "json", "mermaid"), default="text")
    p.set_defaults(func=cmd_topology)

    p = sub.add_parser("collect", help="run every probe on this node and save raw output")
    p.add_argument("--out", required=True, metavar="DIR")
    p.add_argument("--suite", choices=("quick", "full"), default="quick")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("validate", help="run checks (live or replayed) and write a results JSON")
    p.add_argument("--replay", metavar="DIR", help="analyse output saved by `bcp collect`")
    p.add_argument("--suite", choices=("quick", "full"), default="quick")
    p.add_argument("--thresholds", default="configs/thresholds.yaml")
    p.add_argument("--out", default="build/runs", metavar="DIR")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("report", help="compare results JSON files as a Markdown table")
    p.add_argument("runs", nargs="+", metavar="RUN.json")
    p.add_argument("--out", metavar="FILE.md")
    p.add_argument("--plot", metavar="FILE.png", help="also plot busbw curves (needs [plot])")
    p.add_argument("--plot-op", default="all_reduce")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("stack", help="track upstream versions in matrix/stack.yaml")
    stack_sub = p.add_subparsers(dest="stack_command", required=True)
    c = stack_sub.add_parser("check", help="compare pinned versions with latest upstream")
    c.add_argument("--file", default="matrix/stack.yaml")
    c.add_argument("--update", action="store_true", help="rewrite pins to the latest versions")
    c.add_argument("--summary-file", metavar="FILE.md", help="write a Markdown summary (PR body)")
    c.add_argument("--fail-on-outdated", action="store_true")
    c.set_defaults(func=cmd_stack)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
