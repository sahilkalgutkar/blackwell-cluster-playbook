"""Plan and run a scripted day on rented GPU hardware: `bcp rental plan|run`.

The day is a list of steps built from rental/day.yaml: preflight, background downloads,
a quick validation that stops the day if the hardware is broken, full validation, an
inference matrix (each config: start a server, wait until it answers, sweep concurrency with
AIPerf, stop it), a training matrix, then a report and a redacted copy of the results.

Every step records its outcome in <run>/state.json, so `bcp rental run --resume` picks up
where a failure or a dropped SSH session left off without repeating paid work.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bcp import stack

SERVER = "bcp-server"
PORT = 8000
HF_IN_CONTAINER = "/root/.cache/huggingface"

ENGINE_NAMES = {
    "vllm": "vLLM",
    "sglang": "SGLang",
    "trtllm": "TensorRT-LLM",
    "dynamo-vllm": "Dynamo + vLLM",
}
STARTUP_MIN = {"vllm": 6, "sglang": 6, "trtllm": 10, "dynamo-vllm": 8}
BLACKWELL_ONLY = {"nvfp4", "mxfp8"}

# Validation checks that mean "this node is broken", as opposed to "slower than the
# threshold we guessed": only these stop the day at the quick-validation gate.
HARD_CHECK_PREFIXES = ("inventory.", "topology.nvlink", "dcgm.")


# ---------------------------------------------------------------------------------------
# Config


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


_TEMPLATE = re.compile(r"\{([a-z0-9][a-z0-9-]*)(:bare)?\}")


def resolve_templates(text: str, versions: dict[str, str | None]) -> str:
    def sub(m: re.Match) -> str:
        name, bare = m.group(1), m.group(2)
        version = versions.get(name)
        if not version:
            raise KeyError(f"'{name}' has no pinned version in matrix/stack.yaml")
        return version[1:] if bare and version.startswith("v") else version

    return _TEMPLATE.sub(sub, text)


def load_day(path: Path, stack_path: Path, rehearse: bool = False) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if rehearse:
        cfg = _merge(cfg, cfg.get("rehearsal") or {})
    cfg.pop("rehearsal", None)
    versions = {c["name"]: c.get("pinned") for c in stack.load(stack_path)}
    cfg["images"] = {k: resolve_templates(v, versions) for k, v in cfg["images"].items()}
    cfg["host_packages"] = [resolve_templates(p, versions) for p in cfg.get("host_packages", [])]
    cfg["rehearse"] = rehearse
    return cfg


# ---------------------------------------------------------------------------------------
# The node


@dataclass
class Node:
    gpu_name: str
    gpu_count: int
    compute_cap: float
    driver: str | None = None
    detected: bool = True

    @property
    def blackwell(self) -> bool:
        return self.compute_cap >= 10.0

    def describe(self) -> str:
        src = "detected" if self.detected else "assumed (no nvidia-smi here)"
        return f"{self.gpu_count}x {self.gpu_name} ({self.sm}), {src}"

    @property
    def sm(self) -> str:
        return f"sm_{round(self.compute_cap * 10)}"


def detect_node() -> Node | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    rows = [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]
    if not rows:
        return None
    name, cap, driver = rows[0][:3]
    return Node(name, len(rows), float(cap), driver)


# ---------------------------------------------------------------------------------------
# Steps


@dataclass
class Step:
    id: str
    phase: str
    title: str
    est_min: float
    kind: str  # shell | validate | background | wait | serve | train | python
    argv: list[str] = field(default_factory=list)
    optional: bool = False
    skip: str | None = None
    timeout_min: float = 60
    data: dict = field(default_factory=dict)


@dataclass
class Ctx:
    cfg: dict
    node: Node
    repo: Path
    run_dir: Path
    rate: float
    hf_home: Path
    bin_dir: Path  # the venv that runs bcp: aiperf and hf get installed here

    @property
    def bench_image(self) -> str:
        return self.cfg["images"]["bench"]


def _docker_run(
    ctx: Ctx, *, detach: bool = False, name: str | None = None, privileged: bool = False
) -> list[str]:
    argv = ["docker", "run", "-d" if detach else "--rm"]
    if name:
        argv += ["--name", name]
    argv += [
        "--gpus",
        "all",
        "--ipc=host",
        "--network",
        "host",
        "--shm-size",
        "16g",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
    ]
    if privileged:
        argv.append("--privileged")  # DCGM diagnostics need it; the node is ours alone
    return argv


def _hf_mount(ctx: Ctx) -> list[str]:
    return [
        "-v",
        f"{ctx.hf_home}:{HF_IN_CONTAINER}",
        "-e",
        f"HF_HOME={HF_IN_CONTAINER}",
        "-e",
        "HF_HUB_OFFLINE=1",
    ]


def _gpu_list(n: int, start: int = 0) -> str:
    return ",".join(str(i) for i in range(start, start + n))


def config_gpus(conf: dict) -> int:
    if "layout" in conf:
        return sum(sum(tps) for tps in conf["layout"].values())
    return int(conf.get("gpus", 1))


def concurrency_for(inf: dict, gpus: int) -> list[int]:
    table = {int(k): v for k, v in inf["concurrency"].items()}
    key = max((k for k in table if k <= gpus), default=min(table))
    return list(table[key])


def render_dynamo_launch(
    model: str, layout: dict, max_model_len: int, system_port: int = 8081, nixl_port: int = 20097
) -> tuple[str, list[int]]:
    """A launch script for one Dynamo layout, run inside the vllm-runtime container.

    layout maps role (agg | prefill | decode) to a list of tensor-parallel sizes, one per
    worker. GPUs are handed out in order. Returns the script and the workers' health ports.
    """
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by `bcp rental`: Dynamo frontend plus one vLLM worker per entry in the",
        f"# layout {json.dumps(layout)}. Runs inside nvcr.io/nvidia/ai-dynamo/vllm-runtime.",
        "set -euo pipefail",
        "trap 'kill 0' EXIT",
        "",
        f'MODEL="{model}"',
        "COMMON=(",
        '  --model "${MODEL}"',
        "  --discovery-backend file",
        f"  --max-model-len {max_model_len}",
        "  --kv-events-config '{\"enable_kv_cache_events\": false}'",
        ")",
        "",
        f"python3 -m dynamo.frontend --http-port {PORT} --discovery-backend file \\",
        "  > /bench/frontend.log 2>&1 &",
    ]
    ports, gpu = [], 0
    roles = {
        "agg": [],
        "prefill": [
            "--disaggregation-mode",
            "prefill",
            "--kv-transfer-config",
            '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}',
        ],
        "decode": [
            "--disaggregation-mode",
            "decode",
            "--kv-transfer-config",
            '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}',
        ],
    }
    for role, role_args in roles.items():
        for tp in layout.get(role, []):
            i = len(ports)
            port = system_port + i
            ports.append(port)
            args = [shlex.quote(a) for a in role_args]
            if tp > 1:
                args += ["--tensor-parallel-size", str(tp)]
            lines += [
                "",
                f"# worker {i}: {role}, TP{tp}, GPUs {_gpu_list(tp, gpu)}",
                # vLLM's NIXL side channel uses one port per TP rank, so space them out.
                f"CUDA_VISIBLE_DEVICES={_gpu_list(tp, gpu)} DYN_SYSTEM_PORT={port} "
                f"VLLM_NIXL_SIDE_CHANNEL_PORT={nixl_port + 8 * i} \\",
                '  python3 -m dynamo.vllm "${COMMON[@]}"' + "".join(f" {a}" for a in args) + " \\",
                f"  > /bench/worker-{i}-{role}.log 2>&1 &",
            ]
            gpu += tp
    lines += [
        "",
        "# Stop everything as soon as any process exits.",
        "wait -n",
        'echo "a Dynamo process exited; see /bench/*.log" >&2',
        "exit 1",
        "",
    ]
    return "\n".join(lines), ports


def engine_command(ctx: Ctx, conf: dict, model: str) -> list[str]:
    """`docker run -d ...` for a single-engine server (not Dynamo)."""
    inf, engine, gpus = ctx.cfg["inference"], conf["engine"], config_gpus(conf)
    image = ctx.cfg["images"][engine]
    argv = _docker_run(ctx, detach=True, name=SERVER) + _hf_mount(ctx)
    argv += ["-e", f"CUDA_VISIBLE_DEVICES={_gpu_list(gpus)}"]
    mml = str(inf["max_model_len"])
    if engine == "vllm":
        argv += [
            "--entrypoint",
            "vllm",
            image,
            "serve",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--tensor-parallel-size",
            str(gpus),
            "--max-model-len",
            mml,
        ]
    elif engine == "sglang":
        argv += [
            image,
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--tp-size",
            str(gpus),
            "--context-length",
            mml,
        ]
        if conf["precision"] == "nvfp4":
            argv += ["--quantization", "modelopt_fp4"]
    elif engine == "trtllm":
        argv += [
            image,
            "trtllm-serve",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--tp_size",
            str(gpus),
            "--max_seq_len",
            mml,
        ]
    else:
        raise ValueError(f"unknown engine {engine!r}")
    return argv + list(conf.get("extra_args", []))


def dynamo_command(ctx: Ctx, conf: dict, bench_dir: Path) -> list[str]:
    argv = _docker_run(ctx, detach=True, name=SERVER) + _hf_mount(ctx)
    return argv + [
        "-v",
        f"{bench_dir}:/bench",
        ctx.cfg["images"]["dynamo-vllm"],
        "bash",
        "/bench/launch.sh",
    ]


def _skip_reason(ctx: Ctx, gpus: int, precision: str) -> str | None:
    if gpus > ctx.node.gpu_count:
        return f"needs {gpus} GPUs, node has {ctx.node.gpu_count}"
    if precision in BLACKWELL_ONLY and not ctx.node.blackwell:
        return f"{precision} needs Blackwell (sm_100+), node is {ctx.node.sm}"
    return None


def build_plan(ctx: Ctx) -> list[Step]:
    cfg, run, repo = ctx.cfg, ctx.run_dir, ctx.repo
    steps: list[Step] = []
    add = steps.append

    add(
        Step(
            "setup.host",
            "setup",
            "Install AIPerf and the Hugging Face CLI into this venv",
            2,
            "shell",
            [str(ctx.bin_dir / "python"), "-m", "pip", "install", "-q", *cfg["host_packages"]],
            timeout_min=15,
        )
    )
    add(
        Step(
            "setup.preflight",
            "setup",
            "Record the node; check Docker, disk and GPUs",
            1,
            "python",
            data={"fn": "preflight"},
        )
    )
    add(
        Step(
            "setup.prefetch",
            "setup",
            "Start pulling images and downloading models (background)",
            0.5,
            "background",
            ["bash", str(run / "prefetch.sh")],
        )
    )
    add(
        Step(
            "setup.bench-image",
            "setup",
            "Build the validation image (nccl-tests, nvbandwidth, DCGM)",
            10,
            "shell",
            bench_build_argv(ctx),
            timeout_min=45,
        )
    )

    val = cfg.get("validation", {})
    for suite in ("quick", "full"):
        if not val.get(suite):
            continue
        out = f"/out/validate/{suite}"
        script = (
            "nv-hostengine >/dev/null 2>&1 || true; "
            f"mkdir -p {out}; "
            f"bcp topology > {out}/topology.txt; "
            f"bcp topology --format mermaid > {out}/topology.mmd; "
            f"bcp collect --suite {suite} --out {out}/raw && "
            f"bcp validate --replay {out}/raw --suite {suite} --out {out}"
        )
        argv = _docker_run(ctx, privileged=True) + [
            "-v",
            f"{run}:/out",
            "-v",
            f"{repo}:/workspace/bcp:ro",
            "-w",
            "/workspace/bcp",
            ctx.bench_image,
            "bash",
            "-c",
            script,
        ]
        gate = suite == "quick"
        title = (
            "Quick validation: stops the day if the hardware is broken"
            if gate
            else "Full validation: DCGM level 3, NCCL to 16 GiB, nvbandwidth"
        )
        add(
            Step(
                f"validate.{suite}",
                "validate",
                title,
                8 if gate else 50,
                "validate",
                argv,
                optional=not gate,
                timeout_min=30 if gate else 120,
                data={"gate": gate, "dir": str(run / "validate" / suite)},
            )
        )

    add(
        Step(
            "setup.prefetch-wait",
            "setup",
            "Wait for downloads to finish",
            2,
            "wait",
            optional=True,
            timeout_min=90,
            data={"done": str(run / "prefetch.done")},
        )
    )

    inf = cfg.get("inference") or {}
    per_point = float(inf.get("est_min_per_point", 1.75))
    for conf in inf.get("configs", []):
        gpus, precision = config_gpus(conf), conf["precision"]
        model = inf["models"][precision]
        conc = concurrency_for(inf, gpus)
        layout = conf.get("layout")
        what = f"{ENGINE_NAMES[conf['engine']]} {precision.upper()} on {gpus} GPU" + (
            "s" if gpus > 1 else ""
        )
        if layout:
            what += " · " + " + ".join(
                f"{len(v)} {k}" + (f" (TP{max(v)})" if max(v) > 1 else "")
                for k, v in layout.items()
            )
        add(
            Step(
                f"inference.{conf['id']}",
                "inference",
                what,
                STARTUP_MIN[conf["engine"]] + per_point * len(conc),
                "serve",
                optional=True,
                timeout_min=30 + 6 * len(conc),
                skip=_skip_reason(ctx, gpus, precision),
                data={"conf": conf, "model": model, "gpus": gpus, "concurrency": conc},
            )
        )

    tr = cfg.get("training") or {}
    for r in tr.get("runs", []):
        out = f"/out/training/{r['id']}.json"
        train = [
            "torchrun",
            "--standalone",
            "--nproc-per-node",
            str(r["gpus"]),
            "training/bench_train.py",
            "--preset",
            r["preset"],
            "--precision",
            r["precision"],
            "--micro-batch",
            str(r.get("micro_batch", 2)),
            "--seq-len",
            str(tr["seq_len"]),
            "--warmup-steps",
            str(tr["warmup_steps"]),
            "--steps",
            str(tr["steps"]),
            "--id",
            r["id"],
            "--out",
            out,
        ]
        argv = _docker_run(ctx) + [
            "-e",
            f"CUDA_VISIBLE_DEVICES={_gpu_list(r['gpus'])}",
            "-v",
            f"{repo}:/workspace/bcp:ro",
            "-v",
            f"{run}:/out",
            "-w",
            "/workspace/bcp",
            cfg["images"]["pytorch"],
            *train,
        ]
        add(
            Step(
                f"training.{r['id']}",
                "training",
                f"{r['preset']} {r['precision'].upper()} on {r['gpus']} GPU"
                + ("s" if r["gpus"] > 1 else ""),
                2.5,
                "train",
                argv,
                optional=True,
                timeout_min=30,
                skip=_skip_reason(ctx, r["gpus"], r["precision"]),
                data={"out": str(run / "training" / f"{r['id']}.json")},
            )
        )

    add(
        Step(
            "report",
            "wrap-up",
            "Write summary.md (and plots if matplotlib is installed)",
            0.5,
            "python",
            data={"fn": "report"},
        )
    )
    add(
        Step(
            "package",
            "wrap-up",
            "Make a redacted copy to commit and a full tarball",
            1,
            "python",
            data={"fn": "package"},
        )
    )
    return steps


def bench_build_argv(ctx: Ctx) -> list[str]:
    versions = {c["name"]: c.get("pinned") for c in stack.load(ctx.repo / "matrix/stack.yaml")}
    args = [
        "docker",
        "build",
        "-t",
        ctx.bench_image,
        "-f",
        str(ctx.repo / "containers/bench/Dockerfile"),
    ]
    for arg, name in (
        ("CUDA_VERSION", "cuda"),
        ("NCCL_TESTS_VERSION", "nccl-tests"),
        ("NVBANDWIDTH_VERSION", "nvbandwidth"),
    ):
        if versions.get(name):
            args += ["--build-arg", f"{arg}={versions[name]}"]
    return args + [str(ctx.repo)]


def images_used(ctx: Ctx, steps: list[Step]) -> list[str]:
    images = []
    for s in steps:
        if s.skip:
            continue
        if s.kind == "serve":
            images.append(ctx.cfg["images"][s.data["conf"]["engine"]])
        elif s.kind == "train":
            images.append(ctx.cfg["images"]["pytorch"])
    return list(dict.fromkeys(images))


def models_used(steps: list[Step]) -> list[str]:
    return list(dict.fromkeys(s.data["model"] for s in steps if s.kind == "serve" and not s.skip))


def render_prefetch(ctx: Ctx, steps: list[Step]) -> str:
    run = ctx.run_dir
    hf = ctx.bin_dir / "hf"
    tokenizer = ctx.cfg.get("inference", {}).get("tokenizer")
    models = models_used(steps)
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by `bcp rental`: pull every image and download every model the day uses,",
        "# images and models in parallel. Failures are listed in prefetch.failed.",
        "set -u",
        f"export HF_HOME={shlex.quote(str(ctx.hf_home))}",
        f"FAILED={shlex.quote(str(run / 'prefetch.failed'))}",
        ': > "${FAILED}"',
        "",
        "(",
    ]
    for image in images_used(ctx, steps):
        q = shlex.quote(image)
        lines.append(
            f'  echo "== {image}"; docker pull -q {q} || echo "image {image}" >> "${{FAILED}}"'
        )
    lines += [f") > {shlex.quote(str(run / 'logs/prefetch-images.log'))} 2>&1 &", "", "("]
    for model in models:
        q = shlex.quote(model)
        lines.append(
            f'  echo "== {model}"; {shlex.quote(str(hf))} download {q} >/dev/null '
            f'|| echo "model {model}" >> "${{FAILED}}"'
        )
    if tokenizer and tokenizer not in models:
        q = shlex.quote(tokenizer)
        lines.append(
            f"  {shlex.quote(str(hf))} download {q} --include '*.json' '*.txt' "
            f'>/dev/null || echo "tokenizer {tokenizer}" >> "${{FAILED}}"'
        )
    lines += [
        f") > {shlex.quote(str(run / 'logs/prefetch-models.log'))} 2>&1 &",
        "",
        "wait",
        f"date -u +%FT%TZ > {shlex.quote(str(run / 'prefetch.done'))}",
        "",
    ]
    return "\n".join(lines)


def estimate(steps: list[Step]) -> float:
    return sum(s.est_min for s in steps if not s.skip)


def _hm(minutes: float) -> str:
    return f"{int(minutes // 60)}h{int(round(minutes % 60)):02d}"


def render_plan(ctx: Ctx, steps: list[Step], commands: bool = False) -> str:
    total = estimate(steps)
    lines = [
        f"Rental day plan{' (REHEARSAL)' if ctx.cfg['rehearse'] else ''}: {ctx.node.describe()}",
        f"Run directory: {ctx.run_dir}",
        "",
        f"{'#':>3}  {'STEP':<26} {'EST':>7}  WHAT",
    ]
    clock = 0.0
    for i, s in enumerate(steps, 1):
        if s.skip:
            lines.append(f"{i:>3}  {s.id:<26} {'skip':>7}  {s.title}  [{s.skip}]")
            continue
        clock += s.est_min
        lines.append(
            f"{i:>3}  {s.id:<26} {s.est_min:>5.1f}m  {s.title}"
            + ("  (stops the day on failure)" if s.data.get("gate") else "")
        )
        if commands:
            for cmd in step_commands(ctx, s):
                lines.append("       $ " + cmd)
    lines += [
        "",
        f"Estimated time: {_hm(total)}  ·  at ${ctx.rate:g}/h: about ${total / 60 * ctx.rate:,.0f}",
        f"Images: {', '.join(images_used(ctx, steps)) or 'none'}",
        f"Models: {', '.join(models_used(steps)) or 'none'}",
    ]
    return "\n".join(lines)


def step_commands(ctx: Ctx, s: Step) -> list[str]:
    if s.kind == "serve":
        conf = s.data["conf"]
        bench = ctx.run_dir / "inference" / conf["id"]
        start = (
            dynamo_command(ctx, conf, bench)
            if conf["engine"] == "dynamo-vllm"
            else engine_command(ctx, conf, s.data["model"])
        )
        sweep = sweep_env(ctx, s)
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in sweep.items() if k != "PATH")
        return [shlex.join(start), f"{env} inference/bench/aiperf-sweep.sh"]
    if s.kind == "python":
        return [f"(bcp internal: {s.data['fn']})"]
    if s.kind == "wait":
        return [f"(wait for {s.data['done']})"]
    return [shlex.join(s.argv)]


def sweep_env(ctx: Ctx, s: Step) -> dict[str, str]:
    inf = ctx.cfg["inference"]
    return {
        "URL": f"http://localhost:{PORT}",
        "MODEL": s.data["model"],
        "TOKENIZER": inf["tokenizer"],
        "ISL": str(inf["workload"]["isl"]),
        "OSL": str(inf["workload"]["osl"]),
        "CONCURRENCY": " ".join(map(str, s.data["concurrency"])),
        "OUT": str(ctx.run_dir / "inference" / s.data["conf"]["id"]),
        "HF_HOME": str(ctx.hf_home),
        "PATH": f"{ctx.bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
    }


def render_scripts(ctx: Ctx, steps: list[Step], out: Path) -> list[Path]:
    """Write the generated scripts (prefetch, Dynamo launchers) for review or shellcheck."""
    out.mkdir(parents=True, exist_ok=True)
    written = [out / "prefetch.sh"]
    written[0].write_text(render_prefetch(ctx, steps))
    for s in steps:
        if s.kind == "serve" and "layout" in s.data["conf"]:
            script, _ = render_dynamo_launch(
                s.data["model"], s.data["conf"]["layout"], ctx.cfg["inference"]["max_model_len"]
            )
            path = out / f"launch-{s.data['conf']['id']}.sh"
            path.write_text(script)
            written.append(path)
    return written


# ---------------------------------------------------------------------------------------
# Running


def gate_verdict(doc: dict) -> tuple[list[str], list[str]]:
    """(hard failures, soft failures) from a `bcp validate` results document."""
    hard, soft = [], []
    for c in doc.get("checks", []):
        if c["status"] != "fail":
            continue
        name = c["name"]
        is_hard = name.startswith(HARD_CHECK_PREFIXES) or name.endswith(".correctness")
        (hard if is_hard else soft).append(f"{name}: {c.get('measured', '')}")
    return hard, soft


class StepFailed(Exception):
    pass


class Runner:
    def __init__(self, ctx: Ctx, steps: list[Step], log: Callable[[str], None] = print) -> None:
        self.ctx, self.steps, self.log = ctx, steps, log
        self.state_path = ctx.run_dir / "state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {"started_at": _now(), "steps": {}}
        )

    # -- bookkeeping
    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n")

    def elapsed_min(self) -> float:
        started = datetime.fromisoformat(self.state["started_at"])
        return (datetime.now(timezone.utc) - started).total_seconds() / 60

    def _clock(self) -> str:
        m = self.elapsed_min()
        return f"{_hm(m)} elapsed, ~${m / 60 * self.ctx.rate:,.0f}"

    # -- main loop
    def run(self, resume: bool = False, only: set[str] | None = None) -> int:
        ctx = self.ctx
        (ctx.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (ctx.run_dir / "plan.json").write_text(
            json.dumps(
                {
                    "node": asdict(ctx.node),
                    "config": ctx.cfg,
                    "steps": [asdict(s) for s in self.steps],
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        (ctx.run_dir / "prefetch.sh").write_text(render_prefetch(ctx, self.steps))
        self._save()
        failures = []
        for i, step in enumerate(self.steps, 1):
            prev = self.state["steps"].get(step.id, {})
            if only and step.phase not in only and step.id not in only:
                continue
            if step.skip:
                self._record(step, "skipped", detail=step.skip)
                continue
            if resume and prev.get("status") == "ok":
                continue
            self.log(
                f"\n[{i}/{len(self.steps)} · {self._clock()}] {step.id}: {step.title}"
                f" (est {step.est_min:g} min)"
            )
            t0 = time.time()
            try:
                detail = getattr(self, f"_do_{step.kind}")(step) or ""
                self._record(step, "ok", t0, detail)
                if detail:
                    self.log(f"    {detail}")
            except StepFailed as exc:
                self._record(step, "failed", t0, str(exc))
                self.log(f"    FAILED: {exc}")
                failures.append(step.id)
                if not step.optional:
                    self.log(
                        f"\nStopped. Fix the problem, then: bcp rental run --resume "
                        f"--run-dir {ctx.run_dir}"
                    )
                    return 1
        self.log(
            f"\nDone · {self._clock()}"
            + (
                f" · {len(failures)} optional step(s) failed: {', '.join(failures)}"
                if failures
                else ""
            )
        )
        self.log(f"Summary: {ctx.run_dir / 'summary.md'}")
        self.log("Copy the results off the node, then TERMINATE THE INSTANCE.")
        return 0

    def _record(self, step: Step, status: str, t0: float | None = None, detail: str = "") -> None:
        entry = {"status": status, "finished": _now(), "detail": detail}
        if t0 is not None:
            entry["minutes"] = round((time.time() - t0) / 60, 1)
        self.state["steps"][step.id] = entry
        self._save()

    def _logfile(self, step: Step, suffix: str = "") -> Path:
        return self.ctx.run_dir / "logs" / f"{step.id}{suffix}.log"

    def _sh(
        self,
        argv: list[str],
        log: Path,
        timeout_min: float,
        env: dict[str, str] | None = None,
        ok: tuple[int, ...] = (0,),
    ) -> int:
        with log.open("a") as f:
            f.write(f"$ {shlex.join(argv)}\n")
            f.flush()
            try:
                code = subprocess.run(
                    argv,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_min * 60,
                    env={**os.environ, **(env or {})},
                ).returncode
            except subprocess.TimeoutExpired as exc:
                raise StepFailed(f"timed out after {timeout_min:g} min (log: {log})") from exc
            except FileNotFoundError as exc:
                raise StepFailed(f"{argv[0]} not found") from exc
        if code not in ok:
            raise StepFailed(f"exit {code}; last lines of {log}:\n" + _tail(log))
        return code

    # -- step kinds
    def _do_shell(self, step: Step) -> str:
        self._sh(step.argv, self._logfile(step), step.timeout_min)
        return ""

    def _do_background(self, step: Step) -> str:
        log = self._logfile(step).open("a")
        subprocess.Popen(step.argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return "running in the background; logs/prefetch-*.log"

    def _do_wait(self, step: Step) -> str:
        done = Path(step.data["done"])
        deadline = time.time() + step.timeout_min * 60
        while not done.exists():
            if time.time() > deadline:
                raise StepFailed(f"downloads not finished after {step.timeout_min:g} min")
            time.sleep(15)
        failed = self.ctx.run_dir / "prefetch.failed"
        missing = failed.read_text().split("\n") if failed.exists() else []
        missing = [m for m in missing if m.strip()]
        if missing:
            raise StepFailed("could not fetch: " + "; ".join(missing))
        return "all images and models are local"

    def _do_validate(self, step: Step) -> str:
        out = Path(step.data["dir"])
        code = self._sh(step.argv, self._logfile(step), step.timeout_min, ok=(0, 1))
        docs = sorted(out.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not docs:
            raise StepFailed(f"no results JSON in {out}")
        doc = json.loads(docs[-1].read_text())
        hard, soft = gate_verdict(doc)
        summary = f"verdict {doc['verdict'].upper()} ({docs[-1].name})"
        if soft:
            self.log("    below threshold (recorded, not fatal): " + "; ".join(soft))
        if hard and step.data.get("gate"):
            if self.ctx.cfg["rehearse"]:
                self.log("    REHEARSAL: these would stop a real day: " + "; ".join(hard))
            else:
                raise StepFailed(
                    "the node looks broken: "
                    + "; ".join(hard)
                    + ". Ask the provider for a replacement before spending more time on it."
                )
        return summary + ("" if code == 0 else " (some checks failed)")

    def _do_train(self, step: Step) -> str:
        log = self._logfile(step)
        try:
            self._sh(step.argv, log, step.timeout_min)
        except StepFailed:
            if "--parallel" in step.argv:
                raise
            # FSDP2 is the default; DDP is the simpler fallback if TE + FSDP2 misbehave.
            self.log("    FSDP2 run failed; retrying with DDP")
            self._sh([*step.argv, "--parallel", "ddp"], log, step.timeout_min)
        out = Path(step.data["out"])
        if not out.exists():
            raise StepFailed(f"no result written to {out}")
        doc = json.loads(out.read_text())
        return (
            f"{doc['tokens_per_s_per_gpu']:,.0f} tokens/s/GPU, {doc['tflops_per_gpu']} "
            f"TFLOP/s/GPU, MFU {doc['mfu']}"
        )

    def _do_serve(self, step: Step) -> str:
        ctx, conf = self.ctx, step.data["conf"]
        bench = ctx.run_dir / "inference" / conf["id"]
        bench.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": conf["id"],
            "engine": conf["engine"],
            "precision": conf["precision"],
            "model": step.data["model"],
            "gpus": step.data["gpus"],
            "layout": conf.get("layout"),
            "image": ctx.cfg["images"][conf["engine"]],
            "workload": ctx.cfg["inference"]["workload"],
            "concurrency": step.data["concurrency"],
            "started": _now(),
            "status": "starting",
            "order": self.steps.index(step),
        }
        write_meta = lambda: (bench / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")  # noqa: E731
        write_meta()
        health_ports: list[int] = []
        if "layout" in conf:
            script, health_ports = render_dynamo_launch(
                step.data["model"], conf["layout"], ctx.cfg["inference"]["max_model_len"]
            )
            (bench / "launch.sh").write_text(script)
            start = dynamo_command(ctx, conf, bench)
        else:
            start = engine_command(ctx, conf, step.data["model"])
        log = self._logfile(step)
        subprocess.run(["docker", "rm", "-f", SERVER], capture_output=True)
        try:
            self._sh(start, log, 30)  # docker pulls here if the prefetch missed the image
            waited = wait_ready(
                step.data["model"],
                health_ports,
                float(ctx.cfg["inference"].get("server_timeout_min", 30)),
            )
            self.log(f"    server ready after {waited:.0f} s")
            meta["status"] = "benchmarking"
            write_meta()
            self._sh(
                ["bash", str(ctx.repo / "inference/bench/aiperf-sweep.sh")],
                log,
                step.timeout_min,
                env=sweep_env(ctx, step),
            )
            meta["status"] = "ok"
        except StepFailed:
            meta["status"] = "failed"
            raise
        finally:
            meta["finished"] = _now()
            write_meta()
            with (bench / "server.log").open("w") as f:
                subprocess.run(["docker", "logs", SERVER], stdout=f, stderr=subprocess.STDOUT)
            subprocess.run(["docker", "rm", "-f", SERVER], capture_output=True)
        return f"swept concurrency {step.data['concurrency']}"

    def _do_python(self, step: Step) -> str:
        return getattr(self, f"_fn_{step.data['fn']}")()

    # -- in-process steps
    def _fn_preflight(self) -> str:
        return preflight(self.ctx, self.steps, self.log)

    def _fn_report(self) -> str:
        from bcp import bench

        path = bench.write_summary(self.ctx.run_dir)
        return f"wrote {path}"

    def _fn_package(self) -> str:
        from bcp import bench

        share, tarball = bench.package(self.ctx.run_dir)
        m = self.elapsed_min()
        self.log(f"    redacted copy (safe to commit): {share}")
        self.log(f"    full tarball (keep private):    {tarball}")
        self.log(f"    copy them off with e.g.:  scp -r <user>@<node>:{share} ./runs/")
        return f"{_hm(m)} so far ≈ ${m / 60 * self.ctx.rate:,.0f}"


def wait_ready(model: str, health_ports: list[int], timeout_min: float) -> float:
    """Wait until the server container runs, every worker is healthy and a request succeeds."""
    t0 = time.time()
    deadline = t0 + timeout_min * 60
    body = json.dumps(
        {"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "Say hello."}]}
    ).encode()
    pending = list(health_ports)
    while time.time() < deadline:
        running = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", SERVER],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if running != "true":
            logs = subprocess.run(
                ["docker", "logs", "--tail", "30", SERVER], capture_output=True, text=True
            )
            raise StepFailed("server exited during startup:\n" + logs.stdout + logs.stderr)
        pending = [p for p in pending if not _http_ok(f"http://localhost:{p}/health")]
        if not pending and _http_ok(f"http://localhost:{PORT}/v1/chat/completions", body):
            return time.time() - t0
        time.sleep(10)
    raise StepFailed(
        f"server not ready after {timeout_min:g} min"
        + (f" (workers on ports {pending} not healthy)" if pending else "")
    )


def _http_ok(url: str, body: bytes | None = None) -> bool:
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"} if body else {}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def preflight(ctx: Ctx, steps: list[Step], log: Callable[[str], None]) -> str:
    host_dir = ctx.run_dir / "host"
    host_dir.mkdir(parents=True, exist_ok=True)
    problems, warnings = [], []

    def run(argv: list[str]) -> subprocess.CompletedProcess | None:
        try:
            return subprocess.run(argv, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            return None

    for name, argv in {
        "nvidia-smi-q.txt": ["nvidia-smi", "-q"],
        "nvidia-smi-topo.txt": ["nvidia-smi", "topo", "-m"],
        "docker-version.txt": ["docker", "version"],
    }.items():
        res = run(argv)
        (host_dir / name).write_text(res.stdout + res.stderr if res else "not available\n")

    if not ctx.node.detected:
        problems.append("nvidia-smi not found or not working")
    info = run(["docker", "info", "--format", "{{json .}}"])
    if not info or info.returncode != 0:
        problems.append(
            "`docker info` failed; if it is a permissions error: "
            "sudo usermod -aG docker $USER, then log out and back in"
        )
        docker_root = None
    else:
        details = json.loads(info.stdout)
        docker_root = details.get("DockerRootDir")
        if "nvidia" not in json.dumps(details.get("Runtimes", {})):
            warnings.append("no nvidia runtime in `docker info`; --gpus may still work via CDI")

    need = float(ctx.cfg.get("disk_free_gb_min", 400))
    for label, path in (
        ("run dir", ctx.run_dir),
        ("HF cache", ctx.hf_home),
        ("Docker", Path(docker_root) if docker_root else None),
    ):
        if path is None:
            continue
        p = path
        while not p.exists():
            p = p.parent
        free = shutil.disk_usage(p).free / 1e9
        if free < need:
            problems.append(f"{label} ({path}) has {free:.0f} GB free, the day needs ~{need:.0f}")

    if ctx.node.gpu_count == 8 and ctx.node.compute_cap in (9.0, 10.0):
        fm = run(["systemctl", "is-active", "nvidia-fabricmanager"])
        if fm and fm.stdout.strip() != "active":
            warnings.append(
                "nvidia-fabricmanager is not active: NVSwitch systems need it "
                "for NVLink (the quick validation will confirm)"
            )
    apps = run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"])
    if apps and apps.stdout.strip():
        warnings.append("processes are already using the GPUs:\n" + apps.stdout.strip())
    if (
        ctx.node.driver
        and ctx.node.driver.split(".")[0].isdigit()
        and int(ctx.node.driver.split(".")[0]) < 580
    ):
        warnings.append(f"driver {ctx.node.driver}: CUDA 13 containers want R580 or newer")
    if not (os.environ.get("TMUX") or os.environ.get("STY")):
        warnings.append(
            "not inside tmux/screen: a dropped SSH session will stop the day "
            "(recover with --resume)"
        )
    hf_token = bool(os.environ.get("HF_TOKEN"))
    doc = {
        "node": asdict(ctx.node),
        "host": os.uname().nodename,
        "checked_at": _now(),
        "problems": problems,
        "warnings": warnings,
        "hf_token_set": hf_token,
    }
    (host_dir / "preflight.json").write_text(json.dumps(doc, indent=2) + "\n")
    for w in warnings:
        log(f"    warning: {w}")
    if problems:
        raise StepFailed("; ".join(problems))
    return f"{ctx.node.describe()}, driver {ctx.node.driver}"


def _tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_dir(repo: Path, rehearse: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repo / "runs" / f"{'rehearsal' if rehearse else 'day'}-{stamp}"


def make_ctx(
    config: Path, rehearse: bool, run_dir: Path | None, rate: float | None, node: Node | None
) -> Ctx:
    repo = config.resolve().parent.parent
    cfg = load_day(config, repo / "matrix/stack.yaml", rehearse)
    if node is None:
        node = detect_node()
    if node is None:
        expect = cfg.get("expect_node") or {}
        node = Node(
            expect.get("gpu_name", "NVIDIA B200"),
            int(expect.get("gpu_count", 8)),
            float(expect.get("compute_cap", 10.0)),
            detected=False,
        )
    hf_home = Path(os.path.expanduser(os.environ.get("HF_HOME") or cfg["hf_home"]))
    return Ctx(
        cfg=cfg,
        node=node,
        repo=repo,
        run_dir=(run_dir or default_run_dir(repo, rehearse)).resolve(),
        rate=float(rate if rate is not None else cfg.get("rate_usd_per_hour", 0)),
        hf_home=hf_home,
        bin_dir=Path(sys.executable).parent,
    )
