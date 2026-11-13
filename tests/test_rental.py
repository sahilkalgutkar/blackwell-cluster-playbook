import json
import sys
from pathlib import Path

import pytest

from bcp import rental
from bcp.cli import main

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "rental" / "day.yaml"
B200 = rental.Node("NVIDIA B200", 8, 10.0, "580.95.05")
H100x2 = rental.Node("NVIDIA H100 80GB HBM3", 2, 9.0, "580.95.05")


def ctx_for(node, tmp_path, rehearse=False):
    return rental.make_ctx(CONFIG, rehearse, tmp_path / "run", 65, node)


def test_images_and_packages_follow_the_stack_matrix(tmp_path):
    ctx = ctx_for(B200, tmp_path)
    images = ctx.cfg["images"]
    assert images["vllm"].startswith("vllm/vllm-openai:v")
    assert not images["trtllm"].split(":")[1].startswith("v")  # NGC tags have no "v"
    assert not images["dynamo-vllm"].split(":")[1].startswith("v")
    assert any(p.startswith("aiperf==") and "==v" not in p for p in ctx.cfg["host_packages"])


def test_resolve_templates():
    versions = {"dynamo": "v1.6.2", "cuda": "13.5.0"}
    assert rental.resolve_templates("x:{dynamo:bare}", versions) == "x:1.6.2"
    assert rental.resolve_templates("x:{dynamo}", versions) == "x:v1.6.2"
    with pytest.raises(KeyError, match="nvidia-driver"):
        rental.resolve_templates("{nvidia-driver}", {"nvidia-driver": None})


def test_full_day_on_8x_b200_runs_everything_in_a_few_hours(tmp_path):
    ctx = ctx_for(B200, tmp_path)
    steps = rental.build_plan(ctx)
    ids = [s.id for s in steps]
    assert not [s.id for s in steps if s.skip]
    assert ids[:4] == ["setup.host", "setup.preflight", "setup.prefetch", "setup.bench-image"]
    assert (
        ids.index("validate.quick")
        < ids.index("validate.full")
        < ids.index("setup.prefetch-wait")
        < ids.index("inference.vllm-fp8")
    )
    assert ids[-2:] == ["report", "package"]
    gate = next(s for s in steps if s.id == "validate.quick")
    assert gate.data["gate"] and not gate.optional
    assert all(s.optional for s in steps if s.phase in ("inference", "training"))
    assert 3 * 60 < rental.estimate(steps) < 6 * 60


def test_rehearsal_on_two_hopper_gpus_skips_what_the_box_cannot_do(tmp_path):
    ctx = ctx_for(H100x2, tmp_path, rehearse=True)
    steps = rental.build_plan(ctx)
    skipped = {s.id: s.skip for s in steps if s.skip}
    assert "Blackwell" in skipped["inference.vllm-nvfp4"]
    assert "Blackwell" in skipped["training.1b-mxfp8"]
    assert "validate.full" not in [s.id for s in steps]
    assert rental.models_used(steps) == ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-FP8"]
    assert rental.estimate(steps) < 2 * 60


def test_main_config_on_small_node_skips_big_layouts(tmp_path):
    steps = rental.build_plan(ctx_for(H100x2, tmp_path))
    skipped = {s.id: s.skip for s in steps if s.skip}
    assert skipped["inference.dynamo-2p2d"] == "needs 4 GPUs, node has 2"
    assert skipped["training.8b-bf16"] == "needs 8 GPUs, node has 2"
    assert "inference.vllm-fp8" not in skipped


def test_dynamo_launch_assigns_distinct_gpus_and_ports():
    script, ports = rental.render_dynamo_launch(
        "Qwen/Qwen3-32B-FP8", {"prefill": [1, 1], "decode": [2]}, 4096
    )
    assert ports == [8081, 8082, 8083]
    assert "CUDA_VISIBLE_DEVICES=0 " in script and "CUDA_VISIBLE_DEVICES=1 " in script
    assert "CUDA_VISIBLE_DEVICES=2,3 " in script
    assert script.count("--disaggregation-mode prefill") == 2
    assert script.count("kv_consumer") == 1
    assert "--tensor-parallel-size 2" in script
    assert script.count("--discovery-backend file") == 2  # frontend and COMMON
    nixl = [int(w.split("=")[1]) for w in script.split() if w.startswith("VLLM_NIXL")]
    assert len(set(nixl)) == 3
    assert all(b - a >= 8 for a, b in zip(nixl, nixl[1:], strict=False))


def test_engine_commands(tmp_path):
    ctx = ctx_for(B200, tmp_path)
    sg = rental.engine_command(ctx, {"engine": "sglang", "precision": "nvfp4", "gpus": 1}, "m")
    assert sg[sg.index("--quantization") + 1] == "modelopt_fp4"
    trt = rental.engine_command(ctx, {"engine": "trtllm", "precision": "fp8", "gpus": 2}, "m")
    assert trt[trt.index("--tp_size") + 1] == "2" and "CUDA_VISIBLE_DEVICES=0,1" in trt
    vl = rental.engine_command(ctx, {"engine": "vllm", "precision": "fp8", "gpus": 1}, "m")
    assert vl[vl.index("--entrypoint") + 1] == "vllm" and "HF_HUB_OFFLINE=1" in vl


def test_concurrency_picks_the_table_for_the_gpu_count(tmp_path):
    inf = {"concurrency": {1: [1, 2], 4: [8]}}
    assert rental.concurrency_for(inf, 1) == [1, 2]
    assert rental.concurrency_for(inf, 2) == [1, 2]
    assert rental.concurrency_for(inf, 8) == [8]


def test_gate_stops_only_for_broken_hardware():
    doc = {
        "checks": [
            {"name": "topology.nvlink_symmetry", "status": "fail", "measured": "GPU5 NV16"},
            {"name": "nccl.all_reduce.busbw", "status": "fail", "measured": "600 GB/s"},
            {"name": "nccl.alltoall.correctness", "status": "fail", "measured": "#wrong 3"},
            {"name": "dcgm.diag", "status": "pass"},
        ]
    }
    hard, soft = rental.gate_verdict(doc)
    assert [h.split(":")[0] for h in hard] == [
        "topology.nvlink_symmetry",
        "nccl.alltoall.correctness",
    ]
    assert soft == ["nccl.all_reduce.busbw: 600 GB/s"]


def _step(sid, code, optional=False):
    argv = [sys.executable, "-c", f"import sys; sys.exit({code})"]
    return rental.Step(sid, "test", sid, 1, "shell", argv, optional=optional)


def test_runner_stops_on_failure_and_resumes(tmp_path):
    ctx = ctx_for(B200, tmp_path)
    steps = [
        _step("a", 0),
        _step("b", 0, optional=True),
        _step("b2", 3, optional=True),
        _step("c", 1),
        _step("d", 0),
    ]
    log = []
    assert rental.Runner(ctx, steps, log.append).run() == 1
    state = json.loads((ctx.run_dir / "state.json").read_text())["steps"]
    assert state["a"]["status"] == "ok" and state["b2"]["status"] == "failed"
    assert state["c"]["status"] == "failed" and "d" not in state
    assert any("--resume" in line for line in log)

    steps[3] = _step("c", 0)
    log.clear()
    assert rental.Runner(ctx, steps, log.append).run(resume=True) == 0
    ran = [line.split("] ")[1].split(":")[0] for line in log if line.startswith("\n[")]
    assert ran == ["b2", "c", "d"]  # a and b already succeeded


def test_cli_plan_renders_scripts(tmp_path, capsys):
    out = tmp_path / "rendered"
    assert (
        main(
            [
                "rental",
                "plan",
                "--config",
                str(CONFIG),
                "--gpus",
                "8",
                "--render-dir",
                str(out),
                "--commands",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "Estimated time" in text and "aiperf-sweep.sh" in text
    names = sorted(p.name for p in out.iterdir())
    assert "prefetch.sh" in names and "launch-dynamo-2p2d.sh" in names


def test_cli_run_refuses_without_gpus(monkeypatch, capsys):
    monkeypatch.setattr(rental, "detect_node", lambda: None)
    assert main(["rental", "run", "--config", str(CONFIG)]) == 2
    assert "rental plan" in capsys.readouterr().err
