import json
import tarfile
from pathlib import Path

from bcp import bench


def aiperf_doc(tps, ttft_p50, ttft_p99, itl_p50):
    # Shape documented in aiperf docs/reference/json-export-schema.md
    return {
        "schema_version": "1.5",
        "output_token_throughput": {"unit": "tokens/sec", "avg": tps},
        "request_throughput": {"unit": "requests/sec", "avg": tps / 256},
        "time_to_first_token": {
            "unit": "ms",
            "avg": ttft_p50,
            "p50": ttft_p50,
            "p99": ttft_p99,
            "count": 10,
        },
        "inter_token_latency": {
            "unit": "ms",
            "avg": itl_p50,
            "p50": itl_p50,
            "p99": itl_p50 * 2,
            "count": 10,
        },
        "request_count": {"unit": "requests", "avg": 64.0},
    }


def write_config(run: Path, cid, engine, precision, gpus, points, layout=None):
    d = run / "inference" / cid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "id": cid,
                "engine": engine,
                "precision": precision,
                "gpus": gpus,
                "layout": layout,
                "model": "m",
                "image": "i",
                "status": "ok",
                "workload": {"isl": 2048, "osl": 256},
            }
        )
    )
    for c, (tps, p50, p99, itl) in points.items():
        sub = d / f"c{c}" / "nested"  # AIPerf may nest its exports
        sub.mkdir(parents=True)
        (sub / "profile_export_aiperf.json").write_text(json.dumps(aiperf_doc(tps, p50, p99, itl)))


def training_doc(rid, preset, precision, gpus, tps_per_gpu, mfu):
    return {
        "id": rid,
        "preset": preset,
        "precision": precision,
        "parallel": "fsdp2",
        "gpus": gpus,
        "seq_len": 4096,
        "steps": 40,
        "warmup_steps": 10,
        "micro_batch": 2,
        "tokens_per_s_per_gpu": tps_per_gpu,
        "tflops_per_gpu": 900.0,
        "mfu": mfu,
        "peak_mem_gb": 100.0,
        "model": {"params": 8.0e9},
    }


def test_parse_aiperf_per_gpu_and_interactivity():
    p = bench.parse_aiperf(aiperf_doc(8000, 300, 900, 20.0), 64, 4)
    assert p["output_tps_per_gpu"] == 2000
    assert p["tps_per_user"] == 50
    assert p["ttft_p99_ms"] == 900 and p["errors"] == 0


def test_inference_summary_picks_sla_point_and_compares(tmp_path):
    run = tmp_path / "day"
    write_config(
        run,
        "vllm-fp8",
        "vllm",
        "fp8",
        1,
        {8: (3000, 200, 400, 10), 64: (9000, 900, 1800, 25), 128: (10000, 2500, 5000, 40)},
    )
    write_config(
        run,
        "vllm-nvfp4",
        "vllm",
        "nvfp4",
        1,
        {8: (4000, 150, 300, 8), 128: (15000, 2000, 4000, 30)},
    )
    write_config(
        run,
        "agg",
        "dynamo-vllm",
        "fp8",
        4,
        {64: (20000, 500, 1500, 20)},
        layout={"agg": [1, 1, 1, 1]},
    )
    write_config(
        run,
        "2p2d",
        "dynamo-vllm",
        "fp8",
        4,
        {64: (26000, 400, 1200, 18)},
        layout={"prefill": [1, 1], "decode": [1, 1]},
    )
    configs = bench.load_inference(run)
    fp8 = next(c for c in configs if c["id"] == "vllm-fp8")
    assert bench.peak(fp8["points"])["concurrency"] == 128
    assert bench.best_under_sla(fp8["points"], 2000)["concurrency"] == 64

    md = bench.inference_markdown(configs, 2000)
    assert "vllm: NVFP4 vs FP8 peak throughput per GPU: **1.50x**" in md
    assert "`2p2d` vs `agg` (best under the TTFT limit): **1.30x**" in md
    assert "9,000 @ c64" in md


def test_training_summary_scaling_and_precision(tmp_path):
    runs = [
        training_doc("s1", "3b", "bf16", 1, 10000, 0.45),
        training_doc("s8", "3b", "bf16", 8, 9200, 0.41),
        training_doc("b", "8b", "bf16", 8, 4000, 0.40),
        training_doc("f", "8b", "mxfp8", 8, 5600, 0.56),
    ]
    md = bench.training_markdown(runs)
    assert "92% of 1 GPU" in md
    assert "8b on 8 GPUs: MXFP8 vs BF16 **1.40x**" in md
    assert "45.0%" in md


def test_redactor_removes_identifiers_but_keeps_fabric_semantics():
    text = (
        "GPU UUID : GPU-1a2b3c4d-1111-2222-3333-444455556666\n"
        "    Serial Number                     : 1654123456789\n"
        "        ClusterUUID                   : 9d1f2a3b-4c5d-6e7f-8091-a2b3c4d5e6f7\n"
        "        ClusterUUID                   : 00000000-0000-0000-0000-000000000000\n"
        "host gpu-node-17.example.internal at 10.20.30.40 mac 0c:42:a1:00:11:22\n"
        "CUDA Version: 13.4.3.001, bind 127.0.0.1, short name gpu-node-17\n"
    )
    out = bench.Redactor(["gpu-node-17.example.internal"]).text(text)
    assert "1a2b3c4d" not in out and "GPU-REDACTED" in out
    assert "1654123456789" not in out and "Serial Number                     : REDACTED" in out
    assert "9d1f2a3b" not in out and "00000000-0000-0000-0000-000000000001" in out
    assert "00000000-0000-0000-0000-000000000000" in out  # "not in a fabric" stays visible
    assert "10.20.30.40" not in out and "127.0.0.1" in out and "13.4.3.001" in out
    assert "0c:42:a1" not in out and "gpu-node-17" not in out and "node01" in out


def test_package_writes_redacted_share_and_tarball(tmp_path):
    run = tmp_path / "day-1"
    (run / "host").mkdir(parents=True)
    (run / "host" / "preflight.json").write_text(json.dumps({"host": "gpu-node-17"}))
    (run / "validate" / "quick").mkdir(parents=True)
    (run / "validate" / "quick" / "gpu-node-17-2026.json").write_text(
        json.dumps({"run": {"host": "gpu-node-17"}, "note": "Serial Number : 99"})
    )
    share, tarball = bench.package(run)
    files = [p.relative_to(share).as_posix() for p in share.rglob("*") if p.is_file()]
    assert "validate/quick/node01-2026.json" in files
    assert "gpu-node-17" not in (share / "validate/quick/node01-2026.json").read_text()
    with tarfile.open(tarball) as tar:
        assert "day-1/validate/quick/gpu-node-17-2026.json" in tar.getnames()


def test_write_summary_handles_a_partial_run(tmp_path):
    run = tmp_path / "day"
    run.mkdir()
    (run / "state.json").write_text(
        json.dumps(
            {
                "started_at": "2026-11-20T09:00:00+00:00",
                "steps": {"inference.trtllm-fp8": {"status": "failed", "detail": "exit 1\nmore"}},
            }
        )
    )
    out = bench.write_summary(run, plot=False)
    text = out.read_text()
    assert "_No inference runs._" in text and "`inference.trtllm-fp8` | failed | exit 1 |" in text
