"""The FLOP and MFU arithmetic in training/bench_train.py, without needing torch."""

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "bench_train", Path(__file__).resolve().parent.parent / "training" / "bench_train.py"
)
bt = importlib.util.module_from_spec(spec)
sys.modules["bench_train"] = bt  # dataclasses look the module up while it is loading
spec.loader.exec_module(bt)


def test_8b_preset_matches_llama_3_1_8b_parameter_count():
    shape = bt.ModelShape(**bt.PRESETS["8b"])
    assert shape.total_params() == pytest.approx(8.03e9, rel=0.01)
    assert shape.head_dim == 128


def test_flops_per_token_is_6n_plus_attention():
    shape = bt.ModelShape(**bt.PRESETS["8b"])
    n = shape.matmul_params()
    attn = 12 * 32 * 32 * 128 * 4096
    assert shape.flops_per_token(4096) == 6 * n + attn


def test_peak_lookup_prefers_the_specific_name():
    assert bt.peak_tflops("NVIDIA GB200", "bf16") == (2500, 2500)
    assert bt.peak_tflops("NVIDIA B200", "mxfp8") == (2250, 4500)
    assert bt.peak_tflops("NVIDIA H100 80GB HBM3", "nvfp4") == (989, None)
    assert bt.peak_tflops("Some GPU", "bf16") == (None, None)


def test_summarize():
    shape = bt.ModelShape(**bt.PRESETS["debug"])
    out = bt.summarize(
        [0.5, 0.4, 0.6],
        tokens_per_step=4000,
        world=2,
        shape=shape,
        seq_len=64,
        peak_bf16=1.0,
        peak_prec=None,
    )
    assert out["step_time_s_median"] == 0.5
    assert out["tokens_per_s"] == 8000 and out["tokens_per_s_per_gpu"] == 4000
    tflops = 8000 * shape.flops_per_token(64) / 2 / 1e12
    assert out["tflops_per_gpu"] == round(tflops, 1)
    assert out["mfu"] == round(tflops / 1.0, 4) and out["mfu_at_precision_peak"] is None
