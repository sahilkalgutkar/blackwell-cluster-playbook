#!/usr/bin/env python3
"""Training throughput for a Llama-style model on one node: tokens/s, TFLOP/s and MFU.

    torchrun --standalone --nproc-per-node 8 training/bench_train.py \
        --preset 8b --precision mxfp8 --micro-batch 2 --seq-len 4096 --out runs/8b-mxfp8.json

Synthetic tokens, so nothing to download. Data parallel with FSDP2 (default) or DDP; linear
layers come from Transformer Engine when it is installed, so BF16, FP8 (per-tensor current
scaling), MXFP8 and NVFP4 runs differ only in the quantization recipe.

MFU follows the torchtitan/PaLM convention: 6 x non-embedding parameters per token, plus
12 x layers x heads x head_dim x seq_len for attention counted as full (non-causal). Causal
kernels skip about half of that attention work, so MFU reads slightly high; the same formula
is used for every run, which is what matters for comparisons. Peaks are dense (no sparsity)
figures from NVIDIA datasheets, always BF16 for `mfu`, so FP8 runs can exceed what BF16 can
reach; `mfu_at_precision_peak` uses the FP8/FP4 peak instead.

The FLOP and peak tables are importable without torch (tests run them on any machine).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from contextlib import nullcontext, suppress
from dataclasses import asdict, dataclass

PRESETS: dict[str, dict] = {
    # Dimensions match Llama 3.x (untied embeddings everywhere, for a uniform FLOP count).
    "debug": dict(dim=256, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=512, vocab=2048),
    "1b": dict(dim=2048, n_layers=16, n_heads=32, n_kv_heads=8, ffn_dim=8192, vocab=128256),
    "3b": dict(dim=3072, n_layers=28, n_heads=24, n_kv_heads=8, ffn_dim=8192, vocab=128256),
    "8b": dict(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, ffn_dim=14336, vocab=128256),
}

# Dense TFLOP/s per GPU. Order matters: the first substring found in the device name wins.
PEAK_TFLOPS: list[tuple[str, dict[str, float]]] = [
    ("GB300", {"bf16": 2500, "fp8": 5000, "fp4": 15000}),
    ("B300", {"bf16": 2250, "fp8": 4500, "fp4": 13500}),
    ("GB200", {"bf16": 2500, "fp8": 5000, "fp4": 10000}),
    ("B200", {"bf16": 2250, "fp8": 4500, "fp4": 9000}),
    ("H200", {"bf16": 989, "fp8": 1979}),
    ("H100", {"bf16": 989, "fp8": 1979}),
    ("A100", {"bf16": 312}),
]

PRECISIONS = ("bf16", "fp8", "fp8-delayed", "mxfp8", "nvfp4")
PEAK_KEY = {"bf16": "bf16", "fp8": "fp8", "fp8-delayed": "fp8", "mxfp8": "fp8", "nvfp4": "fp4"}


@dataclass
class ModelShape:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    ffn_dim: int
    vocab: int

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def matmul_params(self) -> int:
        """Parameters that take part in matmuls: every linear layer, including the output head."""
        hd = self.head_dim
        attn = self.dim * (self.n_heads + 2 * self.n_kv_heads) * hd + self.n_heads * hd * self.dim
        mlp = self.dim * 2 * self.ffn_dim + self.ffn_dim * self.dim
        return self.n_layers * (attn + mlp) + self.dim * self.vocab

    def total_params(self) -> int:
        norms = (2 * self.n_layers + 1) * self.dim
        return self.matmul_params() + self.vocab * self.dim + norms

    def flops_per_token(self, seq_len: int) -> float:
        attention = 12 * self.n_layers * self.n_heads * self.head_dim * seq_len
        return 6 * self.matmul_params() + attention


def peak_tflops(device_name: str, precision: str) -> tuple[float | None, float | None]:
    """(BF16 peak, peak at this run's precision) for the device, or (None, None) if unknown."""
    for key, peaks in PEAK_TFLOPS:
        if key in device_name.upper():
            return peaks.get("bf16"), peaks.get(PEAK_KEY[precision])
    return None, None


def summarize(
    step_times: list[float],
    *,
    tokens_per_step: int,
    world: int,
    shape: ModelShape,
    seq_len: int,
    peak_bf16: float | None,
    peak_prec: float | None,
) -> dict:
    step = statistics.median(step_times)
    tps = tokens_per_step / step
    tflops_per_gpu = tps * shape.flops_per_token(seq_len) / world / 1e12

    def frac(peak: float | None) -> float | None:
        return round(tflops_per_gpu / peak, 4) if peak else None

    return {
        "step_time_s_median": round(step, 5),
        "step_time_s_p90": round(sorted(step_times)[int(0.9 * (len(step_times) - 1))], 5),
        "tokens_per_s": round(tps, 1),
        "tokens_per_s_per_gpu": round(tps / world, 1),
        "tflops_per_gpu": round(tflops_per_gpu, 1),
        "mfu": frac(peak_bf16),
        "mfu_at_precision_peak": frac(peak_prec),
        "peak_tflops_bf16": peak_bf16,
        "peak_tflops_precision": peak_prec,
    }


# ---------------------------------------------------------------------------------------
# Everything below needs torch.


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = args.device == "cuda"
    if cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    distributed = world > 1 or "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl" if cuda else "gloo", device_id=device if cuda else None)

    te = None
    if args.linear in ("auto", "te") and cuda:
        try:
            import transformer_engine.pytorch as te  # noqa: F811
        except ImportError:
            if args.linear == "te":
                raise
    if args.precision != "bf16" and te is None:
        raise SystemExit(f"--precision {args.precision} needs Transformer Engine on a GPU")

    shape = ModelShape(**PRESETS[args.preset])
    torch.manual_seed(1234)
    model = build_model(shape, args.seq_len, te, device, args.activation_checkpointing)
    if args.compile:
        for block in model.layers:
            block.compile()
    model = wrap(model, args.parallel, world, cuda, distributed, args.precision)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0, foreach=True)
    recipe = make_recipe(args.precision) if te else None
    group = dist.group.WORLD if distributed else None

    def quant_ctx():
        if recipe is None:
            return nullcontext()
        autocast = getattr(te, "autocast", None)
        if autocast is not None:
            return autocast(enabled=True, recipe=recipe, amax_reduction_group=group)
        return te.fp8_autocast(enabled=True, fp8_recipe=recipe, fp8_group=group)

    # FSDP2 casts parameters to BF16 itself; DDP (and NVFP4, which wants BF16 in both passes)
    # use torch autocast. On CPU everything stays FP32.
    use_amp = cuda and (args.parallel == "ddp" or args.precision == "nvfp4")

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    gen = torch.Generator(device=device).manual_seed(rank)
    tokens = torch.randint(
        0, shape.vocab, (args.micro_batch, args.seq_len + 1), device=device, generator=gen
    )
    inputs, labels = tokens[:, :-1], tokens[:, 1:]

    def sync():
        if cuda:
            torch.cuda.synchronize()

    losses, times = [], []
    total = args.warmup_steps + args.steps
    for step in range(total):
        sync()
        t0 = time.perf_counter()
        with amp(), quant_ctx():
            logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, shape.vocab), labels.reshape(-1)
        )
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        sync()
        dt = time.perf_counter() - t0
        losses.append(loss.item())
        if step >= args.warmup_steps:
            times.append(dt)
        if rank == 0 and (step < 2 or (step + 1) % 10 == 0 or step == total - 1):
            print(
                f"step {step + 1}/{total}  {dt * 1000:8.1f} ms  loss {losses[-1]:.4f}", flush=True
            )

    if not all(math.isfinite(x) for x in losses):
        print("non-finite loss", file=sys.stderr)
    peak_bf16, peak_prec = (None, None)
    device_name = torch.cuda.get_device_name(device) if cuda else platform.processor() or "cpu"
    if cuda:
        peak_bf16, peak_prec = peak_tflops(device_name, args.precision)
        if args.peak_tflops_bf16:
            peak_bf16 = args.peak_tflops_bf16
    tokens_per_step = args.micro_batch * args.seq_len * world
    result = {
        "schema_version": 1,
        "id": args.id or f"{args.preset}-{args.precision}-{world}gpu",
        "preset": args.preset,
        "precision": args.precision,
        "parallel": args.parallel,
        "linear": "te" if te else "torch",
        "compile": args.compile,
        "activation_checkpointing": args.activation_checkpointing,
        "gpus": world,
        "seq_len": args.seq_len,
        "micro_batch": args.micro_batch,
        "tokens_per_step": tokens_per_step,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "model": {
            **asdict(shape),
            "params": shape.total_params(),
            "matmul_params": shape.matmul_params(),
            "flops_per_token": shape.flops_per_token(args.seq_len),
        },
        **summarize(
            times,
            tokens_per_step=tokens_per_step,
            world=world,
            shape=shape,
            seq_len=args.seq_len,
            peak_bf16=peak_bf16,
            peak_prec=peak_prec,
        ),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if cuda else None,
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "losses_finite": all(math.isfinite(x) for x in losses),
        "device_name": device_name,
        "versions": versions(torch, te),
    }
    if rank == 0:
        text = json.dumps(result, indent=2) + "\n"
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                f.write(text)
        print(
            f"{result['id']}: {result['tokens_per_s_per_gpu']:,.0f} tokens/s/GPU, "
            f"{result['tflops_per_gpu']} TFLOP/s/GPU, MFU {result['mfu']}",
            flush=True,
        )
    if distributed:
        dist.destroy_process_group()
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--preset", choices=sorted(PRESETS), default="8b")
    p.add_argument("--precision", choices=PRECISIONS, default="bf16")
    p.add_argument("--parallel", choices=("fsdp2", "ddp"), default="fsdp2")
    p.add_argument(
        "--linear",
        choices=("auto", "te", "torch"),
        default="auto",
        help="linear layer implementation (auto: Transformer Engine if importable)",
    )
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--micro-batch", type=int, default=2, help="sequences per GPU per step")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--compile", action="store_true", help="torch.compile each block")
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--peak-tflops-bf16", type=float, help="override the datasheet peak")
    p.add_argument("--id")
    p.add_argument("--out", help="write the result JSON here (rank 0)")
    return p.parse_args(argv)


def versions(torch, te) -> dict:
    out = {"torch": torch.__version__, "python": platform.python_version()}
    if torch.cuda.is_available():
        out["cuda"] = torch.version.cuda
        with suppress(Exception):  # depends on the build
            out["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
    if te is not None:
        import transformer_engine

        out["transformer_engine"] = getattr(transformer_engine, "__version__", "?")
    return out


def make_recipe(precision: str):
    if precision == "bf16":
        return None
    from transformer_engine.common import recipe as r

    return {
        "fp8": lambda: r.Float8CurrentScaling(),
        "fp8-delayed": lambda: r.DelayedScaling(fp8_format=r.Format.HYBRID),
        "mxfp8": lambda: r.MXFP8BlockScaling(),
        "nvfp4": lambda: r.NVFP4BlockScaling(),
    }[precision]()


def build_model(shape: ModelShape, seq_len: int, te, device, checkpointing: bool):
    import torch
    import torch.utils.checkpoint
    from torch import nn
    from torch.nn import functional as F

    def linear(i: int, o: int):
        if te is not None:
            return te.Linear(i, o, bias=False, params_dtype=torch.float32, device=device)
        return nn.Linear(i, o, bias=False, device=device)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hd, nh, nkv = shape.head_dim, shape.n_heads, shape.n_kv_heads
            self.attn_norm = nn.RMSNorm(shape.dim, eps=1e-5, device=device)
            self.wqkv = linear(shape.dim, (nh + 2 * nkv) * hd)
            self.wo = linear(nh * hd, shape.dim)
            self.mlp_norm = nn.RMSNorm(shape.dim, eps=1e-5, device=device)
            self.w13 = linear(shape.dim, 2 * shape.ffn_dim)  # gate and up, fused
            self.w2 = linear(shape.ffn_dim, shape.dim)

        def forward(self, x, cos, sin):
            b, s, _ = x.shape
            hd, nh, nkv = shape.head_dim, shape.n_heads, shape.n_kv_heads
            q, k, v = self.wqkv(self.attn_norm(x)).split([nh * hd, nkv * hd, nkv * hd], dim=-1)
            q = rope(q.view(b, s, nh, hd).transpose(1, 2), cos, sin)
            k = rope(k.view(b, s, nkv, hd).transpose(1, 2), cos, sin)
            v = v.view(b, s, nkv, hd).transpose(1, 2)
            if nkv != nh:  # grouped-query attention: share each K/V head across a group
                k = k.repeat_interleave(nh // nkv, dim=1)
                v = v.repeat_interleave(nh // nkv, dim=1)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.wo(attn.transpose(1, 2).reshape(b, s, nh * hd))
            gate, up = self.w13(self.mlp_norm(x)).chunk(2, dim=-1)
            return x + self.w2(F.silu(gate) * up)

    def rope(t, cos, sin):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat((t1 * cos - t2 * sin, t2 * cos + t1 * sin), dim=-1).type_as(t)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(shape.vocab, shape.dim, device=device)
            self.layers = nn.ModuleList(Block() for _ in range(shape.n_layers))
            self.norm = nn.RMSNorm(shape.dim, eps=1e-5, device=device)
            # The output head stays in high precision, as in most FP8 training setups.
            self.head = nn.Linear(shape.dim, shape.vocab, bias=False, device=device)
            inv = 1.0 / (
                500000.0
                ** (torch.arange(0, shape.head_dim, 2, device=device).float() / shape.head_dim)
            )
            freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv)
            self.register_buffer("cos", freqs.cos(), persistent=False)
            self.register_buffer("sin", freqs.sin(), persistent=False)

        def forward(self, tokens):
            x = self.embed(tokens)
            for layer in self.layers:
                if checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, self.cos, self.sin, use_reentrant=False
                    )
                else:
                    x = layer(x, self.cos, self.sin)
            return self.head(self.norm(x))

    return Model()


def wrap(model, parallel: str, world: int, cuda: bool, distributed: bool, precision: str):
    import torch

    if not distributed:
        return model
    if parallel == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()] if cuda else None
        )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh("cuda" if cuda else "cpu", (world,))
    mp = (
        MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        if cuda
        else MixedPrecisionPolicy()
    )
    # FSDP2 replaces parameters with DTensors and drops the custom attributes Transformer
    # Engine sets on them; put them back afterwards (as TE's own FSDP2 tests do).
    saved = {n: dict(vars(p)) for n, p in model.named_parameters()}
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    for name, param in model.named_parameters():
        for key, value in saved.get(name, {}).items():
            if not key.startswith("_"):
                setattr(param, key, value)
    return model


if __name__ == "__main__":
    sys.exit(main())
