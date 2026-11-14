# Training throughput

[`bench_train.py`](bench_train.py) trains a Llama-style model on synthetic tokens and reports
tokens/s, TFLOP/s per GPU and MFU. `bcp rental run` runs it in the NGC PyTorch container
(PyTorch 2.15, Transformer Engine 2.20) for the matrix in `rental/day.yaml`:

- **Scaling:** a 3B model on 1, 2, 4 and 8 GPUs at a fixed batch per GPU, with FSDP2 sharding
  over NVLink. Scaling efficiency is tokens/s per GPU relative to the 1-GPU run.
- **Precision:** an 8B model on 8 GPUs with Transformer Engine linear layers in BF16, FP8
  (per-tensor current scaling), MXFP8 and NVFP4. MXFP8 and NVFP4 use Blackwell's
  block-scaled tensor cores; on older GPUs they are skipped.

```bash
# on a GPU node, inside nvcr.io/nvidia/pytorch:26.10-py3
torchrun --standalone --nproc-per-node 8 training/bench_train.py \
    --preset 8b --precision mxfp8 --micro-batch 2 --seq-len 4096 --out runs/8b-mxfp8.json

# anywhere with PyTorch (CPU), as CI does
torchrun --standalone --nproc-per-node 2 training/bench_train.py \
    --device cpu --preset debug --seq-len 64 --steps 3 --parallel fsdp2
```

The model is deliberately plain (RMSNorm, rotary embeddings, grouped-query attention through
`scaled_dot_product_attention`, SwiGLU) so the numbers reflect the hardware and the
precision, not a framework's tricks. That also means a tuned framework such as Megatron-LM or
torchtitan will reach higher MFU on the same node; quote results as "this benchmark", not as
the ceiling of the GPU.

| Flag | Default | Notes |
|---|---|---|
| `--preset` | `8b` | `debug`, `1b`, `3b`, `8b` (Llama 3.x shapes, untied embeddings) |
| `--precision` | `bf16` | `fp8`, `fp8-delayed`, `mxfp8`, `nvfp4` need Transformer Engine |
| `--parallel` | `fsdp2` | or `ddp`; `bcp rental` retries with DDP if an FSDP2 run fails |
| `--compile` | off | `torch.compile` each block |
| `--activation-checkpointing` | off | trade compute for memory |
