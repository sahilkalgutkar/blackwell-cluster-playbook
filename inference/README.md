# Inference: NVIDIA Dynamo on Blackwell

| Path | What it does | Status |
|---|---|---|
| `dynamo/disagg-vllm-b200.yaml` | Prefill/decode split serving of Qwen3-32B on one HGX B200 node | Draft, follows the Dynamo v1.6.2 CRD |
| `bench/aiperf-sweep.sh` | Concurrency sweep with AIPerf; saves every run's artifacts | Draft |

## What to measure

Run the sweep against three layouts on the same node and publish the curves in `results/`:

1. **Aggregated**: one vLLM worker doing both prefill and decode (Dynamo's `agg.yaml` upstream).
2. **Disaggregated 2P + 1D** (this folder's manifest).
3. **Disaggregated 1P + 1D (TP2)**, or whatever the first two suggest.

Plot throughput per GPU (tokens/s/GPU) against time-to-first-token and inter-token latency.
The point where disaggregation starts to win, and why, is the interesting result.

## Going bigger: wide expert parallelism on NVL72

Large mixture-of-experts models (DeepSeek-R1/V4, Kimi K2) spread experts across many GPUs,
and the all-to-all traffic that creates is exactly what the 72-GPU NVLink domain is for.
Dynamo maintains recipes for this rather than one-off manifests:

- `recipes/deepseek-r1/` and `recipes/deepseek-v4/` in
  [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo/tree/main/recipes), including GB200
  variants
- [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) for the expert-parallel
  communication kernels

Before running one on rented GB200 time, check the multi-node NVLink domain first with
`bcp validate` on each tray and `k8s/dra/nccl-computedomain.yaml` across trays.
