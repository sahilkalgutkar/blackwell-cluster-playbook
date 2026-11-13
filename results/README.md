# Results

Real-hardware runs only. Synthetic fixture runs never go here.

**No real runs yet.** The plan for the first one is in
[docs/rental-runbook.md](../docs/rental-runbook.md).

Each run added here should include:

- the results JSON written by `bcp validate`
- a short write-up: provider, instance type, date, driver/CUDA/NCCL versions, and anything
  unexpected
- a regenerated comparison table: `bcp report results/runs/*.json --out results/summary.md`
