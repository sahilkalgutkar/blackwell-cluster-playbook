# Results

Real-hardware runs only. Synthetic fixture runs never go here.

**No real runs yet.** The plan for the first one is in
[docs/rental-runbook.md](../docs/rental-runbook.md).

Each run added here should include:

- the redacted copy `bcp rental package` writes (`runs/day-<time>-share/`), as
  `results/<date>-<provider>-<shape>/`: `summary.md`, the validation, inference and training
  JSON, and the plot
- a short write-up: provider, instance type, region, date, driver/CUDA/NCCL versions, and
  anything unexpected
- a regenerated comparison table across runs:
  `bcp report results/*/validate/*/*.json --out results/summary.md`
