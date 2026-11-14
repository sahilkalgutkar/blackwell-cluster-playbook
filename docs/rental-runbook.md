# Getting real numbers from one day of rented hardware

GPU time is billed by the hour, so the whole day is scripted: `bcp rental run` does every
step in [`rental/day.yaml`](../rental/day.yaml) unattended and leaves results you can
analyse, and re-analyse, after the instance is gone.

| What you get | From |
|---|---|
| A validated node: DCGM diagnostics, NCCL collectives, NVLink bandwidth, pass/fail per check | `bcp collect` + `bcp validate`, quick then full |
| Inference: BF16 vs FP8 vs NVFP4; vLLM vs SGLang vs TensorRT-LLM; Dynamo with prefill and decode together vs split | 11 servers, each swept over 5 concurrency levels with AIPerf |
| Training: tokens/s, TFLOP/s and MFU, scaling 1 → 8 GPUs, BF16 vs FP8 vs MXFP8 vs NVFP4 | [`training/bench_train.py`](../training/bench_train.py) |
| `summary.md`, a throughput-vs-interactivity plot, and a redacted copy to commit | `bcp rental report`, `bcp rental package` |

**Budget:** about 4.5 hours on an 8x B200 (HGX) node. On-demand B200 was roughly $6–9 per
GPU-hour at specialist clouds in November 2026, so $50–70 an hour for the node and about
$300 for the day. Run `bcp rental plan --rate <your price>` for your own number.

## 1. Before you rent (free)

```bash
make test
bcp rental plan --commands      # every command the day will run, with time and cost
```

- [ ] Read the plan. Edit `rental/day.yaml` to drop what you don't care about: each
      inference config is about 15 minutes, each training run about 3.
- [ ] CI is green, including the `images` workflow: it builds the validation container, so a
      broken Dockerfile shows up there and not on the paid node.
- [ ] Pick the instance: 8x B200 HGX, Ubuntu with Docker and the NVIDIA container toolkit
      (the default image at most GPU clouds), about 400 GB of free disk, outbound internet.
      The models are ungated, so no Hugging Face token is needed.

## 2. Rehearse on a cheap box (about $10)

Rent one or two H100s or A100s for an hour and run the same thing shrunk down:

```bash
bcp rental run --rehearse
```

Small models, two concurrency levels, short training runs. Anything the box can't do (NVFP4
and MXFP8 need Blackwell, 4-GPU layouts need 4 GPUs) is skipped and listed. The point is to
find a wrong flag, a missing package or a changed container entrypoint at $3 an hour rather
than $65. Fix what breaks, push, and rehearse again if you changed much.

## 3. The day

```bash
ssh <user>@<node>
tmux new -s day                 # a dropped SSH session must not stop the run
git clone https://github.com/sahilkalgutkar/blackwell-cluster-playbook
cd blackwell-cluster-playbook
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[plot]'
bcp rental plan                 # check it detected 8x B200
bcp rental run
```

| Time | Phase | Notes |
|---|---|---|
| 0:00 | Setup | Installs AIPerf, checks Docker, disk and GPUs, starts pulling the container images and ~120 GB of models in the background, builds the validation image |
| 0:15 | Quick validation | **The gate.** If a GPU is missing, an NVLink is down, DCGM fails or NCCL returns wrong results, the day stops here: ask the provider for another node. Numbers merely below the threshold are recorded, not fatal |
| 0:25 | Full validation | DCGM level 3, NCCL up to 16 GiB, nvbandwidth. Downloads finish meanwhile |
| 1:15 | Inference | One server at a time: start it, wait until it answers, sweep, save its logs, stop it |
| 4:00 | Training | 3B model on 1, 2, 4 and 8 GPUs; 8B model on 8 GPUs in four precisions |
| 4:30 | Wrap-up | `summary.md`, then the redacted copy and a full tarball |

Every step's outcome is saved in `runs/day-<time>/state.json`. A failed inference or training
run is recorded and the day moves on; only setup and the gate stop it.

**If something goes wrong:**

- SSH dropped: `tmux attach -t day`. If the process itself died, continue with
  `bcp rental run --resume --run-dir runs/day-<time>`; finished steps are skipped.
- One config failed: its server log is in `runs/day-<time>/inference/<id>/server.log`. After a
  fix, rerun just that step:
  `bcp rental run --resume --run-dir runs/day-<time> --only inference.<id>`.
- Out of budget: Ctrl-C (the running server is stopped and removed), then
  `bcp rental run --resume --run-dir runs/day-<time> --only wrap-up` to keep what you have.

## 4. Before you terminate

The last step prints where everything is:

```bash
# from your laptop
scp -r <user>@<node>:<repo>/runs/day-<time>-share ./runs/   # redacted: safe to commit
scp <user>@<node>:<repo>/runs/day-<time>.tgz ./runs/         # everything: keep private
```

Then terminate the instance, and check the provider's console that its storage is gone too.

## 5. Afterwards

- Skim the redacted copy before committing it. Redaction replaces GPU UUIDs, serial numbers,
  IP and MAC addresses and the hostname, by pattern; `host/nvidia-smi-q.txt` is the file
  worth a second look.
- Commit it as `results/<date>-<provider>-8xb200/` with a short write-up: provider, instance
  type, region, driver, and anything that surprised you. The write-up is worth as much as the
  numbers.
- Replace the starting points in `configs/thresholds.yaml` with ~90% of what this healthy node
  measured, and say so in the commit.

## What the numbers do and don't say

- Every engine runs with its defaults apart from the context length: an out-of-the-box
  comparison on one model (Qwen3-32B) and one load shape (2,048 tokens in, 256 out), not a
  tuned shoot-out. Tuning one engine and not the others would make the comparison unfair.
- AIPerf sends the same synthetic prompts, from the same tokenizer and seed, to every server.
- MFU uses the dense BF16 peak and the common 6N + attention FLOP count, described at the top
  of `training/bench_train.py`. FP8 runs are also reported against the FP8 peak.
- One node, one day. Say so when you quote a number.
