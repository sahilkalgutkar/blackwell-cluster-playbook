# blackwell-cluster-playbook

Bring-up, validation and benchmarking for NVIDIA Blackwell GPU clusters, from a single
HGX B200 node to a GB200/GB300 NVL72 rack.

[![ci](https://github.com/sahilkalgutkar/blackwell-cluster-playbook/actions/workflows/ci.yml/badge.svg)](https://github.com/sahilkalgutkar/blackwell-cluster-playbook/actions/workflows/ci.yml)
[![stack-watch](https://github.com/sahilkalgutkar/blackwell-cluster-playbook/actions/workflows/stack-watch.yml/badge.svg)](https://github.com/sahilkalgutkar/blackwell-cluster-playbook/actions/workflows/stack-watch.yml)

New GPU nodes fail in boring, expensive ways: one NVLink down, a NIC on the wrong socket,
an NVL72 tray that never joined the rack's NVLink domain. This repo is a checklist for
deciding whether to trust a node with a real job, plus the configs to put those nodes behind
Slurm or Kubernetes and serve models on them.

- **`bcp`**, a small Python CLI that detects the platform (HGX baseboard, DGX, NVL72 tray),
  runs DCGM, nccl-tests and nvbandwidth, and turns the output into pass/fail checks and a
  results JSON.
- **HPC**: Ansible for Slurm + Pyxis/Enroot, with Slurm block topology for NVL72 racks, plus
  multi-node nccl-tests and HPL jobs.
- **Kubernetes**: GPU Operator, the DRA driver's ComputeDomains for multi-node NVLink, KAI
  Scheduler queues, and DCGM alerts.
- **Inference**: NVIDIA Dynamo serving with separate prefill and decode workers, benchmarked
  with AIPerf.
- **A pinned stack** in [`matrix/stack.yaml`](matrix/stack.yaml) that a weekly workflow
  checks against upstream releases, opening a PR when something moves.

## Status

Be clear-eyed about what has run where:

| Part | State |
|---|---|
| `bcp` CLI, parsers, checks | Working. 49 tests, end-to-end demo in CI on synthetic fixtures |
| Real-hardware results | **None yet.** See [docs/rental-runbook.md](docs/rental-runbook.md) |
| `hpc/`, `k8s/`, `inference/` | Drafts. Syntax- and schema-checked in CI, not yet run on hardware |

Everything under [`fixtures/`](fixtures/) is synthetic: modelled numbers in the exact format
the real tools print. Every report built from it says so.

## Quick start (no GPU needed)

```bash
git clone https://github.com/sahilkalgutkar/blackwell-cluster-playbook && cd blackwell-cluster-playbook
make demo
```

`make demo` runs the full pipeline against four fixture nodes and expects the broken one to
fail:

```text
$ bcp validate --replay fixtures/dgx-b200-degraded
Verdict: FAIL  (synthetic-dgx-b200-degraded, replay:fixtures/dgx-b200-degraded) · SYNTHETIC fixture
Platform: HGX 8-GPU baseboard (NVSwitch) · 8 x NVIDIA B200

STATUS  CHECK                                          MEASURED                   EXPECTED
pass    inventory.gpu_count                            8                          == 8
fail    topology.nvlink_symmetry                       GPU5 NV16                  all NV18
          ↳ a GPU with fewer active NVLinks than its peers
fail    dcgm.diag                                      fail                       pass
          ↳ pcie (GPU 5): GPU 5: NVLink 7 and 13 are down (link state inactive)
fail    nccl.all_reduce.busbw                          565.3 GB/s @ 8.0 GiB       >= 650 GB/s
pass    nccl.all_gather.busbw                          615.6 GB/s @ 8.0 GiB       >= 550 GB/s
pass    nccl.reduce_scatter.busbw                      610.5 GB/s @ 8.0 GiB       >= 550 GB/s
fail    nccl.alltoall.busbw                            469.5 GB/s @ 8.0 GiB       >= 500 GB/s
pass    nvbandwidth.device_to_device_memcpy_read_ce    750.0 GB/s avg             >= 650 GB/s
...
```

The fixture models a common failure: two NVLinks down on one GPU. A real degraded link tends
to show up in several places at once (the topology matrix, DCGM, collective bandwidth) while
point-to-point copy tests can look nearly normal, which is why the checks cover all of them.
The numbers here are modelled, so treat the size of the drop as illustrative.

## On a real node

```bash
pip install -e .
bcp topology                                  # what is this machine?
bcp collect --suite full --out runs/raw/node01 # ~45 min on 8 GPUs; saves raw output only
bcp validate --replay runs/raw/node01          # analyse now, or later on your laptop
bcp report build/runs/*.json --out results/summary.md
```

`collect` keeps raw output, so a few hours of rented time can be analysed (and re-analysed
when the thresholds improve) long after the instance is gone. It needs `nvidia-smi`, `dcgmi`,
[nccl-tests](https://github.com/NVIDIA/nccl-tests) (point `BCP_NCCL_TESTS_DIR` at its
`build/`) and [nvbandwidth](https://github.com/NVIDIA/nvbandwidth) on the node. Anything
missing is reported as skipped rather than failing the run.

### What `bcp topology` sees on an NVL72 tray

```mermaid
flowchart LR
  subgraph numa0["NUMA 0 · CPUs 0-71"]
    GPU0["GPU0<br/>GB200"]
    GPU1["GPU1<br/>GB200"]
    NIC0(["NIC0<br/>mlx5_0"])
    NIC1(["NIC1<br/>mlx5_1"])
    NIC4(["NIC4<br/>mlx5_4"])
  end
  subgraph numa1["NUMA 1 · CPUs 72-143"]
    GPU2["GPU2<br/>GB200"]
    GPU3["GPU3<br/>GB200"]
    NIC2(["NIC2<br/>mlx5_2"])
    NIC3(["NIC3<br/>mlx5_3"])
    NIC5(["NIC5<br/>mlx5_5"])
  end
  FABRIC{{"NVLink Switch trays<br/>rack-wide domain, clique 32766"}}
  GPU0 ===|NV18| FABRIC
  GPU1 ===|NV18| FABRIC
  GPU2 ===|NV18| FABRIC
  GPU3 ===|NV18| FABRIC
  GPU0 ---|PIX| NIC0
  GPU1 ---|PIX| NIC1
  GPU2 ---|PIX| NIC2
  GPU3 ---|PIX| NIC3
```

Only 4 GPUs are local to the tray, but `nvidia-smi -q` reports that each one has joined a
rack-wide NVLink fabric (state `Completed`, non-zero cluster UUID), so `bcp` identifies it
as part of an NVL72 domain rather than a 4-GPU server. `--format mermaid` output pastes
straight into GitHub Markdown.

## How the platforms relate

DGX, HGX, MGX and NVL72 describe different things (a product line, a baseboard, a reference
architecture, a rack) and overlap a lot. [docs/platforms.md](docs/platforms.md) sorts them
out and explains the terms the checks use: NVSwitch, IMEX, ComputeDomain, NVLS, busbw.

## Keeping the stack current

```bash
bcp stack check            # pinned vs latest stable upstream, for 16 tracked components
bcp stack check --update   # rewrite the pins (comments in stack.yaml are kept)
```

The [stack-watch workflow](.github/workflows/stack-watch.yml) runs this every Monday and
opens a PR with the bumps. Release candidates are ignored; repos that publish several
projects (NVIDIA/nccl also releases `nccl4py`) or reset their numbering (the DRA driver,
after it moved to `kubernetes-sigs`) are handled by `tag_pattern` and `order: date`.

## Layout

```text
src/bcp/            CLI: probes, parsers, topology, checks, report, stack tracking
configs/            pass/fail thresholds per GPU model (starting points; tune per site)
fixtures/           synthetic tool output for H100, DGX B200, GB200 NVL72, degraded B200
matrix/stack.yaml   pinned component versions, checked weekly
containers/         nccl-tests image (MPI build) for Slurm and Kubernetes
hpc/                Ansible (Slurm, Pyxis/Enroot, NVL72 block topology), sbatch jobs
k8s/                GPU Operator, DRA ComputeDomain, KAI queues, DCGM alerts
inference/          Dynamo prefill/decode deployment, AIPerf sweep
docs/               platform guide, rented-hardware runbook
results/            real-hardware runs only
```

## Roadmap

- [ ] First real run: 8x B200 (`bcp collect --suite full`, HPL, Dynamo sweep)
- [ ] Multi-node NVLink across GB200 trays vs the same job over InfiniBand
- [ ] `bcp collect --redact` to strip serials, UUIDs and hostnames before committing raw output
- [ ] Rubin (Vera Rubin NVL72) profile once real `nvidia-smi` output is available
- [ ] Grafana dashboard for the DCGM alerts
- [ ] Parse AIPerf exports into `bcp report`

## License

Apache-2.0. See [LICENSE](LICENSE).
