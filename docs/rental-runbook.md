# Getting real numbers from a few hours of rented hardware

GPU time is billed by the hour, so do everything that doesn't need a GPU before the
instance starts, and leave with raw output you can analyse later.

## Before you rent

- [ ] `make demo` passes locally.
- [ ] The nccl-tests image (`containers/nccl-tests`) is built and pushed, or you have the
      build commands ready to run on the node.
- [ ] You know what you will publish: at minimum `bcp collect --suite full` for one node,
      ideally one inference sweep.
- [ ] Pick the instance: an 8x B200 HGX node is the most widely available Blackwell shape.
      NVL72 capacity is mostly sold reserved; look for trial credits or startup programs if
      you want tray-to-tray numbers.
- [ ] Set a phone timer for the end of your budget.

## On the node (about 3 hours for 8x B200)

| Time | Step |
|---|---|
| 0:00 | `nvidia-smi`, then `git clone` this repo and `pip install -e .` |
| 0:05 | `bcp topology --format mermaid > runs/topology.mmd`; check the platform it detected |
| 0:10 | Install what's missing: DCGM 4 (`dcgmi`; package names are in NVIDIA's DCGM docs), build nccl-tests (`make -j` in a clone, then `export BCP_NCCL_TESTS_DIR=$PWD/build`) and nvbandwidth (CMake + Boost program_options) |
| 0:25 | `bcp collect --suite quick --out runs/raw/quick`, a 5-minute sanity pass |
| 0:30 | `bcp collect --suite full --out runs/raw/full`, about 45 minutes |
| 1:20 | Optional: HPL from the NGC HPC Benchmarks container (`hpc/jobs/hpl.sbatch` shows the command) |
| 1:40 | Optional: serve a model with Dynamo (its local quickstart works without Kubernetes) and run `inference/bench/aiperf-sweep.sh` |
| 2:40 | `tar czf runs.tgz runs/` and copy it off the node, then **terminate the instance** |

## Afterwards

```bash
bcp validate --replay runs/raw/full --suite full
bcp report build/runs/*.json --out results/summary.md --plot results/all_reduce.png
```

Before committing anything under `results/`:

- Remove serial numbers, GPU UUIDs, hostnames and IP addresses from the raw files.
  `nvidia-smi -q` in particular contains serials.
- Write down the provider, instance type, region, date, driver and anything that surprised
  you. The write-up is worth as much as the numbers.
- Replace the threshold starting points in `configs/thresholds.yaml` with ~90% of what a
  healthy node measured, and say so in the commit.
