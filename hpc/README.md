# HPC: Slurm + Pyxis/Enroot

| Path | What it does | Status |
|---|---|---|
| `ansible/` | Installs munge, Slurm (controller + GPU nodes), Enroot and Pyxis | Draft: syntax-checked in CI, not yet run on hardware |
| `ansible/roles/slurm/templates/topology.conf.j2` | Slurm `topology/block`: one block per NVL72 rack (NVLink domain) | Draft |
| `jobs/nccl-multinode.sbatch` | nccl-tests across nodes; output parses with `bcp validate --replay` | Draft |
| `jobs/hpl.sbatch` | HPL / HPL-MxP / HPCG from the NGC HPC Benchmarks container | Draft |

## Why these pieces

- **Pyxis + Enroot** let `srun --container-image=...` run NGC containers unprivileged. This is
  the standard way to run containers under Slurm on DGX SuperPOD and most GPU clouds.
- **`AutoDetect=nvml`** in `gres.conf` lets slurmd find the GPUs and their CPU/NIC affinity
  itself, so jobs get cores on the right socket.
- **`topology/block`** describes NVL72 racks to Slurm. A job that fits in one rack stays on
  NVLink; one that doesn't spills onto InfiniBand/Ethernet. Measuring that difference with
  `jobs/nccl-multinode.sbatch` is one of the most useful results this repo can publish.

## Run

```bash
cp hpc/ansible/inventory.example.ini hpc/ansible/inventory.ini
dd if=/dev/urandom bs=1 count=1024 of=hpc/ansible/files/munge.key
ansible-playbook -i hpc/ansible/inventory.ini hpc/ansible/site.yml
```

`inventory.ini` and `munge.key` are git-ignored.
