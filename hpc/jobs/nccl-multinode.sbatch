#!/bin/bash
# Multi-node nccl-tests through Pyxis/Enroot. Output lands in runs/raw/<jobid>/ in the same
# layout `bcp collect` uses, so `bcp validate --replay runs/raw/<jobid>` can parse it.
#
#   NCCL_TESTS_IMAGE=ghcr.io/<you>/nccl-tests:cuda13.5 sbatch -N 2 hpc/jobs/nccl-multinode.sbatch
#
# NVL72: with TopologyPlugin=topology/block (see hpc/ansible/group_vars/all.yml) add
# `--segment=<nodes>` to keep the job inside one NVLink domain, then compare the numbers
# against a run that spans two racks to see what the multi-node NVLink domain buys you.
#SBATCH --job-name=nccl-tests
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-nccl.log

set -euo pipefail
: "${NCCL_TESTS_IMAGE:?set NCCL_TESTS_IMAGE to an image built from containers/nccl-tests}"
OUT="runs/raw/${SLURM_JOB_ID}"
mkdir -p "${OUT}"

export NCCL_DEBUG=VERSION
# GB200/GB300 NVL72: NCCL uses multi-node NVLink automatically when the IMEX domain is up.
# Uncomment to fail loudly instead of silently falling back to the network:
# export NCCL_MNNVL_ENABLE=1

# Pyxis wants registry#repo/image:tag, so NCCL_TESTS_IMAGE must include the registry host.
IMAGE="${NCCL_TESTS_IMAGE/\//#}"
for op in all_reduce all_gather reduce_scatter alltoall; do
  echo "== ${op}" >&2
  srun --mpi=pmix --container-image="${IMAGE}" \
    "${op}_perf" -b 8 -e 16G -f 2 -g 1 | tee "${OUT}/nccl_${op}.txt"
done

echo "Parse with: bcp validate --replay ${OUT}" >&2
