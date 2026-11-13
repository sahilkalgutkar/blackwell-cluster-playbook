#!/bin/bash
# HPL (and HPL-MxP / HPCG) from NVIDIA's HPC Benchmarks container on NGC.
#
#   HPL_IMAGE='nvcr.io#nvidia/hpc-benchmarks:<tag>' \
#   HPL_DAT=/workspace/hpl-linux-x86_64/sample-dat/<file>.dat \
#   sbatch -N 1 hpc/jobs/hpl.sbatch
#
# Pick the newest tag on NGC and record it in matrix/stack.yaml (hpc-benchmarks). Sample
# .dat files differ per release and CPU architecture (x86_64 vs aarch64 for Grace); list them
# with: srun --container-image="$HPL_IMAGE" ls /workspace
# Swap hpl.sh for hpl-mxp.sh or hpcg.sh to run the other benchmarks in the same container.
#SBATCH --job-name=hpl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j-hpl.log

set -euo pipefail
: "${HPL_IMAGE:?set HPL_IMAGE, e.g. nvcr.io#nvidia/hpc-benchmarks:<tag>}"
: "${HPL_DAT:?set HPL_DAT to a sample .dat inside the container}"

srun --mpi=pmix --container-image="${HPL_IMAGE}" /workspace/hpl.sh --dat "${HPL_DAT}"
