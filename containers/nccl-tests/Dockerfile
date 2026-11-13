# nccl-tests built with MPI, for multi-node runs under Slurm (Pyxis) or Kubernetes (MPIJob).
#
#   docker build -t ghcr.io/<you>/nccl-tests:cuda13.5 containers/nccl-tests
#
# Versions follow matrix/stack.yaml. The CUDA devel image ships NCCL; to test a newer NCCL
# than the image has, add a step that installs libnccl2/libnccl-dev at NCCL_VERSION.
ARG CUDA_VERSION=13.5.0
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ARG NCCL_TESTS_VERSION=v2.21.1

# openssh-server: Kubeflow MPIJob workers run sshd (k8s/dra/nccl-computedomain.yaml).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git openmpi-bin libopenmpi-dev openssh-client openssh-server \
 && rm -rf /var/lib/apt/lists/* \
 && ssh-keygen -A && mkdir -p /run/sshd

RUN git clone --depth 1 --branch "${NCCL_TESTS_VERSION}" https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
 && MPI_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v mpicc)")")")" \
 && make -C /opt/nccl-tests -j"$(nproc)" MPI=1 MPI_HOME="${MPI_HOME}" CUDA_HOME=/usr/local/cuda

ENV PATH=/opt/nccl-tests/build:${PATH}
