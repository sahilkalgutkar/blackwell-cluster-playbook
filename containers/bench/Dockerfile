# Everything `bcp validate` needs on one node, so a rented machine only has to provide Docker
# and the NVIDIA container toolkit: nccl-tests (single process, no MPI), nvbandwidth,
# DCGM 4 (dcgmi + nv-hostengine) and bcp itself.
#
#   docker build -t bcp-bench:local -f containers/bench/Dockerfile .
#   docker run --rm --gpus all --privileged bcp-bench:local \
#       bash -c 'nv-hostengine; bcp validate --suite quick'
#
# `bcp rental run` builds it on the node with versions from matrix/stack.yaml. CUDA_ARCHS
# covers Hopper (for rehearsals on H100) and Blackwell; sm_100 code also runs on B300.
ARG CUDA_VERSION=13.5.0
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ARG NCCL_TESTS_VERSION=v2.21.1
ARG NVBANDWIDTH_VERSION=v0.11.0
ARG CUDA_ARCHS="90;100"

# The CUDA image already has NVIDIA's apt repository, which carries DCGM 4.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git cmake build-essential libboost-program-options-dev \
      python3 python3-venv pciutils \
      datacenter-gpu-manager-4-cuda13 \
 && rm -rf /var/lib/apt/lists/*

RUN gencode="$(echo "${CUDA_ARCHS}" | tr ';' '\n' \
      | sed 's/.*/-gencode=arch=compute_&,code=sm_&/' | tr '\n' ' ')" \
 && git clone --depth 1 --branch "${NCCL_TESTS_VERSION}" \
      https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
 && make -C /opt/nccl-tests -j"$(nproc)" MPI=0 CUDA_HOME=/usr/local/cuda \
      NVCC_GENCODE="${gencode}"

RUN git clone --depth 1 --branch "${NVBANDWIDTH_VERSION}" \
      https://github.com/NVIDIA/nvbandwidth.git /opt/nvbandwidth \
 && cmake -S /opt/nvbandwidth -B /opt/nvbandwidth/build \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
 && cmake --build /opt/nvbandwidth/build -j"$(nproc)" \
 && install -m 0755 /opt/nvbandwidth/build/nvbandwidth /usr/local/bin/nvbandwidth

COPY pyproject.toml README.md /opt/bcp/
COPY src /opt/bcp/src
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir /opt/bcp

ENV PATH=/opt/venv/bin:/opt/nccl-tests/build:${PATH} \
    BCP_NCCL_TESTS_DIR=/opt/nccl-tests/build/
WORKDIR /opt/bcp
