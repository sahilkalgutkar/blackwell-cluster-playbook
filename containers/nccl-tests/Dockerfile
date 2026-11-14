# nccl-tests built with MPI, for multi-node runs under Slurm (Pyxis) or Kubernetes (MPIJob).
#
#   docker build -t ghcr.io/<you>/nccl-tests:cuda13.5 containers/nccl-tests
#
# Versions follow matrix/stack.yaml. The CUDA devel image ships NCCL; to test a newer NCCL
# than the image has, add a step that installs libnccl2/libnccl-dev at NCCL_VERSION.
ARG CUDA_VERSION=13.5.0
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ARG NCCL_TESTS_VERSION=v2.21.1
# Hopper (H100 rehearsals) and Blackwell; sm_100 code also runs on B300. nccl-tests' default
# list has six targets and took about 25 minutes to build in CI.
ARG CUDA_ARCHS="90;100"

# openssh-server: Kubeflow MPIJob workers run sshd (k8s/dra/nccl-computedomain.yaml).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git openmpi-bin libopenmpi-dev openssh-client openssh-server \
 && rm -rf /var/lib/apt/lists/* \
 && ssh-keygen -A && mkdir -p /run/sshd

# Ubuntu keeps Open MPI's include/ and lib/ under the multiarch directory, not /usr; uname -m
# makes this x86_64-linux-gnu on HGX and aarch64-linux-gnu on Grace (GB200).
#
# OMPI_SKIP_MPICXX: included from C++, Open MPI 4.1's mpi.h also pulls in its C++ bindings,
# and the device API (GIN) tests would then need libmpi_cxx to link. They use only the C API.
# The flags go in through the environment, which nccl-tests' common.mk appends to; on make's
# command line they would replace its own flags instead.
RUN MPI_HOME="/usr/lib/$(uname -m)-linux-gnu/openmpi" \
 && { test -f "${MPI_HOME}/include/mpi.h" || { echo "mpi.h not found under ${MPI_HOME}" >&2; exit 1; }; } \
 && gencode="$(echo "${CUDA_ARCHS}" | tr ';' '\n' \
      | sed 's/.*/-gencode=arch=compute_&,code=sm_&/' | tr '\n' ' ')" \
 && git clone --depth 1 --branch "${NCCL_TESTS_VERSION}" https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
 && CXXFLAGS="-DOMPI_SKIP_MPICXX=1" NVCUFLAGS="-DOMPI_SKIP_MPICXX=1" \
    make -C /opt/nccl-tests -j"$(nproc)" MPI=1 MPI_HOME="${MPI_HOME}" CUDA_HOME=/usr/local/cuda \
      NVCC_GENCODE="${gencode}"

ENV PATH=/opt/nccl-tests/build:${PATH}
