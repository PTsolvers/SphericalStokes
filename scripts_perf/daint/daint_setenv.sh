#!/bin/bash

module load daint-gpu
module load Julia/1.7.2-CrayGNU-21.09-cuda
module load cray-hdf5-parallel

# export JULIA_MPI_PATH=/apps/daint/UES/omlins/craympich_juliafix2
# export JULIA_MPI_LIBRARY=/apps/daint/UES/omlins/craympich_juliafix2/lib/libmpich_gnu_82.so

export JULIA_HDF5_PATH=$HDF5_ROOT
export JULIA_CUDA_MEMORY_POOL=none

export IGG_CUDAAWARE_MPI=1
export MPICH_RDMA_ENABLED_CUDA=1

echo ready
