#!/bin/bash

export JULIA_CUDA_USE_BINARYBUILDER=false
export JULIA_MPI_BINARY=system

export JULIA_HDF5_PATH=$HDF5_ROOT

export JULIA_CUDA_MEMORY_POOL=none
export IGG_CUDAAWARE_MPI=1
export JULIA_NUM_THREADS=4

module purge > /dev/null 2>&1

module load julia/julia-1.7.1
module load cuda/11.4
module load openmpi/gcc83-314-c112
module load hdf5

