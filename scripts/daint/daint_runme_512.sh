#!/bin/bash -l
#SBATCH --job-name="SphStokes_512"
#SBATCH --output=SphStokes_512.%j.o
#SBATCH --error=SphStokes_512.%j.e
#SBATCH --time=00:7:00
#SBATCH --nodes=512
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account c23

module load daint-gpu
module load Julia/1.7.2-CrayGNU-21.09-cuda
module load cray-hdf5-parallel

export JULIA_HDF5_PATH=$HDF5_ROOT
export JULIA_CUDA_MEMORY_POOL=none

export IGG_CUDAAWARE_MPI=1
export MPICH_RDMA_ENABLED_CUDA=1

scp SphericalStokes_mxpu_pareff.jl data_io2.jl daint_submit_pareff.sh $SCRATCH/SphericalStokes/scripts/

pushd $SCRATCH/SphericalStokes/scripts

srun daint_submit_pareff.sh
