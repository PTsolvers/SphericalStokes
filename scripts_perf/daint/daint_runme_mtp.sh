#!/bin/bash -l
#SBATCH --job-name="SphStokes_mtp"
#SBATCH --output=SphStokes_mtp.%j.o
#SBATCH --error=SphStokes_mtp.%j.e
#SBATCH --time=00:10:00
#SBATCH --nodes=1
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

scp SphericalStokes_mxpu_mtp.jl data_io2.jl daint_submit_mtp.sh $SCRATCH/SphericalStokes/scripts/

pushd $SCRATCH/SphericalStokes/scripts

srun daint_submit_mtp.sh
