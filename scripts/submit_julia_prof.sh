#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

source setenv_octopus.sh

julia_=$(which julia)
nvprof_=$(which nvprof)

USE_GPU=$1
DO_SAVE=$2
DO_SAVE_P=$3
RESOL=$4

$nvprof_ --openacc-profiling off --profile-from-start off --export-profile sph3D.%q{OMPI_COMM_WORLD_RANK}.prof -f $(USE_GPU=$USE_GPU DO_SAVE=$DO_SAVE DO_SAVE_P=$DO_SAVE_P NR=$RESOL NTH=$RESOL NPH=$RESOL $julia_ --project --check-bounds=no -O3  SphericalStokes_mxpu_prof.jl)
