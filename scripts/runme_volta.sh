#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

source setenv_octopus.sh

mpirun_=$(which mpirun)
julia_=$(which julia)

USE_GPU=true

DO_SAVE=true

DO_SAVE_P=false

RESOL=255

np=$1

export CUDA_VISIBLE_DEVICES=1,2,3,4
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

USE_GPU=$USE_GPU DO_SAVE=$DO_SAVE DO_SAVE_P=$DO_SAVE_P NR=$RESOL NTH=$RESOL NPH=$RESOL $mpirun_ -np $np --bind-to socket $julia_ --project --check-bounds=no -O3  SphericalStokes_mxpu.jl
