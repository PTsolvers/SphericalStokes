#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

source setenv_octopus.sh

mpirun_=$(which mpirun)

USE_GPU=true

DO_SAVE=false

DO_SAVE_P=false

RESOL=511

np=$1

export CUDA_VISIBLE_DEVICES=1,2,3,4

$mpirun_ -np $np --bind-to socket ./submit_julia_prof.sh $USE_GPU $DO_SAVE $DO_SAVE_P $RESOL
