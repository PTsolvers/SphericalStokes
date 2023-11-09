#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

source setenv_octopus.sh

mpirun_=$(which mpirun)
julia_=$(which julia)

RESOL=511

NPROCS=(1 2 4 8)

USE_GPU=true

DO_VIZ=false

DO_SAVE=false

DO_SAVE_P=true


if [ "$DO_SAVE_P" = "true" ]; then

    FILE=../out_perf/out_SphericalStokes_pareff.txt
    
    if [ -f "$FILE" ]; then
        echo "Systematic results (file $FILE) already exists. Remove to continue."
        exit 0
    else 
        echo "Launching systematics (saving results to $FILE)."
    fi
fi

for nproc in "${NPROCS[@]}"; do

    for ie in {1..5}; do

        echo "== Running script SphericalStokes_mxpu.jl, resol=$RESOL, nprocs=$nproc (test $ie)"
        USE_GPU=$USE_GPU DO_SAVE=$DO_SAVE DO_SAVE_P=$DO_SAVE_P NR=$RESOL NTH=$RESOL NPH=$RESOL $mpirun_ -np $nproc --bind-to socket $julia_ --project --check-bounds=no -O3  SphericalStokes_mxpu_pareff.jl

    done

done