#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

source setenv_octopus.sh

julia_=$(which julia)

RESOL=( 31 63 127 255 511 )

USE_GPU=true

DO_VIZ=false

DO_SAVE=false

DO_SAVE_P=true


if [ "$DO_SAVE_P" = "true" ]; then

    FILE=../out_perf/out_SphericalStokes_mtp.txt
    
    if [ -f "$FILE" ]; then
        echo "Systematic results (file $FILE) already exists. Remove to continue."
        exit 0
    else 
        echo "Launching systematics (saving results to $FILE)."
    fi
fi

for i in "${RESOL[@]}"; do

    for ie in {1..5}; do

        echo "== Running script SPH_3D_PERF3.jl, resol=$i (test $ie)"
        USE_GPU=$USE_GPU DO_SAVE=$DO_SAVE DO_SAVE_P=$DO_SAVE_P NR=$i NTH=$i NPH=$i $julia_ --project -O3 --check-bounds=no SphericalStokes_mxpu_mtp.jl
    
    done

done