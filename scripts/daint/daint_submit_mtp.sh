#!/bin/bash

# RESOL=( 31 63 127 255 383 )
RESOL=( 383 )

RUN="SphericalStokes_mxpu_mtp"

USE_GPU=true

DO_VIZ=false

DO_SAVE=false

DO_SAVE_P=true

# if [ "$DO_SAVE_P" = "true" ]; then

#     FILE=../out_perf/out_SphericalStokes_mtp.txt
    
#     if [ -f "$FILE" ]; then
#         echo "Systematic results (file $FILE) already exists. Remove to continue."
#         exit 0
#     else 
#         echo "Launching systematics (saving results to $FILE)."
#     fi
# fi

for i in "${RESOL[@]}"; do

    for ie in {1..3}; do

        echo "== Running script SphericalStokes_mxpu_mtp.jl, resol=$i (test $ie)"

        USE_GPU=$USE_GPU DO_SAVE=$DO_SAVE DO_SAVE_P=$DO_SAVE_P NR=$i NTH=$i NPH=$i LD_PRELOAD="/usr/lib64/libcuda.so:/usr/local/cuda/lib64/libcudart.so" julia --project -O3 --check-bounds=no "$RUN".jl
    
    done

done
