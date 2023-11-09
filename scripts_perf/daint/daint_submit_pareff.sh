#!/bin/bash

RESOL=383

RUN="SphericalStokes_mxpu_pareff"

USE_GPU=true

DO_VIZ=false

DO_SAVE=false

DO_SAVE_P=true

# if [ "$DO_SAVE_P" = "true" ]; then

#     FILE=../out_perf/out_SphericalStokes_pareff.txt
    
#     if [ -f "$FILE" ]; then
#         echo "Systematic results (file $FILE) already exists. Remove to continue."
#         exit 0
#     else 
#         echo "Launching systematics (saving results to $FILE)."
#     fi
# fi

for ie in {1..2}; do

    # echo "== Running script SphericalStokes_mxpu_pareff.jl, resol=$RESOL, (test $ie)"

    USE_GPU=$USE_GPU DO_SAVE=$DO_SAVE DO_SAVE_P=$DO_SAVE_P NR=$RESOL NTH=$RESOL NPH=$RESOL LD_PRELOAD="/usr/lib64/libcuda.so:/usr/local/cuda/lib64/libcudart.so" julia --project --check-bounds=no -O3  "$RUN".jl

done
