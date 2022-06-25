#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

# /scratch-1/luraess/opt/nvidia/nsight-systems/2022.2.1/bin/nsys profile --cpuctxsw=none --sample=none --capture-range=cudaProfilerApi --trace=cuda --output=outprof.%q{OMPI_COMM_WORLD_RANK}.nsys-rep -f true /scratch-1/soft/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.4.0/julia-1.7.1-mk3ybkevqzzmqchacwvvqpky4srgezhc/bin/julia --project -O3 --check-bounds=no SPH_3D_PERF3_mxpu_prof.jl
# /scratch-1/luraess/opt/nvidia/nsight-systems/2022.2.1/bin/nsys profile --cpuctxsw=none --sample=none --trace=cuda --output=outprof.%q{OMPI_COMM_WORLD_RANK}.nsys-rep -f true /scratch-1/soft/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.4.0/julia-1.7.1-mk3ybkevqzzmqchacwvvqpky4srgezhc/bin/julia --project -O3 --check-bounds=no SPH_3D_PERF3_mxpu_prof.jl
nsys profile --cpuctxsw=none --sample=none --trace=cuda --output=outprof.%q{OMPI_COMM_WORLD_RANK} -f true julia --project -O3 --check-bounds=no SPH_3D_PERF3_mxpu_prof.jl
