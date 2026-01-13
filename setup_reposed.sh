#!/bin/bash
# source setup_reposed.sh
# 1. Load the specific C++ Toolchain required for the solvers
module load CMake/3.24.3-GCCcore-12.2.0
module load Ninja/1.11.1-GCCcore-12.2.0
module load Eigen/3.4.0-GCCcore-12.2.0
module load Ceres-Solver/2.2.0-foss-2022b

# 2. Add the linker fix for -ldl
export LIBRARY_PATH=$LIBRARY_PATH:/usr/lib64:/lib64:/usr/lib/x86_64-linux-gnu

# 3. Set Python Paths relative to the root for your specific structure
export PYTHONPATH=$PYTHONPATH:$(pwd)/external/RePoseD
export PYTHONPATH=$PYTHONPATH:$(pwd)/external/madpose
export PYTHONPATH=$PYTHONPATH:$(pwd)/external/PoseLib-mdrp

# 4. Activate the correct conda environment
conda activate reposed

echo "------------------------------------------------"
echo "RePoseD / MDRP Environment Ready"
echo "Active Python: $(which python)"
echo "------------------------------------------------"