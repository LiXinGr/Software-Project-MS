!/bin/bash

# 1. Force load the modules (Even if direnv unloaded them)
module load Anaconda3/2024.02-1
module load GCCcore/12.2.0
module load CMake/3.24.3-GCCcore-12.2.0
module load Ninja/1.11.1-GCCcore-12.2.0
module load Eigen/3.4.0-GCCcore-12.2.0
module load Ceres-Solver/2.2.0-foss-2022b

# 2. Add the linker fix
export LIBRARY_PATH=$LIBRARY_PATH:/usr/lib64:/lib64:/usr/lib/x86_64-linux-gnu

# 3. SET PYTHON PATH - Only for the evaluation scripts
# We MUST NOT include the madpose/poselib folders here
PROJECT_ROOT="/home.stud/gorbuden/datagrid/Software-Project-MS"
export PYTHONPATH=$PROJECT_ROOT/external/RePoseD:$PYTHONPATH

# 4. Activate the environment
conda activate reposed

echo "------------------------------------------------"
echo "RePoseD Environment Ready"
echo "Verification: "
python -c "import poselib; import madpose; print('   >>> Verification Success!')"
echo "------------------------------------------------"