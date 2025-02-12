#!/bin/bash
#SBATCH --job-name=p2e2m68
#SBATCH --error=p2e2m68.%J.err
#SBATCH --output=p2e2m68.%J.out
#SBATCH --partition=testp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=72:00:00

# Activate the conda environment
source /nlsasfs/home/precipitation/midhunm/Conda/bin/activate tf2 # Edit here
conda list

####################### EDIT BELOW #######################
PREFIX="${1:-p02a}"   
EPOCHS="${2:-201}"
SCRIPTNAME='main.py' 
REFD_PATH="/nlsasfs/home/precipitation/midhunm/AI4KLIM/DATASET/DATA_P02/grid_masks"
DATA_PATH="/nlsasfs/home/precipitation/midhunm/AI4KLIM/DATASET/DATA_P02"
WRKDIR=$PWD
EXP_ID=e02
####################### EDIT ABOVE #######################

# Display help message if "--help" is passed
if [[ "$1" == "--help" ]]; then
    echo "Usage: $(basename "$0") [PREFIX] [RUN_ID] [EPOCHS]"
    echo ""
    echo "Arguments:"
    echo "  PREFIX    Optional. Prefix for the run. Default: p05"
    echo "  EPOCHS    Optional. Number of epochs for training. Default: 101"
    exit 0
fi

# Detect working directory and normalize paths
PWD=$(pwd)
if [[ "$PWD" == /nlsasfs/* ]]; then
    WRKDIR="$PWD"
elif [[ "$PWD" == /mnt/* ]]; then
    WRKDIR="/nlsasfs${PWD#/mnt}"
else
    echo "Error: Cannot determine correct path mapping for $PWD"
    exit 1
fi

echo ""
echo "Working directory set to: $WRKDIR"
echo ""

# Check if main.py exists in the normalized path
if [[ ! -f "$WRKDIR/$SCRIPTNAME" ]]; then
    echo "Error: Cannot find $SCRIPTNAME in $WRKDIR"
    exit 1
fi

# Execute the Python script
echo ''
echo "Starting ... ${PREFIX} for ${EPOCHS} epochs."
python3 "$WRKDIR/$SCRIPTNAME" \
        --pwd "$WRKDIR" \
        --exp_id "$EXP_ID" \
        --prefix "$PREFIX" \
        --epochs "$EPOCHS" \
        --dpath "$DATA_PATH" \
        --rpath "$REFD_PATH"

# Print completion message
echo "Job exit at $(date '+%Y-%m-%d %H:%M:%S')"

