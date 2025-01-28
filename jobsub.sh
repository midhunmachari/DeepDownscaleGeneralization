#!/bin/bash
#SBATCH --job-name=e1m18
#SBATCH --error=e1m18.%J.err
#SBATCH --output=e1m18.%J.out
#SBATCH --partition=testp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=72:00:00

####################### EDIT BELOW #######################
PREFIX="${1:-p02a}"   
EPOCHS="${2:-201}"
SCRIPTNAME='main.py' 

EXP_ID=e01
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

WRKDIR=$PWD

echo "Working directory set to: $WRKDIR"
echo ""

# Check if main.py exists in the normalized path
if [[ ! -f "$WRKDIR/$SCRIPTNAME" ]]; then
    echo "Error: Cannot find $SCRIPTNAME in $WRKDIR"
    exit 1
fi

# Activate the conda environment
source /nlsasfs/home/precipitation/midhunm/Conda/bin/activate tf2
conda list

# Execute the Python script
echo ''
echo "Starting ... ${PREFIX} for ${EPOCHS} epochs."
python3 "$WRKDIR/$SCRIPTNAME" \
        --pwd "$WRKDIR" \
        --exp_id "$EXP_ID" \
        --prefix "$PREFIX" \
        --epochs "$EPOCHS"

# Print completion message
echo "Job exit at $(date '+%Y-%m-%d %H:%M:%S')"

