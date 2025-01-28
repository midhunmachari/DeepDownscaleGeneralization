#!/bin/bash
#SBATCH --job-name=q0e1s1
#SBATCH --error=q0e1s1.%J.err
#SBATCH --output=q0e1s1.%J.out
#SBATCH --partition=testp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=72:00:00

####################### EDIT BELOW #######################
EXPID_LIST=(e01)
INPID_LIST=(s01)

PREFIX="${1:-p02}"   
EPOCHS="${2:-21}"
SCRIPTNAME='main.py' 
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

for EXP_ID in "${EXPID_LIST[@]}"; do
    for INP_ID in "${INPID_LIST[@]}"; do
    
        # Execute the Python script
        echo ''
        echo "Starting ... ${PREFIX}_${INP_ID}_${EXP_ID} for ${EPOCHS} epochs."
        python3 "$WRKDIR/$SCRIPTNAME" \
                --pwd "$WRKDIR" \
                --inp_id "$INP_ID" \
                --exp_id "$EXP_ID" \
                --prefix "$PREFIX" \
                --epochs "$EPOCHS"

    done
done

# Print completion message
echo "Job exit at $(date '+%Y-%m-%d %H:%M:%S')"

