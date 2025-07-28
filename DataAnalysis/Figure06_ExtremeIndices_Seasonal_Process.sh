#!/bin/bash

SOURCEPATH="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/GENDATA/TESTING/MASTER"

# ##################### DATA_MONMEAN #####################
# DIR="DATA_MONMEAN"
# SOURCEPATH="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/GENDATA/TESTING"
# DESTINPATH="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/PLTDATA/FIGURE05/$DIR"

# mkdir -p "$DESTINPATH"

# # Using for loop over the found files
# for filepath in $(find "$SOURCEPATH" -type f -name "*.nc" | sort); do
#     filename="$(basename "$filepath")"
#     echo "Processing ... $filepath"
#     cdo -v -L -setname,"prec" -monmean "$filepath" "$DESTINPATH/$filename"
# done

##################### DATA_MONMEAN #####################
DIR="DATA_RX1DAY"
DESTINPATH="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/PLTDATA/FM06_ExtremeIndices_Seasonal/$DIR"

mkdir -p "$DESTINPATH"

for filepath in $(find "$SOURCEPATH" -type f -name "*.nc" | sort); do
    filename="$(basename "$filepath")"
    echo "Processing ... $filepath"
    cdo -v -L -setname,"prec" -etccdi_rx1day,freq=month "$filepath" "$DESTINPATH/$filename"
done

##################### DATA_R99/95/90/75/50 #####################

for pctl in 99 95 90; do
  DIR="DATA_R${pctl}"
  DESTINPATH="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/PLTDATA/FM06_ExtremeIndices_Seasonal/$DIR"

  mkdir -p "$DESTINPATH"

  for filepath in $(find "$SOURCEPATH" -type f -name "*.nc" | sort); do
      filename="$(basename "$filepath")"
      echo "Processing ... $filepath"
      cdo -v -L -monpctl,$pctl "$filepath" -monmin "$filepath" -monmax "$filepath" "$DESTINPATH/$filename"
  done
done
