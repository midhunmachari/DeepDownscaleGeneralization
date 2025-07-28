#!/bin/bash

# Base directories
BASE_DIR="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE"
MASK_DIR="${BASE_DIR}/UTILITY/MASKFILES"
INPUT_DIR="${BASE_DIR}/GENDATA/VALIDATION/MASTER"
OUTPUT_BASE="${BASE_DIR}/PLTDATA/FM02_QQPlot_Fldmean_Series"

# Loop over channels B01, B02, B03
for dom in B01 B02 B03; do
  maskfile="${MASK_DIR}/MASK_${dom}_SM.nc"
  outdir="${OUTPUT_BASE}/${dom}"
  mkdir -p "${outdir}"

  # Process all .nc files inside the subdirectory named “dom”
  for f in "${INPUT_DIR}/${dom}"/*.nc; do
    fname=$(basename "${f}")
    cdo -v -fldmean \
        -ifthen "${maskfile}" \
        "${f}" \
        "${outdir}/${fname}"
  done
done
