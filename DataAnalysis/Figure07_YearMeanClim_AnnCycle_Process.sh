#!/bin/bash

MASKPATH="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/UTILITY/MASKFILES"
SOURCEPATH="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/GENDATA/TESTING/MASTER"
DESTINPATH="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/PLTDATA/FM07_YearMeanClim_AnnCycle"

# ########################################## DATA_YEARMEAN_FLDMEAN_2001_2023 ########################################

# # #----------------------------------------------------------------------------------------------------
# OUTFILE="DATA_YEARMEAN_FLDMEAN_2001_2023.nc" 

# mkdir -p ${DESTINPATH}
# #----------------------------------------------------------------------------------------------------

# DIR=TEMP

# if [ "$1" = "c" ]; then
#     rm -rfv $DIR
# fi

# if [ "$2" = "d" ]; then
#     rm -v ${DESTINPATH}/${OUTFILE}
# fi

# mkdir -p "${DIR}"

# for exp_id in ref e01_m00 e01_m01 e01_m02 e01_m03 e01_m04 e01_m05 e01_m07 e01_m08 e01_end e01_eng; do
# 	filepath=$(find ${SOURCEPATH} -maxdepth 1 -name "*${exp_id}*.nc" -type f | sed 's/\.\///')
# 	filename=$(basename "$filepath")
# 	echo ''
# 	echo "Processing ... ${filename}: ${filepath}"
# 	cdo -v \
# 	-setname,${exp_id^^} \
# 	-yearmean \
# 	-fldmean \
# 	-selyear,2001/2023 \
# 	-ifthen "${MASKPATH}/MASK_C01_SM.nc" \
# 	"${SOURCEPATH}/${filename}" \
# 	"$DIR/TEMP_${exp_id^^}.nc"
# done

# cdo -v -L merge ${DIR}/TEMP*.nc ${DESTINPATH}/${OUTFILE}
# rm -rfv $DIR

# ########################################## DATA_YEARMEAN_FLDMEAN_2001_2023 ########################################

# #----------------------------------------------------------------------------------------------------
OUTFILE="DATA_MONMEAN_FLDMEAN_2001_2023.nc" 

mkdir -p ${DESTINPATH}
#----------------------------------------------------------------------------------------------------

DIR=TEMP

if [ "$1" = "c" ]; then
    rm -rfv $DIR
fi

if [ "$2" = "d" ]; then
    rm -v ${DESTINPATH}/${OUTFILE}
fi

mkdir -p "${DIR}"

for exp_id in ref e01_m00 e01_m01 e01_m02 e01_m03 e01_m04 e01_m05 e01_m07 e01_m08 e01_end e01_eng; do
	filepath=$(find ${SOURCEPATH} -maxdepth 1 -name "*${exp_id}*.nc" -type f | sed 's/\.\///')
	filename=$(basename "$filepath")
	echo ''
	echo "Processing ... ${filename}: ${filepath}"
	cdo -v \
	-setname,${exp_id^^} \
	-monmean \
	-fldmean \
	-selyear,2001/2023 \
	-ifthen "${MASKPATH}/MASK_C01_SM.nc" \
	"${SOURCEPATH}/${filename}" \
	"$DIR/TEMP_${exp_id^^}.nc"
done

cdo -v -L merge ${DIR}/TEMP*.nc ${DESTINPATH}/${OUTFILE}
rm -rfv $DIR

# ######################################### DATA_YMONMEAN_FLDMEAN_2001_2023 ########################################

# # #----------------------------------------------------------------------------------------------------
# OUTFILE="DATA_YMONMEAN_FLDMEAN_2001_2023.nc" 
# mkdir -p ${DESTINPATH}
# #----------------------------------------------------------------------------------------------------

# DIR=TEMP

# if [ "$1" = "c" ]; then
#     rm -rfv $DIR
# fi

# if [ "$2" = "d" ]; then
#     rm -v ${DESTINPATH}/${OUTFILE}
# fi

# mkdir -p "${DIR}"

# for exp_id in ref e01_m00 e01_m01 e01_m02 e01_m03 e01_m04 e01_m05 e01_m07 e01_m08 e01_end e01_eng; do
# 	filepath=$(find ${SOURCEPATH} -maxdepth 1 -name "*${exp_id}*.nc" -type f | sed 's/\.\///')
# 	filename=$(basename "$filepath")
# 	echo ''
# 	echo "Processing ... ${filename}: ${filepath}"
# 	cdo -v \
# 	-setname,${exp_id^^} \
# 	-ymonmean \
# 	-fldmean \
# 	-selyear,2001/2023 \
# 	-ifthen "${MASKPATH}/MASK_C01_SM.nc" \
# 	"${SOURCEPATH}/${filename}" \
# 	"$DIR/TEMP_${exp_id^^}.nc"
# done

# cdo -v -L merge ${DIR}/TEMP*.nc ${DESTINPATH}/${OUTFILE}
# rm -rfv $DIR

# ######################################### DATA_YDAYMEAN_FLDMEAN_2001_2023 ########################################

# # #----------------------------------------------------------------------------------------------------
# OUTFILE="DATA_YDAYMEAN_FLDMEAN_2001_2023.nc" 
# mkdir -p ${DESTINPATH}
# #----------------------------------------------------------------------------------------------------

# DIR=TEMP

# if [ "$1" = "c" ]; then
#     rm -rfv $DIR
# fi

# if [ "$2" = "d" ]; then
#     rm -v ${DESTINPATH}/${OUTFILE}
# fi

# mkdir -p "${DIR}"

# for exp_id in ref e01_m00 e01_m01 e01_m02 e01_m03 e01_m04 e01_m05 e01_m07 e01_m08 e01_end e01_eng; do
# 	filepath=$(find ${SOURCEPATH} -maxdepth 1 -name "*${exp_id}*.nc" -type f | sed 's/\.\///')
# 	filename=$(basename "$filepath")
# 	echo ''
# 	echo "Processing ... ${filename}: ${filepath}"
# 	cdo -v \
# 	-setname,${exp_id^^} \
# 	-ydaymean \
# 	-fldmean \
# 	-selyear,2001/2023 \
# 	-ifthen "${MASKPATH}/MASK_C01_SM.nc" \
# 	"${SOURCEPATH}/${filename}" \
# 	"$DIR/TEMP_${exp_id^^}.nc"
# done

# cdo -v -L merge ${DIR}/TEMP*.nc ${DESTINPATH}/${OUTFILE}
# rm -rfv $DIR

# #---------------------------------------# End of the Script #---------------------------------------#

