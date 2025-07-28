#!/bin/bash

SOURCEPATH="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/GENDATA/REF_DATA"
MASKPATH="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/UTILITY/MASKFILES"
DESTINPATH="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/PLTDATA/FM01_StudyArea_RegionStats"

# ######################################### IND_B123C5_EDA_FLDMEAN_2001_2023 ########################################

# #----------------------------------------------------------------------------------------------------
# OUTFILE="IND_B123C5_EDA_FLDMEAN_2001_2023.nc "
# #----------------------------------------------------------------------------------------------------

# DIR=TEMP

# if [ "$1" = "c" ]; then
#     rm -rfv $DIR
# fi

# if [ "$2" = "d" ]; then
#     rm -v ${DESTINPATH}/${OUTFILE}
# fi

# mkdir -p "${DIR}"

# for dom in B01 B02 B03 C01; do
# 	cdo -v \
# 	-setname,${dom} \
# 	-fldmean \
# 	-selyear,2001/2023 \
# 	-ifthen "${MASKPATH}/MASK_${dom}_SM.nc" \
# 	"${SOURCEPATH}/${dom}_010_IMRG_PREC_2001_2023.nc" \
# 	"$DIR/TEMP_${dom}.nc"
# done

# cdo -v -L merge ${DIR}/TEMP*.nc ${DESTINPATH}/${OUTFILE}
# rm -rfv $DIR

# ########################################## IND_B123C5_EDA_YEARMEAN_FLDMEAN_2001_2023 ########################################

# #----------------------------------------------------------------------------------------------------
# OUTFILE="IND_B123C5_EDA_YEARMEAN_FLDMEAN_2001_2023.nc" 
# #----------------------------------------------------------------------------------------------------

# DIR=TEMP

# if [ "$1" = "c" ]; then
#     rm -rfv $DIR
# fi

# if [ "$2" = "d" ]; then
#     rm -v ${DESTINPATH}/${OUTFILE}
# fi

# mkdir -p "${DIR}"

# for dom in B01 B02 B03 C01; do
# 	cdo -v \
# 	-setname,${dom} \
# 	-yearmean \
# 	-fldmean \
# 	-selyear,2001/2023 \
# 	-ifthen "${MASKPATH}/MASK_${dom}_SM.nc" \
# 	"${SOURCEPATH}/${dom}_010_IMRG_PREC_2001_2023.nc" \
# 	"$DIR/TEMP_${dom}.nc"
# done

# cdo -v -L merge ${DIR}/TEMP*.nc ${DESTINPATH}/${OUTFILE}
# rm -rfv $DIR

# ######################################### IND_B123C5_EDA_YMONMEAN_FLDMEAN_2001_2023 ########################################

# #----------------------------------------------------------------------------------------------------
# OUTFILE="IND_B123C5_EDA_YMONMEAN_FLDMEAN_2001_2023.nc" 
# #----------------------------------------------------------------------------------------------------

# DIR=TEMP

# if [ "$1" = "c" ]; then
#     rm -rfv $DIR
# fi

# if [ "$2" = "d" ]; then
#     rm -v ${DESTINPATH}/${OUTFILE}
# fi

# mkdir -p "${DIR}"

# for dom in B01 B02 B03 C01; do
# 	cdo -v \
# 	-setname,${dom} \
# 	-ymonmean \
# 	-fldmean \
# 	-selyear,2001/2023 \
# 	-ifthen "${MASKPATH}/MASK_${dom}_SM.nc" \
# 	"${SOURCEPATH}/${dom}_010_IMRG_PREC_2001_2023.nc" \
# 	"$DIR/TEMP_${dom}.nc"
# done

# cdo -v -L merge ${DIR}/TEMP*.nc ${DESTINPATH}/${OUTFILE}
# rm -rfv $DIR

########################################## IND_B123C5_EDA_RX1DAY_FLDMEAN_2001_2023 ########################################

#----------------------------------------------------------------------------------------------------
OUTFILE="IND_B123C5_EDA_RX1DAY_FLDMEAN_2001_2023.nc" 
#----------------------------------------------------------------------------------------------------

DIR=TEMP

if [ "$1" = "c" ]; then
    rm -rfv $DIR
fi

if [ "$2" = "d" ]; then
    rm -v ${DESTINPATH}/${OUTFILE}
fi

mkdir -p "${DIR}"

for dom in B01 B02 B03 C01; do
	cdo -v \
	-setname,${dom} \
	-fldmean \
	-etccdi_rx1day \
	-selyear,2001/2023 \
	-ifthen "${MASKPATH}/MASK_${dom}_SM.nc" \
	"${SOURCEPATH}/${dom}_010_IMRG_PREC_2001_2023.nc" \
	"$DIR/TEMP_${dom}.nc"
done

cdo -v -L merge ${DIR}/TEMP*.nc ${DESTINPATH}/${OUTFILE}
rm -rfv $DIR
#---------------------------------------# End of the Script #---------------------------------------#

