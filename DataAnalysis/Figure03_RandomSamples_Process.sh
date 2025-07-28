#!/bin/bash

SOURCEPATH="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/GENDATA/TESTING/MASTER"
DESTINPATH="/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/PLTDATA/FM03_RandomSamples"
mkdir -p "${DESTINPATH}"

datalist=(ref inp e01_m00 e01_m01 e01_m02 e01_m03 e01_m04 e01_m05 e01_m07 e01_m08 e01_ena e01_end e01_eng)

# timestep_list=(567 1235 1287 1376 2392 3568 6371 6765 7967)

timestep_list=(152  154  155  156  503  566  567  568  588  886  891  
940 1234 1235 1283 1284 1285 1286 1287 1351 1352 1375 1376 1976 2025 
2325 2353 2391 2392 2393 2397 2398 2438 2440 3103 3567 3568 3851 3852 
4193 4194 4222 4591 4599 4973 4974 5289 5320 5324 5344 5345 5620 5640 
5684 5711 5994 6007 6034 6048 6066 6067 6137 6138 6358 6370 6371 6372 
6373 6763 6764 6765 6766 7108 7169 7486 7836 7837 7838 7967 8169 8200 
8253 8254 8268)

for timestep in "${timestep_list[@]}"; do

    OUTFILE="C01_RANDOMDAY_t${timestep}.nc"

    DIR=TEMP

    if [ "$1" = "c" ]; then
        rm -rfv "${DIR}"
    fi

    if [ "$2" = "d" ]; then
        rm -v "${DESTINPATH}/${OUTFILE}"
    fi

    mkdir -p "${DIR}"

    for data_id in "${datalist[@]}"; do
        filepath=$(find "${SOURCEPATH}" -maxdepth 1 -name "*${data_id}*.nc" -type f | sed 's|^\./||')
        filename=$(basename "${filepath}")
        echo "${filename}" "${data_id^^}"

        cdo -v -L \
        -setname,"${data_id^^}" \
        -seltimestep,${timestep} \
        "${SOURCEPATH}/${filename}" \
        "${PWD}/${DIR}/TEMP_${data_id}.nc"
        
		echo ''
    done

    echo "${OUTFILE}"
    cdo -v -L merge "${DIR}"/TEMP*.nc "${DESTINPATH}/${OUTFILE}"

done

#---------------------------------------# End of the Script #---------------------------------------#
