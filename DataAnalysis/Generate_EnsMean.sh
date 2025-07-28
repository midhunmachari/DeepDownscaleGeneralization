#!/bin/bash

############ ena ############
for dom in b01 b02 b03
do

    echo -e "\nProcessing ensemble mean ENS: p02a_hp-b32-r7e4-wmae_e01_ena_b13_${dom}_det00_out_2015_2023.nc ..."
    ls -1 *${dom}*.nc
    cdo -v -L -ensmean *${dom}*.nc p02a_hp-b32-r7e4-wmae_e01_ena_b13_${dom}_det00_out_2015_2023.nc
    
    echo -e "\nProcessing ensemble mean DL-ENS: p02a_hp-b32-r7e4-wmae_e01_end_b13_${dom}_det00_out_2015_2023.nc ..."
    ls -1 *m0[0-5]*${dom}*.nc
    cdo -v -L -ensmean *m0[0-5]*${dom}*.nc p02a_hp-b32-r7e4-wmae_e01_end_b13_${dom}_det00_out_2015_2023.nc

    echo -e "\nProcessing ensemble mean GAN-ENS: p02a_hp-b32-r7e4-wmae_e01_eng_b13_${dom}_det00_out_2015_2023.nc ..."
    ls -1 *m0[7-8]*${dom}*.nc
    cdo -v -L -ensmean *m0[7-8]*${dom}*.nc p02a_hp-b32-r7e4-wmae_e01_eng_b13_${dom}_det00_out_2015_2023.nc
 
done




