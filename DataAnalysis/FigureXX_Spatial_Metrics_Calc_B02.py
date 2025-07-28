#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:18:22 2024

@author: midhunm
"""

from ai4klima.evalmetrics import calculate_evalmetrics_spatial, create_directory

def main(dom, period, DATA_PATH, SAVE_PATH):

    create_directory(SAVE_PATH)

    checkdict = {
        'e01': 'det00',
        # 'e02': 'emn12',
    }

    for exp_id, mode in checkdict.items():
        data_dict = {  
            'REF'    : [f'{DATA_PATH}/p02a_ref_{dom}_{period}.nc', 'IMERG'],
            'SRCNN'  : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m00_b13_{dom}_{mode}_out_{period}.nc', 'M00: SRCNN'], 
            'FSRCNN' : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m01_b13_{dom}_{mode}_out_{period}.nc', 'M01: FSRCNN'],
            'SRDRN'  : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m03_b13_{dom}_{mode}_out_{period}.nc', 'M03: SRDRN'],
            'EDRN'   : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m02_b13_{dom}_{mode}_out_{period}.nc', 'M02: EDRN'],
            'U-NET'  : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m04_b13_{dom}_{mode}_out_{period}.nc', 'M04: U-NET'], 
            'AU-NET' : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m05_b13_{dom}_{mode}_out_{period}.nc', 'M05: AU-NET'], 
            'U-GAN'  : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m07_b13_{dom}_{mode}_out_{period}.nc', 'M07: U-GAN'], 
            'AU-GAN' : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m08_b13_{dom}_{mode}_out_{period}.nc', 'M08: AU-GAN'],
            'ENS'    : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_ena_b13_{dom}_{mode}_out_{period}.nc', 'ENA: ENS'], 
            'DL-ENS' : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_end_b13_{dom}_{mode}_out_{period}.nc', 'END: DL-ENS'], 
            'GAN-ENS': [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_eng_b13_{dom}_{mode}_out_{period}.nc', 'ENG: GAN-ENS'],  
            }
    

    calculate_evalmetrics_spatial(data_dict, data_path=DATA_PATH, save_path=SAVE_PATH, varname='prec', tag=f"P02A.{exp_id}.{mode}.{dom.upper()}.{period}")

if __name__ == "__main__":
    dom = 'b02'             # Edit here
    period = '2015_2023'    # Edit here
    
    DATA_PATH = f"/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/GENDATA/VALIDATION/MASTER/{dom.upper()}"
    SAVE_PATH = f"/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/PLTDATA/FM08_Spatial_Metrics"
    main(dom = dom, period = period, DATA_PATH = DATA_PATH, SAVE_PATH = f"{SAVE_PATH}/EVAL_METRICS_SPATIAL_E01_{dom.upper()}_{period}")
    













