#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:18:22 2024

@author: midhunm
"""

from ai4klima.evalmetrics import make_percentile_events_scores_table, make_threshold_events_scores_table

def main(DATA_PATH, SAVE_PATH):

    checkdict = {
        'e01': 'det00',
        # 'e02': 'emn12',
    }

    for exp_id, mode in checkdict.items():
        data_dict = {  
            'REF'    : [f'{DATA_PATH}/p02a_ref_c01_2001_2023.nc', 'IMERG'],
            'SRCNN'  : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m00_b13_c01_{mode}_out_2001_2023.nc', 'M00: SRCNN'  ], 
            'FSRCNN' : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m01_b13_c01_{mode}_out_2001_2023.nc', 'M01: FSRCNN' ],
            'EDRN'   : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m02_b13_c01_{mode}_out_2001_2023.nc', 'M02: EDRN'   ],
            'SRDRN'  : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m03_b13_c01_{mode}_out_2001_2023.nc', 'M03: SRDRN'  ],
            'U-NET'  : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m04_b13_c01_{mode}_out_2001_2023.nc', 'M04: U-NET'  ], 
            'AU-UNET': [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m05_b13_c01_{mode}_out_2001_2023.nc', 'M05: AU-NET' ], 
            'U-GAN'  : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m07_b13_c01_{mode}_out_2001_2023.nc', 'M07: U-GAN'  ], 
            'AU-GAN' : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m08_b13_c01_{mode}_out_2001_2023.nc', 'M08: AU-GAN' ],
            'DL-ENS' : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_end_b13_c01_{mode}_out_2001_2023.nc', 'END: DL-ENS' ], 
            'GAN-ENS': [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_eng_b13_c01_{mode}_out_2001_2023.nc', 'ENG: GAN-ENS'], 
            'ENS'    : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_ena_b13_c01_{mode}_out_2001_2023.nc', 'ENA: ENS'    ], 
            }
        
        MASK_PATH = "/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/UTILITY/MASKFILES"
        mask_dict = {
            'ALL': [f'{MASK_PATH}/MASK_C01_ALL.nc', 'ALL GRIDS'],
            'SM': [f'{MASK_PATH}/MASK_C01_SM.nc', 'SEA MASK'], 
            'SMM': [f'{MASK_PATH}/MASK_C01_SMM.nc', 'SEA MOUNTAIN MASK'], \
        }

        try:

            # ----------------------------------
            # Events above Percentile Threshold
            # ----------------------------------

            make_percentile_events_scores_table(
                data_dict,
                percentile_thresh=99.9,
                mask_dict=mask_dict, 
                suffix=f'BINARY_METRICS_A999P.C01.2001-23.{exp_id}', 
                csv_save_path = SAVE_PATH
                )

            make_percentile_events_scores_table(
                data_dict,
                percentile_thresh=99,
                mask_dict=mask_dict, 
                suffix=f'BINARY_METRICS_A099P.C01.2001-23.{exp_id}', 
                csv_save_path = SAVE_PATH
                )
    
            make_percentile_events_scores_table(
                data_dict,
                percentile_thresh=95,
                mask_dict=mask_dict, 
                suffix=f'BINARY_METRICS_A095P.C01.2001-23.{exp_id}', 
                csv_save_path = SAVE_PATH
                )

            make_percentile_events_scores_table(
                data_dict,
                percentile_thresh=90,
                mask_dict=mask_dict, 
                suffix=f'BINARY_METRICS_A090P.C01.2001-23.{exp_id}', 
                csv_save_path = SAVE_PATH
                )

            make_percentile_events_scores_table(
                data_dict,
                percentile_thresh=75,
                mask_dict=mask_dict, 
                suffix=f'BINARY_METRICS_A075P.C01.2001-23.{exp_id}', 
                csv_save_path = SAVE_PATH
                )

            make_percentile_events_scores_table(
                data_dict,
                percentile_thresh=50,
                mask_dict=mask_dict, 
                suffix=f'BINARY_METRICS_A050P.C01.2001-23.{exp_id}', 
                csv_save_path = SAVE_PATH
                )

            # ----------------------------------
            # Events above Intensity Threshold
            # ----------------------------------

            make_threshold_events_scores_table(
                data_dict,
                threshold=1,
                mask_dict=mask_dict, 
                suffix=f'BINARY_METRICS_I001MM.C01.2001-23.{exp_id}', 
                csv_save_path = SAVE_PATH
                )
            
            make_threshold_events_scores_table(
                data_dict,
                threshold=5,
                mask_dict=mask_dict, 
                suffix=f'BINARY_METRICS_I005MM.C01.2001-23.{exp_id}', 
                csv_save_path = SAVE_PATH
                )

            make_threshold_events_scores_table(
                data_dict,
                threshold=10,
                mask_dict=mask_dict, 
                suffix=f'BINARY_METRICS_I010MM.C01.2001-23.{exp_id}', 
                csv_save_path = SAVE_PATH
                )

            make_threshold_events_scores_table(
                data_dict,
                threshold=20,
                mask_dict=mask_dict, 
                suffix=f'BINARY_METRICS_I020MM.C01.2001-23.{exp_id}', 
                csv_save_path = SAVE_PATH
                )

            make_threshold_events_scores_table(
                data_dict,
                threshold=50,
                mask_dict=mask_dict, 
                suffix=f'BINARY_METRICS_I050MM.C01.2001-23.{exp_id}', 
                csv_save_path = SAVE_PATH
                )

            make_threshold_events_scores_table(
                data_dict,
                threshold=100,
                mask_dict=mask_dict, 
                suffix=f'BINARY_METRICS_I100MM.C01.2001-23.{exp_id}', 
                csv_save_path = SAVE_PATH
                )
            
        except Exception as e:
            print(f"Error occurred while making the evaluation table: {e}")



if __name__ == "__main__":

    DATA_PATH = "/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/GENDATA/TESTING/MASTER"
    SAVE_PATH = "/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/PLTDATA"
    main(DATA_PATH, SAVE_PATH)


