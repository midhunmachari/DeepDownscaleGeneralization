from ai4klima.evalmetrics import make_eval_table, create_directory

def main(dom, tag, DATA_PATH):

    checkdict = {
        'e01': 'det00',
        # 'e02': 'emn12',
    }

    for exp_id, mode in checkdict.items():
        data_dict = {  
            'REF'    : [f'{DATA_PATH}/{dom}_010_IMRG_PREC_2015_2023.nc', 'IMERG'],
            'SRCNN'  : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m00_b13_{dom.lower()}_{mode}_out_2015_2023.nc', 'M00: SRCNN'  ], 
            'FSRCNN' : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m01_b13_{dom.lower()}_{mode}_out_2015_2023.nc', 'M01: FSRCNN' ],
            'EDRN'   : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m02_b13_{dom.lower()}_{mode}_out_2015_2023.nc', 'M02: EDRN'   ],
            'SRDRN'  : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m03_b13_{dom.lower()}_{mode}_out_2015_2023.nc', 'M03: SRDRN'  ],
            'U-NET'  : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m04_b13_{dom.lower()}_{mode}_out_2015_2023.nc', 'M04: U-NET'  ], 
            'AU-UNET': [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m05_b13_{dom.lower()}_{mode}_out_2015_2023.nc', 'M05: AU-NET' ], 
            'U-GAN'  : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m07_b13_{dom.lower()}_{mode}_out_2015_2023.nc', 'M07: U-GAN'  ], 
            'AU-GAN' : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_m08_b13_{dom.lower()}_{mode}_out_2015_2023.nc', 'M08: AU-GAN' ],
            'DL-ENS' : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_end_b13_{dom.lower()}_{mode}_out_2015_2023.nc', 'END: DL-ENS' ], 
            'GAN-ENS': [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_eng_b13_{dom.lower()}_{mode}_out_2015_2023.nc', 'ENG: GAN-ENS'], 
            'ENS'    : [f'{DATA_PATH}/p02a_hp-b32-r7e4-wmae_{exp_id}_ena_b13_{dom.lower()}_{mode}_out_2015_2023.nc', 'ENA: ENS'    ], 
            }
        
        MASK_PATH = "/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/UTILITY/MASKFILES" # Edit here
        mask_dict = {
            'ALL': [f'{MASK_PATH}/MASK_{dom}_ALL.nc', 'ALL GRIDS'],
            'SM':  [f'{MASK_PATH}/MASK_{dom}_SM.nc', 'SEA MASK'], 
        }

        SAVE_PATH = "/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/PLTDATA" # Edit here
        create_directory(SAVE_PATH)

        try:
            make_eval_table(data_dict, mask_dict=mask_dict, jjas_only=False, csv_save_path=SAVE_PATH, tag=f'{tag}.{exp_id}.{mode}') # Edit here
            print("Evaluation table created successfully.")
        except Exception as e:
            print(f"Error occurred while making the evaluation table: {e}")

if __name__ == "__main__":

    for dom in ['B01', 'B02', 'B03']:
        print(f"\nStart processing domainL {dom}")
        
        PATH = "/home/midhunm/AI4KLIM/EXPMNTS/P02A_DeepDown_Generalization/ANALYSE/GENDATA/VALIDATION" # Edit here
        path_dict = {
            f"P02A.ALLDAY.{dom}.2015-23" : f"{PATH}/MASTER",
        }

        for tag, data_path in path_dict.items():
            main(dom, tag, data_path)

