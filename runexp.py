#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:52:48 2024

@author: midhunm
"""
import os
import gc
import numpy as np
import xarray as xr
import tensorflow as tf
from utils import configure_model, loadstack_data_pairs, load_checkpoints

from ai4klima.tensorflow.train import ModelTraining, Pix2Pix, SRGAN


class RunExperiments:
    
    def __init__(
            self, 
            prefix, 
            inp_id, 
            exp_id, 
            data_path, 
            save_path, 
            refd_path, 
            models_dict, 
            gen_lr, 
            dis_lr, 
            loss_obj, 
            ckpts_dict=None
            ):
        
        self.prefix = prefix
        self.expset_id = inp_id
        self.exp_id = exp_id
        self.data_path = data_path
        self.save_path = save_path
        self.refd_path = refd_path

        self.loss_fn = loss_obj
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr

        self.train_bounds = (None, 5113)
        self.val_bounds = (5113, 6209)
        self.test_bounds  = (6209, None)

        self.models_dict = models_dict
        self.ckpts_dict  = ckpts_dict
        
        self.boxes_dict = {
            'b01' : 'B01',
            'b02' : 'B02',
            'b03' : 'B03',
            'c01' : 'C01',
            'c02' : 'C02',
            'c03' : 'C03',
            'c04' : 'C04',
            'c05' : 'C05',
            'c06' : 'C06',
            }
        
    def get_ref_path(self, box):
        return f"{self.refd_path}/{box}_GRID_INFO.nc" # Edit here 
    
    ################################################# EXP: E01 #################################################
    def experiment_01(self, epochs, bs):
        """
        E01: TRAIN AND VALIDATE ON B1, B2, B3. DEPLOY IT TO GENERATE TEST ON EACH DOMAIN.
        """
        print('\n' + '#' * 100)
        print("Processing data ...")
        
        # Load training and validation data
        boxes = ['B01', 'B02', 'B03']    
        train_data = loadstack_data_pairs(boxes, self.data_path, bounds=self.train_bounds, expset=self.expset_id)
        val_data = loadstack_data_pairs(boxes, self.data_path, bounds=self.val_bounds, expset=self.expset_id)
        
        for model_id in self.models_dict.keys():

            expname = f"{self.prefix}_{self.expset_id}_{self.exp_id}_{model_id}"
            print(f'\nInitiate experiment: {expname}')
            print('-'*100)

            gen_opt, dis_opt = self.models_dict[model_id]
            
            # Model Training and Plot Learning Curves
            mt = self.model_trainer(prefix = expname,
                                    suffix = 'b11', 
                                    model_id = model_id, 
                                    gen_opt = gen_opt, 
                                    dis_opt = dis_opt, 
                                    train_data = train_data, 
                                    val_data = val_data, 
                                    epochs = epochs, 
                                    bs = bs, 
                                    gen_ckpt=None, 
                                    dis_ckpt=None
                                    )

            # Generate the test netcdf; Loop through all boxes
            for box_id, box in self.boxes_dict.items():
                
                # Load testing data
                X_test, _ = loadstack_data_pairs(box, self.data_path, bounds=self.test_bounds, expset=self.expset_id) # Edit here
                # Generate the test netcdf
                mt.generate_data_and_build_netcdf(
                    X_test, 
                    model_path = None,
                    refd_path=self.get_ref_path(box), 
                    save_raw_npy=True, # Edit here
                    build_netcdf=True, # Edit here
                    varname = 'prec', 
                    start_date = "2018-01-01",  # Edit here
                    end_date   = "2023-12-31",  # Edit here
                    tag = box_id, # Edite here
                )

            del mt
            gc.collect()
            
            print('\nEND OF EXPERIMENT-01: TRAIN AND VALIDATE ON B1, B2, B3. DEPLOY IT TO GENERATE TEST ON EACH DOMAINS')
    
    ################################################# EXP: E02 #################################################
    def experiment_02(self, epochs, bs):
        """
        E02: RETRAIN R01 MODELS AND GENERATE TEST ON EACH DOMAIN SEPARATE 
        """
        
        for box_id, box in self.boxes_dict.items():
            
            print('\n' + '#' * 100)
            print("Processing data ...")
            
            train_data = loadstack_data_pairs(box, self.data_path, bounds=self.train_bounds, expset=self.expset_id)
            val_data   = loadstack_data_pairs(box, self.data_path, bounds=self.val_bounds,   expset=self.expset_id)
            
            for model_id in self.models_dict.keys():

                print('-'*100) 
                expname = f"{self.prefix}_{self.expset_id}_{self.exp_id}_{model_id}"
                print(f'\nInitiate experiment: {expname}')
                print('-'*100) 

                # Unpack model architecture
                gen_opt, dis_opt = self.models_dict[model_id]

                # Unpack checkpoint filenames
                gen_ckpt, dis_ckpt = self.ckpts_dict[model_id]

                # Model Training and Plot Learning Curves
                mt = self.model_trainer(prefix = expname,
                                        suffix = box_id,  # Edit here
                                        model_id = model_id, 
                                        gen_opt = gen_opt, 
                                        dis_opt = dis_opt, 
                                        train_data = train_data, 
                                        val_data = val_data, 
                                        epochs = epochs, 
                                        bs = bs, 
                                        gen_ckpt=gen_ckpt,  # Edit here
                                        dis_ckpt=dis_ckpt,  # Edit here
                                        )

                # Load testing data
                X_test, _ = loadstack_data_pairs(box, self.data_path, bounds=self.test_bounds, expset=self.expset_id) # Edit here
                # Generate the test netcdf
                mt.generate_data_and_build_netcdf(
                    X_test, 
                    model_path = None,
                    refd_path=self.get_ref_path(box), 
                    save_raw_npy=True, # Edit here
                    build_netcdf=True, # Edit here
                    varname = 'prec', 
                    start_date = "2018-01-01",  # Edit here
                    end_date   = "2023-12-31",  # Edit here
                    tag = box_id, # Edite here
                    )
                
            # del mt
            # gc.collect()
            
        print('\nEND OF EXPERIMENT-02: RETRAIN R01 MODELS AND GENERATE TEST ON EACH DOMAIN SEPARATE')

    ################################################# EXP: E03 #################################################
    def experiment_03(self, epochs, bs):
        """
        E03: TRAIN, VALIDATE AND GENERATE TEST ON EACH DOMAIN SEPARATE.
        """

        for box_id, box in self.boxes_dict.items():
            
            print('\n' + '#' * 100)
            print("Processing data ...")
            
            train_data = loadstack_data_pairs(box, self.data_path, bounds=self.train_bounds, expset=self.expset_id)
            val_data   = loadstack_data_pairs(box, self.data_path, bounds=self.val_bounds,   expset=self.expset_id)
        
            # Assuming X and y have the same number of samples
            for model_id in self.models_dict.keys():
                
                print('-'*100)
                expname = f"{self.prefix}_{self.expset_id}_{self.exp_id}_{model_id}"
                print(f'\nInitiate experiment: {expname}')
                print('-'*100)

                # Unpack model architecture
                gen_opt, dis_opt = self.models_dict[model_id]

                # Model Training and Plot Learning Curves
                mt = self.model_trainer(prefix = expname,
                                        suffix = box_id,  # Edit here
                                        model_id = model_id, 
                                        gen_opt = gen_opt, 
                                        dis_opt = dis_opt, 
                                        train_data = train_data, 
                                        val_data = val_data, 
                                        epochs = epochs, 
                                        bs = bs, 
                                        gen_ckpt=None,  # Edit here
                                        dis_ckpt=None,  # Edit here
                                        )

                # Load testing data
                X_test, _ = loadstack_data_pairs(box, self.data_path, bounds=self.test_bounds, expset=self.expset_id) # Edit here
                # Generate the test netcdf
                mt.generate_data_and_build_netcdf(
                    X_test, 
                    model_path = None,
                    refd_path=self.get_ref_path(box), 
                    save_raw_npy=True, # Edit here
                    build_netcdf=True, # Edit here
                    varname = 'prec', 
                    start_date = "2018-01-01",  # Edit here
                    end_date   = "2023-12-31",  # Edit here
                    tag = box_id, # Edite here
                )

            # del mt
            # gc.collect()

        print('\nEND OF EXPERIMENT-03: TRAIN, VALIDATE AND GENERATE TEST ON EACH DOMAIN SEPARATE')

    ################################################# EXP: E04 #################################################
    def experiment_04(self, epochs, bs):
        """
        E04: TRAIN ON GLOBAL POOL AND GENERATE TEST ON EACH DOMAIN.
        """
        print('\n' + '#' * 100)
        print("Processing data ...")
        
        # Load training and validation data
        boxes = ['B01', 'B02', 'B03', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06']
        train_data = loadstack_data_pairs(boxes, self.data_path, bounds=self.train_bounds, expset=self.expset_id)
        val_data = loadstack_data_pairs(boxes, self.data_path, bounds=self.val_bounds, expset=self.expset_id)
        
        for model_id in self.models_dict.keys():
            
            print('-'*100)
            expname = f"{self.prefix}_{self.expset_id}_{self.exp_id}_{model_id}"
            print(f'\nInitiate experiment: {expname}')
            print('-'*100)

            gen_opt, dis_opt = self.models_dict[model_id]

            # Model Training and Plot Learning Curves
            mt = self.model_trainer(prefix = expname,
                                    suffix = 'b44',  # Edit here
                                    model_id = model_id, 
                                    gen_opt = gen_opt, 
                                    dis_opt = dis_opt, 
                                    train_data = train_data, 
                                    val_data = val_data, 
                                    epochs = epochs, 
                                    bs = bs, 
                                    gen_ckpt=None,  # Edit here
                                    dis_ckpt=None,  # Edit here
                                    )
            
            # Generate the test netcdf; Loop through all boxes
            for box_id, box in self.boxes_dict.items():
                
                # Load testing data
                X_test, _ = loadstack_data_pairs(box, self.data_path, bounds=self.test_bounds, expset=self.expset_id) # Edit here
                # Generate the test netcdf
                mt.generate_data_and_build_netcdf(
                    X_test, 
                    model_path = None,
                    refd_path=self.get_ref_path(box), 
                    save_raw_npy=True, # Edit here
                    build_netcdf=True, # Edit here
                    varname = 'prec', 
                    start_date = "2018-01-01",  # Edit here
                    end_date   = "2023-12-31",  # Edit here
                    tag = box_id, # Edite here
                )
        
            # del mt
            # gc.collect()
            
        print('\nEND OF EXPERIMENT-04: TRAIN ON GLOBAL POOL AND GENERATE TEST ON EACH DOMAIN.')

    ################################################# MODEL TRAINER #################################################
    def model_trainer(self, prefix, suffix, model_id, gen_opt, dis_opt, train_data, val_data, epochs, bs, gen_ckpt=None, dis_ckpt=None):

        X_train, y_train = train_data
        X_val, y_val = val_data

        print(f"X_train shape: {X_train[:,:,:,0].shape}, X_train max.: {X_train[:,:,:,0].max()}")
        print(f"X_val shape: {X_val[:,:,:,0].shape}, X_val max.: {X_val[:,:,:,0].max()}")
        print(f"y_train shape: {y_train[:,:,:,0].shape}, y_train max.: {y_train[:,:,:,0].max()}")
        print(f"y_val shape: {y_val[:,:,:,0].shape}, y_val max.: {y_val[:,:,:,0].max()}")

        try:
            
            ################################# TRAIN: FSRCNN, SRDRN, UNET, AUNET #################################
            if model_id in ['m01', 'm02', 'm03', 'm04']:
                """
                Train UNET/Attention-UNET with WMAE Loss Function -> Deterministic Modelling
                """
                if gen_ckpt is None:
                    gen_arch = configure_model(gen_opt, X_train.shape[1:]) # Edit here
                else:
                    gen_arch = load_checkpoints(gen_ckpt)
                    # Configure model architecture and compare
                    gen_config = configure_model(self.model_id, X_train.shape[1:])
                    if not isinstance(gen_arch, type(gen_config)):
                        raise ValueError(f"The loaded model does not match the expected configuration for model_id: {model_id}")

                mt = ModelTraining(
                    prefix = prefix, 
                    save_path = self.save_path,
                    generator = gen_arch,
                    loss_fn = self.loss_fn, # Edit here
                    lr_init = self.gen_lr,
                    log_tensorboard = True,
                    enable_function = True,
                    suffix = suffix,
                    )
                mt.train_by_fit(
                    train_data = (X_train, y_train), 
                    val_data = (X_val, y_val), 
                    epochs = epochs,  # Edit here
                    batch_size = bs, 
                    monitor="val_mean_absolute_error",
                    mode = "min",
                    min_lr = 1e-10,
                    save_ckpt = True,
                    ckpt_interval = 1,
                    save_ckpt_best = True,
                    # lrdecay_scheduler = True,
                    # lrdecay_factor = 0.1,
                    # lrdecay_wait = 3,
                    # lrdecay_interval = 1,
                    reducelr_on_plateau = True,
                    reducelr_factor = 0.1,
                    reducelr_patience = 12,
                    early_stopping=True,
                    early_stopping_patience = 32,
                    )

            ################################# TRAIN: SRGAN #################################
            elif model_id in ['m05', 'm06', 'm07']: # SRGAN

                """
                Train SRGAN -> Generative Modelling
                """
                if gen_ckpt is None:
                    gen_arch = configure_model(gen_opt, X_train.shape[1:]) # Edit here
                else:
                    gen_arch = load_checkpoints(gen_ckpt)
                    # Configure model architecture and compare
                    gen_config = configure_model(self.model_id, X_train.shape[1:])
                    if not isinstance(gen_arch, type(gen_config)):
                        raise ValueError(f"The loaded model does not match the expected configuration for model_id: {model_id}")
                
                if dis_ckpt is None:
                    dis_arch = configure_model(dis_opt, y_train.shape[1:]) # Edit here
                else:
                    dis_arch = load_checkpoints(dis_ckpt)
                    # Configure model architecture and compare
                    dis_config = configure_model(dis_opt, y_train.shape[1:]) # Edit here
                    if not isinstance(gen_arch, type(dis_config)):
                        raise ValueError(f"The loaded model does not match the expected configuration for model_id: {model_id}")
                
                mt = SRGAN(
                    prefix = prefix,  
                    save_path = self.save_path,
                    generator= gen_arch, 
                    discriminator = dis_arch,
                    gen_lr_init = self.gen_lr,
                    dis_lr_init = self.dis_lr,
                    # extract_features=False, 
                    # feature_extractor='VGG19',  
                    # output_layer='block3_conv4', 
                    # pretrained_weights="imagenet",
                    cl_opt='WMAE',
                    lambda_value=1e-3,
                    log_tensorboard = True,
                    enable_function=True,
                    suffix = suffix,
                    )
                mt.train(
                    train_data = (X_train, y_train),
                    val_data = (X_val, y_val),
                    epochs = epochs, 
                    batch_size = bs,
                    monitor= "val_mean_absolute_error", 
                    mode = "min",
                    min_lr_gen = 1e-10,
                    min_lr_dis = 1e-10,
                    save_ckpt = True,
                    ckpt_interval = 1,
                    save_ckpt_best = True,
                    # lrdecay_gen = True,
                    # lrdecay_factor_gen = 0.1,
                    # lrdecay_wait_gen = 3,
                    # lrdecay_interval_gen = 1,
                    # lrdecay_dis = False,
                    # lrdecay_factor_dis = None,
                    # lrdecay_wait_dis = None,
                    # lrdecay_interval_dis = None,
                    reducelr_on_plateau_gen = True,
                    reducelr_factor_gen = 0.1,
                    reducelr_patience_gen = 12,
                    # reducelr_on_plateau_dis = False,
                    # reducelr_factor_dis = None,
                    # reducelr_patience_dis = None,
                    early_stopping=False,
                    early_stopping_patience = 32,
                    )

            # ################################# TRAIN: PIX2PIX #################################
            # elif model_id=='m07':

            #     """
            #     Train Pix2Pix -> Generative Modelling
            #     """
            #     if gen_ckpt is None:
            #         gen_arch = configure_model(gen_opt, X_train.shape[1:]) # Edit here
            #     else:
            #         gen_arch = load_checkpoints(gen_ckpt)
            #         # Configure model architecture and compare
            #         gen_config = configure_model(self.model_id, X_train.shape[1:])
            #         if not isinstance(gen_arch, type(gen_config)):
            #             raise ValueError(f"The loaded model does not match the expected configuration for model_id: {model_id}")
                
            #     if dis_ckpt is None:
            #         dis_arch = configure_model(dis_opt, X_train.shape[1:], y_train.shape[1:]) # Edit here
            #     else:
            #         dis_arch = load_checkpoints(dis_ckpt)
            #         # Configure model architecture and compare
            #         dis_config = configure_model(dis_opt, X_train.shape[1:], y_train.shape[1:]) # Edit here
            #         if not isinstance(dis_arch, type(dis_config)):
            #             raise ValueError(f"The loaded model does not match the expected configuration for model_id: {model_id}")
                
            #     mt = Pix2Pix(
            #         prefix = prefix,  
            #         save_path = self.save_path,
            #         generator = gen_arch, 
            #         discriminator = dis_arch, 
            #         gen_lr_init = self.gen_lr,
            #         dis_lr_init = self.dis_lr,
            #         l1_opt = 'WMAE',
            #         lambda_value=100,
            #         log_tensorboard = True,
            #         enable_function=True,
            #         suffix = suffix,
            #         )
            #     mt.train(
            #         train_data = (X_train, y_train),
            #         val_data = (X_val, y_val),
            #         epochs = epochs, 
            #         batch_size = bs,
            #         monitor= "val_mean_absolute_error", 
            #         mode = "min",
            #         min_lr_gen = 1e-10,
            #         min_lr_dis = 1e-10,
            #         save_ckpt = True,
            #         ckpt_interval = 1,
            #         save_ckpt_best = True,
            #         # lrdecay_gen = True,
            #         # lrdecay_factor_gen = 0.1,
            #         # lrdecay_wait_gen = 3,
            #         # lrdecay_interval_gen = 1,
            #         # lrdecay_dis = False,
            #         # lrdecay_factor_dis = None,
            #         # lrdecay_wait_dis = None,
            #         # lrdecay_interval_dis = None,
            #         reducelr_on_plateau_gen = True,
            #         reducelr_factor_gen = 0.1,
            #         reducelr_patience_gen = 12,
            #         # reducelr_on_plateau_dis = False,
            #         # reducelr_factor_dis = None,
            #         # reducelr_patience_dis = None,
            #         early_stopping=False,
            #         early_stopping_patience = 32,
            #         )
            
            # Plot the training curves
            mt.plot_training_curves()

            # Export the ModelTrainer object
            print("Exporting ModelTrainer...", mt)
            return mt
                
            ########### COMMON POST PROCS ###########
                
        except Exception as e:
            print(f"An error occurred with run_id: {model_id}: {e}")
