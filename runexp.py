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
from utils import configure_model, loadstack_data_pairs

from ai4klima.tensorflow.train import ModelTraining, SRGAN


class RunExperiments:
    
    def __init__(
            self, 
            prefix, 
            exp_id, 
            data_path, 
            save_path, 
            refd_path, 
            models_dict, 
            gen_lr, 
            dis_lr, 
            loss_obj, 
            ):
        
        self.prefix = prefix
        self.exp_id = exp_id
        self.data_path = data_path
        self.save_path = save_path
        self.refd_path = refd_path

        self.loss_fn = loss_obj
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr

        self.train_bounds = (None, 5113)
        self.val_bounds = (5113, None)
        self.test_bounds  = (None, None)

        self.models_dict = models_dict
        
        self.boxes_dict = {
            'b01' : 'B01',
            'b02' : 'B02',
            'b03' : 'B03',
            'c01' : 'C01',
            }
        
    def get_ref_path(self, box):
        return f"{self.refd_path}/{box}_GRID_INFO.nc" # Edit here 
    
    ################################################# EXP: E01 DETERMINISTIC #################################################
    def experiment(self, epochs, bs, add_input_noise):
        """
        E01: TRAIN AND VALIDATE ON B1, B2, B3. DEPLOY IT TO GENERATE TEST ON EACH DOMAIN.
        """
        print('\n' + '#' * 100)
        print("Processing data ...")
        
        # Load training and validation data
        boxes = ['B01', 'B02', 'B03']    
        train_data = loadstack_data_pairs(boxes, self.data_path, bounds=self.train_bounds)
        val_data   = loadstack_data_pairs(boxes, self.data_path, bounds=self.val_bounds)

        for model_id in self.models_dict.keys():

            expname = f"{self.prefix}_{self.exp_id}_{model_id}"
            print(f'\nInitiate experiment: {expname}')
            print('-'*100)

            gen_opt, dis_opt = self.models_dict[model_id]
            
            # Model Training and Plot Learning Curves
            mt = self.model_trainer(prefix = expname,
                                    suffix = 'b13', 
                                    model_id = model_id, 
                                    gen_opt = gen_opt, 
                                    dis_opt = dis_opt, 
                                    train_data = train_data, 
                                    val_data = val_data, 
                                    epochs = epochs, 
                                    bs = bs, 
                                    add_input_noise = add_input_noise,
                                    )

            # Generate the test netcdf; Loop through all boxes
            for box_id, box in self.boxes_dict.items(): 
                # Load testing data
                X_test, _, S_test = loadstack_data_pairs(box, self.data_path, bounds=self.test_bounds, expset=self.expset_id) # Edit her

                # Generate the test netcdf
                mt.generate_data_and_build_netcdf(
                    [X_test, S_test], 
                    model_path = None,
                    refd_path=self.get_ref_path(box), 
                    save_raw_npy=True, # Edit here
                    build_netcdf=True, # Edit here
                    varname = 'prec', 
                    start_date = "2001-01-01",  # Edit here
                    end_date   = "2023-12-31",  # Edit here
                    tag = box_id, # Edite here
                )

            if add_input_noise:  
                ### Prepare 10 ensemble preciction over NE
                for i in range(1,13):
                    X_test, _, S_test = loadstack_data_pairs('C01', self.data_path, bounds=self.test_bounds, add_noise=True, noise_stddev=0.1) # Edit here

                    # Generate the test netcdf
                    mt.generate_data_and_build_netcdf(
                        [X_test, S_test], 
                        model_path = None,
                        refd_path=self.get_ref_path(box), 
                        save_raw_npy=True, # Edit here
                        build_netcdf=True, # Edit here
                        varname = 'prec', 
                        start_date = "2001-01-01",  # Edit here
                        end_date   = "2023-12-31",  # Edit here
                        tag = f"{box_id}_ens{i:02d}", # Edite here
                    )

            del mt
            gc.collect()
            
            print('\nEND OF EXPERIMENT-01: TRAIN AND VALIDATE ON B1, B2, B3. DEPLOY IT TO GENERATE TEST ON EACH DOMAINS')
    
    ################################################# MODEL TRAINER #################################################
    def model_trainer(self, prefix, suffix, model_id, gen_opt, dis_opt, train_data, val_data, epochs, bs, add_input_noise):

        X_train, y_train, S_train = train_data
        X_val, y_val, S_val = val_data

        print(f"X_train shape: {X_train[:,:,:,0].shape}, X_train max.: {X_train[:,:,:,0].max()}")
        print(f"X_val shape: {X_val[:,:,:,0].shape}, X_val max.: {X_val[:,:,:,0].max()}")
        print(f"y_train shape: {y_train[:,:,:,0].shape}, y_train max.: {y_train[:,:,:,0].max()}")
        print(f"y_val shape: {y_val[:,:,:,0].shape}, y_val max.: {y_val[:,:,:,0].max()}")
        print(f"S_train shape: {S_train[:,:,:,0].shape}, S_train max.: {S_train[:,:,:,0].max()}")
        print(f"S_val shape: {S_val[:,:,:,0].shape}, S_val max.: {S_val[:,:,:,0].max()}")

        try:
            
            ################################# TRAIN: FSRCNN, SRDRN, UNET, AUNET #################################
            if model_id in ['m01', 'm02', 'm03', 'm04']:
                """
                Train UNET/Attention-UNET with WMAE Loss Function -> Deterministic Modelling
                """

                gen_arch = configure_model(gen_opt, X_train.shape[1:], y_train.shape[1:], S_train.shape[1:], add_input_noise) # Edit here

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
                    train_data = (X_train, S_train, y_train), 
                    val_data = (X_val, S_val, y_val), 
                    epochs = epochs,  # Edit here
                    batch_size = bs, 
                    monitor="val_mean_absolute_error",
                    mode = "min",
                    min_lr = 1e-10,
                    save_ckpt = True,
                    ckpt_interval = 1,
                    save_ckpt_best = True,
                    reducelr_on_plateau = True,
                    reducelr_factor = 0.1,
                    reducelr_patience = 12,
                    early_stopping=True,
                    early_stopping_patience = 32,
                    )

            ################################# TRAIN: SRGAN #################################
            elif model_id in ['m05', 'm06', 'm07', 'm08']: # SRGAN

                """
                Train SRGAN -> Generative Modelling
                """
                gen_arch = configure_model(gen_opt, X_train.shape[1:], y_train.shape[1:], S_train.shape[1:], add_input_noise) # Edit here
                dis_arch = configure_model(dis_opt, X_train.shape[1:], y_train.shape[1:], S_train.shape[1:], add_input_noise) # Edit here
                
                mt = SRGAN(
                    prefix = prefix,  
                    save_path = self.save_path,
                    generator= gen_arch, 
                    discriminator = dis_arch,
                    gen_lr_init = self.gen_lr,
                    dis_lr_init = self.dis_lr,
                    cl_opt='WMAE',
                    lambda_value=1e-3,
                    log_tensorboard = True,
                    enable_function=True,
                    suffix = suffix,
                    )
                mt.train(
                    train_data = (X_train, S_train, y_train),
                    val_data = (X_val, S_train, y_val),
                    epochs = epochs, 
                    batch_size = bs,
                    monitor= "val_mean_absolute_error", 
                    mode = "min",
                    min_lr_gen = 1e-10,
                    min_lr_dis = 1e-10,
                    save_ckpt = True,
                    ckpt_interval = 1,
                    save_ckpt_best = True,
                    reducelr_on_plateau_gen = True,
                    reducelr_factor_gen = 0.1,
                    reducelr_patience_gen = 12,
                    early_stopping=False,
                    early_stopping_patience = 32,
                    )

            # Plot the training curves
            mt.plot_training_curves()

            # Export the ModelTrainer object
            print("Exporting ModelTrainer...", mt)
            return mt
                
            ########### COMMON POST PROCS ###########
                
        except Exception as e:
            print(f"An error occurred with run_id: {model_id}: {e}")
