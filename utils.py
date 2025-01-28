import os
import numpy as np
import tensorflow as tf
from ai4klima.tensorflow.models import FSRCNN, SRDRN, UNet_ups, AttentionUNet_ups, Discriminator, PatchDiscriminator_ups


def loadstack_data_pairs(boxes, DATA_PATH, bounds=None, expset=None, concat=True):
    """
    Load data for the given boxes, stack the inputs, and return arrays for inputs and targets.

    Args:
        boxes (list or str): List of box names or a single box name as a string to process.
        DATA_PATH (str): Path to the directory containing the data files.
        bounds (tuple, optional): Tuple specifying the start and end indices for slicing the data. Default is None.
        expset (str, optional): Experiment set identifier (e.g., 'SET1', 'SET2'). Default is 'SET1'.
        concat (bool, optional): If True, return concatenated arrays. Otherwise, return lists of arrays. Default is True.

    Returns:
        tuple: Inputs and targets arrays (either concatenated or as lists).
    """
    
    # Convert a single box string to a list
    if isinstance(boxes, str):
        boxes = [boxes]
        concat = False  # Disable concatenation for a single box

    # Check if concatenation is valid
    if len(boxes) == 1:
        print("\nSingle box detected. Skipping concatenation.")
        concat = False

    inputs_list = []
    target_list = []

    print(f"\nExperiment input set: {expset}")
    
    for box in boxes:
        print(f"\n\tProcessing ... {box}")

        if expset == 's01': # SET S1: IMRG PREC
            
            # Load input channels for SET0
            inputs = np.load(f"{DATA_PATH}/{box}_080_IMRG_PREC_2001_2023_GMEAN_LOG.npy")
            inputs = np.expand_dims(inputs, axis=-1)
            
            # Load and expand target channel
            target = np.load(f"{DATA_PATH}/{box}_010_IMRG_PREC_2001_2023_LOG.npy")
            target = np.expand_dims(target, axis=-1)
        
        elif expset == 's02': # SET S2: IMRG PREC, IMRG DCLM, GTOP ELEV as INPUTS
            
            # Load input channels for SET1
            channels = [ #Edit here
                f"{DATA_PATH}/{box}_080_IMRG_PREC_2001_2023_GMEAN_LOG.npy",
                f"{DATA_PATH}/{box}_080_IMRG_DCLM_2001_2023_GMEAN_LOG.npy",
                f"{DATA_PATH}/{box}_080_GTOP_ELEV_2001_2023_GMEAN_CLOG.npy",    
            ]
            inputs = np.stack([np.load(ch) for ch in channels], axis=3)
            
            # Load and expand target channel
            target = np.load(f"{DATA_PATH}/{box}_010_IMRG_PREC_2001_2023_LOG.npy")
            target = np.expand_dims(target, axis=-1)
        
        elif expset == 's03': # SET S3: ERA5 VARS MULTICHANNELS, GTOP ELEV as INPUTS
            
            # Load input channels for SET2
            channels = [ #Edit here
                f"{DATA_PATH}/{box}_080_ERA5_PREC_2001_2023_RCON_LOG.npy",
                f"{DATA_PATH}/{box}_080_ERA5_Q500_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_Q850_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_T500_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_T850_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_U500_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_U850_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_V500_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_V850_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_W500_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_W850_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_Z500_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_Z850_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_MSL_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_T2M_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_GTOP_ELEV_2001_2023_GMEAN_CLOG.npy",
            ]
            inputs = np.stack([np.load(ch) for ch in channels], axis=3)
            
            # Load and expand target channel
            target = np.load(f"{DATA_PATH}/{box}_010_IMRG_PREC_2001_2023_LOG.npy")
            target = np.expand_dims(target, axis=-1)
        
        elif expset == 's04': # SET S4: ERA5 VARS MULTICHANNELS AUXVARS ONLY
            
            # Load input channels for SET2
            channels = [ #Edit here
                f"{DATA_PATH}/{box}_080_ERA5_Q500_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_Q850_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_T500_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_T850_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_U500_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_U850_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_V500_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_V850_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_W500_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_W850_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_Z500_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_Z850_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_MSL_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_ERA5_T2M_2001_2023_RBIL_STD.npy",
                f"{DATA_PATH}/{box}_080_GTOP_ELEV_2001_2023_GMEAN_CLOG.npy",
            ]
            inputs = np.stack([np.load(ch) for ch in channels], axis=3)
            
            # Load and expand target channel
            target = np.load(f"{DATA_PATH}/{box}_010_IMRG_PREC_2001_2023_LOG.npy")
            target = np.expand_dims(target, axis=-1)
        
        else:
            raise ValueError(f"Unsupported expset: {expset}")
        
        # Apply bounds if specified
        if bounds is not None:
            inputs = inputs[bounds[0]:bounds[1]]
            target = target[bounds[0]:bounds[1]]
        
        # Append to lists
        inputs_list.append(inputs)
        target_list.append(target)
        
        print(f"\tShape of {box} inputs array: {inputs.shape}")
        print(f"\tShape of {box} target array: {target.shape}")
    
    if concat:
        # Concatenate all inputs and targets
        inputs_array = np.concatenate(inputs_list, axis=0)
        target_array = np.concatenate(target_list, axis=0)
        
        print(f"\nFinal concatenated inputs shape: {inputs_array.shape}")
        print(f"Final concatenated targets shape: {target_array.shape}")
        
        return inputs_array, target_array
    else:
        # Return lists if concatenation is not required
        return inputs_list[0], target_list[0] if len(boxes) == 1 else (inputs_list, target_list)

def configure_model(model_id, inputs_shape, target_shape=None):
    
    if model_id == 'fsrcnn': # FSRCNN
        return FSRCNN(
            input_shape = inputs_shape, 
            ups_factors = (2,2,2),
            k_size = 3, 
            n = 128,
            d = 64, 
            s = 32,
            m = 4,
            isgammaloss = False,
            )
    elif model_id == 'srdrn': # SRDRN
        return SRDRN(
            input_shape = inputs_shape,
            ups_factors = (2,2,2),
            n_filters = 64,
            n_res_blocks = 16, 
            n_ups_filters = 256,
            n_classes = 1,
            last_kernel_size = 3,
            activation = 'prelu',
            regularizer = tf.keras.regularizers.l2(0.01),
            initializer = tf.keras.initializers.RandomNormal(stddev=0.02), 
            interpolation='nearest',
            isgammaloss = False,
            )
    elif model_id == 'unet': # UNET
        return UNet_ups(
            input_shape = inputs_shape,  
            ups_size = (8, 8),
            layer_N=[64, 96, 128, 160],
            input_stack_num=2, 
            pool=True, 
            activation='prelu',
            n_classes = 1,
            dropout_rate=0,
            isgammaloss=False,
            ) 
    elif model_id=='aunet': # ATT-UNET
        return AttentionUNet_ups(
            input_shape = inputs_shape, 
            ups_size = (8,8),
            layer_N=[64, 96, 128, 160],
            input_stack_num=2, 
            pool=True, 
            activation='prelu',
            n_classes=1,
            dropout_rate=0,
            isgammaloss=False,
            )
    elif model_id=='sigmoid_dis': # Sigmoid Discriminator
        return Discriminator(
            inputs_shape = inputs_shape, 
            layer_N=[64, 96, 128, 160],
            input_stack_num=2, 
            pool=True, 
            activation='leaky',
            initializer = tf.random_normal_initializer(0., 0.02)
            )
    elif model_id=='patch_dis': # Patch Discriminator
        return PatchDiscriminator_ups(
            inputs_shape = inputs_shape, 
            target_shape = target_shape,
            ups_size = (8,8),
            layer_N=[64, 96, 128, 160],
            input_stack_num=2, 
            pool=True, 
            activation='leaky',
            initializer = tf.random_normal_initializer(0., 0.02)
            ) 
    else:
        raise ValueError(f"Invalid model_id: {model_id}. Crosscheck!")
    
def load_checkpoints(checkpoint_path):
    # Load the pre-trained model if a checkpoint is provided
    if checkpoint_path:
        if os.path.exists(checkpoint_path):  # Check if the file exists
            print(f"Loading pre-trained model from {checkpoint_path}")
            try:
                # Load the model from the checkpoint
                model = tf.keras.models.load_model(checkpoint_path, compile = False) # Edit here
                print("Pre-trained model loaded successfully.")
            except Exception as e:
                raise ValueError(f"Error loading pre-trained model: {e}")
        else:
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")
    else:
        print("No pre-trained model checkpoint provided. Moving Next.")
    
    return model
        

        
