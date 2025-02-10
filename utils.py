import os
import numpy as np
import tensorflow as tf
from ai4klima.tensorflow.models import FSRCNN, SRDRN, EDRN, MegaUNet, Discriminator

def loadstack_data_pairs(boxes, DATA_PATH, bounds=None, concat=True,
                        add_noise=False, noise_stddev=0.1):
    """
    Load data for the given boxes, stack the inputs, and return arrays for inputs and targets.

    Args:
        boxes (list or str): List of box names or a single box name as a string to process.
        DATA_PATH (str): Path to the directory containing the data files.
        bounds (tuple, optional): Tuple specifying the start and end indices for slicing the data. Default is None.
        concat (bool, optional): If True, return concatenated arrays. Otherwise, return lists of arrays. Default is True.
        add_noise (bool, optional): If True, add Gaussian noise to the inputs. Default is False.
        noise_stddev (float, optional): Standard deviation of the Gaussian noise to be added. Default is 0.1.

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
    static_list = []
    
    for box in boxes:
        print(f"\n\tProcessing ... {box}")
        
        # Load input channels for SET1
        channels = [
            f"{DATA_PATH}/{box}_080_IMRG_PREC_2001_2023_GMEAN_LOG.npy",
            f"{DATA_PATH}/{box}_080_IMRG_DCLM_2001_2023_GMEAN_LOG.npy",  
        ]
        
        # Stack inputs along the last axis
        inputs = np.stack([np.load(ch) for ch in channels], axis=-1)
        
        # Add Gaussian noise to the inputs if required
        if add_noise:
            print(f"\tAdding Gaussian noise to inputs with stddev={noise_stddev}")
            noise = np.random.normal(loc=0, scale=noise_stddev, size=inputs.shape)
            inputs += noise
        
        # Load and expand target channel
        target = np.load(f"{DATA_PATH}/{box}_010_IMRG_PREC_2001_2023_LOG.npy")
        target = np.expand_dims(target, axis=-1)

        # Load and expand static orography channel
        static = np.load(f"{DATA_PATH}/{box}_080_GTOP_ELEV_2001_2023_GMEAN_CLOG.npy")
        static = np.expand_dims(static, axis=-1)
        
        # Apply bounds if specified
        if bounds is not None:
            inputs = inputs[bounds[0]:bounds[1]]
            target = target[bounds[0]:bounds[1]]
            static = static[bounds[0]:bounds[1]]

        # Append to lists
        inputs_list.append(inputs)
        target_list.append(target)
        static_list.append(static)
        
        print(f"\tShape of {box} inputs array: {inputs.shape}")
        print(f"\tShape of {box} target array: {target.shape}")
        print(f"\tShape of {box} static array: {static.shape}")
    
    if concat:
        # Concatenate all inputs, targets, and static data
        inputs_array = np.concatenate(inputs_list, axis=0)
        target_array = np.concatenate(target_list, axis=0)
        static_array = np.concatenate(static_list, axis=0)
        
        print(f"\nFinal concatenated inputs shape: {inputs_array.shape}")
        print(f"Final concatenated targets shape: {target_array.shape}")
        print(f"Final concatenated static shape: {static_array.shape}")
        
        return inputs_array, target_array, static_array
    else:
        # Return lists if concatenation is not required
        if len(boxes) == 1:
            return inputs_list[0], target_list[0], static_list[0]
        else:
            return inputs_list, target_list, static_list


def configure_model(model_id, input_shape, target_shape, input_shape_2=None, add_input_noise=False, input_noise_stddev=0.1):
    
    if model_id == 'fsrcnn': # FSRCNN
        return FSRCNN(
            input_shape = input_shape,
            target_shape = target_shape,
            input_shape_2 = input_shape_2, 
            ups_factors = (2,2,2),
            output_activation = 'linear',
            k_size = 3, 
            n = 128,
            d = 64, 
            s = 32,
            m = 4,
            activation = 'prelu',
            ups_method = 'bilinear', 
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            )
    
    elif model_id == 'edrn': # EDRN 
        return EDRN(
            input_shape = input_shape,
            target_shape = target_shape,
            input_shape_2 = input_shape_2, 
            ups_blocks_factors = (2,2,2),
            output_activation='linear',
            n_filters = 64,
            n_res_blocks = 16, 
            n_ups_filters = 128,
            last_kernel_size = 9,
            activation = 'prelu',
            ups_method = 'bilinear', 
            add_input_noise = add_input_noise, # Edit here
            input_noise_stddev = input_noise_stddev,
            )
    
    elif model_id == 'srdrn': # SRDRN 
        return SRDRN(
            input_shape = input_shape,
            target_shape = target_shape, 
            input_shape_2 = input_shape_2, 
            ups_blocks_factors = (2,2,2),
            output_activation='linear',
            n_filters = 64,
            n_res_blocks = 16, 
            n_ups_filters = 128,
            last_kernel_size = 9,
            activation = 'prelu',
            ups_method = 'bilinear', 
            add_input_noise = add_input_noise, # Edit here
            input_noise_stddev = input_noise_stddev,
            )
    
    elif model_id == 'unet': # UNET 
        return MegaUNet(
            input_shape = input_shape, 
            target_shape = target_shape,
            input_shape_2 = input_shape_2, 
            # lr_ups_size = (8,8),
            output_activation = 'linear',
            convblock_opt = 'conv',
            layer_N = [64, 96, 128, 160],
            activation = 'prelu',
            ups_method = 'bilinear',
            add_input_noise = add_input_noise, # Edit here
            input_noise_stddev = input_noise_stddev,
            last_conv_filters = 32,
            attention_on = False,
            ) 
    
    elif model_id=='aunet': # ATT-UNET 
        return MegaUNet(
            input_shape = input_shape,
            target_shape = target_shape,
            input_shape_2 = input_shape_2, 
            # lr_ups_size = (8,8),
            output_activation = 'linear',
            convblock_opt = 'conv',
            layer_N = [64, 96, 128, 160],
            activation = 'prelu',
            ups_method = 'bilinear',
            add_input_noise = add_input_noise,
            input_noise_stddev = input_noise_stddev,
            last_conv_filters = 32,
            attention_on = True,
        )
    
    elif model_id=='sigmoid_dis': # Sigmoid Discriminator 
        return Discriminator(
            inputs_shape = target_shape, 
            layer_N = [64, 96, 128, 160],
            activation = 'leaky',
            )
    
    else:
        raise ValueError(f"Invalid model_id: {model_id}. Crosscheck!")
        
#%% Test the model configurations
    
# for model in ['fsrcnn', 'edrn', 'srdrn', 'unet', 'aunet', 'sigmoid_dis']:
    
#     m = configure_model(model_id = model, 
#                         input_shape = (16, 16, 2), 
#                         target_shape = (128, 128, 1), 
#                         input_shape_2 = (16, 16, 1), 
#                         add_input_noise=False,
#                         input_noise_stddev=0.1
#                         )

#     print(m.summary())

###
 # redundant       
def loadstack_input_target_pairs(boxes, DATA_PATH, bounds=None, concat=True):
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
    
    for box in boxes:
        print(f"\n\tProcessing ... {box}")
        
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