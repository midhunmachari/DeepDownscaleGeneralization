"""
author: Midhun Murukesh

Experiment description:  Intercomparison
"""

#%%
import gc
import argparse
import tensorflow as tf
import itertools

tf.keras.backend.clear_session()
tf.config.run_functions_eagerly(True)

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

from runexp import RunExperiments
# from ai4klima.tensorflow.losses import weighted_mae
import tensorflow.keras.backend as K

def weighted_mae(y_true, y_pred):
  weights= tf.clip_by_value(y_true, K.log(0.1+1), K.log(100.0+1))
  return K.mean(tf.multiply(weights, tf.abs(tf.subtract(y_pred, y_true))))

    
#%% The training block

lr_dict = {
    'r1e4' : [1e-4, 1e-4],
    'r7e4' : [7e-4, 1e-4],
    # 'r2e4' : [2e-4, 2e-4],
    # 'r3e4' : [3e-4, 3e-4],
    # 'r1e5' : [1e-5, 1e-5],
    # 'r1e6' : [1e-6, 1e-6],
    }

bs_dict = {
   'b32' : 32,
    # 'b64' : 64,
    # 'b16' : 16,
    # 'b08' : 8,
    }

losses_dict = {
    'wmae': weighted_mae,
    }

def main(prefix, exp_id, epochs, data_path, save_path, refd_path, models_dict):

    if not (lr_dict and bs_dict and models_dict and losses_dict):
        raise ValueError("One or more required dictionaries are empty. Please check the input.")

    ################################
    # START THE EXPERIMENT ITERATOR
    ################################
    for (bs_id, bs), (lr_id, (gen_lr, dis_lr)), (loss_id, loss) in itertools.product(bs_dict.items(), lr_dict.items(), losses_dict.items()):

        expname = f"{prefix}_hp-{bs_id}-{lr_id}-{loss_id}"

        re = RunExperiments( 
            prefix = expname, 
            exp_id = exp_id, 
            data_path = data_path, 
            save_path = save_path, 
            refd_path = refd_path, 
            models_dict = models_dict, 
            gen_lr = gen_lr, 
            dis_lr = dis_lr, 
            loss_obj = loss,
            )
        
        if exp_id == 'e01': # Train deterministially
            re.experiment(epochs, bs, add_input_noise=False)
        elif exp_id == 'e02': # Train with noise
            re.experiment(epochs, bs, add_input_noise=True)
        else:
            raise ValueError(f"Invalid option {exp_id}. Availbale options: 'e01' or 'e02'.")

        del re        # Delete the experiment object

        gc.collect()  # Trigger garbage collection
        print(f"Resources cleared for experiment: {expname}")
            

#%%# Execute the main
            
if __name__ == "__main__":
    
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add the arguments to the parser
    parser.add_argument('--prefix', type=str, default='rdx', help='Prefix of current experiment')
    parser.add_argument('--exp_id', type=str, default=None, help='Current experiment')
    parser.add_argument('--epochs', type=int, default=10   , help='Number of epochs to train')
    parser.add_argument('--pwd'   , type=str, default='./' , help='Path to present working directory')
    parser.add_argument('--dpath' , type=str, default='./' , help='Path to Data directory')
    parser.add_argument('--rpath'   , type=str, default='./' , help='Path to reference data grid directory')
    # Parse the command-line arguments
    args = parser.parse_args()

    models_dict = {
        'm01': ['fsrcnn',  None],         # FSRCNN
        'm02': ['edrn'  ,  None],         # EDRN
        'm03': ['srdrn' ,  None],         # SRDRN
        'm04': ['unet'  ,  None],         # UNET
        'm05': ['aunet' ,  None],         # AUNET
        'm06': ['srdrn' , 'sigmoid_dis'], # SR-GAN  
        'm07': ['unet'  , 'sigmoid_dis'], # UNET-GAN
        'm08': ['aunet' , 'sigmoid_dis'], # AUNET-GAN    
        }

    main(
        prefix = args.prefix,
        exp_id = args.exp_id,
        epochs = args.epochs,
        data_path = args.dpath,
        save_path = f"{args.pwd}/..",
        refd_path = args.rpath,
        models_dict = models_dict,
        )
    

