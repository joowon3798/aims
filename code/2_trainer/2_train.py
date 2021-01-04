from os import makedirs
from input_data import *
from models import *
from training_utilities_custom import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

# The lines that the user needs to modify are marked with (*)

"""Directories
When cloning the STEM_Defect_Analysis_Data directory into your workspace,
the parent_dir & all other directory paths need to be reset.
"""
# (*) Parent directory: the common directory path under which all files are located
parent_dir = '/data/aims/STEM_Defect_Analysis_Data/'
# (*) Diretory in which the train set is located
data_dir   = parent_dir +'trainSets/201222a/parsed_label_2vacancy/'
# (*) Specify the train session name, this will be the prefix of the folder that gets created to save the train results
sess_name  = '201222a_SV2'

# pixel width & height of image (assume square)
N          = 256    

# (*) factor that describes how many channels we want per layer in FCN
# number of channels/layer = default value per layer * k_fac
k_fac      = 4

# of labels (defects) that we are learning at once
nb_classes = 2

# (*) total number of steps to train on
num_steps  = 100

# Create directory to store the train results
# (diagnostics.dat file containing values of the metrics, the model, and the weights)
sess_dir = parent_dir + "trainResults/" + sess_name + "/"
makedirs(sess_dir, exist_ok=True)

# The following two lines were added to save the whole model
# Currently, there is no need to refer to the model in the "checkpoint" directory that gets created
# This part is still being worked on
checkpoint_filepath = sess_dir + 'checkpoint/'
makedirs(checkpoint_filepath, exist_ok=True)

# The names of the files to save
model_weights_fn = sess_dir + "weights.h5"
model_fn         = sess_dir + "model.json"
diagnostics_fn   = sess_dir + "diagnostics.dat"

# Create model
model = construct_model(N, k_fac, nb_classes, sess_dir, model_fn, model_weights_fn)
step = setup_diagnostics(diagnostics_fn)

# Checkpoint to save the model after every step
# Unfortunately, the current implementation overwrites everything into the same .pb file.
# Need to use a lambda function to save the model (alternatively, only weights) for every step into a readable format. 
checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        save_weights_only = False,
        monitor = 'val_acc',
        mode = 'max',
        save_best_only = False)
callbacks_list = [checkpoint]


# Train
train(checkpoint_filepath, callbacks_list, step, data_dir, N, nb_classes, model, diagnostics_fn, model_weights_fn, num_steps=num_steps, plots=False)

