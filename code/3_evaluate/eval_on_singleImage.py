# Test trained model with selected weights on real image set
# Assuming you are accessing everything from /data/aims,
# the user only has to change the lines marked with (*)


from os import makedirs
from evaluate import *

"""Directories:
There are 4 directories total.
- parent_dir = contains the common directory path
(ex. if working with the files in /data/aims/STEM_Defect_Analysis_Data/, this will be the parent_dir)
- trainedModel_dir, evalImage_dir, save_dir = parent_dir + subdirectories + filename
"""
# parent_dir: the overarching directory (contains the common file path)
parent_dir = "/data/aims/STEM_Defect_Analysis_Data/"


'''User should change these lines as needed'''

# (*) Specify the type of defect
defectType = "1vac"

# (*) Directory containing the model & weights (If you are testing for 1vac, select the 1vac folder in the results directory)
trainedModel_dir = parent_dir + "trainResults/201223_correctLabels/201223a_betterResults/201223_k4_SV1/"

# (*) Directory containing the image for evaluating the model's performance
evalImage_dir = parent_dir + "simImages_forEval/eval_A0/"

# (*) prefix: to append to the file names that get saved
# One can use any naming convention that suits them; in this example, the name contains
#  the k-factor (3) type of defect (SV1)
prefix = "k4_SV1_"

# (*) Name of folder in which to save the evaluation results
newFolder = "201223_eval"

'''End of section that the user needs to change'''


# The following 2 lines creates the directory to save the evaluation results
# If the specified directory doesn't exist, creates the directory
# If it exists, uses that pre-existing directory
save_dir = parent_dir + "modelPerformanceEval/simImTest/" + newFolder + "/"
makedirs(save_dir, exist_ok=True)

# Load model
model_fn = trainedModel_dir + "model.json"
# Load the trained weights
model_weights_fn = trainedModel_dir + "weights.h5"
# Call the specific image to evaluate the model on
input_file = evalImage_dir + "input.tif"



"""
evaluate(): get the model's predictions on the test image and save the results
"""

'''Specify the settings to evaluate the model'''
# Size of the evaluation image
l_shape = (256, 256)
# stride size to go through the evaluation image
stride = (32, 32)
# avg: to specify whether to evaluate the image as is (0), or to evaluate by averaging over rotations & flips
avg = 0
# Decide whether to display the evaluation results
plot = True
# Decide whether to save the evaluation results
save_data = True

'''Call the evaluate() method'''
prediction = evaluate(model_fn, model_weights_fn, input_file, l_shape, stride, avg=avg, plot=plot, save_data=save_data, save_dir=save_dir, prefix=prefix)




"""
calc_accuracy(): calculates the accuracy of the model's predictions based on the 4 metrics
and plots the TP, FP, FN and saves it as an image
"""

# Get the label image to calcualte the accuracy
label_file_list = [evalImage_dir + "label_{}.tif".format(defectType)]

# Threshold for the labeled images;
# Pixels in the range [0, tol] = dark, background
# Pixels in range (tol, 1) = white spots, labels
tol = .5
nconvs = 0
# Radius of the dots to plot the model predictions
r = 1
# TN = the atoms that the model correctly identified, this is always set to 0 bc we only want to see TP, FP, FN
TN = 0
# Whether to display the plots
plot = True
# Whether to save the plots
save_data = True
# Whether we want to display in detail what is happening on the screen
# Doesn't matter whether it is set to True or False
verbose = True


TP, FP, FN, TN, recall, precision, F1, bal_acc = calc_accuracy(prediction, label_file_list, tol=tol, nconvs=nconvs, r=r, TN=TN, plot=plot, save_data=save_data, save_dir=save_dir, prefix=prefix, verbose=verbose)


