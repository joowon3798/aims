# Model diagnostics that were output fom the training

from evaluate import *

parent_dir = "/data/aims/sychoi/"

results_dir = parent_dir + "results_MoS2/201223_k2_SV2/"
# Currently two subdirectories under results_MoS2, 1vac & 2vac
save_dir = parent_dir + "testModel_results/"

# Load diagnostic files
# List of directories with each trained model as an element in the list
label_list = ["1vacancy", "2vacancy"] 
results_dir_list = ["{}{}/".format(results_dir, label) for label in label_list] 

diagnostics_data = get_diagnostic_data(results_dir)

# Plot diagnostic data using matplotlib
# Plot function = log(1-TP/(TP+FN))
plot_diagnostics(diagnostics_data, label_list, save_dir=save_dir, diag="F1", log=True, invert=True, save=True, N=2)


plot_diagnostics(diagnostics_data, label_list, save_dir=save_dir, diag="F1", log=True, invert=True, save=True, N=2)

plot_diagnostics(diagnostics_data, label_list, save_dir=save_dir, diag="F1", log=True, invert=True, save=True, N=2)



