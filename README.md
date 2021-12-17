# Pruning_Defense

Please visit both README.md under 'data' folder to properly accquire all data necessary.

The following packages need to be installed to make sure the code function properlly.

>h5py

>numpy

>keras

>matplotlib

First, run pruning_defense.py file and a list of backdoor-repaired models will be generated under 'models' folder. 

These models differ from each other as they were generated as repaired model's accuracy dropped by a certain threshold percentage, with respect to the original and pre-repaired model. 

Threshold values can be changed in line 94 of pruning_defense.py. Values are expected in unit of percent.

Second, run eval.py, afte verifying backdoor-repaired models have indeed been generated under 'models' folder.

Different repaired model can be chosen, by modifying 'repaired_model_filename' variable in line 16 of eval.py.
