# Normalising-Flow-DNN
Normalising Flow based Deep Neural Network for Uncertainty Estimation for classification.

The code for the PostNet method is in the folder called postnet. The run.py script is where everything is called. The other files are mainly classes and functions to help compartmentalise it.

The Deep Ensemble method is in the ensemble folder. It has a run_ensemble.py which loads the correct modules and also makes use of some the postnet folders scripts.

The requirements1.txt contains the necessary packages for a virtual environment.

The submit1.sh is used to run the PostNet script on the HPC server and load the necessary cuda modules.
