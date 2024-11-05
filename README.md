# Retrain LigandMPNN
Completing training code for LigandMPNN as well as experimenting with the protein-to-ligand graph message-passing architecture.

![a](https://github.com/user-attachments/assets/f6406514-647f-4d87-8bc1-62cd2c9e3b8e)
![b](https://github.com/user-attachments/assets/e644481a-ff67-4cb9-8169-0a8835a7030f)

**Code organization**
* `train.py` - the main script to train the model.
* `model.py` - modules for the original LigandMPNN model.
* `model_protein2ligand_MPNN.py` - modules for the modified LigandMPNN model with protein-to-ligand graph update layers.
* `model_data_utils.py` - utility functions and data-loading classes for the main training script.
* `preprocess_training_data.py` - the data preprocessing script parsing PDB files to PyTorch tensor files.
* `data_utils.py` - utility functions for the data preprocessing script.

-----------------------------------------------------------------------------------------------------
**1. Download and parse training data.**

>Use a sample training set:

`tar -xvzf sample.tar.gz`

**2. Train the model.**

`sbatch train_sample.sh`
