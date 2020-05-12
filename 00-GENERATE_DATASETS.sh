#!/bin/bash

# generates all training + test set files required to run the training and testing notebooks in this directory. Make sure to run this script (and the notebooks) with the conda environment supplied in conda/. 

cd datasets/backend/features_X
echo "@@@@@@@ Computing features: @@@@@@@@"
ipython calc_reduced_features.py

cd ../labels_y/
echo -e "\n @@@@@@@ Computing labels: @@@@@@@@"
ipython calc_labels.py

cd ../compiled_dataset/
echo -e "\n @@@@@@@ Compiling datasets: @@@@@@@@"
ipython compile_dataset_CSV.py

echo -e "\n @@@@@@@ Done! @@@@@@@@"
echo "All datasets were written to ./datasets/DATASETS/"
