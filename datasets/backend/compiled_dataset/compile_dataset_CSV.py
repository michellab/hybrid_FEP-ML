import pandas as pd
import os
import numpy as np
import shutil
import time
from tqdm import tqdm

# Path variables:
path = './'
datasets_dr = '../../DATASETS/'
SDF_dr = '../freesolv/'
freesolv_loc = '../freesolv_database.txt'
train_dr = '../train_dr/'
test_dr = '../test_dr/'
for dr in [train_dr, test_dr]:
    if os.path.isdir(dr):
        shutil.rmtree(dr)
        print('Existing directory {} overwritten.'.format(dr))
        os.mkdir(dr)
    else:
        os.mkdir(dr)

# Load in FreeSolve.
freesolv_df = pd.read_csv(freesolv_loc, sep='; ', engine='python')
# SAMPl4_Guthrie experimental reference in FreeSolv.
SAMPL4_Guthrie_ref = 'SAMPL4_Guthrie'
# Experimental reference column name.
exp_ref_col = 'experimental reference (original or paper this value was taken from)'


def main():
    feature_sets = [
                        "MolProps",
                        "APFP",
                        "ECFP2",
                        "ECFP4",
                        "ECFP6",
                        "ECFP8",
                        "TOPOL",
                        "X-NOISE",
                        "MolPropsAPFP",
                        "MolPropsECFP6",
                        "MolPropsTOPOL"
                        ]
    for feature_set in tqdm(feature_sets):

        # Load in features, labels and null labels sorted by index
        reduced_X = pd.read_csv('../features_X/'+feature_set+'_reduced_features.csv', index_col='ID').sort_index()
        true_y = pd.read_csv('../labels_y/experimental_labels.csv', index_col='ID').sort_index()
        null_y = pd.read_csv('../labels_y/null_experimental_labels.csv', index_col='ID').sort_index()

        # Split features, labels and null labels into training and external testing sets
        train_X, test_X = split_train_test(reduced_X)
        train_y, test_y = split_train_test(true_y)
        train_null, test_null = split_train_test(null_y)

        # Write absolute data sets to CSV
        create_absolute_train_test(train_X, train_y, 'train_'+feature_set)
        create_absolute_train_test(test_X, test_y, 'test_'+feature_set)
        create_absolute_train_test(train_X, train_null, 'null_train_'+feature_set)
        create_absolute_train_test(test_X, test_null, 'null_test_'+feature_set)



def create_absolute_train_test(features, labels, set_type):

    # filename to be written
    csv = datasets_dr + set_type + '.csv'
    if os.path.exists(csv): os.remove(csv)

    # iterate through rows
    for (id1, X), (id2, y) in zip(features.iterrows(), labels.iterrows()):

        # write row to CSV file
        row = pd.concat([pd.DataFrame([X]), pd.DataFrame([y])], axis=1)
        with open(csv, 'a') as file:
            row.to_csv(path_or_buf=file, mode='a', index=True, index_label='ID', header=file.tell() == 0)

        # copy SDF files
        sdf = str(id1) + '.sdf'
        if set_type == 'train':
            shutil.copyfile(SDF_dr + sdf, train_dr + sdf)
        elif set_type == 'test':
            shutil.copyfile(SDF_dr + sdf, test_dr + sdf)


def split_train_test(dataframe):

    # List comprehension for all non-SAMPL4_Guthrie entries.
    # exclude the discrepant mols:
    excluded = ["mobley_6309289", "mobley_3395921", "mobley_6739648", "mobley_2607611", "mobley_637522", "mobley_172879"]
    train_ids = [freesolv_df.iloc[i][0]
                 for i in range(len(freesolv_df))
                 if freesolv_df.loc[i, exp_ref_col] != SAMPL4_Guthrie_ref ]
    train_ids = [train_id for train_id in train_ids if train_id not in excluded]

    # List comprehension for all SAMPL4_Guthrie entries.
    test_ids = [freesolv_df.iloc[i][0]
                for i in range(len(freesolv_df))
                if freesolv_df.loc[i, exp_ref_col] == SAMPL4_Guthrie_ref]
    test_ids = test_ids + excluded
    train_df = dataframe.drop(test_ids, errors="ignore")
    test_df = dataframe.drop(train_ids)
    
    return train_df, test_df


if __name__ == '__main__':

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script started on {}'.format(time.ctime()))

    main()

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script finished on {}'.format(time.ctime()))
