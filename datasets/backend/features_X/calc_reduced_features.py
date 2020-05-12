# General:
import pandas as pd
import numpy as np
import os
import glob
import time
import pickle

# PCA:
from sklearn.decomposition import PCA
from sklearn import preprocessing

# RDKit:
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdMolDescriptors, AllChem

# Mordred descriptors:
from mordred import Calculator, descriptors

# Path variables:
datasets_dr = '../'
sdf_dr = datasets_dr + 'freesolv/'


# PCA parameter:
pca_threshold = 0.95  # Keeps dimensions up to x% variance explained


def main():

    # feature generation
    mordred_df = get_descriptors()

    fp_APFP_df = get_fingerprints("APFP")
    fp_ECFP2_df = get_fingerprints("ECFP2")
    fp_ECFP4_df = get_fingerprints("ECFP4")
    fp_ECFP6_df = get_fingerprints("ECFP6")  
    fp_ECFP8_df = get_fingerprints("ECFP8")
    fp_TOPOL_df = get_fingerprints("TOPOL")
    fp_NOISE_df = get_fingerprints("X-NOISE")




    mordred_APFP_df = pd.concat([mordred_df, fp_APFP_df], axis=1)
    mordred_ECFP6_df = pd.concat([mordred_df, fp_ECFP6_df], axis=1)
    mordred_TOPOL_df = pd.concat([mordred_df, fp_TOPOL_df], axis=1)
    
    # make collections of these datasets to easily iterate:
    descriptor_sets = [
                        mordred_df, 
                        fp_APFP_df, 
                        fp_ECFP2_df,
			fp_ECFP4_df,
			fp_ECFP6_df,
			fp_ECFP8_df, 
                        fp_TOPOL_df,
                        fp_NOISE_df,
                        mordred_APFP_df, 
                        mordred_ECFP6_df, 
                        mordred_TOPOL_df
                        ]

    descriptor_names = [
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

    # now that we have the seven feature sets, generate training sets on each:
    for dataset, feature_type in zip(descriptor_sets, descriptor_names):
        print("\n Working on", feature_type)
        float_X = convert_to_float(dataset)
        normalised_X = normalise_datasets(float_X, feature_type)
        
        reduce_features(normalised_X, pca_threshold, feature_type)



def reduce_features(normalised_collection, pca_threshold, feature_type):
    """Returns PCA reduced DataFrame according to a pca_threshold parameter.
    Original columns with the highest contribution to PCX are written to CSV."""

    print('Computing PCA, reducing features up to ' + str(round(pca_threshold * 100, 5)) + '% VE...')
    training_data = normalised_collection
    
    # Initialise PCA object, keep components up to x% variance explained:
    PCA.__init__
    pca = PCA(n_components=pca_threshold)

    # Fit to and transform training set.
    train_post_pca = pd.DataFrame(pca.fit_transform(training_data))

    # Reset column names to PCX.
    pca_col = np.arange(1, len(train_post_pca.columns) + 1).tolist()
    pca_col = ['PC' + str(item) for item in pca_col]
    train_post_pca.columns = pca_col
    train_post_pca.index = training_data.index

    print('Number of PCA features after reduction: ' + str(len(train_post_pca.columns)))
    # save PCA object to file for future test set dim. reduction:
    pickle.dump(pca, open("pca_"+feature_type+".pkl","wb"))

    def recovery_pc(normalised_collection, pca_threshold):

        training_data = normalised_collection

        # Normalise data.
        data_scaled = pd.DataFrame(preprocessing.scale(training_data), columns=training_data.columns)

        # Initialise PCA object, keep components up to x% variance explained:
        PCA.__init__
        pca = PCA(n_components=pca_threshold)
        pca.fit_transform(data_scaled)

        index = list(range(1, len(train_post_pca.columns) + 1))
        index = ['PC{}'.format(x) for x in index]

        return_df = pd.DataFrame(pca.components_, columns=data_scaled.columns, index=index)

        return return_df

    # Adapted from https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in
    # -pca-with-sklearn
    recovered_pc = recovery_pc(normalised_collection, pca_threshold)

    # List of column names with highest value in each row.
    recovered_pc_max = recovered_pc.idxmax(axis=1)

    # Recovery 'PCX' indexing.
    pc_index = recovered_pc_max.index.tolist()

    # Write feature names to list.
    pc_feature = recovered_pc_max.values.tolist()

    # Write to DataFrame.
    recovered_pc_dict = {'PCX': pc_index, 'Highest contributing feature': pc_feature}
    recovered_pc_df = pd.DataFrame(recovered_pc_dict)

    # Save recovered PCs to CSV
    save_loc = 'recovered_PCs_'+feature_type+'.csv'
    save_csv(recovered_pc_df, save_loc)

    # Save reduced features to CSV
    save_loc = feature_type+'_reduced_features.csv'
    save_csv(train_post_pca, save_loc)
    
    return train_post_pca


def normalise_datasets(dataframe, feature_type):
    """Returns a normalised DataFrame"""

    # Calculate statistics, compute Z-scores, clean.

    print('Normalising dataframe...')   
    stat = dataframe.describe()
    stat = stat.transpose()

    def norm(x):
        return (x - stat['mean']) / stat['std']

    # save stats to file for test set normalisation:
    save_loc = 'stats_'+feature_type+'.csv'
    save_csv(stat, save_loc)
    # Normalise and return separately.
    normed_data = norm(dataframe).fillna(0).replace([np.inf, -np.inf], 0.0)

    print('Completed normalising dataframe.')
    
    return normed_data


def convert_to_float(dataframe):
    """Returns a DataFrame where all cells are converted to floats"""

    print('Converting dataframe to float...')
    float_df = dataframe._get_numeric_data().astype(float)
    n_dropped_columns = len(dataframe.columns) - len(float_df.columns)

    print('Completed converting dataframe to float. Dropped', n_dropped_columns, ' non-numeric columns.')
    return float_df


def get_fingerprints(fp_type):
    """Returns DataFrame and saves CSV of all calculated RDKit fingerprints for all
    SDF files in the SDF directory (SDF_dr) specified in the path variables."""

    fp_table = []
    for sdf in glob.glob(sdf_dr + '*.sdf'):

        fp_row = []

        # Append ligand ID.
        fp_row.append(sdf.replace(sdf_dr, "").replace(".sdf", ""))

        # Setup fingerprint.
        mol = Chem.rdmolfiles.SDMolSupplier(sdf)[0]
        mol.UpdatePropertyCache(strict=False)

        # Calculate fingerprint.
        if fp_type == "APFP":
            fp = rdMolDescriptors.GetHashedAtomPairFingerprint(mol, 256)
        elif fp_type == "ECFP2":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol,1,nBits=1024)
        elif fp_type == "ECFP4":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
        elif fp_type == "ECFP6":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=1024)
        elif fp_type == "ECFP8":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol,4,nBits=1024)        
        elif fp_type == "TOPOL":
            fp = Chem.RDKFingerprint(mol)
        elif fp_type == "X-NOISE":
            fp = np.random.randint(0,100,200)



        for x in list(fp): 
            fp_row.append(x)

        fp_table.append(fp_row)

    # Column names:
    id_col = ['ID']
    num_cols = len(fp_row) -1
    fp_col = np.arange(0, num_cols).tolist()
    fp_col = [id_col.append("pfp" + str(item)) for item in fp_col]

    fp_df = pd.DataFrame(fp_table, columns=id_col)
    fp_df = fp_df.set_index('ID')

    print('Completed calculating fingerprints.')
    return fp_df


def get_descriptors():
    """Returns DataFrame and saves CSV of all calculated Mordred descriptors for all
    SDF files in the SDF directory (SDF_dr) specified in the path variables."""


    # Read in mordred descriptors to be calculated. In this case, all descriptors.
    descriptors_raw = open(datasets_dr + 'all_mordred_descriptors.txt', 'r')
    descriptors_raw_list = [line.split('\n') for line in descriptors_raw.readlines()]
    descriptors_list = [desc[0] for desc in descriptors_raw_list]
    print('Number of descriptors:', str(len(descriptors_list)))

    # Setup feature calculator.
    print('Calculating Mordred descriptors...')
    calc = Calculator(descriptors, ignore_3D=False)

    # Supply SDF.
    suppl = [sdf for sdf in glob.glob(sdf_dr + '*.sdf')]
    pert_names = [ path.replace(sdf_dr, "").replace(".sdf", "") for path in suppl]

    # generate list of RDKit mols:
    mols = [ Chem.MolFromMolFile(mol) for mol in suppl ]

    mordred_df = calc.pandas(mols)

    mordred_df.index = pert_names
    mordred_df.index.rename("ID", inplace=True)

    return mordred_df


def save_csv(dataframe, pathname):


    dataframe.to_csv(path_or_buf=pathname, index=True)


if __name__ == '__main__':
    main()
