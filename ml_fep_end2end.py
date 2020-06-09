import os
import glob
import time
import pickle

import pandas as pd
import numpy as np
from rdkit import Chem

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


from rdkit.Chem import rdmolfiles, rdMolDescriptors, AllChem
from mordred import Calculator, descriptors


def get_mordred_descriptors(mols):
    calc = Calculator(descriptors, ignore_3D=False)
    mordred_df = calc.pandas(mols)
    return mordred_df


def get_apfp(mols):
    fps = []
    for item in mols:
        fp = rdMolDescriptors.GetHashedAtomPairFingerprint(item, 256)
        fps.append(list(fp))
    fps = np.array(fps)
    return fps

def normalize_feature(fea):
    scaler = StandardScaler()
    fea = scaler.fit_transform(fea)
    fea = np.nan_to_num(fea, nan=0, posinf=0, neginf=0)
    return fea

def reduce_feature(fea):
    pca = PCA(n_components=0.95)
    post_pca = pca.fit_transform(fea)
    return post_pca

def mol2fea(mols):
    mordred_desc = get_mordred_descriptors(mols)
    apfp = get_apfp(mols)
    normalized_feature = normalize_feature(np.concatenate([mordred_desc, apfp], 1).astype(float))
    reduced_feature = reduce_feature(normalized_feature)
    return reduced_feature

def get_mols(file_path):
    df = pd.read_csv(file_path)
    mols = []
    for item in df.ID:
        sdf_file_path = f'./datasets/backend/freesolv/{item}.sdf'
        mol = Chem.MolFromMolFile(sdf_file_path)
        mols.append(mol)
    return mols

def train():
    train_file_path = './datasets/DATASETS/train_MolPropsAPFP.csv'
    test_file_path = './datasets/DATASETS/test_MolPropsAPFP.csv'

    train_mols = get_mols(train_file_path)
    test_mols = get_mols(test_file_path)

    all_fea = mol2fea(train_mols + test_mols)

    train_x = all_fea[:len(train_mols)]
    train_y = pd.read_csv(train_file_path)['dGoffset (kcal/mol)']
    test_x = all_fea[len(train_mols):]
    test_y = pd.read_csv(test_file_path)['dGoffset (kcal/mol)']
    model = SVR(kernel='rbf', gamma=0.001, C=10.0, epsilon=0.01)

    model.fit(train_x, train_y)
    print(model.score(train_x, train_y))
    print(model.score(test_x, test_y))
    mae_score = mean_absolute_error(test_y, model.predict(test_x))
    print(np.sqrt(mae_score))

def ano_train():
    train_file_path = './datasets/DATASETS/train_MolPropsAPFP.csv'
    test_file_path = './datasets/DATASETS/test_MolPropsAPFP.csv'
    train = pd.read_csv(train_file_path).values
    test = pd.read_csv(test_file_path).values
    train_x = train[:, 1:-2]
    train_y = train[:, -2]
    test_x = test[:, 1:-2]
    test_y = test[:, -2]
    model = SVR(gamma=0.001, C=10.0, epsilon=0.01)
    model.fit(train_x, train_y)
    print(model.score(train_x, train_y))
    print(model.score(test_x, test_y))
    mae_score = mean_absolute_error(test_y, model.predict(test_x))
    print(mae_score)
    print(np.sqrt(mae_score))



if __name__ == '__main__':
    ano_train()
    train()


