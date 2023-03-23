import numpy as np 
import os
import pickle
from collections import defaultdict

from gensim.models.word2vec import Word2Vec
from gensim import corpora, models

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt

import torch
from src.datasets.CARADataset import CARADataset


DATASET_PARAMS = {
    "task": "LO_Kinase",
    "subset": "train",
}


def batch_data_process_MTDNN(datalist):
    # bug when batch_size == 1
    if len(datalist) == 1:
        datalist = datalist * 2
    datalist = list(zip(*datalist))
    assay_id, value_type, compound, affinity = datalist
    compound_batch = torch.FloatTensor(compound)
    affinity_batch = torch.Tensor(affinity).reshape(-1, 1)

    return (compound_batch, ),  {"Affinity": torch.FloatTensor(affinity_batch), 
                                              "assay_id": assay_id,
                                              "value_type": value_type,
                                             }


class MTDNNData(object):
    def load_dict(self):
        self.dict_mol = {}

    def get_compound(self, smiles):
        #不用重复计算
        if smiles not in self.dict_mol:
            mol = self.get_mol(smiles)
            compound = self.get_mol_features(mol)
            self.dict_mol[smiles] = compound

        return self.dict_mol[smiles]


    def get_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if self.remove_Hs:
            mol = Chem.RemoveHs(mol)
        # if ExactMolWt(mol) > 1000:
        #     raise ValueError
        # AllChem.EmbedMultipleConfs(mol, 1)
        # if mol.GetNumConformers() == 0:
        #     raise ValueError
        return mol

    def get_mol_features(self, mol):
        """
        Input:
            mol: mol object
        Output:
            if the compound is valid, return its 1024-dimension fingerprint;
            else return string "error"
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,useChirality=True)
        return fp


class DataSet(CARADataset, MTDNNData):
    """
    Class of Container for chemical compounds
    """
    def __init__(self, path, kwargs, remove_Hs=True):
        CARADataset.__init__(self, path, kwargs, remove_Hs)

        # load data
        self.load_dict()  

    def get_one_sample(self, idx):
        # get affinity
        smiles = self.table.loc[idx, 'Smiles']
        affinity = self.table.loc[idx, self.kwargs['label']]
        assay_id = self.table.loc[idx, 'Task ID']
        value_type = self.table.loc[idx, 'Value Type']

        compound = self.get_compound(smiles)
        return assay_id, value_type, compound, affinity
