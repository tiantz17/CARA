import os
import pickle
import torch
import rdkit.Chem as Chem
from src.datasets.CARADataset import CARADataset

DATASET_PARAMS = {
    "task": "LO_Kinase",
    "subset": "train",
}

def batch_data_process_CNN(datalist):

    def padding(data):
        length = [len(i) for i in data]
        encoding = torch.zeros(len(data), max(length)).long()
        for i in range(len(data)):
            encoding[i,:length[i]] = data[i]
        return encoding

    datalist = list(zip(*datalist))
    assay_id, value_type, compound, affinity = datalist
    compound_batch = padding(compound)
    affinity_batch = torch.Tensor(affinity).reshape(-1, 1)
    return (compound_batch, ), {'Affinity': affinity_batch, "assay_id":assay_id, "value_type":value_type}


class DeepDTAData(object):

    def register_init_feature(self):

        self.dict_char_seq = { 
            " ": 0,
            "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
            "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
            "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
            "U": 19, "T": 20, "W": 21, 
            "V": 22, "Y": 23, "X": 24, 
            "Z": 25 
        }
        self.dict_char_mol = {
            " ": 0,
            "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
            "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
            "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
            "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
            "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
            "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
            "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
            "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64,
        }
        self.max_seq_len = 1000
        self.max_mol_len = 100
        
        path = self.root_path + "deepdta/"
        self.dict_canonical_smiles = {}
        if os.path.exists(path + self.kwargs['task'] + "_dict_smiles"):
            self.dict_canonical_smiles = pickle.load(open(path + self.kwargs['task'] + "_dict_smiles", 'rb'))
            print("dict smiles loaded")

    def get_compound(self, smiles):
        #给Smiles，转成Token
        if smiles not in self.dict_canonical_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if self.remove_Hs:
                mol = Chem.RemoveHs(mol)
            self.dict_canonical_smiles[smiles] = Chem.MolToSmiles(mol)

        compound = torch.LongTensor([self.dict_char_mol[i] for i in self.dict_canonical_smiles[smiles][:self.max_mol_len]])

        return compound
        
    def get_protein(self, seq):
        #给protein seq，转成Token
        protein = torch.LongTensor([self.dict_char_seq[i] for i in seq[:self.max_seq_len]])
        
        return protein


class DataSet(CARADataset, DeepDTAData):
    """
    Class of Container for chemical compounds
    """
    def __init__(self, path, kwargs, remove_Hs=True):
        CARADataset.__init__(self, path, kwargs, remove_Hs)

        # load data
        self.register_init_feature()

    def get_one_sample(self, idx):
        # get affinity
        smiles = self.table.loc[idx, 'Smiles']
        affinity = self.table.loc[idx, self.kwargs['label']]
        assay_id = self.table.loc[idx, 'Task ID']
        value_type = self.table.loc[idx, 'Value Type']

        compound = self.get_compound(smiles)

        return assay_id, value_type, compound, affinity
