import os
import pickle
import torch
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from src.datasets.CARADataset import CARADataset

DATASET_PARAMS = {
    "task": "LO_Kinase",
    "subset": "train",
}

def batch_data_process_DeepConvDTI(datalist):
    if len(datalist) == 1:
        datalist = datalist * 2

    def padding(data):
        length = [len(i) for i in data]
        encoding = torch.zeros(len(data), max(length)).long()
        for i in range(len(data)):
            encoding[i,:length[i]] = data[i]
        return encoding

    datalist = list(zip(*datalist))
    assay_id, value_type, compound, protein, affinity = datalist
    compound_batch = torch.Tensor(compound)
    protein_batch = padding(protein)
    affinity_batch = torch.Tensor(affinity).reshape(-1, 1)
    return (compound_batch, protein_batch), {'Affinity': affinity_batch, "assay_id":assay_id, "value_type":value_type}


class DeepConvDTIData(object):

    def register_init_feature(self):
        self.seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
        self.seq_dic = {w: i+1 for i,w in enumerate(self.seq_rdic)}
        self.max_seq_len = 2500

        self.dict_fp = {}
        path = self.root_path + "deepconvdti/"
        if os.path.exists(path + self.kwargs['task'] + "_dict_fp"):
            self.dict_fp = pickle.load(open(path + self.kwargs['task'] + "_dict_fp", 'rb'))

    def get_compound(self, smiles):
        #给Smiles，转成Token
        if smiles not in self.dict_fp:
            mol = Chem.MolFromSmiles(smiles)
            if self.remove_Hs:
                mol = Chem.RemoveHs(mol)
            self.dict_fp[smiles] = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048,useChirality=True)

        compound = self.dict_fp[smiles]

        return compound
        
    def get_protein(self, seq):
        #给protein seq，转成Token
        protein = torch.LongTensor([self.seq_dic[i] for i in seq[:self.max_seq_len]])
        
        return protein


class DataSet(CARADataset, DeepConvDTIData):
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
        seq = self.table.loc[idx, 'Target Sequence']
        affinity = self.table.loc[idx, self.kwargs['label']]
        assay_id = self.table.loc[idx, 'Task ID']
        value_type = self.table.loc[idx, 'Value Type']

        compound = self.get_compound(smiles)
        protein = self.get_protein(seq)

        return assay_id, value_type, compound, protein, affinity
