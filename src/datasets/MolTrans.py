import torch
import numpy as np
import pandas as pd
from subword_nmt.apply_bpe import BPE
import codecs
from src.datasets.CARADataset import CARADataset

DATASET_PARAMS = {
    "task": "LO_Kinase",
    "subset": "train",
}

def batch_data_process_MolTrans(datalist):

    if len(datalist) == 1:
        datalist = datalist * 2

    datalist = list(zip(*datalist))
    assay_id, value_type, compound, protein, compound_mask, protein_mask, affinity = datalist
    compound_batch = torch.Tensor(np.array(compound)).long()
    compound_mask_batch = torch.Tensor(np.array(compound_mask)).long()
    protein_batch = torch.Tensor(np.array(protein)).long()
    protein_mask_batch = torch.Tensor(np.array(protein_mask)).long()
    affinity_batch = torch.Tensor(affinity).reshape(-1, 1)
    return (compound_batch, protein_batch, compound_mask_batch, protein_mask_batch), {'Affinity': affinity_batch, "assay_id":assay_id, "value_type":value_type}


class MolTransData(object):

    def register_init_feature(self):
        path = self.root_path + "moltrans/"
        vocab_path = path + 'protein_codes_uniprot.txt'
        bpe_codes_protein = codecs.open(vocab_path)
        self.pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
        sub_csv = pd.read_csv(path + 'subword_units_map_uniprot.csv')

        idx2word_p = sub_csv['index'].values
        self.words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

        vocab_path = path + 'drug_codes_chembl.txt'
        bpe_codes_drug = codecs.open(vocab_path)
        self.dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
        sub_csv = pd.read_csv(path + 'subword_units_map_chembl.csv')

        idx2word_d = sub_csv['index'].values
        self.words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    def get_compound(self, x):
        max_d = 50

        t1 = self.dbpe.process_line(x).split()  # split
        try:
            i1 = np.asarray([self.words2idx_d[i] for i in t1])  # index
        except:
            i1 = np.array([0])
            #print(x)
        
        l = len(i1)

        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
            input_mask = ([1] * l) + ([0] * (max_d - l))

        else:
            i = i1[:max_d]
            input_mask = [1] * max_d

        return i, np.asarray(input_mask)
        
        
    def get_protein(self, x):
        max_p = 545
        t1 = self.pbpe.process_line(x).split()  # split
        try:
            i1 = np.asarray([self.words2idx_p[i] for i in t1])  # index
        except:
            i1 = np.array([0])
            #print(x)

        l = len(i1)
    
        if l < max_p:
            i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
            input_mask = ([1] * l) + ([0] * (max_p - l))
        else:
            i = i1[:max_p]
            input_mask = [1] * max_p
            
        return i, np.asarray(input_mask)


class DataSet(CARADataset, MolTransData):
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

        compound, compound_mask = self.get_compound(smiles)
        protein, protein_mask = self.get_protein(seq)

        return assay_id, value_type, compound, protein, compound_mask, protein_mask, affinity
