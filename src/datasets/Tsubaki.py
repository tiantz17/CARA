import pickle
import torch
from src.datasets.CARADataset import CARADataset

DATASET_PARAMS = {
    "task": "LO_Kinase",
    "subset": "train",
}

def batch_data_process_Tsubaki(datalist):

    datalist = list(zip(*datalist))
    assay_id, value_type, compound, adj, protein, affinity = datalist
    
    def padding(data):
        length = [len(i) for i in data]
        encoding = torch.zeros(len(data), max(length)).long()
        mask = torch.zeros(len(data), max(length))
        for i in range(len(data)):
            encoding[i,:length[i]] = torch.Tensor(data[i]).long()
            mask[i,:length[i]] = 1
        return encoding, mask

    def padding2d(data):
        length = [len(i) for i in data]
        encoding = torch.zeros(len(data), max(length), max(length))
        mask = torch.zeros(len(data), max(length))
        for i in range(len(data)):
            encoding[i,:length[i],:length[i]] = torch.Tensor(data[i])
            mask[i,:length[i]] = 1
        return encoding, mask

    compound_batch, compound_mask = padding(compound)
    adj_batch, _ = padding2d(adj)
    protein_batch, protein_mask = padding(protein)
    affinity_batch = torch.Tensor(affinity).reshape(-1, 1)
    return (compound_batch, adj_batch, protein_batch, compound_mask, protein_mask), {'Affinity': affinity_batch, "assay_id":assay_id, "value_type":value_type}


class TsubakiData(object):

    def register_init_feature(self):
        path = self.root_path + "tsubaki/"
        self.dict_comp = pickle.load(open(path + self.kwargs['task'] + "_comp_dict", 'rb'))
        self.dict_prot = pickle.load(open(path + "prot_dict", 'rb'))

    def get_compound(self, cid):        
        return self.dict_comp[cid]
        
    def get_protein(self, tid):
        return self.dict_prot[tid]  


class DataSet(CARADataset, TsubakiData):
    """
    Class of Container for chemical compounds
    """
    def __init__(self, path, kwargs, remove_Hs=True):
        CARADataset.__init__(self, path, kwargs, remove_Hs)

        # load data
        self.register_init_feature()

    def get_one_sample(self, idx):
        # get affinity
        cid = self.table.loc[idx, 'Molecule ChEMBL ID']
        tid = self.table.loc[idx, 'Target ChEMBL ID']
        affinity = self.table.loc[idx, self.kwargs['label']]
        assay_id = self.table.loc[idx, 'Task ID']
        value_type = self.table.loc[idx, 'Value Type']

        compound, adj = self.get_compound(cid)
        protein = self.get_protein(tid)

        return assay_id, value_type, compound, adj, protein, affinity
