import numpy as np 
import torch
import time
from src.datasets.CARADatasetMeta import CARADatasetMeta
from src.datasets.DeepCPI import DeepCPIData

import time

DATASET_PARAMS = {
    "task": "VS_Kinase",
    "subset": "train",
    "extra_assay":"",
    "step": 30, # finetune step
    "n_way": 1,
    "k_shot": 50,
    "sample_times": 1000
}


class DataSet(CARADatasetMeta, DeepCPIData):
    """
    Class of Container for chemical compounds
    """
    def __init__(self, path, kwargs, remove_Hs=True):
        CARADatasetMeta.__init__(self, path, kwargs, remove_Hs)

        # load data
        self.load_dict()  
        if self.kwargs['subset'] in ['train', 'ID', 'OOD']:
            self.create_batch_half(valid=True)
            self.create_batch_half()

    def get_one_sample(self, sample_idx):
        # get affinity
        x_spt = [[],[]]
        x_qry = [[],[]]
        y_spt = {}
        y_spt['Affinity'] = []
        y_spt['assay_id'] = []
        y_spt['value_type'] = []
        y_qry = {}
        y_qry['Affinity'] = []
        y_qry['assay_id'] = []
        y_qry['value_type'] = []

        #生成support set的data和label
        if not self.valid:
            support_x = self.support_x_batch[self.index[sample_idx]]
        else:
            support_x = self.support_x_batch_valid[self.index[sample_idx]]
        
        if len(support_x) == 1:
            support_x = support_x * 2
        for idx in support_x:
            smiles = self.table.loc[idx, 'Smiles']
            seq = self.table.loc[idx, 'Target Sequence']
            affinity = self.table.loc[idx, self.kwargs['label']]
            assay_id = self.table.loc[idx, 'Task ID']
            value_type = self.table.loc[idx, 'Value Type']
            compound = self.get_compound(smiles)
            protein = self.get_protein(seq)
            x_spt[0].append(compound)
            x_spt[1].append(protein)
            y_spt['Affinity'].append([affinity])
            y_spt['assay_id'].append(assay_id)
            y_spt['value_type'].append(value_type)
#         print('dict_mol', len(self.dict_mol), 'dict_seq', len(self.dict_seq))
        
        if not self.valid:
            query_x = self.query_x_batch[self.index[sample_idx]]
        else:
            query_x = self.query_x_batch_valid[self.index[sample_idx]]

        if len(query_x) == 1:
            query_x = query_x * 2
        for idx in query_x:
            smiles = self.table.loc[idx, 'Smiles']
            seq = self.table.loc[idx, 'Target Sequence']
            affinity = self.table.loc[idx, self.kwargs['label']]
            assay_id = self.table.loc[idx, 'Task ID']
            value_type = self.table.loc[idx, 'Value Type']
            compound = self.get_compound(smiles)
            protein = self.get_protein(seq)
            x_qry[0].append(compound)
            x_qry[1].append(protein)
            y_qry['Affinity'].append([affinity])
            y_qry['assay_id'].append(assay_id)
            y_qry['value_type'].append(value_type)
            
#         print("support : query,", len(x_spt[0]), len(x_qry[0]))
        x_spt[0] = torch.Tensor(np.array(x_spt[0]))
        x_spt[1] = torch.Tensor(np.array(x_spt[1]))
        x_qry[0] = torch.Tensor(np.array(x_qry[0]))
        x_qry[1] = torch.Tensor(np.array(x_qry[1]))
        y_spt['Affinity'] = torch.Tensor(y_spt['Affinity'])
        y_qry['Affinity'] = torch.Tensor(y_qry['Affinity'])
        
        return x_spt, x_qry, y_spt, y_qry
    

    def get_finetune_sample(self, assay_id):
        x_spt = [[],[]]
        y_spt = {}
        y_spt['Affinity'] = []
        y_spt['assay_id'] = []
        y_spt['value_type'] = []
        x_qry = [[],[]]
        y_qry = {}
        y_qry['Affinity'] = []
        y_qry['assay_id'] = []  
        y_qry['value_type'] = []     
        
        support_list = self.support_assay_index[assay_id]
        if len(support_list) == 1:
            support_list = support_list * 2
        for idx in support_list:
            smiles = self.table_support.loc[idx, 'Smiles']
            seq = self.table_support.loc[idx, 'Target Sequence']
            affinity = self.table_support.loc[idx, self.kwargs['label']]
            assay_id = self.table_support.loc[idx, 'Task ID']
            value_type = self.table_support.loc[idx, 'Value Type']
            compound = self.get_compound(smiles)
            protein = self.get_protein(seq)

            x_spt[0].append(compound)
            x_spt[1].append(protein)
            y_spt['Affinity'].append([affinity])
            y_spt['assay_id'].append(assay_id)
            y_spt['value_type'].append(value_type)

        query_list = self.query_assay_index[assay_id]
        if len(query_list) == 1:
            query_list = query_list * 2
        for idx in query_list:
            smiles = self.table_query.loc[idx, 'Smiles']
            seq = self.table_query.loc[idx, 'Target Sequence']
            affinity = self.table_query.loc[idx, self.kwargs['label']]
            assay_id = self.table_query.loc[idx, 'Task ID']
            value_type = self.table_query.loc[idx, 'Value Type']
            compound = self.get_compound(smiles)
            protein = self.get_protein(seq)

            x_qry[0].append(compound)
            x_qry[1].append(protein)
            y_qry['Affinity'].append([affinity])
            y_qry['assay_id'].append(assay_id)
            y_qry['value_type'].append(value_type)

        x_spt[0] = torch.Tensor(np.array(x_spt[0]))
        x_spt[1] = torch.Tensor(np.array(x_spt[1]))
        x_qry[0] = torch.Tensor(np.array(x_qry[0]))
        x_qry[1] = torch.Tensor(np.array(x_qry[1]))
        y_spt['Affinity'] = torch.Tensor(y_spt['Affinity'])
        y_qry['Affinity'] = torch.Tensor(y_qry['Affinity'])
        
        return x_spt, x_qry, y_spt, y_qry

