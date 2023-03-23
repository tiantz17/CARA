import torch
from src.datasets.CARADatasetMeta import CARADatasetMeta
from src.datasets.GraphDTA import GraphDTAData
from torch_geometric.data import Batch

DATASET_PARAMS = {
    "task": "VS_Kinase",
    "subset": "train",
    "extra_assay":"",
    "step": 30, # finetune step
    "n_way": 5,
    "k_shot":50,
    "sample_times":1000
}



class DataSet(CARADatasetMeta, GraphDTAData):
    """
    Class of Container for chemical compounds
    """
    def __init__(self, path, kwargs, remove_Hs=True):
        CARADatasetMeta.__init__(self, path, kwargs, remove_Hs)

        # load data
        self.register_init_feature()
        if self.kwargs['subset'] == 'train':
            self.create_batch(valid=True)
            self.create_batch()

    def padding(self, data):
        length = [len(i) for i in data]
        encoding = torch.zeros(len(data), 1000).long()
        for i in range(len(data)):
            encoding[i,:length[i]] = data[i]
        return encoding

    def get_one_sample(self, sample_idx):
        # get affinity
        x_spt = [[],[]]
        x_qry = [[],[]]
        y_spt = {}
        y_spt['Affinity'] = []
        y_spt['assay_id'] = []
        y_qry = {}
        y_qry['Affinity'] = []
        y_qry['assay_id'] = []

        #生成support set的data和label
        if not self.valid:
            support_x = self.support_x_batch[self.index[sample_idx]]
        else:
            support_x = self.support_x_batch_valid[self.index[sample_idx]]
        
        for idx in support_x:
            smiles = self.table.loc[idx, 'Smiles']
            seq = self.table.loc[idx, 'Target Sequence']
            affinity = self.table.loc[idx, self.kwargs['label']]
            assay_id = self.table.loc[idx, 'Task ID']
            compound = self.get_compound(smiles)
            protein = self.get_protein(seq)
            x_spt[0].append(compound)
            x_spt[1].append(protein)
            y_spt['Affinity'].append([affinity])
            y_spt['assay_id'].append(assay_id)
        
        if not self.valid:
            query_x = self.query_x_batch[self.index[sample_idx]]
        else:
            query_x = self.query_x_batch_valid[self.index[sample_idx]]

        for idx in query_x:
            smiles = self.table.loc[idx, 'Smiles']
            seq = self.table.loc[idx, 'Target Sequence']
            affinity = self.table.loc[idx, self.kwargs['label']]
            assay_id = self.table.loc[idx, 'Task ID']
            compound = self.get_compound(smiles)
            protein = self.get_protein(seq)
            x_qry[0].append(compound)
            x_qry[1].append(protein)
            y_qry['Affinity'].append([affinity])
            y_qry['assay_id'].append(assay_id)

        # print("support : query,", len(x_spt[0]), len(x_qry[0]))
        x_spt[0] = Batch.from_data_list(x_spt[0])
        x_spt[1] = self.padding(x_spt[1])
        x_qry[0] = Batch.from_data_list(x_qry[0])
        x_qry[1] = self.padding(x_qry[1])
        y_spt['Affinity'] = torch.Tensor(y_spt['Affinity'])
        y_qry['Affinity'] = torch.Tensor(y_qry['Affinity'])

        return x_spt, x_qry, y_spt, y_qry
    
    def get_finetune_sample(self, assay_id):
        x_spt = [[],[]]
        y_spt = {}
        y_spt['Affinity'] = []
        y_spt['assay_id'] = []
        x_qry = [[],[]]
        y_qry = {}
        y_qry['Affinity'] = []
        y_qry['assay_id'] = []       

        support_list = self.support_assay_index[assay_id]

        for idx in support_list:
            smiles = self.table_support.loc[idx, 'Smiles']
            seq = self.table_support.loc[idx, 'Target Sequence']
            affinity = self.table_support.loc[idx, self.kwargs['label']]
            assay_id = self.table_support.loc[idx, 'Task ID']
            compound = self.get_compound(smiles)
            protein = self.get_protein(seq)

            x_spt[0].append(compound)
            x_spt[1].append(protein)
            y_spt['Affinity'].append([affinity])
            y_spt['assay_id'].append(assay_id)

        query_list = self.query_assay_index[assay_id]

        for idx in query_list:
            smiles = self.table_query.loc[idx, 'Smiles']
            seq = self.table_query.loc[idx, 'Target Sequence']
            affinity = self.table_query.loc[idx, self.kwargs['label']]
            assay_id = self.table_query.loc[idx, 'Task ID']
            compound = self.get_compound(smiles)
            protein = self.get_protein(seq)

            x_qry[0].append(compound)
            x_qry[1].append(protein)
            y_qry['Affinity'].append([affinity])
            y_qry['assay_id'].append(assay_id)

        x_spt[0] = Batch.from_data_list(x_spt[0])
        x_spt[1] = self.padding(x_spt[1])
        x_qry[0] = Batch.from_data_list(x_qry[0])
        x_qry[1] = self.padding(x_qry[1])
        y_spt['Affinity'] = torch.Tensor(y_spt['Affinity'])
        y_qry['Affinity'] = torch.Tensor(y_qry['Affinity'])
        
        return x_spt, x_qry, y_spt, y_qry

