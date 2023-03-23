import os
import json
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset
from common.src.datasets.base import Split, Eval


class CARADataset(Dataset, Split, Eval):
    """
    Class of Container for chemical compounds
    """
    def __init__(self, path, kwargs, remove_Hs=True):
        Dataset.__init__(self)
        self.root_path = path
        self.path = self.root_path + "CARA/"
        self.kwargs = kwargs
        self.remove_Hs = remove_Hs
        assert self.kwargs['subset'] in ['train', 'test', 'query', 'support', 'mtl']
        self.kwargs['thre'] = '0.4'
        if 'setting' not in self.kwargs:
            if "VS" in self.kwargs['task']:
                self.kwargs['setting'] = 'NewProtein'
            else:
                self.kwargs['setting'] = 'NewAssay'
        if 'label' not in self.kwargs:
            self.kwargs['label'] = 'pChEMBL Value'

        self.register_table()

    def register_table(self):
        if 'custom' in self.kwargs:
            self.table = pd.read_csv(self.path + "Custom/table_pred_" + self.kwargs["custom"] + ".csv")
            self.index_all = self.table.index
            self.index = self.index_all
            return 
        
        # load table
        table_cpi = pd.read_csv(self.path + "Task/{}.tsv".format(self.kwargs['task']), sep='\t')
        table_seq = pd.read_csv(self.path + "Task/ChEMBL30_seq.csv")[['Task ID', 'Target Sequence']].drop_duplicates()
        table = pd.merge(table_cpi, table_seq, on='Task ID', how='left')
        
        # load data
        if 'test_assay' in self.kwargs and self.kwargs['test_assay'] != '':
            split_json = json.load(open(self.path + "Split/{}_{}.json".format(self.kwargs['task'], self.kwargs['subset'])))  
            split_json = {self.kwargs['test_assay']:split_json[self.kwargs['test_assay']]}
        else:
            split_json = json.load(open(self.path + "Split/{}_{}.json".format(self.kwargs['task'], self.kwargs['subset'])))  
                
        self.table = table.loc[np.concatenate(list(split_json.values()))]

        self.index_all = self.table.index
        self.index = self.index_all

    def reset_index(self, index):
        self.index = index

    def __getitem__(self, idx):
        return self.get_one_sample(self.index[idx])

    def __len__(self):
        return len(self.index)

    def get_one_sample(self, idx):
        pass
        
