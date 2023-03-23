from collections import defaultdict
import os
import json
import random
import pickle
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset
from common.src.datasets.base import Split, Eval


class CARADatasetMeta(Dataset, Split, Eval):
    """
    Class of Container for chemical compounds
    """
    def __init__(self, path, kwargs, remove_Hs=True):
        super(CARADatasetMeta, self).__init__()
        self.root_path = path
        self.path = self.root_path + "CARA/"
        self.kwargs = kwargs
        self.remove_Hs = remove_Hs

        assert self.kwargs['subset'] in ['train', 'test', 'finetune']
        
        if 'shot' in self.kwargs:
            self.kwargs['shot'] = int(self.kwargs['shot'])
        self.kwargs['n_way'] = int(self.kwargs['n_way'])
        self.kwargs['k_shot'] = int(self.kwargs['k_shot'])
        self.kwargs['sample_times'] = int(self.kwargs['sample_times'])
        self.kwargs['thre'] = '0.4'
        self.kwargs['label'] = 'pChEMBL Value'

        if self.kwargs['subset'] in ['train', 'test']:
            self.register_table()
        elif self.kwargs['subset'] in ['finetune']:
            self.register_table_finetune()
        

    def __getitem__(self, idx):
        if self.kwargs['subset'] in ['train']:
            return self.get_one_sample(idx)
        elif self.kwargs['subset'] in ['finetune']:
            return self.get_finetune_sample(idx)

    def __len__(self):
        if self.kwargs['subset'] in ['train']:
            return self.kwargs['sample_times']
        elif self.kwargs['subset'] in ['finetune']:
            return self.assay_num

    def register_table(self):
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

        if self.kwargs['subset'] in ['train']:
            self.index_all = [i for i in range(self.kwargs['sample_times'])]
        else:
            self.index_all = self.table.index
        
        self.index = self.index_all
        
        #assay number
        self.assay_list = self.table['Task ID'].unique().tolist()
        self.assay_num = len(self.assay_list)

        self.assay_index = defaultdict(lambda:[])
        for i in self.table.index:
            self.assay_index[self.table.loc[i, 'Task ID']].append(i)

    def register_table_finetune(self):
        table_cpi = pd.read_csv(self.path + "Task/{}.tsv".format(self.kwargs['task']), sep='\t')
        table_seq = pd.read_csv(self.path + "Task/ChEMBL30_seq.csv")[['Task ID', 'Target Sequence']].drop_duplicates()
        table = pd.merge(table_cpi, table_seq, on='Task ID', how='left')

        split_json_support = json.load(open(self.path + "Split/{}_{}.json".format(self.kwargs['task'], 'support')))
        split_json_query = json.load(open(self.path + "Split/{}_{}.json".format(self.kwargs['task'], 'query')))
        if 'test_assay' in self.kwargs and self.kwargs['test_assay'] != '':
            split_json_support = {self.kwargs['test_assay']:split_json_support[self.kwargs['test_assay']]}
            split_json_query = {self.kwargs['test_assay']:split_json_query[self.kwargs['test_assay']]}
                
                                
        if 'shot' in self.kwargs:
            temp = np.concatenate([item[:self.kwargs['shot']] for item in list(split_json_support.values())])
            self.table_support = table.loc[temp]
        else:
            self.table_support = table.loc[np.concatenate(list(split_json_support.values()))]
            
        self.table_query = table.loc[np.concatenate(list(split_json_query.values()))]

        self.index = [i for i in range(len(self.table_support.index))]
        
        #assay number
        self.assay_list = self.table_query['Task ID'].unique().tolist()
        self.assay_num = len(self.assay_list)

        self.support_assay_index = defaultdict(lambda:[])
        self.query_assay_index = defaultdict(lambda:[])
        for i in self.table_support.index:
            self.support_assay_index[self.table_support.loc[i, 'Task ID']].append(i)
        for i in self.table_query.index:
            self.query_assay_index[self.table_query.loc[i, 'Task ID']].append(i)

    def reset_index(self, index):
        self.index = index

    def reset_valid(self, valid):
        self.valid = valid

    def shuffle_batch(self):
        random.shuffle(self.index)

    def create_batch(self, valid=False):
        #[[episode1], [episode2], [episode3]]
        #episode:= [[c,p,i],[c,p,i]] 同一个assay
        support_x_batch = []  
        query_x_batch = [] 
        count = 0
        dead_loop = False
        while len(support_x_batch) < self.kwargs['sample_times']:
            count += 1
            if (count > self.kwargs['sample_times']) and (count / (1 + len(support_x_batch)) > 10):
                dead_loop = True
            # 随机抽n_way个assay
            selected_assay_set = np.random.choice(self.assay_num, self.kwargs['n_way'], False)  # no duplicate
            np.random.shuffle(selected_assay_set)
            support_x = []
            query_x = []

            for i in range(self.kwargs['n_way']):
                tmp_assay = self.assay_list[selected_assay_set[i]]
                #该assay_id 对应的所有数据
                assay_index = self.assay_index[tmp_assay]
                np.random.shuffle(assay_index)
                num2 = int(min(len(assay_index), 1000))
                support_x.extend(assay_index[:self.kwargs['k_shot']]) 
                query_x.extend(assay_index[self.kwargs['k_shot']:num2]) 
            if not dead_loop:
                if len(query_x) < self.kwargs['k_shot'] * self.kwargs['n_way']:
                    continue
            else:
                if len(query_x) < self.kwargs['n_way']:
                    continue

            support_x_batch.append(support_x)  # append set to current sets
            query_x_batch.append(query_x)  # append sets to current sets  
        if not valid:
            self.support_x_batch = support_x_batch
            self.query_x_batch = query_x_batch
        else:
            self.support_x_batch_valid = support_x_batch
            self.query_x_batch_valid = query_x_batch
            
    def create_batch_half(self, valid=False):
        #[[episode1], [episode2], [episode3]]
        #episode:= [[c,p,i],[c,p,i]] 同一个assay
        support_x_batch = []  
        query_x_batch = [] 
        count = 0
        while len(support_x_batch) < self.kwargs['sample_times']:
            count += 1
            # 随机抽n_way个assay
            selected_assay_set = np.random.choice(self.assay_num, self.kwargs['n_way'], False)  # no duplicate
            np.random.shuffle(selected_assay_set)
            support_x = []
            query_x = []

            for i in range(self.kwargs['n_way']):
                tmp_assay = self.assay_list[selected_assay_set[i]]
                #该assay_id 对应的所有数据
                assay_index = self.assay_index[tmp_assay]
                np.random.shuffle(assay_index)
                num1 = int(min(len(assay_index)//2, self.kwargs['k_shot']))
                num2 = int(min(len(assay_index), 1000))
                support_x.extend(assay_index[:num1]) 
                query_x.extend(assay_index[num1:num2]) 
            if len(support_x) == 0:
                continue
            if len(query_x) == 0:
                continue

            support_x_batch.append(support_x)  # append set to current sets
            query_x_batch.append(query_x)  # append sets to current sets  
        if not valid:
            self.support_x_batch = support_x_batch
            self.query_x_batch = query_x_batch
        else:
            self.support_x_batch_valid = support_x_batch
            self.query_x_batch_valid = query_x_batch

    def get_one_sample(self, sample_idx):
        pass

    def get_finetune_sample(self, sample_idx):
        pass