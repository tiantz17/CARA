import pickle
import logging
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score


# load and split data
class Split(object):
    
    def split_data(self, seed, num_fold):
        self.split_data_fold(seed, num_fold, self.kwargs["setting"], self.kwargs["thre"])

    def split_data_fold(self, seed, num_fold, setting, thre):
        if setting == 'Random': # imputation
            self.split_data_imputation_random(seed, num_fold)
        elif setting == 'NewProtein':
            self.split_data_new_protein(seed, num_fold, thre)
        elif setting == 'NewCompound':
            self.split_data_new_compound(seed, num_fold, thre)
        elif setting == 'NewAssay':
            self.split_data_new_assay(seed, num_fold, thre)
        elif setting == 'BothNew':
            self.split_data_both_new(seed, num_fold, thre)
        else:
            raise NotImplementedError
    
    def split_data_new_assay(self, seed, num_fold, thre):
        logging.info("Using new assay split")
        columns = 'Assay ChEMBL ID'

        index_cv = np.array(self.index_all)
        list_compound = list(set(self.table.loc[self.index_all, columns].values))
        
        np.random.seed(seed)
        np.random.shuffle(list_compound)
        
        self.list_fold_train, self.list_fold_valid, self.list_fold_test = [], [], []
        kf = KFold(n_splits=num_fold, shuffle=True)
        for train_idx, test_idx in kf.split(list_compound):
            list_train_idx, list_valid_idx, list_test_idx = [], [], []
            test_cp = set(np.array(list_compound)[test_idx].tolist())
            train_cp = np.array(list_compound)[train_idx]
            valid_cp = set(np.random.choice(train_cp, int(len(train_cp)/num_fold), replace=False).tolist())
            for idx in index_cv:
                if self.table.loc[idx, columns] in test_cp:
                    list_test_idx.append(idx)
                elif self.table.loc[idx, columns] in valid_cp:
                    list_valid_idx.append(idx)
                else:
                    list_train_idx.append(idx)
            
            list_valid_idx = list(set(list_valid_idx))
            list_test_idx = list(set(list_test_idx))
            
            self.list_fold_train.append(np.array(list_train_idx))
            self.list_fold_valid.append(np.array(list_valid_idx))
            self.list_fold_test.append(np.array(list_test_idx))
        
    def split_data_new_compound(self, seed, num_fold, thre):
        logging.info("Using new compound split")
        columns = 'Molecule ChEMBL ID'

        index_cv = np.array(self.index_all)
        list_compound = list(set(self.table.loc[self.index_all, columns].values))
        
        np.random.seed(seed)
        np.random.shuffle(list_compound)
        
        self.list_fold_train, self.list_fold_valid, self.list_fold_test = [], [], []
        kf = KFold(n_splits=num_fold, shuffle=True)
        for train_idx, test_idx in kf.split(list_compound):
            list_train_idx, list_valid_idx, list_test_idx = [], [], []
            test_cp = set(np.array(list_compound)[test_idx].tolist())
            train_cp = np.array(list_compound)[train_idx]
            valid_cp = set(np.random.choice(train_cp, int(len(train_cp)/num_fold), replace=False).tolist())
            for idx in index_cv:
                if self.table.loc[idx, columns] in test_cp:
                    list_test_idx.append(idx)
                elif self.table.loc[idx, columns] in valid_cp:
                    list_valid_idx.append(idx)
                else:
                    list_train_idx.append(idx)
            
            list_valid_idx = list(set(list_valid_idx))
            list_test_idx = list(set(list_test_idx))
            
            self.list_fold_train.append(np.array(list_train_idx))
            self.list_fold_valid.append(np.array(list_valid_idx))
            self.list_fold_test.append(np.array(list_test_idx))
        
    def split_data_new_protein(self, seed, num_fold, thre):
        logging.info("Using new protein split")
        columns = 'Target Cluster '+str(thre)
        if columns not in self.table.columns:
            columns = 'Target ChEMBL ID'
            
        index_cv = np.array(self.index_all)
        list_protein = list(set(self.table.loc[self.index_all, columns].values))
            
        np.random.seed(seed)
        np.random.shuffle(list_protein)
        
        self.list_fold_train, self.list_fold_valid, self.list_fold_test = [], [], []
        kf = KFold(n_splits=num_fold, shuffle=True)
        for train_idx, test_idx in kf.split(list_protein):
            list_train_idx, list_valid_idx, list_test_idx = [], [], []
            test_pid = set(np.array(list_protein)[test_idx].tolist())
            train_pid = np.array(list_protein)[train_idx]
            valid_pid = set(np.random.choice(train_pid, int(len(train_pid)/num_fold), replace=False).tolist())
            for idx in index_cv:
                if self.table.loc[idx, columns] in test_pid:
                    list_test_idx.append(idx)
                elif self.table.loc[idx, columns] in valid_pid:
                    list_valid_idx.append(idx)
                else:
                    list_train_idx.append(idx)
            
            list_valid_idx = list(set(list_valid_idx))
            list_test_idx = list(set(list_test_idx))
            
            self.list_fold_train.append(np.array(list_train_idx))
            self.list_fold_valid.append(np.array(list_valid_idx))
            self.list_fold_test.append(np.array(list_test_idx))
    
    def split_data_both_new(self, seed, num_fold, thre):
        assert np.sqrt(num_fold) == int(np.sqrt(num_fold))
        
        c_columns = 'Molecule ChEMBL ID'
        p_columns = 'Target Cluster '+str(thre)
        if c_columns not in self.table.columns:
            c_columns = 'Molecule ChEMBL ID'
            p_columns = 'Target ChEMBL ID'
        logging.info("Using both-new split")
                
        index_cv = np.array(self.index_all)
        list_compound = list(set(self.table.loc[self.index_all, c_columns].values))
        list_protein = list(set(self.table.loc[self.index_all, p_columns].values))
                
        np.random.seed(seed)
        np.random.shuffle(list_compound)
        np.random.seed(seed)
        np.random.shuffle(list_protein)
        
        self.list_fold_train, self.list_fold_valid, self.list_fold_test = [], [], []
        kfp = KFold(n_splits=int(np.sqrt(num_fold)), shuffle=True)
        kfc = KFold(n_splits=int(np.sqrt(num_fold)), shuffle=True)
        cpd_split = []
        for train_idx_c, test_idx_c in kfc.split(list_compound):
            cpd_split.append([train_idx_c, test_idx_c])

        for train_idx_p, test_idx_p in kfp.split(list_protein):
            for fold in range(len(cpd_split)):
                train_idx_c, test_idx_c = cpd_split[fold]

                list_train_idx, list_valid_idx, list_test_idx = [], [], []
                test_pid = set(np.array(list_protein)[test_idx_p].tolist())
                train_pid = np.array(list_protein)[train_idx_p]
                valid_pid = set(np.random.choice(train_pid, int(len(train_pid)/np.sqrt(num_fold)), replace=False).tolist())

                test_cid = set(np.array(list_compound)[test_idx_c].tolist())
                train_cid = np.array(list_compound)[train_idx_c]
                valid_cid = set(np.random.choice(train_cid, int(len(train_cid)/np.sqrt(num_fold)), replace=False).tolist())

                for idx in index_cv:
                    if self.table.loc[idx, p_columns] in test_pid and self.table.loc[idx, c_columns] in test_cid:
                        list_test_idx.append(idx)
                    elif self.table.loc[idx, p_columns] in valid_pid and self.table.loc[idx, c_columns] in valid_cid:
                        list_valid_idx.append(idx)
                    elif self.table.loc[idx, p_columns] in train_pid and self.table.loc[idx, c_columns] in train_cid:
                        list_train_idx.append(idx)
                        
                list_valid_idx = list(set(list_valid_idx))
                list_test_idx = list(set(list_test_idx))
            
                self.list_fold_train.append(np.array(list_train_idx))
                self.list_fold_valid.append(np.array(list_valid_idx))
                self.list_fold_test.append(np.array(list_test_idx))

    def split_data_imputation_random(self, seed, num_fold):
        logging.info("Using new interaction random split")
        np.random.seed(seed)
        
        index_cv = np.array(self.index_all)
        
        self.list_fold_train, self.list_fold_valid, self.list_fold_test = [], [], []
        kf = KFold(n_splits=num_fold, shuffle=True)
        for train_idx, test_idx in kf.split(index_cv):
            list_valid_idx = np.random.choice(train_idx, int(len(train_idx)/num_fold), replace=False)
            list_train_idx = list(set(train_idx.tolist())-set(list_valid_idx.tolist()))
            list_test_idx = test_idx
            
            list_valid_idx = list(set(list_valid_idx.tolist()))
            list_test_idx = list(set(list_test_idx.tolist()))
            self.list_fold_train.append(index_cv[list_train_idx])
            self.list_fold_valid.append(index_cv[list_valid_idx])
            self.list_fold_test.append(index_cv[list_test_idx])
    

# evaluation functions
class Eval(object):
    """
    Class for evaluation methods
    """
    def score(self, scoretype):
        if scoretype == "reg":
            return self.score_reg
        elif scoretype == "cls":
            return self.score_cls

    def score_reg(self, pred, label):
        pred = np.array(pred).reshape(-1)
        label = np.array(label).reshape(-1)
        if (len(np.unique(label)) < 2) or (len(np.unique(pred)) < 2):
            return {"r2":np.nan, "mse":np.nan, "pcc":np.nan, "scc":np.nan, 
                    'EF1%': np.nan,
                    'EF5%': np.nan,
                    'prc1%': np.nan,
                    'prc5%': np.nan,
                    "percentPCC": np.nan,
                    "percentR2": np.nan}
        try:
            if len(pred) >= 2:
                r2 = r2_score(label, pred)
            else:
                r2 = np.nan
        except:
            r2 = np.nan
        try:
            mse = mean_squared_error(label, pred)
        except:
            mse = np.nan
        try:
            pcc = pearsonr(label, pred)[0]
        except:
            pcc = np.nan
        try:
            scc = spearmanr(label, pred)[0]
        except:
            scc = np.nan
        try:
            pr2 = 100 * float(r2 > 0.3)
        except:
            pr2 = np.nan
        try:
            ppcc = 100 * float(pcc > 0.5)
        except:
            ppcc = np.nan
        try:
            rank_pred = np.argsort(np.argsort(-pred))
            rank_label = np.argsort(np.argsort(-label))
            pred_1 = 1 / (1 + np.exp(-(pred - 6)))
            N = len(pred_1)

            chi = 0.01 # ratio of important samples
            label_1 = rank_label < chi*N
            n = sum(label_1)
            TP = sum((rank_pred < chi*N) & label_1)
            EF1 = TP / (chi * n)
            prc1 = float(TP>0) *100

            chi = 0.05
            label_1 = rank_label < chi*N
            n = sum(label_1)
            TP = sum((rank_pred < chi*N) & label_1)
            EF5 = TP / (chi * n)
            prc5 = float(TP>0)*100
        except:
            EF1 = np.nan
            EF5 = np.nan
            prc1 = np.nan
            prc5 = np.nan
    
        return {"r2":r2, "mse":mse, "pcc":pcc, "scc":scc, 
                'EF1%': EF1,
                'EF5%': EF5,
                'prc1%': prc1,
                'prc5%': prc5,
                "percentPCC": ppcc,
                "percentR2": pr2,}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def score_cls(self, pred, label):
        pred = np.array(pred).reshape(-1)
        label = np.array(label).reshape(-1)
        try:
            auroc = roc_auc_score(label, pred)
        except:
            auroc = np.nan
        try:
            aupr = average_precision_score(label, pred)
        except:
            aupr = np.nan
        try:
            index = np.argsort(pred)[::-1]
            num = int(np.sum(label))
            epr = precision_score(label[index[:num]], np.array(self.sigmoid(pred[index[:num]])>0.5, dtype=float)) / np.mean(label)
        except:
            epr = np.nan
        return {"auroc":auroc, "aupr":aupr, "epr":epr}

    def score_distance(self, pred, label, a_mask, b_mask):
        results = {}
        for idx in range(label.shape[0]):
            num_vertex = int(np.sum(a_mask[idx]))
            num_residue = int(np.sum(b_mask[idx]))
            pred_i = pred[idx, :num_vertex, :num_residue].reshape(-1)
            label_i = label[idx, :num_vertex, :num_residue].reshape(-1)
            assert len(np.where(label_i == 0)[0]) == 0
            for key,value in self.score_reg(label_i, pred_i).items():
                if key not in results:
                    results[key] = []
                results[key].append(value)
        for item in results:
            results[item] = np.nanmean(results[item])
        return results
    
    def score_distance_triu(self, pred, label, mask):
        results = {}
        mask_index = np.triu_indices(mask.shape[1], 1)
        mask = mask[:,None] * mask[:,:,None]
        loss_mask = np.array(mask[:, mask_index[0], mask_index[1]], dtype=bool)
        
        for idx in range(label.shape[0]):
            mask_i = loss_mask[idx]
            pred_i = pred[idx, mask_i].reshape(-1)
            label_i = label[idx, mask_i].reshape(-1)
            for key,value in self.score_reg(label_i, pred_i).items():
                if key not in results:
                    results[key] = []
                results[key].append(value)
        for item in results:
            results[item] = np.nanmean(results[item])
            
        return results

    def score_int(self, pred, label, interaction_mask, vertex_mask, seq_mask, pocket_mask):
        results = {}
        for idx in range(len(interaction_mask)):
            if not interaction_mask[idx]:
                continue
            num_vertex = int(np.sum(vertex_mask[idx,:]))
            pocket_idx_i = np.where(pocket_mask[idx])[0]
            pred_i = pred[idx][:num_vertex, pocket_idx_i].reshape(-1)
            label_i = label[idx][:num_vertex, pocket_idx_i].reshape(-1)
            if np.sum(label_i) == 0:
                continue
            for key,value in self.score_cls(label_i, pred_i).items():
                if key not in results:
                    results[key] = []
                results[key].append(value)
        for item in results:
            results[item] = np.nanmean(results[item])

        return results

    def score_distance_list(self, list_pred, list_label, list_a_mask, list_b_mask):
        results = {}
        for i in range(len(list_pred)):
            pred = list_pred[i]
            label = list_label[i]
            a_mask = list_a_mask[i]
            b_mask = list_b_mask[i]
            for idx in range(label.shape[0]):
                num_a = int(torch.sum(a_mask[idx]))
                num_b = int(torch.sum(b_mask[idx]))
                pred_i = pred[idx, :num_a, :num_b].reshape(-1)
                label_i = label[idx, :num_a, :num_b].reshape(-1)
                assert len(torch.where(label_i == 0)[0]) == 0
                for key,value in self.score_reg(label_i, pred_i).items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value)
        for item in results:
            results[item] = np.nanmean(results[item])
        return results

    def score_distance_triu_list(self, list_pred, list_label, list_mask):
        results = {}
        for i in range(len(list_pred)):
            pred = list_pred[i]
            label = list_label[i]
            mask = list_mask[i]
            mask_index = np.triu_indices(mask.shape[1], 1)
            mask = mask[:,None] * mask[:,:,None]
            loss_mask = np.array(mask[:, mask_index[0], mask_index[1]], dtype=bool)
            for idx in range(label.shape[0]):
                mask_i = loss_mask[idx]
                pred_i = pred[idx, mask_i].reshape(-1)
                label_i = label[idx, mask_i].reshape(-1)
                for key,value in self.score_reg(label_i, pred_i).items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value)
        for item in results:
            results[item] = np.nanmean(results[item])
        return results

    def score_int_list(self, list_pred, list_label, list_interaction_mask, list_a_mask, list_b_mask):
        results = {}
        for i in range(len(list_pred)):
            pred = list_pred[i]
            label = list_label[i]
            interaction_mask = list_interaction_mask[i]
            a_mask = list_a_mask[i]
            b_mask = list_b_mask[i]
            for idx in range(len(interaction_mask)):
                if not interaction_mask[idx]:
                    continue
                num_a = int(np.sum(a_mask[idx,:]))
                b_idx_i = np.where(b_mask[idx])[0]
                pred_i = pred[idx][:num_a, b_idx_i].reshape(-1)
                label_i = label[idx][:num_a, b_idx_i].reshape(-1)
                if np.sum(label_i) == 0:
                    continue
                for key,value in self.score_cls(label_i, pred_i).items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value)
        for item in results:
            results[item] = np.nanmean(results[item])
        return results

    def score_margin(self, pred, label):
        pred = np.array(pred).reshape(-1)
        label = np.array(label).reshape(-1)
        mask = (pred > label)
        pred = pred[mask]
        label = label[mask]

        if len(pred) == 0:
            return {"mae":0.0, "mse":0.0,}

        try:
            mse = mean_squared_error(label, pred)
        except:
            mse = np.nan

        try:
            mae = mean_absolute_error(label, pred)
        except:
            mae = np.nan

        return {"mae":mae, "mse":mse,}



