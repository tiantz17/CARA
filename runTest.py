import os
import sys
import time
import json
import pickle
import socket
import logging
import argparse
from importlib import import_module

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
torch.multiprocessing.set_sharing_strategy('file_system')


from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class CARATest(object):
    """
    CARA test
    """
    def __init__(self, args):
        """ common parameters """
        self.seed = args.seed
        self.info = args.info
        self.gpu = args.gpu
        self.use_cuda = args.gpu != "-1"
        self.path = args.path
        self.num_workers = args.num_workers

        """ special parameters """
        self.dataset = args.model
        self.meta = 'Meta' in self.dataset
        self.model = self.dataset.split("Meta")[0]
        self.model_path = args.model_path

        """ modules """
        self.DATASET = import_module("src.datasets."+self.dataset)
        self.MODEL = import_module("src.models."+self.model)
        self.META_MODEL = import_module("src.models.ModelMeta")

        """ training parameters """      
        self.dataset_params = self.DATASET.DATASET_PARAMS
        self.model_params = self.MODEL.MODEL_PARAMS
        self.train_params = self.MODEL.TRAIN_PARAMS

        if len(args.dataset_params) > 0:
            update_params = {item.split(':')[0]:item.split(':')[1] for item in args.dataset_params.split(',')}
        else:
            update_params = {}
        self.dataset_params.update(update_params)

        if len(args.model_params) > 0:
            update_params = {item.split(':')[0]:item.split(':')[1] for item in args.model_params.split(',')}
        else:
            update_params = {}
        self.model_params.update(update_params)

        if len(args.train_params) > 0:
            update_params = {item.split(':')[0]:item.split(':')[1] for item in args.train_params.split(',')}
        else:
            update_params = {}
        self.train_params.update(update_params)
        self.train_params['use_cuda'] = self.use_cuda
        self.train_params['regression'] = True

        """ update common parameters"""
        self.num_repeat = int(self.train_params["num_repeat"])
        self.num_fold = int(self.train_params["num_fold"])
        self.list_task = self.train_params["list_task"]

        self.task = self.dataset_params['task']
        if self.meta:
            self.task += 'Meta{}way{}shot'.format(self.dataset_params['n_way'], self.dataset_params['k_shot'])
        if 'test_assay' in self.dataset_params and self.dataset_params['test_assay'] != '':
            self.task += 'Sim'
            
        """ local directory """
        file_folder = "CARATest_task_{}_model_{}_info_{}_{}_cuda{}"
        file_folder = file_folder.format(self.task, self.dataset, \
            self.info, socket.gethostname(), self.gpu)
        file_folder += time.strftime("_%Y%m%d_%H%M%S/", time.localtime())
        self.save_path = self.path + "/predictions/" + file_folder
        self.valid_log_file = self.save_path + "validation.log"
        self.test_log_file = self.save_path + "test.log"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.define_logging()
        logging.info("Local folder created: {}".format(self.save_path))

        """ save hyperparameters """
        self.save_hyperparameter(args)

    def define_logging(self):
        # Create a logger
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %A %H:%M:%S',
            filename=self.save_path + "logging.log",
            filemode='w')
        # Define a Handler and set a format which output to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

    def save_hyperparameter(self, args):
        args.dataset_params = self.dataset_params
        args.model_params = self.model_params
        args.train_params = self.train_params
        json.dump(dict(args._get_kwargs()), open(self.save_path + "config", "w+"), indent=4, sort_keys=True)

    def load_data(self):
        logging.info("Loading data...")
        # load data       
        self.Dataset = self.DATASET.DataSet(self.path + "/data/", self.dataset_params)

        if not self.meta:
            self.Dataloader = DataLoader(self.Dataset, 
                                        batch_size=int(self.train_params["batch_size"]), 
                                        shuffle=False, 
                                        collate_fn=eval("self.DATASET.batch_data_process_"+self.model), 
                                        num_workers=self.num_workers, 
#                                         persistent_workers=False, 
                                        drop_last=False, 
                                        pin_memory=False)

    def split_data(self, seed):
        if not self.meta:
            logging.info("Spliting data into {} folds with seed {}".format(self.num_fold, seed))
            self.Dataset.split_data(seed, self.num_fold)
        
    def get_data_batch(self, batch_items):
        if self.use_cuda: 
            batch_items = [item.to(next(self.Model.parameters()).device) if item is not None and not isinstance(item, list) else \
                [it.to(next(self.Model.parameters()).device) for it in item] if isinstance(item, list) else \
                None for item in batch_items]

        return batch_items  

    def get_label_batch(self, batch_items):
        if self.use_cuda: 
            for key in batch_items.keys():
                if key in self.list_task:
                    batch_items[key] = batch_items[key].to(next(self.Model.parameters()).device)

        return batch_items

    def load_model(self, model_file):
        logging.info("Loading model...")
        if self.use_cuda:
            device = torch.device("cuda:"+self.gpu)
        else:
            device = torch.device("cpu")
        # load model
        if self.meta:
            self.Model = self.META_MODEL.Model(self.train_params, self.model_params, self.MODEL.Model)
        else:
            self.Model = self.MODEL.Model(self.model_params)
        if model_file is not None:
            state_dict = torch.load(model_file, map_location=torch.device("cpu"))
            if not self.meta:
                self.Model.load_state_dict(state_dict)
            else:
                self.Model.learner.net.load_state_dict(state_dict)

        if not self.meta:
            self.Model.load_optimizer(self.train_params)
        else:
            self.Model.learner.net.load_optimizer(self.train_params)
            self.Model.learner.net_pi.load_optimizer(self.train_params)

        self.Model = self.Model.to(device)

    def predict(self):
        logging.info("Start prediction")
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        
        list_results = []
        list_dict_step = []
        list_dict_collect = []
        list_dict_pred = []
        self.load_data()
        list_models = []
        if self.model_path == '':
            list_models = [None] * 5
        else:
            for i in os.listdir(self.model_path):
                if "best_model" not in i:
                    continue
                if i[-2:] != "pt":
                    continue
                list_models.append(i)
            list_models = sorted(list_models)
        for repeat, model_file in enumerate(list_models):
            if model_file is not None:
                self.load_model(self.model_path + model_file)
            else:
                self.load_model(None)
            if self.meta:
                if self.dataset_params['subset'] == 'cvfinetune':
                    self.Dataset.reset_cv_index(repeat)
                dict_steps, dict_collect, dict_pred, results = self.evaluate_meta()
            else:
                dict_collect, dict_pred, results = self.evaluate()
                dict_steps = None
            list_results.append(results)
            list_dict_step.append(dict_steps)
            list_dict_collect.append(dict_collect)
            list_dict_pred.append(dict_pred)
            logging.info("="*60)
            logging.info("Repeat: {}, model: {}".format(repeat, model_file))
            for term in results: 
                logging.info("{}: {}".format(term, results[term]))
        results_all = self.merge_list_results(list_results)
        self.save(list_dict_step, list_dict_collect, list_dict_pred, results_all)

        logging.info("="*60)
        logging.info("All done prediction at {}".format(self.model))
        logging.info("Results per repeat:")
        for repeat, results in enumerate(list_results):
            logging.info("="*60)
            logging.info("Repeat: {}".format(repeat))
            for term in results: 
                logging.info("{}: {}".format(term, results[term]))
        logging.info("-"*60)
        logging.info("Results all:")
        for term in results_all: 
            logging.info("{}: {}".format(term, results_all[term]))

    def merge_list_results(self, list_results):
        dict_one_repeat = {}
        for results in list_results:
            for term in results:
                if term not in dict_one_repeat:
                    dict_one_repeat[term] = {}
                for score in results[term]:
                    if score not in dict_one_repeat[term]:
                        dict_one_repeat[term][score] = []
                    if isinstance(results[term][score], list):
                        dict_one_repeat[term][score].append(results[term][score][0])
                    else:
                        dict_one_repeat[term][score].append(results[term][score])
        for term in dict_one_repeat:
            for score in dict_one_repeat[term]:
                average = float(np.nanmean(dict_one_repeat[term][score]))
                std = float(np.nanstd(dict_one_repeat[term][score]))
                dict_one_repeat[term][score] = [average, std]

        return dict_one_repeat

    def check_data_leakage(self):
        overlap = np.intersect1d(list(self.index_train), list(self.index_valid))
        assert len(overlap) == 0, "Data leakage observed for valid set: {}".format(len(overlap))

    def evaluate(self):
        self.Model.eval()
        dict_results = {}
        dict_valuetypes = {}
        task = "Affinity"
        score_func = self.Dataset.score(self.Model.task_eval[task])
        list_labels = []
        list_preds = []
        with torch.no_grad():
            for _, data in enumerate(self.Dataloader):
                data_tuple, label_dict = data
                data_tuple = self.get_data_batch(data_tuple)
                label_dict = self.get_label_batch(label_dict)
                loss, pred_dict = self.Model(data_tuple, label_dict)

                preds = pred_dict[task].cpu().data.numpy()
                labels = label_dict[task].cpu().data.numpy()
                assay_id = label_dict['assay_id']
                value_type = label_dict['value_type']

                list_labels.append(labels)
                list_preds.append(preds)
                for k in range(len(assay_id)):
                    id = assay_id[k]
                    if id not in dict_results.keys():
                        dict_results[id] = []
                        dict_valuetypes[id] = []
                    dict_results[id].append(np.array([preds[k], labels[k]]))
                    dict_valuetypes[id].append(value_type[k])

                if self.info == "debug":
                    break  
            dict_results = {item:np.array(dict_results[item]) for item in dict_results}
#             temp = np.concatenate(list(dict_results.values()))
#             res = score_func(temp[:,0], temp[:,1])
            dict_score = {}
            dict_pred = {}
            # per-assay evaluation
            for assay in dict_results.keys():
                pred = dict_results[assay][:, 0]
                label = dict_results[assay][:, 1]
                value_type = np.array(dict_valuetypes[assay])
                list_value_type = np.unique(value_type)
                if len(list_value_type) == 0:
                    res = score_func(pred, label)
                else:
                    num_dict = {}
                    temp_dict = {}
                    for vt in list_value_type:
                        temp_index = value_type==vt
                        if sum(temp_index) < 1:
                            continue
                        num_dict[vt] = sum(temp_index)
                        temp_dict[vt] = score_func(pred[temp_index], label[temp_index])
                        
                    max_num = 0
                    for vt in num_dict:
                        if num_dict[vt] > max_num:
                            max_num = num_dict[vt]
                            
                    res = {}
                    count = 0
                    for vt in temp_dict:
                        if num_dict[vt] == max_num:
                            count += 1
                            for item in temp_dict[vt]:
                                if item not in res:
                                    res[item] = temp_dict[vt][item]
                                else:
                                    res[item] += temp_dict[vt][item]
                    for item in res:
                        res[item] /= count
                
                dict_score[assay] = res
                dict_pred[assay] = {'pred': pred, 'label': label}
            results = self.merge_list_results([{"PerAssayAffinity":dict_score[item]} for item in dict_score])
            list_labels = np.concatenate(list_labels)
            list_preds = np.concatenate(list_preds)
            results['AllAssayAffinity'] = score_func(list_preds, list_labels)

        return dict_score, dict_pred, results

    def evaluate_meta(self):
        dict_results = {}
        task = "Affinity"
        score_func = self.Dataset.score(self.Model.task_eval[task])
        list_labels = []
        list_preds = []
        dict_preds_step = {}
        save_steps = False

        for i, assay_id in enumerate(self.Dataset.assay_list):
            torch.cuda.empty_cache()
#             assay_id = "CHEMBL1614458_Potency"
            # data_time = time.time()
            x_spt, x_qry, y_spt, y_qry = self.Dataset.get_finetune_sample(assay_id)
            
            if (len(x_spt[0]) == 0) or (len(x_qry[0]) == 0):
                continue
            if 'finetune' in self.dataset_params['subset']:
                pred_collect, preds_list, steps_list = self.Model.pred_step_by_step(x_spt, y_spt, x_qry, y_qry, int(self.dataset_params['step']))
                save_steps = True
            elif self.dataset_params['subset'] == 'query':
                pred_collect = self.Model.pred(x_spt, y_spt, x_qry, y_qry, 0)

            preds = np.array(pred_collect[task]["pred"]).reshape((-1, 1))
            labels = np.array(pred_collect[task]["label"]).reshape((-1, 1))

            list_labels.append(labels)
            list_preds.append(preds)
            dict_results[assay_id] = np.concatenate([preds, labels], axis=1)
            if save_steps:
                dict_preds_step[assay_id] = (preds_list, steps_list)
            if self.info == "debug":
                break  

        dict_score = {}
        dict_pred = {}
        dict_steps = {}
        for assay in dict_results.keys():
            label = dict_results[assay][:, 1]
            pred = dict_results[assay][:, 0]  
            res = score_func(pred, label)    
            dict_score[assay] = res
            dict_pred[assay] = {'pred': pred, 'label': label}
            if save_steps:
                for pred, step in zip(*dict_preds_step[assay]):
                    if assay not in dict_steps:
                        dict_steps[assay] = {}
                    pred = np.array(pred).reshape(-1)
                    res = score_func(pred, label)
                    dict_steps[assay][step] = res

        
        results = self.merge_list_results([{"PerAssayAffinity":dict_score[item]} for item in dict_score])
        list_labels = np.concatenate(list_labels)
        list_preds = np.concatenate(list_preds)
        results['AllAssayAffinity'] = score_func(list_preds, list_labels)

        return dict_steps, dict_score, dict_pred, results

    def get_results_template(self):
        results = {}
        for task in self.list_task:
            results[task] = {"pred":[], "label":[]}
        return results

    def save(self, dict_step, dict_collect, dict_pred, results):
        if dict_step[0] is not None:
            pickle.dump(dict_step, open(self.save_path + "dict_step", "wb"))
        pickle.dump(dict_collect, open(self.save_path + "dict_collect", "wb"))
        pickle.dump(dict_pred, open(self.save_path + "dict_pred", "wb"))
        json.dump(results, open(self.save_path + "results", "w"), indent=4, sort_keys=True)
        logging.info("Prediction results saved at " + self.save_path)


def main():
    parser = argparse.ArgumentParser()
    # define environment
    parser.add_argument("--gpu", default="0", help="which GPU to use", type=str)
    parser.add_argument("--seed", default=1234, help="random seed", type=int)
    parser.add_argument("--info", default="fast", help="output folder special marker", type=str)
    parser.add_argument("--path", default="./", help="data path", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)

    # define model
    parser.add_argument("--model", default="DeepCPI", help="model", type=str)
    parser.add_argument("--model_path", default="", help="path to saved model", type=str)

    # define parameters
    parser.add_argument("--dataset_params", default="", help="dict of dataset parameters", type=str)
    parser.add_argument("--model_params", default="", help="dict of model parameters", type=str)
    parser.add_argument("--train_params", default="", help="dict of training parameters", type=str)
    args = parser.parse_args()

    """ check """
    # dataset
    if not os.path.exists(args.path + "/src/datasets/" + args.model.split("Meta")[0] + ".py"):
        raise NotImplementedError("Dataset {} not found!".format(args.model))
    
    # model
    if not os.path.exists(args.path + "/src/models/" + args.model.split("Meta")[0] + ".py"):
        raise NotImplementedError("Model {} not found!".format(args.model))

    """ claim class instance """
    tester = CARATest(args)
        
    """ Test """
    tester.predict()


if __name__ == "__main__":
    main()

