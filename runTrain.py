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
# torch.multiprocessing.set_sharing_strategy('file_system')


from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class CARATrain(object):
    """
    CARA train
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
        self.train_params['batch_size'] = int(self.train_params['batch_size'])
        self.train_params['learning_rate'] = float(self.train_params['learning_rate'])
        self.train_params['weight_decay'] = float(self.train_params['weight_decay'])


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
        file_folder = "CARATrain_task_{}_model_{}_info_{}_{}_cuda{}"
        file_folder = file_folder.format(self.task, self.dataset, \
            self.info, socket.gethostname(), self.gpu)
        file_folder += time.strftime("_%Y%m%d_%H%M%S/", time.localtime())
        self.save_path = self.path + "/models/" + file_folder
        self.valid_log_file = self.save_path + "validation.log"
        self.test_log_file = self.save_path + "test.log"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.define_logging()
        logging.info("Local folder created: {}".format(self.save_path))

        self.run_path = self.path + "/runs/" + file_folder
        if not os.path.exists(self.run_path):
            os.makedirs(self.run_path)
        self.writer = SummaryWriter(self.run_path)

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
                                        shuffle=True, 
                                        collate_fn=eval("self.DATASET.batch_data_process_"+self.model), 
                                        num_workers=self.num_workers, 
                                        # persistent_workers=False, 
                                        drop_last=False, 
                                        pin_memory=True)
        else:
            self.Dataloader = DataLoader(self.Dataset, 
                                        batch_size=1, 
                                        shuffle=False, 
                                        collate_fn=(lambda x:x), 
                                        num_workers=self.num_workers, 
                                        # persistent_workers=False, 
                                        drop_last=False, 
                                        pin_memory=True)
            

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

    def load_model(self):
        logging.info("Loading model...")
        # load model
        if self.meta:
            self.Model = self.META_MODEL.Model(self.train_params, self.model_params, self.MODEL.Model)
        else:
            self.Model = self.MODEL.Model(self.model_params)
        # print model
        logging.info("Model:\n" + str(self.Model))
        
        if len(self.model_path) > 0:
            state_dict = torch.load(self.model_path + 'best_model_{}_{}.pt'.format(self.repeat, self.fold), map_location=torch.device("cpu"))
            if not self.meta:
                self.Model.load_state_dict(state_dict)
            else:
                self.Model.learner.load_state_dict(state_dict)

        if not self.meta:
            self.Model.load_optimizer(self.train_params)
        else:
            self.Model.learner.net.load_optimizer(self.train_params)
            self.Model.learner.net_pi.load_optimizer(self.train_params)

        if self.use_cuda: 
            self.Model = self.Model.cuda()
            
    def train_and_save(self):
        logging.info("Start training")
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.seeds = np.random.randint(0, 1024, self.num_repeat)
        
        list_results_all = []
        list_results_per_fold = []

        if "fold" in self.train_params:
            list_folds = [int(fold) for fold in self.train_params['fold'].split("_")]
        else:
            list_folds = list(range(self.num_fold))

        for self.repeat in range(self.num_repeat):
            self.load_data()
            self.split_data(self.seeds[self.repeat])
            list_results = []
            for self.fold in list_folds:
                self.load_model()
                best_results = self.train_one_fold()
                list_results.append(best_results)
                logging.info("="*60)
                logging.info("Repeat: {}, Fold: {}".format(self.repeat+1, self.fold+1))
                for term in best_results: 
                    logging.info("{}: {}".format(term, best_results[term]))
            results_one_repeat = self.merge_list_results(list_results)
            logging.info("="*60)
            logging.info("Done repeat: {}".format(self.repeat+1))
            for term in results_one_repeat: 
                logging.info("{}: {}".format(term, results_one_repeat[term]))
            list_results_all.append(results_one_repeat)
            list_results_per_fold.append(list_results)
        results_all_repeat = self.merge_list_results(list_results_all)
        logging.info("="*60)
        logging.info("All done {} repeat {} fold training".format(self.num_repeat, self.num_fold))
        logging.info("Results per fold:")
        for repeat, list_results in enumerate(list_results_per_fold):
            for fold, best_results in enumerate(list_results):
                logging.info("Repeat: {}, Fold: {}".format(repeat+1, fold+1))
                for term in best_results: 
                    logging.info("{}: {}".format(term, best_results[term]))
        logging.info("-"*60)
        logging.info("Results per repeat:")
        for repeat, results_one_repeat in enumerate(list_results_all):
            logging.info("Repeat: {}".format(repeat+1))
            for term in results_one_repeat: 
                logging.info("{}: {}".format(term, results_one_repeat[term]))
        logging.info("-"*60)
        logging.info("Results summary:")
        for term in results_all_repeat: 
            logging.info("{}: {}".format(term, results_all_repeat[term]))

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
                average = np.nanmean(dict_one_repeat[term][score])
                std = np.nanstd(dict_one_repeat[term][score])
                dict_one_repeat[term][score] = [average, std]

        return dict_one_repeat

    def train_one_fold(self):
        logging.info("Start training for repeat {} fold {}".format(self.repeat+1, self.fold+1))
        if not self.meta:
            self.index_train = np.concatenate([self.Dataset.list_fold_train[self.fold], self.Dataset.list_fold_valid[self.fold]])
            self.index_valid = self.Dataset.list_fold_test[self.fold]
            logging.info("Training dataset size: {}".format(len(self.index_train)))
            logging.info("Validation dataset size: {}".format(len(self.index_valid)))

        best_score = -np.inf
        best_results = {}
        patience = 0
        for self.epoch in range(int(self.train_params["max_epoch"])):
            if 'debug' in self.info and self.epoch > 0:
                break
            patience += 1
            if self.meta:
                # train
                self.Dataset.create_batch(valid=True)
                self.Dataset.create_batch()
#                 self.Dataset.create_batch_half(valid=True)
#                 self.Dataset.create_batch_half()
                self.Dataset.shuffle_batch()
                self.train_one_epoch_meta()
                # valid
                results_valid = self.evaluate_meta("Valid")
            else:
                # train
                self.train_one_epoch()
                # valid
                results_valid = self.evaluate("Valid")


            log_v = open(self.valid_log_file, "a+")
            print("Repeat:", self.repeat+1, "Fold:", self.fold+1, "Epoch:", self.epoch+1, file=log_v)
            print(results_valid, file=log_v)
            log_v.close()

            for task in results_valid:
                for term in results_valid[task]:
                    self.writer.add_scalar('Fold_{}_Valid/{}_{}'.format(self.fold, task, term), results_valid[task][term], self.epoch)

            score = results_valid[self.train_params["task"]][self.train_params["goal"]]

            if score > best_score:
                best_score = score
                best_results = results_valid
                patience = 0
                self.save()
            if patience >= self.train_params["early_stop"]:
                logging.info("Early stopped after {} epoches".format(self.epoch+1))
                break

        return best_results

    def check_data_leakage(self):
        overlap = np.intersect1d(list(self.index_train), list(self.index_valid))
        assert len(overlap) == 0, "Data leakage observed for valid set: {}".format(len(overlap))

    def train_one_epoch(self):
        # torch.autograd.set_detect_anomaly(True)
        self.Model.train()
        self.Dataset.reset_index(self.index_train)
        collect = "fast" not in self.info
        if collect:
            dict_collect = self.get_results_template()
        # start_time = time.time()
        begin_time = time.time()
        for batch, data in enumerate(self.Dataloader):
            # data_time = time.time()
            self.Model.optimizer.zero_grad()
            data_tuple, label_dict = data
            data_tuple = self.get_data_batch(data_tuple)
            label_dict = self.get_label_batch(label_dict)
            loss, pred_dict = self.Model(data_tuple, label_dict)
            loss.backward()

            nn.utils.clip_grad_norm_(self.Model.parameters(), 5.0)
            self.Model.optimizer.step()
#             model_time = time.time()

#             logging.info("Batch: {}, Data time: {}, Model time: {}".format(batch+1, data_time-start_time, model_time-data_time))
#             start_time = time.time()
            
            if collect:
                for task in dict_collect:
                    dict_collect[task]["pred"].extend(pred_dict[task].cpu().data.numpy())
                    dict_collect[task]["label"].extend(label_dict[task].cpu().data.numpy())

                # show training
                if (batch + 1) % 10 == 0:
                    results = {}
                    for term in dict_collect:
                        score_func = self.Dataset.score(self.Model.task_eval[term])
                        results[term] = score_func(dict_collect[term]["pred"], dict_collect[term]["label"])

                    stop_time = time.time()
                    logging.info("-"*60)
                    logging.info("Epoch: {}, Batch: {}, Loss: {}, Time elapsed: {}".format(self.epoch+1, batch+1, loss, stop_time-begin_time))
                    for term in results:
                        logging.info("{}: {}".format(term, str(results[term])))
                    dict_collect = self.get_results_template()
                    begin_time = time.time()
                    torch.cuda.empty_cache()

                if "debug" in self.info:
                    break

    def evaluate(self, dataset):
        self.Model.eval()
        self.check_data_leakage()
        assert dataset in ["Train", "Valid"]
        if dataset == "Train":
            self.Dataset.reset_index(self.index_train)
        elif dataset == "Valid":
            self.Dataset.reset_index(self.index_valid)

        dict_collect = self.get_results_template()
        dict_results = {}
        results = {}
        with torch.no_grad():
            for _, data in enumerate(self.Dataloader):
                data_tuple, label_dict = data
                data_tuple = self.get_data_batch(data_tuple)
                label_dict = self.get_label_batch(label_dict)
                loss, pred_dict = self.Model(data_tuple, label_dict)

                assay_id = label_dict['assay_id']
                
                if 'Affinity' in dict_collect:
                    task = 'Affinity'
                    preds = pred_dict[task].cpu().data.numpy()
                    labels = label_dict[task].cpu().data.numpy()
                    for k in range(len(assay_id)):
                        id = assay_id[k]
                        if id not in dict_results.keys():
                            dict_results[id] = []
                        dict_results[id].append(np.array([preds[k], labels[k]]))

                for task in dict_collect:
                    dict_collect[task]["pred"].extend(pred_dict[task].cpu().data.numpy())
                    dict_collect[task]["label"].extend(label_dict[task].cpu().data.numpy())

                if self.info == "debug":
                    break   

            if 'Affinity' in dict_collect:
                task = 'Affinity'
                dict_results = {item:np.array(dict_results[item]) for item in dict_results}
                temp = np.concatenate(list(dict_results.values()))
                score_func = self.Dataset.score(self.Model.task_eval[task])
                res = score_func(temp[:,0], temp[:,1])
                dict_score = {}
                for assay in dict_results.keys():
                    pred = dict_results[assay][:, 0]
                    label = dict_results[assay][:, 1]
                    res = score_func(pred, label)
                    if 'pcc' in res:
                        res['percentPCC'] = float(res['pcc'] > 0.5) * 100
                    dict_score[assay] = res

                results = self.merge_list_results([{"PerAssayAffinity":dict_score[item]} for item in dict_score])
                results['PerAssayAffinity'] = {term:value[0] for term,value in results['PerAssayAffinity'].items()}

            for term in dict_collect:
                score_func = self.Dataset.score(self.Model.task_eval[term])
                results[term] = score_func(dict_collect[term]["pred"], dict_collect[term]["label"])

            logging.info("="*60)
            logging.info("Epoch: {}, {}".format(self.epoch+1, dataset))
            for term in results:
                logging.info("{}: {}".format(term, str(results[term])))

        return results

    def get_results_template(self):
        results = {}
        for task in self.list_task:
            results[task] = {"pred":[], "label":[]}
        return results

    def save(self):
        if not self.meta:
            torch.save(self.Model, self.save_path + "best_model_{}_{}.pth".format(self.repeat, self.fold))
            torch.save(self.Model.state_dict(), self.save_path + "best_model_{}_{}.pt".format(self.repeat, self.fold))
        else:
            torch.save(self.Model.learner.net, self.save_path + "best_model_{}_{}.pth".format(self.repeat, self.fold))
            torch.save(self.Model.learner.net.state_dict(), self.save_path + "best_model_{}_{}.pt".format(self.repeat, self.fold))
        logging.info("Best model saved at " + self.save_path)

    def train_one_epoch_meta(self):
        """
        Features: (compound_batch, protein_batch),  {"Affinity": torch.FloatTensor(affinity_batch), "assay_id": assay_id}
        Normal Train:
            data_tuple, label_dict = data
            data_tuple = [(Features),(Features)...]
            label_dict = self.get_label_batch(label_dict)

        Meta Train:
            task_list, label_dict = data
            task_list = [x_spt, y_spt, x_qry, y_qry]
            x_spt, x_qry: [(Features),(Features)...]
            y_spt, y_qry: [Affinity, Affinity...]
        """
        collect = "fast" not in self.info
        begin_time = time.time()
        self.Dataset.reset_valid(False)
        
#         print(self.Dataset.table.columns)
#         for col in ['Assay ChEMBL ID', 'Molecule ChEMBL ID', 'Smiles', 'Target Sequence']:
#             print(col, self.Dataset.table[col].unique().shape)

#         start_time = time.time()

        for i, data in enumerate(self.Dataloader):
            x_spt, x_qry, y_spt, y_qry = data[0]
#             data_time = time.time()
            loss, pred_dict = self.Model(x_spt, y_spt, x_qry, y_qry)
#             model_time = time.time()
            
#             logging.info("Batch: {}, Data time: {}, Model time: {}".format(i+1, data_time-start_time, model_time-data_time))
#             start_time = time.time()
            if collect:
                dict_collect = pred_dict

                # show training
                if (i + 1) % 100 == 0:
                    results = {}
                    for term in dict_collect:
                        score_func = self.Dataset.score(self.Model.task_eval[term])
                        results[term] = score_func(dict_collect[term]["pred"], dict_collect[term]["label"])

                    stop_time = time.time()
                    logging.info("-"*60)
                    logging.info("Epoch: {}, Batch: {}, Loss: {}, Time elapsed: {}".format(self.epoch+1, i+1, loss, stop_time-begin_time))
                    for term in results:
                        logging.info("{}: {}".format(term, str(results[term])))
                    begin_time = time.time()
                    torch.cuda.empty_cache()

                if "debug" in self.info:
                    break

    def evaluate_meta(self, dataset):
        self.Dataset.reset_valid(True)
        dict_collect = self.get_results_template()
        dict_per_assay = {}
        for i, data in enumerate(self.Dataloader):
            x_spt, x_qry, y_spt, y_qry = data[0]
            pred_collect = self.Model.pred(x_spt, y_spt, x_qry, y_qry, 2)

            for task in dict_collect:
                pred = pred_collect[task]["pred"]
                label = pred_collect[task]["label"]
                dict_collect[task]["pred"].extend(pred)
                dict_collect[task]["label"].extend(label)
        
                score_func = self.Dataset.score(self.Model.task_eval[task])
                res = score_func(pred, label)
                if 'pcc' in res:
                    res['percentPCC'] = float(res['pcc'] > 0.5) * 100
                dict_per_assay[i] = res
            if self.info == "debug":
                break  

        dict_per_assay = self.merge_list_results([{"PerAssayAffinity":dict_per_assay[item]} for item in dict_per_assay])
        
        results = {}
        for term in dict_collect:
            score_func = self.Dataset.score(self.Model.task_eval[term])
            results[term] = score_func(dict_collect[term]["pred"], dict_collect[term]["label"])

        results['PerAssayAffinity'] = {term:value[0] for term,value in dict_per_assay['PerAssayAffinity'].items()}

        logging.info("="*60)
        logging.info("Epoch: {}, {}".format(self.epoch+1, dataset))
        for term in results:
            logging.info("{}: {}".format(term, str(results[term])))


        return results


def main():
    parser = argparse.ArgumentParser()
    # define environment
    parser.add_argument("--gpu", default="0", help="which GPU to use", type=str)
    parser.add_argument("--seed", default=1234, help="random seed", type=int)
    parser.add_argument("--info", default="fast", help="output folder special marker", type=str)
    parser.add_argument("--path", default="./", help="data path", type=str)
    parser.add_argument("--num_workers", default=8, help="num_workers", type=int)

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

    """ set gpu """
    if args.gpu != "-1":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    """ claim class instance """
    trainer = CARATrain(args)
        
    """ Train """
    trainer.train_and_save()


if __name__ == "__main__":
    main()


    

