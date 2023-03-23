import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.src.models.utils import weights_init

from common.src.models.base import GraphNeuralNetwork, ConvNeuralNetwork, AffinityNeuralNetworkMONN

MODEL_PARAMS = {
    'node_input_dim': 82,
    'edge_input_dim': 6,
    'seq_input_dim': 20,
    'hidden_comp': 128,
    'hidden_prot': 128,
    'hidden_aff': 128,
    'GNN_depth': 4,
    'k_head': 1,
    'CNN_depth': 4,
    'kernel_size': 5,
    'ANN_depth':2,
}

TRAIN_PARAMS = {
    'num_repeat': 1,
    'num_fold': 5,
    'batch_size': 128,
    'max_epoch': 500,
    'early_stop': 20,
    'learning_rate': 5e-4,
    'weight_decay': 1e-5, 
    'loss_weight': 0,    
    'step_size': 20,
    'gamma': 0.5,
    'num_updates':2,
    'list_task': ['Affinity'],
    'loss_weight': {'Affinity': 1.0},
    'task_eval': {'Affinity': "reg"},
    'task': 'PerAssayAffinity',
    'goal': 'pcc',
}


class Model(nn.Module):
    """
    MONN for affinity prediction
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.node_input_dim = int(args["node_input_dim"])
        self.edge_input_dim = int(args["edge_input_dim"])
        self.seq_input_dim = int(args["seq_input_dim"])
        self.hidden_comp = int(args["hidden_comp"])
        self.hidden_prot = int(args["hidden_prot"])
        self.hidden_aff = int(args["hidden_aff"])
        self.GNN_depth = int(args["GNN_depth"])
        self.k_head = int(args['k_head'])
        self.CNN_depth = int(args['CNN_depth'])
        self.kernel_size = int(args['kernel_size'])
        self.ANN_depth = int(args['ANN_depth'])

        # GNN for compound
        self.GNN = GraphNeuralNetwork(
            self.node_input_dim,
            self.hidden_comp, 
            self.edge_input_dim,
            self.k_head,
            self.GNN_depth,
        )

        # CNN for protein
        self.CNN = ConvNeuralNetwork(
            self.seq_input_dim,
            self.hidden_prot,
            self.kernel_size,
            self.CNN_depth,
        )

        # Affinity
        self.ANN = AffinityNeuralNetworkMONN(
            self.hidden_comp,
            self.hidden_prot,
            self.hidden_aff,
            self.ANN_depth,
        )
        self.apply(weights_init)

    def forward(self, input, label):
        comp, prot = input
        ca, cag = self.GNN(comp)
        # prot.x = self.CNN(prot)
        # prot.to_compact()
        pa = prot._to_compact(self.CNN(prot))

        affinity = self.ANN(ca, cag, pa, comp.batch, prot.batch)
        pred = {'Affinity': affinity}
        loss = self.get_loss(pred, label)

        return loss, pred

    def load_optimizer(self, train_params):
        self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, self.parameters())), 
                                    lr=train_params["learning_rate"], 
                                    weight_decay=train_params["weight_decay"])

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                   step_size=train_params["step_size"], 
                                                   gamma=train_params["gamma"])

        self.loss = {
            'Affinity': nn.MSELoss() if train_params['regression'] else nn.BCEWithLogitsLoss(),
        }
        self.loss_weight = train_params['loss_weight']
        self.task_eval = train_params['task_eval']

    def get_loss(self, dict_pred, dict_label):
        loss = 0.0
        for task in dict_pred:
            loss = loss + self.loss[task](dict_pred[task], dict_label[task]) * self.loss_weight[task]
        return loss