import torch
import torch.nn as nn
import torch.optim as optim
import json
from common.src.models.utils import weights_init

MODEL_PARAMS = {
    'mol_input_dim': 1024,
    'encode1': 1024,
    'encode2': 391,
    'task': 'LO_All',
    'subset': 'support',
}

TRAIN_PARAMS = {
    'num_repeat': 1,
    'num_fold': 5,
    'batch_size': 128,
    'max_epoch': 500,
    'early_stop': 20,
    'learning_rate': 5e-5,
    'weight_decay': 0.002, 
    'loss_weight': 0,    
    'step_size': 20,
    'num_updates': 2,
    'gamma': 0.5,
    'list_task': ['Affinity'],
    'loss_weight': {'Affinity': 1.0},
    'task_eval': {'Affinity': "reg"},
    'task': 'PerAssayAffinity',
    'goal': 'pcc',
}


class Model(nn.Module):
    """
    CPI model for Graph and Sequence
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.mol_input_dim = int(args["mol_input_dim"])
        self.encode1 = int(args["encode1"])
        self.encode2 = int(args["encode2"])
        task = args['task']
        subset = args['subset']
        self.path = './T6/data/T6/Split/{}_{}.json'.format(task, subset)
        self.list_assay = list(json.load(open(self.path, 'r')).keys())
        
        """Compound Encoding Module"""
        self.CompoundEncoding = nn.Sequential(
            nn.Linear(self.mol_input_dim, self.encode1), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.encode1, self.encode2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        """Output Module"""
        self.OutputDict = nn.ModuleDict()
        for assay in self.list_assay:
            self.OutputDict[assay] = nn.Linear(self.encode2, 1)

        self.apply(weights_init)

    def forward(self, input, label):
        comp = input[0]
        ca = self.CompoundEncoding(comp)
        affinity = []
        for assay, input in zip(label['assay_id'], ca):
            affinity.append(self.OutputDict[assay](input.reshape(1, -1)))

        affinity = torch.cat(affinity).reshape(-1, 1)
        pred = {'Affinity': affinity}
        loss = self.get_loss(pred, label)

        return loss, pred
    
    def predict(self, input):
        comp = input
        ca = self.CompoundEncoding(comp)
        affinity = self.Output(ca)

        return affinity

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

