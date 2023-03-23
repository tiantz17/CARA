import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common.src.models.utils import weights_init



MODEL_PARAMS = {
    'mol_input_dim': 128,
    'seq_input_dim': 128,
    'mol_kernal_size': 6,
    'seq_kernal_size': 8,
    'hidden': 256,
    'output1': 1024,
    'output2': 1024,
    'output3': 512,
}

TRAIN_PARAMS = {
    'num_repeat': 1,
    'num_fold': 5,
    'batch_size': 128,
    'max_epoch': 500,
    'early_stop': 20,
    'learning_rate': 1e-4,
    'weight_decay': 0,
    'loss_weight': 0,    
    'step_size': 20,
    'gamma': 0.5,
    'num_updates': 2,
    'list_task': ['Affinity'],
    'loss_weight': {'Affinity': 1.0},
    'task_eval': {'Affinity': "reg"},
    'task': 'PerAssayAffinity',
    'goal': 'pcc',
}

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class Max(nn.Module):
    def __init__(self, dim):
        super(Max, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(self.dim)[0]


class Model(nn.Module):
    """
    MONN for affinity prediction
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.mol_input_dim = int(args["mol_input_dim"])
        self.seq_input_dim = int(args["seq_input_dim"])
        self.mol_kernal_size = int(args["mol_kernal_size"])
        self.seq_kernal_size = int(args["seq_kernal_size"])
        self.hidden = int(args["hidden"])
        self.output1 = int(args["output1"])
        self.output2 = int(args["output2"])
        self.output3 = int(args["output3"])
        
        """Compound Encoding Module"""
        self.CompoundEncoding = nn.Sequential(
            nn.Embedding(65, self.mol_input_dim), 
            Transpose(1, 2),
            nn.Conv1d(self.mol_input_dim, self.hidden, self.mol_kernal_size, padding=int(np.ceil((self.mol_kernal_size-1)/2))),
            nn.ReLU(),
            nn.Conv1d(self.hidden, self.hidden*2, self.mol_kernal_size, padding=int(np.ceil((self.mol_kernal_size-1)/2))),
            nn.ReLU(),
            nn.Conv1d(self.hidden*2, self.hidden*3, self.mol_kernal_size, padding=int(np.ceil((self.mol_kernal_size-1)/2))),
            nn.ReLU(),
            Max(-1),
        )

        """Output Module"""
        self.Output = nn.Sequential(
            nn.Linear(self.hidden*3, self.output1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.output1, self.output2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.output2, self.output3),
            nn.ReLU(),
            nn.Linear(self.output3, 1),
        )
        self.apply(weights_init)


    def forward(self, input, label):
        compound_a,  = input
        # Get features
        ca = self.CompoundEncoding(compound_a)
        affinity = self.Output(ca)

        pred = {"Affinity": affinity}
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