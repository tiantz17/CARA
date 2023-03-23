import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common.src.models.utils import weights_init



MODEL_PARAMS = {
    'drug_len': 2048,
    'prot_len': 2500,
    'filters': 64,
    'protein_dim': 20,
}

TRAIN_PARAMS = {
    'num_repeat': 1,
    'num_fold': 5,
    'batch_size': 128,
    'max_epoch': 500,
    'early_stop': 20,
    'learning_rate': 5e-4,
    'weight_decay': 1e-6,
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
        self.drug_len = int(args['drug_len'])
        self.prot_len = int(args['prot_len'])
        self.filters = int(args['filters'])
        self.protein_dim = int(args['protein_dim'])

        self.drug_encoder = nn.Sequential(
            nn.Linear(self.drug_len, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.protein_embedding = nn.Sequential(
            nn.Embedding(26, self.protein_dim),
            Transpose(1, 2),# convert to B,C,L
            nn.Dropout2d(0.2),
        )
        self.CNN = nn.ModuleList()
        for kernal_size in [10, 15, 20, 25]:
            self.CNN.append(nn.Sequential(
                nn.Conv1d(self.protein_dim, self.filters, kernal_size, padding=int(np.ceil((kernal_size-1)/2))),
                nn.BatchNorm1d(self.filters),
                nn.ReLU(),
                Max(-1),
            ))

        self.prot_encoder = nn.Sequential(
            nn.Linear(self.filters*4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fc_decoder = nn.Sequential(
            nn.Linear(512+64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.apply(weights_init)

    def forward(self, input, label):
        drug_feature, protein_feature = input

        drug = self.drug_encoder(drug_feature)
        prot = self.protein_embedding(protein_feature)
        prot = torch.cat([encoder(prot) for encoder in self.CNN], dim=1)
        prot = self.prot_encoder(prot)

        feature = torch.cat([drug, prot], dim=1)
        affinity = self.fc_decoder(feature)

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

