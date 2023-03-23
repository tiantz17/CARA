import torch
import torch.nn as nn
import torch.optim as optim
from common.src.models.utils import weights_init

MODEL_PARAMS = {
    'mol_input_dim': 200,
    'seq_input_dim': 100,
    'encode1': 1024,
    'encode2': 256,
    'output1': 512,
    'output2': 128,                 
    'output3': 32,
}

TRAIN_PARAMS = {
    'num_repeat': 1,
    'num_fold': 5,
    'batch_size': 128,
    'max_epoch': 500,
    'early_stop': 20,
    'learning_rate': 5e-4,
    'weight_decay': 0, 
#     'learning_rate': 1e-3,
#     'weight_decay': 1e-6, 
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
        self.seq_input_dim = int(args["seq_input_dim"])
        self.encode1 = int(args["encode1"])
        self.encode2 = int(args["encode2"])
        self.output1 = int(args["output1"])
        self.output2 = int(args["output2"])
        self.output3 = int(args['output3'])
        
        if 'mini' in args and args['mini'] == 'mini':
            self.CompoundEncoding = nn.Sequential(
                nn.Linear(self.mol_input_dim, self.encode2), 
                nn.BatchNorm1d(self.encode2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )

            self.ProteinEncoding = nn.Sequential(
                nn.Linear(self.seq_input_dim, self.encode2), 
                nn.BatchNorm1d(self.encode2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )

            self.Output = nn.Sequential(
                nn.Linear(self.encode2*2, self.output1),
                nn.BatchNorm1d(self.output1),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.output1, self.output2),
                nn.BatchNorm1d(self.output2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.output2, 1),
            )
        
        elif 'mini' in args and args['mini'] == 'mini2':
            self.CompoundEncoding = nn.Sequential(
                nn.Linear(self.mol_input_dim, self.encode2), 
                nn.BatchNorm1d(self.encode2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )

            self.ProteinEncoding = nn.Sequential(
                nn.Linear(self.seq_input_dim, self.encode2), 
                nn.BatchNorm1d(self.encode2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )

            self.Output = nn.Sequential(
                nn.Linear(self.encode2*2, self.output1),
                nn.BatchNorm1d(self.output1),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.output1, self.output2),
                nn.BatchNorm1d(self.output2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.output2, 1),
            )

        else:
            """Compound Encoding Module"""
            self.CompoundEncoding = nn.Sequential(
                nn.Linear(self.mol_input_dim, self.encode1), 
                nn.BatchNorm1d(self.encode1),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.encode1, self.encode2),
                nn.BatchNorm1d(self.encode2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )

            """Protein Encoding Module"""
            self.ProteinEncoding = nn.Sequential(
                nn.Linear(self.seq_input_dim, self.encode1), 
                nn.BatchNorm1d(self.encode1),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.encode1, self.encode2),
                nn.BatchNorm1d(self.encode2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )

            """Output Module"""
            self.Output = nn.Sequential(
                nn.Linear(self.encode2*2, self.output1),
                nn.BatchNorm1d(self.output1),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.output1, self.output2),
                nn.BatchNorm1d(self.output2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.output2, self.output3),
                nn.BatchNorm1d(self.output3),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.output3, 1),
            )

        self.apply(weights_init)

    def forward(self, input, label):
        comp, prot = input
        ca = self.CompoundEncoding(comp)
        pa = self.ProteinEncoding(prot)
        affinity = self.Output(torch.cat((ca, pa), dim=-1))
        pred = {'Affinity': affinity}
        loss = self.get_loss(pred, label)

        return loss, pred
    
    def predict(self, input):
        comp, prot = input
        ca = self.CompoundEncoding(comp)
        pa = self.ProteinEncoding(prot)
        affinity = self.Output(torch.cat((ca, pa), dim=-1))

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

        
class FeatureNet(nn.Module):
    """
    CPI model for Graph and Sequence
    """
    def __init__(self, args):
        super(FeatureNet, self).__init__()
        self.mol_input_dim = int(args["mol_input_dim"])
        self.seq_input_dim = int(args["seq_input_dim"])
        self.encode1 = int(args["encode1"])
        self.encode2 = int(args["encode2"])
        self.output1 = int(args["output1"])
        self.output2 = int(args["output2"])
        self.output3 = int(args['output3'])
        
        """Compound Encoding Module"""
        self.CompoundEncoding = nn.Sequential(
            nn.Linear(self.mol_input_dim, self.encode1), 
            nn.BatchNorm1d(self.encode1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.encode1, self.encode2),
            nn.BatchNorm1d(self.encode2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        """Protein Encoding Module"""
        self.ProteinEncoding = nn.Sequential(
            nn.Linear(self.seq_input_dim, self.encode1), 
            nn.BatchNorm1d(self.encode1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.encode1, self.encode2),
            nn.BatchNorm1d(self.encode2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        
        """Output Module"""
        self.Output = nn.Sequential(
            nn.Linear(self.encode2*2, self.output1),
            nn.BatchNorm1d(self.output1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.output1, self.output2),
            nn.BatchNorm1d(self.output2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.output2, self.output3),
        )

    def forward(self, input):
        comp, prot = input
        ca = self.CompoundEncoding(comp)
        pa = self.ProteinEncoding(prot)
        pred = self.Output(torch.cat((ca, pa), dim=-1))

        return pred

    def load_optimizer(self, train_params):
        self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, self.parameters())), 
                                    lr=train_params["learning_rate"], 
                                    weight_decay=train_params["weight_decay"])

class PolicyNet(nn.Module):
    """
    CPI model for Graph and Sequence
    """
    def __init__(self, args):
        super(PolicyNet, self).__init__()
        self.output1 = int(args["output1"])
        self.output2 = int(args["output2"])
        self.output3 = int(args['output3'])
        
        """Output Module"""
        self.Output = nn.Sequential(
            nn.BatchNorm1d(self.output3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.output3, 1),
        )


    def forward(self, input):
        pred = self.Output(input)

        return pred

    def load_optimizer(self, train_params):
        self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, self.parameters())), 
                                    lr=train_params["learning_rate"], 
                                    weight_decay=train_params["weight_decay"])

class RegressionNet(nn.Module):
    """
    CPI model for Graph and Sequence
    """
    def __init__(self, args):
        super(RegressionNet, self).__init__()
        self.output1 = int(args["output1"])
        self.output2 = int(args["output2"])
        self.output3 = int(args['output3'])
        
        """Output Module"""
        self.Output = nn.Sequential(
            nn.BatchNorm1d(self.output3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.output3, 1),
        )

    def forward(self, input, label):
        affinity = self.Output(input)
        pred = {'Affinity': affinity}
        loss = self.get_loss(pred, label)

        return loss, pred

    def load_optimizer(self, train_params):
        self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, self.parameters())), 
                                    lr=train_params["learning_rate"], 
                                    weight_decay=train_params["weight_decay"])

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


