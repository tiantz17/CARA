import time
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.src.models.utils import weights_init

from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool

from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap



MODEL_PARAMS = {
    'name': "GINConvNet", 
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
    DeepDTA for affinity prediction
    """
    def __init__(self, args):
        super(Model, self).__init__()
        
        if args['name'] == 'GATNet':
            self.model = GATNet()
        elif args['name'] == 'GAT_GCN':
            self.model = GAT_GCN()
        elif args['name'] == 'GCNNet':
            self.model = GCNNet()
        elif args['name'] == 'GINConvNet':
            self.model = GINConvNet()
 
        self.apply(weights_init)

    def forward(self, input, label):
        compound_a, protein_a = input
        
        affinity = self.model(compound_a, protein_a)
        
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



# GAT  model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc_xt1 = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, target):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        # protein input feed-forward:
        # target = data.target
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt)
        conv_xt = self.relu(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc_xt1(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


# GCN-CNN based model
class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task

    def forward(self, data, target):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # target = data.target
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
    
    
# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data, target):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        # target = data.target

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = nn.Sequential(nn.Linear(num_features_xd, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(dim)

        nn2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(dim)

        nn3 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = nn.BatchNorm1d(dim)

        nn4 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = nn.BatchNorm1d(dim)

        nn5 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = nn.BatchNorm1d(dim)

        self.fc1_xd = nn.Linear(dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data, target):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # target = data.target

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out