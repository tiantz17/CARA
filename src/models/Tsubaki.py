import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.src.models.utils import weights_init



MODEL_PARAMS = {
    'dim': 10,
    'n_word': 8614,
    'n_fingerprint': 72199,
    'window': 5,
    'layer_gnn': 3,
    'layer_cnn': 3,
    'layer_output': 3,
}

TRAIN_PARAMS = {
    'num_repeat': 1,
    'num_fold': 5,
    'batch_size': 128,
    'max_epoch': 500,
    'early_stop': 20,
    'learning_rate': 1e-3,
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


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.dim = int(args['dim'])
        self.n_word = int(args['n_word'])
        self.n_fingerprint = int(args['n_fingerprint'])
        self.window = int(args['window'])
        self.layer_gnn = int(args['layer_gnn'])
        self.layer_cnn = int(args['layer_cnn'])
        self.layer_output = int(args['layer_output'])

        self.embed_fingerprint = nn.Embedding(self.n_fingerprint, self.dim)
        self.embed_word = nn.Embedding(self.n_word, self.dim)
        self.W_gnn = nn.ModuleList([nn.Linear(self.dim, self.dim)
                                    for _ in range(self.layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*self.window+1,
                     stride=1, padding=self.window) for _ in range(self.layer_cnn)])
        self.W_attention = nn.Linear(self.dim, self.dim)
        self.W_out = nn.ModuleList([nn.Linear(2*self.dim, 2*self.dim)
                                    for _ in range(self.layer_output)])
        self.W_interaction = nn.Linear(2*self.dim, 1)
        
        self.apply(weights_init)

    def gnn(self, xs, A, x_mask, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        feature = torch.sum(xs * x_mask, 1) / torch.sum(x_mask, 1)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return feature

    def attention_cnn(self, x, xs, x_mask, layer):
        """The attention mechanism is applied to the last layer of CNN."""
        xs = torch.unsqueeze(xs, 1)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(xs, 1)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))

        # weights = torch.tanh(F.linear(h, hs))
        weights = torch.tanh(torch.matmul(hs, h.unsqueeze(2)))
        ys = weights * hs

        feature = torch.sum(ys * x_mask, 1) / torch.sum(x_mask, 1)
        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return feature

    def forward(self, input, label):

        fingerprints, adjacency, words, cmask, pmask = input

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, cmask.unsqueeze(2), self.layer_gnn)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, pmask.unsqueeze(2), self.layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(self.layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        affinity = self.W_interaction(cat_vector)


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

    # def __call__(self, data, train=True):

    #     inputs, correct_interaction = data[:-1], data[-1]
    #     predicted_interaction = self.forward(inputs)

    #     if train:
    #         loss = F.cross_entropy(predicted_interaction, correct_interaction)
    #         return loss
    #     else:
    #         correct_labels = correct_interaction.to('cpu').data.numpy()
    #         ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
    #         predicted_labels = list(map(lambda x: np.argmax(x), ys))
    #         predicted_scores = list(map(lambda x: x[1], ys))
    #         return correct_labels, predicted_labels, predicted_scores
