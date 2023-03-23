import torch
import torch.nn as nn

# Custom losses
class Masked_MSELoss(nn.Module):
    def __init__(self):
        super(Masked_MSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
    def forward(self, pred, label, vertex_mask, seq_mask):
        batch_size = pred.size(0)
        loss_all = self.criterion(pred, label)
        loss_mask = torch.matmul(vertex_mask.view(batch_size,-1,1), seq_mask.view(batch_size,1,-1))
        loss = torch.sum(loss_all*loss_mask) / torch.sum(loss_mask)
        return loss


class Masked_MSELoss_triu(nn.Module):
    def __init__(self):
        super(Masked_MSELoss_triu, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
    def forward(self, pred, label, mask):
        # batch_size = pred.size(0)
        loss_all = self.criterion(pred, label)
        mask_index = torch.triu_indices(mask.shape[1], mask.shape[1], 1)
        mask_index = mask_index.cuda() if mask.is_cuda else mask_index
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        loss_mask = mask[:, mask_index[0], mask_index[1]]
        loss = torch.sum(loss_all*loss_mask) / torch.sum(loss_mask)
        return loss


class Masked_BCELoss(nn.Module):
    def __init__(self):
        super(Masked_BCELoss, self).__init__()
        self.criterion = nn.BCELoss(reduction='none')
    def forward(self, pred, label, pairwise_mask, vertex_mask, seq_mask):
        batch_size = pred.size(0)
        loss_all = self.criterion(pred, label)
        loss_mask = torch.matmul(vertex_mask.view(batch_size,-1,1), seq_mask.view(batch_size,1,-1))*pairwise_mask.view(-1, 1, 1)
        loss = torch.sum(loss_all*loss_mask) / torch.sum(pairwise_mask).clamp(min=1e-10)
        return loss


class Margin_MSELoss(nn.Module):
    def __init__(self):
        super(Margin_MSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        
    def forward(self, pred, label):
        mask = (pred > label).float()
        loss_all = self.criterion(pred, label)
        den = torch.sum(mask)
        if den == 0:
            den = 1
        loss = torch.sum(loss_all * mask) / den

        return loss


class Masked_Margin_BCELoss(nn.Module):
    def __init__(self):
        super(Masked_Margin_BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, pred, label):
        mask = (pred > label).float()
        zeros = torch.zeros_like(label)
        loss_all = self.criterion(pred-label, zeros)
        den = torch.sum(mask)
        if den == 0:
            den = 1
        loss = torch.sum(loss_all * mask) / den

        return loss


class Margin_BCELoss(nn.Module):
    def __init__(self):
        super(Margin_BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, label):
        zeros = torch.zeros_like(label)
        loss = self.criterion(pred-label, zeros)

        return loss
