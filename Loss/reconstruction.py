import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x_hat, x, mask=None):
        if mask is None:
            loss = self.criterion(x_hat, x)
        else:
            loss = self.criterion(x_hat, x)
            loss = loss * mask.view(loss.size())

        return loss.mean()


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x_hat, x, mask=None):
        if mask is None:
            loss = self.criterion(x_hat, x)
        else:
            loss = self.criterion(x_hat, x)
            loss = loss * mask.view(loss.size())

        return loss.mean()


class CategoryLoss(nn.Module):
    def __init__(self):
        super(CategoryLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x_hat, x, mask=None):
        dim = x.size(-1)

        if mask is None:
            if dim == 1:
                loss = self.bce(x_hat, x)
            else:
                loss = self.ce(x_hat.view(-1, dim), torch.argmax(x, dim=-1).view(-1))
        else:
            if dim == 1:
                loss = self.bce(x_hat, x)
            else:
                loss = self.ce(x_hat.view(-1, dim), torch.argmax(x, dim=-1).view(-1))

            loss = loss * mask.view(loss.size())

        return loss.mean()

