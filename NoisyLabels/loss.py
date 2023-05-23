import torch
import torch.nn as nn
import torch.nn.functional as F


class SCELoss(nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, pred, labels):
        device = pred.device

        # Cross entropy
        CE = F.cross_entropy(pred, labels)

        # Reverse cross entropy
        pred = F.softmax(pred, dim=1).clamp(min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(device).clamp(min=1e-4, max=1.0)
        RCE = -torch.sum(pred * torch.log(label_one_hot), dim=1)

        return self.alpha * CE + self.beta * RCE.mean()