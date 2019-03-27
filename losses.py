import torch
from torch import nn
import numpy as np


class PairwiseMarginRankingLoss(nn.Module):
    def __init__(self, margin=0.0, size_average=None, reduce=None, reduction='mean'):
        """
        Pairwise Margin Ranking Loss. All setted parameters redirected to nn.MarginRankingLoss.
        The only difference is that pairs automatically generated for margin ranking loss.
        All possible pairs of different class are generated.
        """
        super().__init__()
        self.margin_loss = nn.MarginRankingLoss(margin, size_average, reduce, reduction)

    def forward(self, prediction, label):
        """
        Get pairwise margin ranking loss.
        Args:
            prediction (torch.FloatTensor): tensor of shape Bx1 of predicted probabilities
            label (torch.FloatTensor): tensor of shape Bx1 of true labels for pair generation
        """
        # initialize
        out_pred_1 = []
        out_pred_2 = []
        out_targ =[]

        # positive-negative selectors
        mask_0 = label == 0
        mask_1 = label == 1

        # selected predictions
        pred_0 = torch.masked_select(prediction, mask_0)
        pred_1 = torch.masked_select(prediction, mask_1)
        pred_1_n = pred_1.size()[0]
        pred_0_n = pred_0.size()[0]

        # create pairs
        pred_00 = pred_0.unsqueeze(0).repeat(1, pred_1_n)
        pred_11 = pred_1.unsqueeze(1).repeat(1, pred_0_n).view(pred_00.size())
        out01 = -1 * torch.ones(pred_1_n*pred_0_n).to(prediction.device)

        return self.margin_loss(pred_00.view(-1), pred_11.view(-1), out01)
