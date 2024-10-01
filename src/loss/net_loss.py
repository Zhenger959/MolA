'''
Author: Jiaxin Zheng
Date: 2023-08-31 11:56:12
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 15:50:28
Description: 
                    
'''
import torch
import torch.nn as nn

from src.loss import EdgeLoss,EdgeFocalLoss,SequenceLoss

class Criterion(nn.Module):

    def __init__(self, atom_loss,edge_loss=None,atom_bbox_loss=None,atom_prop_loss=None,edge_info_loss=None,loss_weight=None,loss_weight_freq=-1):
        super(Criterion, self).__init__()
        self.loss_weight=loss_weight
        self.loss_weight_freq=loss_weight_freq
        criterion = {}
        if atom_loss is not None:
            criterion['chartok_coords']=atom_loss
        if edge_loss is not None:
            criterion['edges']=edge_loss
        if edge_info_loss is not None:
            criterion['edge_infos']=edge_info_loss
        if atom_bbox_loss is not None:
            criterion['atom_bbox']=atom_bbox_loss
        if atom_prop_loss is not None:
            criterion['atom_prop']=atom_prop_loss
        self.criterion = nn.ModuleDict(criterion)

    def forward(self, results, refs):
        losses = {}
        for format_ in results:
            predictions, targets, *_ = results[format_]
            loss_ = self.criterion[format_](predictions, targets)
            if type(loss_) is dict:
                losses.update(loss_)
            else:
                if loss_.numel() > 1:
                    loss_ = loss_.mean()
                losses[format_] = loss_
        return losses