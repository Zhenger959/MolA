'''
Author: Jiaxin Zheng
Date: 2023-08-31 11:53:35
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 15:50:20
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss import FocalLoss

class EdgeLoss(nn.Module):

    def __init__(self,cls_num=7):
        super(EdgeLoss, self).__init__()
        weight = torch.ones(cls_num) * 10
        weight[0] = 1
        self.criterion = nn.CrossEntropyLoss(weight, ignore_index=-100)

    def forward(self, outputs, targets):
        results = {}
        if 'coords' in outputs:
            pred = outputs['coords']
            max_len = pred.size(1)
            target = targets['coords'][:, :max_len]
            mask = target.ge(0)
            loss = F.l1_loss(pred, target, reduction='none')
            results['coords'] = (loss * mask).sum() / mask.sum()
        if 'edges' in outputs:
            pred = outputs['edges']
            max_len = pred.size(-1)
            target = targets['edges'][:, :max_len, :max_len]
            results['edges'] = self.criterion(pred, target)
        return results

class EdgeFocalLoss(nn.Module):

    def __init__(self,args):
        super(EdgeFocalLoss, self).__init__()
        self.criterion = FocalLoss(args)

    def forward(self, outputs, targets):
        results = {}
        if 'coords' in outputs:
            pred = outputs['coords']
            max_len = pred.size(1)
            target = targets['coords'][:, :max_len]
            mask = target.ge(0)
            loss = F.l1_loss(pred, target, reduction='none')
            results['coords'] = (loss * mask).sum() / mask.sum()
            
        if 'edges' in outputs:
            pred = outputs['edges']
            max_len = pred.size(-1)
            target = targets['edges'][:, :max_len, :max_len]
            
            pred=pred.reshape(-1,pred.shape[1])
            target=target.reshape(-1,1)
            results['edges'] = self.criterion(pred, target)
        return results

class EdgeInfoLoss(nn.Module):

    def __init__(self,bond_order_cls_num=17,bond_display_cls_num=15):
        super(EdgeInfoLoss, self).__init__()
        bond_order_weight = torch.ones(bond_order_cls_num) * 10
        bond_order_weight[0] = 1
        # bond_display_weight = torch.ones(bond_display_cls_num)
        bond_display_weight = torch.ones(bond_display_cls_num) * 10
        bond_display_weight[0] = 1
        
        self.order_criterion = nn.CrossEntropyLoss(bond_order_weight, ignore_index=-100)
        self.display_criterion = nn.CrossEntropyLoss(bond_display_weight, ignore_index=-100)

    def forward(self, outputs, targets):

        edge_orders_pred = outputs['edge_orders']
        edge_displays_pred = outputs['edge_displays']

        max_len = edge_orders_pred.size(-1)
        
        edge_orders_target = targets['edge_orders'][:, :max_len, :max_len]
        edge_displays_target = targets['edge_displays'][:, :max_len, :max_len]
        
        results={}

        results['edge_orders'] = self.order_criterion(edge_orders_pred, edge_orders_target)
        results['edge_displays'] = self.display_criterion(edge_displays_pred, edge_displays_target)
        
        return results