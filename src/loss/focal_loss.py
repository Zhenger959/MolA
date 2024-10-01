'''
Author: Jiaxin Zheng
Date: 2023-08-31 11:27:53
LastEditors: Jiaxin Zheng
LastEditTime: 2023-09-21 19:39:49
Description: 
'''
import torch
import torch.nn as nn
from torch.autograd import Variable

class FocalLoss(nn.Module):
    """
    Warning: small gamma could produce NaN during back prop.
    """
    def __init__(self, args):
        super(FocalLoss, self).__init__()
        self.alpha = args['alpha']
        self.gamma = args['gamma']
        self.reduction=args['reduction']
        self.ignore_index = args['ignore_index']
        self.ignore_indices = args['ignore_indices']
        if len(args['ignore_indices'])>0:
            self.ignore_index = args['ignore_indices'][0]
        
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        """
        :param preds: The prediction. shape=(data,class_prob)
        :param targets: The targets. shape=(data,class)
        
        Example:
        ```
            torch.manual_seed(0)
            pred=torch.rand(4,7,2,2)
            target=torch.randint(0,7,(4,1,2,2))
            
            pred=pred.view(-1,pred.shape[1])
            target=target.view(-1,1)
            loss = criterion(pred, target)
        ```
        """
        
        mask_indices=(targets==self.ignore_index).squeeze(1)
        preds=preds[~mask_indices]
        targets=targets[~mask_indices]
        
        b,c=preds.shape
        class_mask = torch.Tensor(preds.data.new(b, c).fill_(0))
        targets_onehot = class_mask.scatter_(1, targets, 1.)
        positive_label_mask = torch.eq(targets_onehot, torch.Tensor([1]).to(preds.device))
        sigmoid_cross_entropy = self.loss(preds, targets_onehot)
        
        probs = torch.sigmoid(preds)
        probs_gt = torch.where(positive_label_mask, probs, 1.0 - probs)

        modulator = torch.pow(1.0 - probs_gt, self.gamma)
        loss = modulator * sigmoid_cross_entropy
        weighted_loss = torch.where(positive_label_mask, self.alpha * loss,
                                 (1.0 - self.alpha) * loss)
        if self.reduction=='sum':
            return weighted_loss.sum()
        elif self.reduction=='mean':
            return weighted_loss.mean()
        else:
            return weighted_loss