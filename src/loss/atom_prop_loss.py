'''
Author: Jiaxin Zheng
Date: 2023-08-31 11:54:38
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 15:49:59
Description: 
'''
import torch
import torch.nn as nn

class AtomPropCriterion(nn.Module):

    def __init__(self, cls_num=16,ignore_index=-1,negative_type=0):
        super(AtomPropCriterion, self).__init__()
        weight = torch.ones(cls_num) * 10
        weight[negative_type] = 1
        self.ignore_index=ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, results, refs):

        format_='atom_prop'
        # predictions, targets, *_ = results[format_]
        # targets=targets[format_]
        predictions=results[format_]
        targets=refs[format_]
        for atom_prop_i in predictions.keys(): 
            
            batch_size, max_len, vocab_size = predictions[atom_prop_i].size()
            output = predictions[atom_prop_i].reshape(-1, vocab_size)
            target = targets[atom_prop_i].reshape(-1)
                    
            loss_ = self.criterion(output, target)

            return loss_