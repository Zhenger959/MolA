'''
Author: Jiaxin Zheng
Date: 2023-08-31 11:51:27
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 15:50:45
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # assuming output is raw logits
        # convert to log_probs
        log_probs = F.log_softmax(output, dim=-1)

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        # reduction mean or sum?
        return F.kl_div(log_probs, model_prob, reduction='batchmean')


class SequenceLoss(nn.Module):

    def __init__(self, tokenizer,label_smoothing,  ignore_index=-100, ignore_indices=[]):
        super(SequenceLoss, self).__init__()
        vocab_size=len(tokenizer)
        if ignore_indices:
            ignore_index = ignore_indices[0]
        self.ignore_index = ignore_index
        self.ignore_indices = ignore_indices
        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        else:
            self.criterion = LabelSmoothingLoss(label_smoothing, vocab_size, ignore_index)

    def forward(self, output, target):
        """
        :param output: [batch, len, vocab]
        :param target: [batch, len]
        :return:
        """
        batch_size, max_len, vocab_size = output.size()
        output = output.reshape(-1, vocab_size)
        target = target.reshape(-1)
        for idx in self.ignore_indices:
            if idx != self.ignore_index:
                target.masked_fill_((target == idx), self.ignore_index)
        loss = self.criterion(output, target)
        return loss