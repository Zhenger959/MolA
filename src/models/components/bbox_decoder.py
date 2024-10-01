'''
Author: Jiaxin Zheng
Date: 2023-09-21 18:24:06
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 15:12:34
Description: 
'''
import torch
import torch.nn as nn

from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction

class BboxPredictorBase(nn.Module):
    def __init__(self,encoder_dim,decoder_dim):
        super(BboxPredictorBase, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim*4), nn.GELU(),
            nn.Linear(decoder_dim*4, 4),
            # nn.ReLU()
            nn.Sigmoid()
        )
        
    def forward(self, img_embed,atom_embed,  indices=None):
        b, l, dim = atom_embed.size()
        if indices is None:
            index = [i for i in range(3, l, 3)]
            atom_embed = atom_embed[:, index]
        else:
            batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
            indices = indices.view(-1)
            atom_embed = atom_embed[batch_id, indices].view(b, -1, dim)
        
        x_embed=atom_embed
            
        logits=self.head(x_embed)
        
        results={}
        results['atom_bbox']=logits
        return results,x_embed
    
class BboxPredictor(nn.Module):
    def __init__(self,encoder_dim,decoder_dim):
        super(BboxPredictor, self).__init__()
        self.img_mlp = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim)
        )
        
        self.context_attn = MultiHeadedAttention(
            head_count=8, model_dim=decoder_dim, dropout=0.1
        )
        self.drop = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(decoder_dim, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(decoder_dim, decoder_dim*2, 0.1,
                                                    ActivationFunction.relu
                                                    )
        self.bbox_head = nn.Linear(decoder_dim,4)
        
    def forward(self, img_embed,hidden, indices=None):
        b,c,d=img_embed.size()
        img_embed=self.img_mlp(img_embed)
        
        b, l, dim = hidden.size()
        if indices is None:
            index = [i for i in range(3, l, 3)]
            hidden = hidden[:, index]
        else:
            batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
            indices = indices.view(-1)
            hidden = hidden[batch_id, indices].view(b, -1, dim)
        b, l, dim = hidden.size()
        results = {}
        
        # TODO:  padding 的indice不参与backward
        query_norm = self.layer_norm(hidden)
        src_pad_mask=torch.zeros(b,1,c).bool().to(img_embed.device)
        mid, attns = self.context_attn(
            img_embed,
            img_embed,
            query_norm,
            mask=src_pad_mask,
            attn_type="context",
        )
        outputs = self.feed_forward(self.drop(mid).view(hidden.shape) + hidden)
        
        logits=self.bbox_head(outputs)
        results['atom_bbox']=logits
        return results,outputs