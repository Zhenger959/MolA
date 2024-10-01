'''
Author: Jiaxin Zheng
Date: 2023-09-01 15:03:01
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 19:40:26
Description: 
'''
import torch 
import torch.nn as nn


from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction

class EdgePredictor(nn.Module):

    def __init__(self, encoder_dim,decoder_dim,cls_num=7):
        super(EdgePredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, cls_num)
        )


    def forward(self, img_embed,hidden, indices=None,bias=None):
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
        hh = torch.cat([hidden.unsqueeze(2).expand(b, l, l, dim), hidden.unsqueeze(1).expand(b, l, l, dim)], dim=3)
        results['edges'] = self.mlp(hh).permute(0, 3, 1, 2)
        return results
    
class EdgeImgAttnPredictor(nn.Module):

    def __init__(self, encoder_dim,decoder_dim,cls_num=7):
        super(EdgeImgAttnPredictor, self).__init__()

        self.img_mlp = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim*2)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, cls_num)
        )
        self.context_attn = MultiHeadedAttention(
            head_count=8, model_dim=decoder_dim*2, dropout=0.1
        )
        self.drop = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(decoder_dim*2, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(decoder_dim*2, decoder_dim*4, 0.1,
                                                    ActivationFunction.relu
                                                    )

    def forward(self, img_embed,hidden, indices=None,bias=None):
        """
        :img_embed: Image Embedding. shape=(1,144,encoder_dim=1024)
        :hidden: The prediction. [data,class_prob]
        
        Example:
        ```
        ```
        """
        b,c,d=img_embed.size()
        img_embed=self.img_mlp(img_embed)
        
        b, l, dim = hidden.size()
        if indices is None:
            index = [i for i in range(3, l, 3)]
            hidden = hidden[:, index]
            bias = bias[:, index] if bias is not None else None
        else:
            batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
            indices = indices.view(-1)
            hidden = hidden[batch_id, indices].view(b, -1, dim)
            bias = bias[batch_id, indices].view(b, -1,1) if bias is not None else None
        b, l, dim = hidden.size()
        results = {}
        bias=bias.unsqueeze(2).expand(b, l, l, 1)  if bias is not None else None
        hh = torch.cat([hidden.unsqueeze(2).expand(b, l, l, dim), hidden.unsqueeze(1).expand(b, l, l, dim)], dim=3)        
        query_norm = self.layer_norm(hh)
        
        src_pad_mask=torch.zeros(b,1,c).bool().to(img_embed.device)
        
        mid, attns = self.context_attn(
            img_embed,
            img_embed,
            query_norm,
            mask=src_pad_mask,
            attn_type="context",
        )
        outputs = self.feed_forward(self.drop(mid).view(hh.shape) + hh)
        if bias is not None:
            outputs = outputs +bias
        logits=self.mlp(outputs)
        results['edges'] = logits.permute(0, 3, 1, 2)
        return results

class EdgeInfoImgPredictor(nn.Module):

    def __init__(self, encoder_dim,decoder_dim,order_cls_num=17,display_cls_num=15):
        super(EdgeInfoImgPredictor, self).__init__()

        self.order_head = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, order_cls_num)
        )
        self.display_head = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, display_cls_num)
        )

    def forward(self, img_embed,hidden, indices=None,bias=None):
        """
        :img_embed: Image Embedding. shape=(1,144,encoder_dim=1024)
        :hidden: The prediction. [data,class_prob]
        
        Example:
        ```
        ```
        """
        
        b, l, dim = hidden.size()
        if indices is None:
            index = [i for i in range(3, l, 3)]
            hidden = hidden[:, index]
            bias = bias[:, index] if bias is not None else None
        else:
            batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
            indices = indices.view(-1)
            hidden = hidden[batch_id, indices].view(b, -1, dim)
            bias = bias[batch_id, indices].view(b, -1,1) if bias is not None else None
        b, l, dim = hidden.size()
        results = {}
        bias=bias.unsqueeze(2).expand(b, l, l, 1)  if bias is not None else None
        
        hh = torch.cat([hidden.unsqueeze(2).expand(b, l, l, dim), hidden.unsqueeze(1).expand(b, l, l, dim)], dim=3)        

        order_logits=self.order_head(hh)
        display_logits=self.display_head(hh)
        
        results['edge_orders'] = order_logits.permute(0, 3, 1, 2)
        results['edge_displays'] = display_logits.permute(0, 3, 1, 2)
        
        return results
    
class EdgeInfoImgAttnPredictor(nn.Module):

    def __init__(self, encoder_dim,decoder_dim,order_cls_num=17,display_cls_num=15):
        super(EdgeInfoImgAttnPredictor, self).__init__()

        self.img_mlp = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim*2)
        )
        self.context_attn = MultiHeadedAttention(
            head_count=8, model_dim=decoder_dim*2, dropout=0.1
        )
        self.drop = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(decoder_dim*2, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(decoder_dim*2, decoder_dim*4, 0.1,
                                                    ActivationFunction.relu
                                                    )
        self.order_head = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, order_cls_num)
        )
        self.display_head = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, display_cls_num)
        )

    def forward(self, img_embed,hidden, indices=None,bias=None):
        """
        :img_embed: Image Embedding. shape=(1,144,encoder_dim=1024)
        :hidden: The prediction. [data,class_prob]
        
        Example:
        ```
        ```
        """
        b,c,d=img_embed.size()
        img_embed=self.img_mlp(img_embed)
        
        b, l, dim = hidden.size()
        if indices is None:
            index = [i for i in range(3, l, 3)]
            hidden = hidden[:, index]
            bias = bias[:, index] if bias is not None else None
        else:
            batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
            indices = indices.view(-1)
            hidden = hidden[batch_id, indices].view(b, -1, dim)
            bias = bias[batch_id, indices].view(b, -1,1) if bias is not None else None
        b, l, dim = hidden.size()
        results = {}
        bias=bias.unsqueeze(2).expand(b, l, l, 1)  if bias is not None else None
        hh = torch.cat([hidden.unsqueeze(2).expand(b, l, l, dim), hidden.unsqueeze(1).expand(b, l, l, dim)], dim=3)        
        query_norm = self.layer_norm(hh)
        
        src_pad_mask=torch.zeros(b,1,c).bool().to(img_embed.device)
        
        mid, attns = self.context_attn(
            img_embed,
            img_embed,
            query_norm,
            mask=src_pad_mask,
            attn_type="context",
        )
        outputs = self.feed_forward(self.drop(mid).view(hh.shape) + hh)
        if bias is not None:
            outputs = outputs +bias
        order_logits=self.order_head(outputs)
        display_logits=self.display_head(outputs)
        
        results['edge_orders'] = order_logits.permute(0, 3, 1, 2)
        results['edge_displays'] = display_logits.permute(0, 3, 1, 2)
        
        return results