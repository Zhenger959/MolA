'''
Author: Jiaxin Zheng
Date: 2023-09-01 15:03:01
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 15:50:56
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.modules import MultiHeadedAttention

from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction

from src.models.components  import TransformerDecoderMainBase,TransformerMainBase,Embeddings
from src.inference import GreedySearch,BeamSearch
from src.tokenizer.token import PAD_ID,EOS_ID,SOS_ID,MASK_ID

from src import utils

log = utils.get_pylogger(__name__)

class AtomPredictor(TransformerDecoderMainBase):

    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.args=args  # self_attn_type=scaled-dot  ckpt_path=''
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer)
        
        self.embeddings = Embeddings(
            word_vec_size=args.dec_hidden_size,
            word_vocab_size=self.vocab_size,
            word_padding_idx=PAD_ID,
            position_encoding=True,
            dropout=args.hidden_dropout)
        self.output_layer = nn.Linear(args.dec_hidden_size, self.vocab_size, bias=True)

    def dec_embedding(self, tgt, step=None):
        pad_idx = self.embeddings.word_padding_idx
        tgt_pad_mask = tgt.data.eq(pad_idx).transpose(1, 2)
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3
        return emb, tgt_pad_mask

    def forward(self, encoder_out, labels, label_lengths):
        # train
        memory_bank = self.enc_transform(encoder_out)

        tgt = labels.unsqueeze(-1)  # (b, t, 1)
        tgt_emb, tgt_pad_mask = self.dec_embedding(tgt)
        dec_out, *_ = self.decoder(tgt_emb=tgt_emb, memory_bank=memory_bank, tgt_pad_mask=tgt_pad_mask)

        logits = self.output_layer(dec_out)  # (b, t, h) -> (b, t, v)
        return logits[:, :-1], labels[:, 1:], dec_out

    def decode(self, encoder_out, beam_size: int, n_best: int, min_length: int = 1, max_length: int = 256,
               labels=None,use_before=False):
        # inference
        batch_size, max_len, _ = encoder_out.size()
        memory_bank = self.enc_transform(encoder_out)
        orig_labels = labels

        if beam_size == 1:
            decode_strategy = GreedySearch(
                sampling_temp=0.0, keep_topk=1, batch_size=batch_size, min_length=min_length, max_length=max_length,
                pad=PAD_ID, bos=SOS_ID, eos=EOS_ID,
                return_attention=False, return_hidden=True)
        else:
            decode_strategy = BeamSearch(
                beam_size=beam_size, n_best=n_best, batch_size=batch_size, min_length=min_length, max_length=max_length,
                pad=PAD_ID, bos=SOS_ID, eos=EOS_ID,
                return_attention=False)

        # adapted from onmt.translate.translator
        results = {
            "predictions": None,
            "scores": None,
            "attention": None
        }

        # (2) prep decode_strategy. Possibly repeat src objects.
        _, memory_bank = decode_strategy.initialize(memory_bank=memory_bank)

        for step in range(decode_strategy.max_length):
            tgt = decode_strategy.current_predictions.view(-1, 1, 1)
            if labels is not None:
                label = labels[:, step].view(-1, 1, 1)
                mask = label.eq(MASK_ID).long()
                tgt = tgt * mask + label * (1 - mask)
            if use_before:
                tgt_emb, tgt_pad_mask =self.dec_embedding(decode_strategy.alive_seq.unsqueeze(-1))
            else:
                tgt_emb, tgt_pad_mask = self.dec_embedding(tgt)
            dec_out, dec_attn, *_ = self.decoder(tgt_emb=tgt_emb, memory_bank=memory_bank,
                                                 tgt_pad_mask=tgt_pad_mask, step=step)

            attn = dec_attn.get("std", None)

            dec_logits = self.output_layer(dec_out)  # [b, t, h] => [b, t, v]
            dec_logits = dec_logits.squeeze(1)
            log_probs = F.log_softmax(dec_logits, dim=-1)

            if self.args.output_constraint:
                output_mask = [self.tokenizer.get_output_mask(id) for id in tgt.view(-1).tolist()]
                output_mask = torch.tensor(output_mask, device=log_probs.device)
                log_probs.masked_fill_(output_mask, -10000)
            
            if use_before:
                log_probs=log_probs[:,-1,:] if log_probs.shape[1]>1 and len(log_probs.shape)==3 else log_probs
            label = labels[:, step + 1] if labels is not None and step + 1 < labels.size(1) else None
            decode_strategy.advance(log_probs, attn, dec_out, label)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices
            if any_finished:
                # Reorder states.
                memory_bank = memory_bank.index_select(0, select_indices)
                if labels is not None:
                    labels = labels.index_select(0, select_indices)
                self.map_state(lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = decode_strategy.scores  # fixed to be average of token scores
        results["token_scores"] = decode_strategy.token_scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        results["hidden"] = decode_strategy.hidden
        if orig_labels is not None:
            for i in range(batch_size):
                pred = results["predictions"][i][0]
                label = orig_labels[i][1:len(pred) + 1]
                mask = label.eq(MASK_ID).long()
                pred = pred[:len(label)]
                results["predictions"][i][0] = pred * mask + label * (1 - mask)

        return results["predictions"], results['scores'], results["token_scores"], results["hidden"]

    # adapted from onmt.decoders.transformer
    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        if self.decoder.state["cache"] is not None:
            _recursive_map(self.decoder.state["cache"])

class AtomTransformerPredictor(TransformerMainBase):

    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.args=args  # self_attn_type=scaled-dot  ckpt_path=''
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer)
        
        self.embeddings = Embeddings(
            word_vec_size=args.dec_hidden_size,
            word_vocab_size=self.vocab_size,
            word_padding_idx=PAD_ID,
            position_encoding=True,
            dropout=args.hidden_dropout)
        self.output_layer = nn.Linear(args.dec_hidden_size, self.vocab_size, bias=True)

    def dec_embedding(self, tgt, step=None):
        pad_idx = self.embeddings.word_padding_idx
        tgt_pad_mask = tgt.data.eq(pad_idx).transpose(1, 2)
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3
        return emb, tgt_pad_mask

    def forward(self, encoder_out, labels, label_lengths):
        # train
        batch_size, max_len, _ = encoder_out.size()
        enc_out = self.enc_transform(encoder_out)
        max_len = torch.Tensor(batch_size*[max_len]).to(encoder_out.device)
        # log.info(encoder_out.size())
        memory_bank, _, src_len = self.encoder(enc_out,max_len)
        
        tgt = labels.unsqueeze(-1)  # (b, t, 1)
        tgt_emb, tgt_pad_mask = self.dec_embedding(tgt)
        dec_out, *_ = self.decoder(tgt_emb=tgt_emb, memory_bank=memory_bank, tgt_pad_mask=tgt_pad_mask)

        logits = self.output_layer(dec_out)  # (b, t, h) -> (b, t, v)
        return logits[:, :-1], labels[:, 1:], dec_out

    def decode(self, encoder_out, beam_size: int, n_best: int, min_length: int = 1, max_length: int = 256,
               labels=None,use_before=False):
        # inference
        batch_size, max_len, _ = encoder_out.size()
        enc_out = self.enc_transform(encoder_out)
        max_len = torch.Tensor(batch_size*[max_len]).to(encoder_out.device)
        # log.info(max_len)
        memory_bank, _, src_len = self.encoder(enc_out,max_len)
        orig_labels = labels

        if beam_size == 1:
            decode_strategy = GreedySearch(
                sampling_temp=0.0, keep_topk=1, batch_size=batch_size, min_length=min_length, max_length=max_length,
                pad=PAD_ID, bos=SOS_ID, eos=EOS_ID,
                return_attention=False, return_hidden=True)
        else:
            decode_strategy = BeamSearch(
                beam_size=beam_size, n_best=n_best, batch_size=batch_size, min_length=min_length, max_length=max_length,
                pad=PAD_ID, bos=SOS_ID, eos=EOS_ID,
                return_attention=False)

        # adapted from onmt.translate.translator
        results = {
            "predictions": None,
            "scores": None,
            "attention": None
        }

        # (2) prep decode_strategy. Possibly repeat src objects.
        _, memory_bank = decode_strategy.initialize(memory_bank=memory_bank)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            tgt = decode_strategy.current_predictions.view(-1, 1, 1)
            if labels is not None:
                label = labels[:, step].view(-1, 1, 1)
                mask = label.eq(MASK_ID).long()
                tgt = tgt * mask + label * (1 - mask)
            if use_before:
                tgt_emb, tgt_pad_mask =self.dec_embedding(decode_strategy.alive_seq.unsqueeze(-1))
            else:
                tgt_emb, tgt_pad_mask = self.dec_embedding(tgt)
            dec_out, dec_attn, *_ = self.decoder(tgt_emb=tgt_emb, memory_bank=memory_bank,
                                                 tgt_pad_mask=tgt_pad_mask, step=step)

            attn = dec_attn.get("std", None)

            dec_logits = self.output_layer(dec_out)  # [b, t, h] => [b, t, v]
            dec_logits = dec_logits.squeeze(1)
            log_probs = F.log_softmax(dec_logits, dim=-1)

            if self.args.output_constraint:
                output_mask = [self.tokenizer.get_output_mask(id) for id in tgt.view(-1).tolist()]
                output_mask = torch.tensor(output_mask, device=log_probs.device)
                log_probs.masked_fill_(output_mask, -10000)
            
            if use_before:
                log_probs=log_probs[:,-1,:] if log_probs.shape[1]>1 and len(log_probs.shape)==3 else log_probs
            label = labels[:, step + 1] if labels is not None and step + 1 < labels.size(1) else None
            decode_strategy.advance(log_probs, attn, dec_out, label)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices
            if any_finished:
                # Reorder states.
                memory_bank = memory_bank.index_select(0, select_indices)
                if labels is not None:
                    labels = labels.index_select(0, select_indices)
                self.map_state(lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = decode_strategy.scores  # fixed to be average of token scores
        results["token_scores"] = decode_strategy.token_scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        results["hidden"] = decode_strategy.hidden
        if orig_labels is not None:
            for i in range(batch_size):
                pred = results["predictions"][i][0]
                label = orig_labels[i][1:len(pred) + 1]
                mask = label.eq(MASK_ID).long()
                pred = pred[:len(label)]
                results["predictions"][i][0] = pred * mask + label * (1 - mask)

        return results["predictions"], results['scores'], results["token_scores"], results["hidden"]

    # adapted from onmt.decoders.transformer
    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        if self.decoder.state["cache"] is not None:
            _recursive_map(self.decoder.state["cache"])         

class AtomPropPredictor(nn.Module):
    def __init__(self,encoder_dim,decoder_dim,cls_num=8,prop_name='atoms_valence_before'):
        super(AtomPropPredictor, self).__init__()
        self.prop_name=prop_name
        self.head = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim*4), nn.GELU(),
            nn.Linear(decoder_dim*4, cls_num)
        )
        
    def forward(self, atom_embed,  indices=None):
        b, l, dim = atom_embed.size()
        if indices is None:
            index = [i for i in range(3, l, 3)]
            atom_embed = atom_embed[:, index]
        else:
            batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
            indices = indices.view(-1)
            atom_embed = atom_embed[batch_id, indices].view(b, -1, dim)
            
        results={'atom_prop':{}}
        
        x_embed=atom_embed
            
        logits=self.head(x_embed)
        results['atom_prop'][self.prop_name]=logits
        return results,x_embed

# TODO
class AtomOCR(nn.Module):
    def __init__(self,encoder_dim,decoder_dim,cls_num=8,prop_name='atoms_valence_before'):
        super(AtomOCR, self).__init__()
        self.prop_name=prop_name
        self.head = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim*4), nn.GELU(),
            nn.Linear(decoder_dim*4, cls_num)
        )
        
    def forward(self, atom_embed,  indices=None):
        b, l, dim = atom_embed.size()
        if indices is None:
            index = [i for i in range(3, l, 3)]
            atom_embed = atom_embed[:, index]
        else:
            batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
            indices = indices.view(-1)
            atom_embed = atom_embed[batch_id, indices].view(b, -1, dim)
            
        results={'atom_prop':{}}
        
        x_embed=atom_embed
            
        logits=self.head(x_embed)
        results['atom_prop'][self.prop_name]=logits
        return results,x_embed
    
class GraphPropNet(nn.Module):
    def __init__(self,encoder_dim,decoder_dim):
        super(GraphPropNet, self).__init__()
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
        self.bias_head = nn.Linear(decoder_dim, 1)
    
    def forward(self, img_embed,x_embed, indices=None):
        b,c,d=img_embed.size()
        img_embed=self.img_mlp(img_embed)
        
        query_norm = self.layer_norm(x_embed)
        src_pad_mask=torch.zeros(b,1,c).bool().to(img_embed.device)
        mid, attns = self.context_attn(
            img_embed,
            img_embed,
            query_norm,
            mask=src_pad_mask,
            attn_type="context",
        )
        outputs = self.feed_forward(self.drop(mid).view(x_embed.shape) + x_embed)
        
        bias=self.bias_head(outputs)
        return bias