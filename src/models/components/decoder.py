'''
Author: Jiaxin Zheng
Date: 2023-09-01 15:03:01
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 15:52:39
Description: 

'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import to_device

def get_edge_prediction(edge_prob,cls_num=7,return_score=True):
    
    if cls_num==7:
        sym_dict = {5: 6, 6: 5}
    elif cls_num==29:
        sym_dict = {5: 6, 6: 5, 8: 7, 7: 8, 10: 9, 9: 10, 12: 11, 11: 12}
    
    if edge_prob is None:
        return [], []
    n = len(edge_prob)
    if n == 0:
        return [], []
    
    sym_dict_keys_list=list(sym_dict.keys())
    sym_dict_values_list=list(sym_dict.values())
    k_list=list(set(range(cls_num)).difference(set(sym_dict_keys_list)))
    
    if isinstance(edge_prob,list):
        edge_prob=np.array(edge_prob)
    edge_prob_T=edge_prob.transpose(1,0,2)
    edge_prob_avg=(edge_prob+edge_prob_T)/2
    edge_prob_sym_1=(edge_prob[:,:,sym_dict_keys_list]+edge_prob_T[:,:,sym_dict_values_list])/2
    edge_prob_sym_2=(edge_prob[:,:,sym_dict_values_list]+edge_prob_T[:,:,sym_dict_keys_list])/2
    
    edge_prob[:,:,k_list]=edge_prob_avg[:,:,k_list]
    edge_prob[:,:,sym_dict_keys_list]=edge_prob_sym_1
    edge_prob[:,:,sym_dict_values_list]=edge_prob_sym_2
    
    if return_score:
        prediction = np.argmax(edge_prob, axis=2).tolist()
        score = np.max(edge_prob, axis=2).tolist()
        return prediction, score
    else:
        return edge_prob 

def get_edge_info_prediction(edge_orders_prob,edge_displays_prob,display_cls_num=15,return_score=True):
     
    bond_display_sym_dict = {3: 4, 4: 3, 6: 7, 7: 6, 9: 10, 10: 9, 11: 12, 12: 11}
    if edge_orders_prob is None or edge_displays_prob is None:
        if return_score:
            return [], [],[],[]
        else:
            return [], []
    n = len(edge_orders_prob)
    if n == 0:
        if return_score:
            return [], [],[],[]
        else:
            return [], []
     
    sym_dict_keys_list=list(bond_display_sym_dict.keys())
    sym_dict_values_list=list(bond_display_sym_dict.values())
    k_list=list(set(range(display_cls_num)).difference(set(sym_dict_keys_list)))
    
    if isinstance(edge_orders_prob,list):
        edge_orders_prob=np.array(edge_orders_prob)
    if isinstance(edge_displays_prob,list):
        edge_displays_prob=np.array(edge_displays_prob)
    
    if len(edge_displays_prob.shape)==4 and return_score==False:
        
        edge_orders_prob_T=edge_orders_prob.permute(0,1,3,2)
        edge_orders_prob_avg=(edge_orders_prob+edge_orders_prob_T)/2
        
        edge_displays_prob_T=edge_displays_prob.permute(0,1,3,2)
        edge_displays_prob_avg=(edge_displays_prob+edge_displays_prob_T)/2
        
        edge_displays_prob_sym_1=(edge_displays_prob[:,sym_dict_keys_list,:,:]+edge_displays_prob_T[:,sym_dict_values_list,:,:])/2
        edge_displays_prob_sym_2=(edge_displays_prob[:,sym_dict_values_list,:,:]+edge_displays_prob_T[:,sym_dict_keys_list,:,:])/2
        
        edge_displays_prob[:,k_list,:,:]=edge_displays_prob_avg[:,k_list,:,:]
        edge_displays_prob[:,sym_dict_keys_list,:,:]=edge_displays_prob_sym_1
        edge_displays_prob[:,sym_dict_values_list,:,:]=edge_displays_prob_sym_2
        
    else:
        edge_orders_prob_T=edge_orders_prob.transpose(1,0,2)
        edge_orders_prob_avg=(edge_orders_prob+edge_orders_prob_T)/2
        
        edge_displays_prob_T=edge_displays_prob.transpose(1,0,2)
        edge_displays_prob_avg=(edge_displays_prob+edge_displays_prob_T)/2
        
        
        edge_displays_prob_sym_1=(edge_displays_prob[:,:,sym_dict_keys_list]+edge_displays_prob_T[:,:,sym_dict_values_list])/2
        edge_displays_prob_sym_2=(edge_displays_prob[:,:,sym_dict_values_list]+edge_displays_prob_T[:,:,sym_dict_keys_list])/2
        
        edge_displays_prob[:,:,k_list]=edge_displays_prob_avg[:,:,k_list]
        edge_displays_prob[:,:,sym_dict_keys_list]=edge_displays_prob_sym_1
        edge_displays_prob[:,:,sym_dict_values_list]=edge_displays_prob_sym_2
    
    if return_score:
        edge_orders_prediction = np.argmax(edge_orders_prob_avg, axis=2).tolist()
        edge_orders_score = np.max(edge_orders_prob_avg, axis=2).tolist()
        
        edge_displays_prediction = np.argmax(edge_displays_prob, axis=2).tolist()
        edge_displays_score = np.max(edge_displays_prob, axis=2).tolist()
        return edge_orders_prediction, edge_orders_score,edge_displays_prediction,edge_displays_score
    else:
        return edge_orders_prob_avg,edge_displays_prob
    
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        decoder = {}
        if type(args.atom_predictor)!=str:
            decoder['chartok_coords']=args.atom_predictor
        if type(args.atom_prop_predictor)!=str:
            decoder['atom_prop']=args.atom_prop_predictor
        if type(args.edge_predictor)!=str:
            decoder['edges']=args.edge_predictor
        if type(args.edge_info_predictor)!=str:
            decoder['edge_infos']=args.edge_info_predictor
        if type(args.atom_bbox_predictor)!=str:
            decoder['atom_bbox']=args.atom_bbox_predictor
        if type(args.graph_net)!=str:
            decoder['graph_net']=args.graph_net
        self.decoder = nn.ModuleDict(decoder)
        self.compute_confidence = args.compute_confidence

    def forward(self, encoder_out, hiddens, refs):  # TODO: 这里refs={}
        # train
        results = {}
        refs = to_device(refs, encoder_out.device)
        bias = None
        
        if 'chartok_coords' in self.args.formats:
            labels, label_lengths = refs['chartok_coords']
            results['chartok_coords'] = self.decoder['chartok_coords'](encoder_out, labels, label_lengths)
            dec_out = results['chartok_coords'][2]
            
        if 'atom_ocr' in self.args.formats:
            # å
            predictions = self.decoder['atom_ocr'](encoder_out,dec_out, indices=refs['superatom_indices'][0],bias=bias)    
        
        if 'atom_bbox' in self.args.formats:
            predictions,abbox_out = self.decoder['atom_bbox'](encoder_out,dec_out, indices=refs['superatom_indices'][0])
            # print('superatom_indices',refs['superatom_indices'][0].shape)
            targets = {'atom_bbox': refs['superatom_bbox']}
            # print('superatom_bbox',refs['superatom_bbox'].shape)
            results['atom_bbox'] = (predictions, targets)
            
        if 'atom_prop' in self.args.formats:
            predictions,x_embed=self.decoder['atom_prop'](dec_out,indices=refs['atom_indices'][0])
            prop_name = self.decoder['atom_prop'].prop_name
            targets = {'atom_prop': {prop_name:refs[prop_name]}}#,'degree':refs['degree'],'num_Hs_total':refs['num_Hs_total'],'num_Hs_implicit':refs['num_Hs_implicit']}}
            results['atom_prop'] = (predictions, targets)
        
        if 'graph_net' in self.args.formats:
            # bias = self.decoder['graph_net'](encoder_out,x_embed, indices=refs['atom_indices'][0])
            bias = self.decoder['graph_net'](encoder_out,dec_out, indices=refs['atom_indices'][0])
            
        if 'edges' in self.args.formats:
            predictions = self.decoder['edges'](encoder_out,dec_out, indices=refs['atom_indices'][0],bias=bias)    
            
            prob=predictions['edges']
            prob = get_edge_prediction(prob,self.args.edge_cls_num,return_score=False)
            predictions['edges'] = prob
            
            targets = {'edges': refs['edges']}
            if 'coords' in predictions:
                targets['coords'] = refs['coords']
            results['edges'] = (predictions, targets)
        
        if 'edge_infos' in self.args.formats:
            predictions = self.decoder['edge_infos'](encoder_out,dec_out, indices=refs['atom_indices'][0],bias=bias)    

            predictions['edge_orders'],predictions['edge_displays'] = get_edge_info_prediction(predictions['edge_orders'],predictions['edge_displays'],return_score=False)            
            targets = {'edges': refs['edges'],'edge_orders':refs['edge_orders'],'edge_displays':refs['edge_displays']}
            
            if 'coords' in predictions:
                targets['coords'] = refs['coords']
            results['edge_infos'] = (predictions, targets)
               
        return results

    def decode(self, encoder_out, hiddens=None, refs=None, beam_size=1, n_best=1):
        """inferenceå
        """
        # inference
        results = {}
        predictions = []
        bias =None  # TODO

        if 'chartok_coords' in self.args.formats:
            type='chartok_coords'
            results[type] = self.decoder[type].decode(encoder_out, beam_size, n_best, max_length=self.args.max_len)
            outputs, scores, token_scores, *_ = results[type]
            
            # process atom logits by tokenizer
            beam_preds = [[self.decoder['chartok_coords'].tokenizer.sequence_to_smiles(x.tolist()) for x in pred]
                            for pred in outputs]
            predictions = [{type: pred[0]} for pred in beam_preds]
            swap_token=self.decoder['chartok_coords'].tokenizer.swap_token
            if self.compute_confidence:  # TODO: calculate during tokenizer.sequence_to_smiles
                for i in range(len(predictions)):
                    if not swap_token:
                        indices = np.array(predictions[i][type]['indices']) - 3  # -1: y score, -2: x score, -3: symbol score
                        x_indices = np.array(predictions[i][type]['indices']) - 2
                        y_indices = np.array(predictions[i][type]['indices']) - 1
                    else:
                        indices = np.array(predictions[i][type]['indices'])
                        
                        x_indices = []
                        y_indices = []
                        for symbol, index in zip(predictions[i][type]['symbols'], indices):               
                            x_indices.append(index-len(symbol)-1)
                            y_indices.append(index-len(symbol))
                        x_indices = np.array(x_indices)
                        y_indices = np.array(y_indices)
                    
                    x_coord_scores = np.array(token_scores[i][0])[x_indices] if len(x_indices)>0 and len(token_scores[i])>0 else np.array([])
                    y_coord_scores = np.array(token_scores[i][0])[y_indices] if len(y_indices)>0 and len(token_scores[i])>0 else np.array([])
                    coord_scores = np.sqrt(x_coord_scores*y_coord_scores).tolist()
                    
                    x_coord_scores = x_coord_scores.tolist()
                    y_coord_scores = y_coord_scores.tolist()
                    # not consider the x and y token
                    if type == 'chartok_coords':
                        atom_scores = []
                        for symbol, index in zip(predictions[i][type]['symbols'], indices):
                            atom_score = (np.prod(token_scores[i][0][index - len(symbol) + 1:index + 1])
                                            ** (1 / len(symbol))).item()
                            atom_scores.append(atom_score)
                    else:
                        atom_scores = np.array(token_scores[i][0])[indices].tolist()
                    predictions[i][type]['atom_scores'] = atom_scores
                    predictions[i][type]['x_coord_scores'] = x_coord_scores
                    predictions[i][type]['y_coord_scores'] = y_coord_scores
                    predictions[i][type]['coord_scores'] = coord_scores
                    predictions[i][type]['average_token_score'] = scores[i][0]

        if 'atom_bbox' in self.args.formats:
            atom_format = 'chartok_coords'
            dec_out = results[atom_format][3]  # batch x n_best x len x dim
            
            for i in range(len(dec_out)):
                hidden = dec_out[i][0].unsqueeze(0)  # 1 * len * dim
                indices = torch.LongTensor(predictions[i][atom_format]['superatom_indices']).unsqueeze(0)  # 1 * k
                pred,abbox_out = self.decoder['atom_bbox'](encoder_out[i].unsqueeze(0),hidden, indices) 
                type='atom_bbox'
                predictions[i][type]={}
                predictions[i][type]['atom_bbox'] = pred
                # predictions[i][type]['abbox_out'] = abbox_out[0].unsqueeze(0)
                
        if 'atom_prop' in self.args.formats:            
            atom_format = 'chartok_coords'
            type='atom_prop'
            dec_out = results[atom_format][3]  # batch x n_best x len x dim
            
            # TODO: batch predict
            for i in range(len(dec_out)):
                hidden = dec_out[i][0].unsqueeze(0)  # 1 * len * dim
                # abbox_out=predictions[i]['atom_bbox']['abbox_out']
                indices = torch.LongTensor(predictions[i][atom_format]['indices']).unsqueeze(0)  # 1 * k
                res,x_embed = self.decoder['atom_prop'](hidden,indices=indices)  # k * k
                prop_name = self.decoder['atom_prop'].prop_name
                # process atom properties result
                predictions[i][type]={}
                try:
                    predictions[i][type][prop_name] = np.argmax(res[type][prop_name].detach().cpu().numpy(),axis=-1)
                except:
                    predictions[i][type][prop_name] = np.argmax(res[type][prop_name].detach().to(torch.float).cpu().numpy(),axis=-1)
                predictions[i][type]['x_embed'] = x_embed
                # predictions[i][type]['degree'] = results[type]['degree']
                # predictions[i][type]['num_Hs_total'] = results[type]['num_Hs_total']
                # predictions[i][type]['num_Hs_implicit'] = results[type]['num_Hs_implicit'] 
        
        if 'graph_net' in self.args.formats:
            atom_format = 'chartok_coords'
            dec_out = results[atom_format][3]  # batch x n_best x len x dim
            
            for i in range(len(dec_out)):
                # hidden = predictions[i]['atom_prop']['x_embed']   # 1 * len * dim
                hidden = dec_out[i][0].unsqueeze(0)
                indices = torch.LongTensor(predictions[i][atom_format]['indices']).unsqueeze(0)  # 1 * k
                bias = self.decoder['graph_net'](encoder_out[i].unsqueeze(0),hidden, indices)
                type='graph_net'
                predictions[i][type]={}
                predictions[i][type]['bias']=bias
            
        if 'edges' in self.args.formats:
            atom_format = 'chartok_coords'
            dec_out = results[atom_format][3]  # batch x n_best x len x dim
            
            for i in range(len(dec_out)):
                hidden = dec_out[i][0].unsqueeze(0)  # 1 * len * dim
                indices = torch.LongTensor(predictions[i][atom_format]['indices']).unsqueeze(0)  # 1 * k
                if 'graph_net' in self.args.formats:
                    bias=predictions[i]['graph_net']['bias']=bias
                else:
                    bias=None
                pred = self.decoder['edges'](encoder_out[i].unsqueeze(0),hidden, indices,bias) 
                    
                # process edge logits
                prob = F.softmax(pred['edges'].squeeze(0).permute(1, 2, 0), dim=2).tolist()  # k * k * 7
                edge_pred, edge_score = get_edge_prediction(prob,self.args.edge_cls_num)
                
                type='edges'
                predictions[i][type]={}
                predictions[i][type]['edges'] = edge_pred
                
                if self.compute_confidence:
                    predictions[i][type]['edge_scores'] = edge_score
                    predictions[i][type]['edge_score_product'] = np.sqrt(np.prod(edge_score)).item()
                    predictions[i][type]['overall_score'] = predictions[i][atom_format]['average_token_score'] * \
                                                        predictions[i][type]['edge_score_product']
                    # predictions[i][atom_format].pop('average_token_score')
                    predictions[i][type].pop('edge_score_product')
       
        if 'edge_infos' in self.args.formats:
            atom_format = 'chartok_coords'
            dec_out = results[atom_format][3]  # batch x n_best x len x dim
            # edge_dict = {(1, 0): 1, (2, 0): 2, (4, 0): 3, (128, 0): 4, (1, 3): 5, (1, 4): 6, (1, 6): 7, (1, 7): 8, (1, 9): 9, (1, 10): 10, (1, 11): 11, (1, 12): 12, (1, 1): 13, (1, 2): 14, (1, 5): 15, (1, 8): 16, (1, 13): 17, (1, 14): 18, (2, 1): 19, (2, 5): 20, (2, 8): 21, (8, 0): 22, (16, 0): 23, (32, 0): 24, (256, 0): 25, (512, 0): 26, (1024, 0): 27, (2048, 0): 28}
            edge_dict = {(1, 0): 1, (2, 0): 2, (4, 0): 3, (128, 0): 4, (1, 3): 5, (1, 4): 6, (1, 6): 7, (1, 7): 8, (1, 9): 9, (1, 10): 10, (1, 11): 11, (1, 12): 12, (1, 1): 13, (1, 2): 14, (1, 5): 15, (1, 8): 16, (1, 13): 17, (1, 14): 18, (2, 1): 19, (2, 5): 20, (2, 8): 21, (8, 0): 22, (16, 0): 23, (32, 0): 24, (256, 0): 25, (512, 0): 26, (1024, 0): 27, (2048, 0): 28, (4096, 0): 29, (8192, 0): 30, (16384, 0): 31, (32768, 0): 32, (128, 5): 33, (128, 6): 34, (128, 7): 35, (64, 0):36}
            edge_order_reverse_dict = {
                1: 1,
                2: 2,
                3: 4,
                4: 8,
                5: 16,
                6: 32,
                7: 64,
                8: 128,
                9: 256,
                10: 512,
                11: 1024,
                12: 2048,
                13: 4096,
                14: 8192,
                15: 16384,
                16: 32768,
                0: 0}
            
            for i in range(len(dec_out)):
                hidden = dec_out[i][0].unsqueeze(0)  # 1 * len * dim
                indices = torch.LongTensor(predictions[i][atom_format]['indices']).unsqueeze(0)  # 1 * k

                bias=None
                pred = self.decoder['edge_infos'](encoder_out[i].unsqueeze(0),hidden, indices,bias) 
                    
                # process edge logits
                edge_orders_pred = F.softmax(pred['edge_orders'].squeeze(0).permute(1, 2, 0), dim=2).tolist()
                edge_displays_pred = F.softmax(pred['edge_displays'].squeeze(0).permute(1, 2, 0), dim=2).tolist()
                
                edge_orders_pred, edge_orders_score, edge_displays_pred, edge_displays_score = get_edge_info_prediction(edge_orders_pred,edge_displays_pred)
                
                
                type='edge_infos'
                predictions[i][type]={}
                predictions[i][type]['edge_orders'] = edge_orders_pred 
                predictions[i][type]['edge_displays'] = edge_displays_pred 
                
                # TODO
                edges=np.zeros_like(edge_orders_pred)
                for a1_i in range(edges.shape[0]):
                    for a2_j in range(edges.shape[1]):
                        edge_order=edge_order_reverse_dict[edge_orders_pred[a1_i][a2_j]]
                        edge_displays = edge_displays_pred[a1_i][a2_j]
                        
                        cur_k=tuple((edge_order,edge_displays))
                        
                        if cur_k in edge_dict:
                            edges[a1_i][a2_j]=edge_dict[cur_k]
                        else:
                            if edge_order>1:
                                edges[a1_i][a2_j]=edge_dict[tuple((edge_order,0))]
                                
                type='edges'
                predictions[i][type]={}
                predictions[i][type]['edges'] = edges.tolist()
                
                if self.compute_confidence:
                    edge_score = (np.array(edge_orders_score) + np.array(edge_displays_score))/2
                    predictions[i][type]['edge_orders_scores'] = edge_orders_score
                    predictions[i][type]['edge_displays_scores'] = edge_displays_score
                    predictions[i][type]['edge_scores'] = edge_score.tolist()
                    predictions[i][type]['edge_score_product'] = np.sqrt(np.prod(edge_score)).item()
                    predictions[i][type]['overall_score'] = predictions[i][atom_format]['average_token_score'] * \
                                                        predictions[i][type]['edge_score_product']
                    # predictions[i][atom_format].pop('average_token_score')
                    predictions[i][type].pop('edge_score_product')
                                  
        return predictions