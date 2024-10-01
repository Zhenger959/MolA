'''
Author: Jiaxin Zheng
Date: 2023-09-05 10:29:08
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 15:56:39
Description: 
'''
import sys
import random
import time
from loguru import logger

import torch
import torch.nn.functional as F

import numpy as np
import cv2

from rdkit import Chem
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import normalize_nodes
from src.utils.post_process.build_mol_for_train import build_mol
from src.utils import Indigo
from src.utils.data_utils.augment_utils import merge_bbox_keypoints,merge_bbox_keypoints_expand,get_bbox_list_from_s_list

# def pad_images(imgs):
#     padded_imgs = torch.zeros(len(imgs), 3, 384, 384, dtype=torch.float32)
#     for i, img in enumerate(imgs):
#         padded_imgs[i] = img
#     return padded_imgs

def pad_images(imgs):
    # B, C, H, W
    max_shape = [0, 0]
    for img in imgs:
        for i in range(len(max_shape)):
            max_shape[i] = max(max_shape[i], img.shape[-1 - i])
    stack = []
    for img in imgs:
        pad = []
        for i in range(len(max_shape)):
            pad = pad + [0, max_shape[i] - img.shape[-1 - i]]
        stack.append(F.pad(img, pad, value=0))
    return torch.stack(stack)

def image_transform(transform, image, coords=[], renormalize=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image, keypoints=coords)
    image = augmented['image']
    if len(coords) > 0:
        coords = np.array(augmented['keypoints'])
        if renormalize:
            coords = normalize_nodes(coords, flip_y=False)
        else:
            _, height, width = image.shape
            coords[:, 0] = coords[:, 0] / width
            coords[:, 1] = coords[:, 1] / height
        coords = np.array(coords).clip(0, 1)
        return image, coords
    return image

def image_transform_bbox(transform, image, coords=[], renormalize=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image, bboxes=coords, point_ids= list(range(len(coords))))
    image = augmented['image']
    if len(coords) > 0:
        coords = np.array(augmented['bboxes'])
        # TODO: pseudo coords
        # if renormalize:
        #     coords = normalize_nodes(coords, flip_y=False)
        # else:
        _, height, width = image.shape
        coords[:, 0] = coords[:, 0] / width
        coords[:, 1] = coords[:, 1] / height
        coords[:, 2] = coords[:, 2] / width
        coords[:, 3] = coords[:, 3] / height
        coords = np.array(coords).clip(0, 1)
        return image, coords
    return image

# token augmentation
   
def process_rdkit_data_for_train(image, smiles,graph,src_data,cur_id,transform,tokenizer,pseudo_coords=False):
    ref = {}
    edges,coords=graph['edges'],graph['coords']
    
    if 'rect_point_min' in graph:
        coords = coords + graph['rect_point_min'] + graph['rect_point_max']
        num_atoms = len(graph['coords'])
        num_rects = len(graph['rect_point_min'])
    # transform
    image, coords = image_transform(transform,image, coords, renormalize=pseudo_coords)
    
    if 'rect_point_min' in graph:
        rect_point_min = coords[num_atoms:(num_atoms + num_rects), :].copy()
        rect_point_max = coords[(num_atoms + num_rects):, :].copy()
        coords = coords[:num_atoms, :].copy()

    token_list,atom_indices,coords=tokenizer.get_token_list(smiles,coords)
    ref['atom_center'] = coords[:len(atom_indices), :].copy()

    # edge
    if edges is not None:
        edges= torch.tensor(edges)[:len(atom_indices), :len(atom_indices)]
    else:
        if 'edges' in src_data.columns:
            edge_list = eval(src_data.loc[cur_id, 'edges'])
            n = len(atom_indices)
            edges = torch.zeros((n, n), dtype=torch.long)
            for u, v, t in edge_list:
                if u < n and v < n:
                    if t <= 4:
                        edges[u, v] = t
                        edges[v, u] = t
                    else:
                        edges[u, v] = t
                        edges[v, u] = 11 - t
        else:
            edges = torch.ones(len(atom_indices), len(atom_indices), dtype=torch.long) * (-100)
        
    ref['chartok_coords'],ref['atom_indices'],ref['coords'],ref['edges']=token_list,atom_indices,coords,edges

    if 'rect_point_min' in graph:
        ref['rect_point_min'], ref['rect_point_max'] = rect_point_min[:len(atom_indices), :], rect_point_max[:len(atom_indices), :]
    for atom_info_type in ['rect_rtv_pos', 'is_R_or_super_atom', 'radical_electrons', 'num_Hs_total', 'num_Hs_implicit', 'valence', 'isotope', 'degree', 'charge', 'is_arom', 'atoms_valence', 'atoms_valence_before', 'atoms_valence_after']:
        if atom_info_type in graph:
            if atom_info_type == 'charge': # ”+ 3“是为了不出现-1。现在，电荷量普遍加了3，比真正的电荷量多了3.
                ref[atom_info_type] = np.array(graph[atom_info_type][:len(atom_indices)]) + 3
            else:
                ref[atom_info_type] = np.array(graph[atom_info_type][:len(atom_indices)])

    for bond_info_type in ['edges_on_ring', 'edges_valence']:
        if bond_info_type in graph:
            ref[bond_info_type] = torch.tensor(graph[bond_info_type])[:len(atom_indices), :len(atom_indices)] # edges_on_ring 取值范围及意义 2: on ring, 1: on chain, 0: no bond, 3: 不明确
    
    return image, ref

def process_patent_for_train(image, smiles,coords,edges,src_data,cur_id,transform,tokenizer,pseudo_coords=True):
        
    if coords is not None:
        h, w, _ = image.shape
        coords = np.array(eval(coords))
        if pseudo_coords:
            coords = normalize_nodes(coords)
        coords[:, 0] = coords[:, 0] * w
        coords[:, 1] = coords[:, 1] * h
        image, coords = image_transform(transform,image, coords, renormalize=pseudo_coords)
    else:
        image, coords = image_transform(transform,image, [], renormalize=pseudo_coords)
        coords = None    
                
    ref = {}
            
    
    # token
    token_list,atom_indices,coords=tokenizer.get_token_list(smiles,coords)
    
    # edge
    if edges is not None:
        edge_list = eval(edges)
        n = len(atom_indices)
        edges = torch.zeros((n, n), dtype=torch.long)
        for u, v, t in edge_list:
            if u < n and v < n:
                if t <= 4:
                    edges[u, v] = t
                    edges[v, u] = t
                else:
                    edges[u, v] = t
                    edges[v, u] = 11 - t
    else:
        edges = torch.ones(len(atom_indices), len(atom_indices), dtype=torch.long) * (-100)
    
    # valence before&after
    mapping = np.array([0, 1, 2, 3, 1.5, 1, 1]) * 2
    edges_valence = np.round(mapping[edges]).astype(int)
    
    edges_valence_tril = np.tril(edges_valence)
    
    atoms_valence_before = np.sum(edges_valence_tril, axis=1)
    atoms_valence = np.sum(edges_valence, axis=1)
    atoms_valence_after = atoms_valence - atoms_valence_before

    ref['edges_valence'] = edges_valence
    ref['atoms_valence'] = atoms_valence
    ref['atoms_valence_before'] = atoms_valence_before
    ref['atoms_valence_after'] = atoms_valence_after
        
    ref['chartok_coords'],ref['atom_indices'],ref['coords'],ref['edges']=token_list,atom_indices,coords,edges

    return image, ref

def get_edge_dict():
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 4
    AROM = 128
    bond_order_dict = {
        SINGLE: 1, 
        DOUBLE: 2, 
        TRIPLE: 3, 
        8: 4, 
        16: 5, 
        32: 6, 
        AROM: 1.5, 
        256: 2.5, 
        512: 3.5, 
        1024: 4.5, 
        2048: 5.5, 
    }

    edge_dict_list = [(SINGLE, 0), (DOUBLE, 0), (TRIPLE, 0), (AROM, 0)] # 1234
    # 56
    edge_dict_list.append((SINGLE, 3))
    edge_dict_list.append((SINGLE, 4))
    sym_dict = {}
    sym_dict[5] = 6
    sym_dict[6] = 5
    # 78
    edge_dict_list.append((SINGLE, 6))
    edge_dict_list.append((SINGLE, 7))
    sym_dict[8] = 7
    sym_dict[7] = 8
    # 9, 10
    edge_dict_list.append((SINGLE, 9))
    edge_dict_list.append((SINGLE, 10))
    sym_dict[10] = 9
    sym_dict[9] = 10
    # 11, 12
    edge_dict_list.append((SINGLE, 11))
    edge_dict_list.append((SINGLE, 12))
    sym_dict[12] = 11
    sym_dict[11] = 12
    # 13, ...
    for i in [1, 2, 5, 8, 13, 14]:
        edge_dict_list.append((SINGLE, i))
    for i in [1, 5, 8]:
        edge_dict_list.append((DOUBLE, i))
    for key in bond_order_dict:
        if not ((key, 0) in edge_dict_list):
            edge_dict_list.append((key, 0))
    edge_dict = {}
    for i, edge_type_key in enumerate(edge_dict_list):
        edge_dict[edge_type_key] = i + 1

    return edge_dict, sym_dict

def sort_data(token_list,atom_indices,coords,charge,edge_list,order_strategy='random'):
    
    atom_order = list(range(len(atom_indices))) #np.arange(len(atom_indices))
    
    if order_strategy=='random':
        seed = random.randrange(sys.maxsize)
        random.Random(seed).shuffle(atom_order)  # Shuffle the sequence x in place
    if order_strategy=='dis2ori':
        x,y = coords[:,0], coords[:,1]
        dis = np.square(x)+np.square(y)
        atom_order = np.argsort(dis)

    reverse_map = np.argsort(atom_order)
    
    # atom_indices_order = atom_indices[atom_order].tolist()
     
    atom_indices_range =[]
    atom_indices_list = atom_indices.tolist()
    for i,atom_indice in enumerate(atom_indices_list):
        if i==0:
            atom_indices_range.append((1,atom_indice))
        else:
            atom_indices_range.append((atom_indices_list[i-1]+1,atom_indice))
   
    atom_indices_range = np.array(atom_indices_range)[atom_order]
    
    token_list_random=[token_list[0].item()]
    for b,e in atom_indices_range:
        token_list_random = token_list_random+token_list[b:e+1].tolist()
    token_list_random = token_list_random + [token_list[-1].item()]
    
    token_list_random = torch.tensor(token_list_random)
    atom_indices = atom_indices
    coords = coords[atom_order]
    charge = charge[atom_order]
    edge_list = [[int(reverse_map[start]), int(reverse_map[end]), etype] for (start, end, etype) in edge_list]
    
    return token_list_random,atom_indices,coords,charge,edge_list

def process_cdx_for_train(image, smiles, coords, bbox, edges, charge, src_data,cur_id,transform,tokenizer,pseudo_coords=False,order_strategy='random'):
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 4
    AROM = 128
    bond_order_dict = {
        SINGLE: 1, 
        DOUBLE: 2, 
        TRIPLE: 3, 
        8: 4, 
        16: 5, 
        32: 6, 
        AROM: 1.5, 
        256: 2.5, 
        512: 3.5, 
        1024: 4.5, 
        2048: 5.5, 
    }
    
    bond_order_type_dict = {1: 1,
        2: 2,
        4: 3,
        8: 4,
        16: 5,
        32: 6,
        64: 7,
        128: 8,
        256: 9,
        512: 10,
        1024: 11,
        2048: 12,
        4096: 13,
        8192: 14,
        16384: 15,
        32768: 16,
        0: 0}
    ref = {}
    
    # image
    coords = np.array(eval(coords)) + 0.
    bbox = np.array(eval(bbox)) + 0.
    if tokenizer.use_bbox:
        # coords = merge_bbox_keypoints(coords,bbox)
        # image, coords = image_transform_bbox(transform,image, coords, renormalize=pseudo_coords)
        
        coords = merge_bbox_keypoints_expand(coords,bbox)
        image, coords = image_transform(transform,image, coords, renormalize=pseudo_coords)
        coords = get_bbox_list_from_s_list(coords)
    else:
        image, coords = image_transform(transform,image, coords, renormalize=pseudo_coords)

    # token
    if tokenizer.atom_type_only:
        token_list,atom_indices,coords,symbol_labels,superatom_indices,superatom_bbox=tokenizer.get_token_list(smiles,coords)
    else:
        token_list,atom_indices,coords=tokenizer.get_token_list(smiles,coords)
    n = len(atom_indices)
    
    # charge
    charge = torch.tensor(eval(charge)) if isinstance(charge,str) else torch.tensor(charge)
    charge[charge<-3] = 0
    charge[charge>8] = 0
    charge = charge + 3
    
    # edge
    edge_list = eval(edges)
    
    # random
    # TODO symbol_labels
    if tokenizer.atom_only and order_strategy!='default':
        token_list,atom_indices,coords,charge,edge_list = sort_data(token_list,atom_indices,coords,charge,edge_list,order_strategy=order_strategy)
    
    # edge
    edges_valence = torch.zeros((n, n), dtype=torch.long)
    for u, v, t in edge_list:
        if u < n and v < n:
            if t[0] in bond_order_dict:
                this_valence = bond_order_dict[t[0]] * 2
            else: # TODO
                this_valence = -1
            edges_valence[u, v] = this_valence
            edges_valence[v, u] = this_valence
    # valence before&after
    
    edges_valence_tril = np.tril(edges_valence)
    
    atoms_valence_before = np.sum(edges_valence_tril, axis=1)
    atoms_valence_before[atoms_valence_before<-1]=15
    atoms_valence_before[atoms_valence_before>14]=15
    
    atoms_valence = np.sum(edges_valence.numpy(), axis=1)
    atoms_valence_after = atoms_valence - atoms_valence_before
    ref['edges_valence'] = edges_valence
    ref['atoms_valence'] = atoms_valence
    ref['atoms_valence_before'] = atoms_valence_before
    ref['atoms_valence_after'] = atoms_valence_after
    atoms_valence_after[atoms_valence_after<-1]=15
    atoms_valence_after[atoms_valence_after>14]=15

    edge_dict = {(1, 0): 1, (2, 0): 2, (4, 0): 3, (128, 0): 4, (1, 3): 5, (1, 4): 6, (1, 6): 7, (1, 7): 8, (1, 9): 9, (1, 10): 10, (1, 11): 11, (1, 12): 12, (1, 1): 13, (1, 2): 14, (1, 5): 15, (1, 8): 16, (1, 13): 17, (1, 14): 18, (2, 1): 19, (2, 5): 20, (2, 8): 21, (8, 0): 22, (16, 0): 23, (32, 0): 24, (256, 0): 25, (512, 0): 26, (1024, 0): 27, (2048, 0): 28, (4096, 0): 29, (8192, 0): 30, (16384, 0): 31, (32768, 0): 32, (128, 5): 33, (128, 6): 34, (128, 7): 35, (64, 0):36}
    sym_dict = {5: 6, 6: 5, 8: 7, 7: 8, 10: 9, 9: 10, 12: 11, 11: 12}
    bond_display_sym_dict = {3: 4, 4: 3, 6: 7, 7: 6, 9: 10, 10: 9, 11: 12, 12: 11}

    # edges
    edges = torch.zeros((n, n), dtype=torch.long)
    edge_orders = torch.zeros((n, n), dtype=torch.long)
    # edge_displays = torch.zeros((n, n), dtype=torch.long)
    edge_displays = torch.ones((n, n), dtype=torch.long)*-100
    
    
    for u, v, t in edge_list:
        if u < n and v < n:
            if tuple(t) in edge_dict:
                edge_type = edge_dict[tuple(t)]
                edge_bond = bond_order_type_dict[tuple(t)[0]]
                edge_display = tuple(t)[1]
                
                if edge_type in sym_dict:
                    edge_type_reverse = sym_dict[edge_type]
                    edge_display_reverse = bond_display_sym_dict[edge_display]
                else:
                    edge_type_reverse = edge_type
                    edge_display_reverse = edge_display
            
            else: # TODO
                edge_type = -100
                edge_type_reverse = -100
                
                edge_bond = -100
                
                edge_display = -100
                edge_display_reverse = -100
                
                
            edges[u, v] = edge_type
            edges[v, u] = edge_type_reverse
            
            edge_orders[u, v] = edge_bond
            edge_orders[v, u] = edge_bond
            
            edge_displays[u, v] = edge_display
            edge_displays[v, u] = edge_display_reverse
               
    # edges bond
    # edges display
    ref['charge']=charge
    ref['chartok_coords'],ref['atom_indices'],ref['coords'],ref['edges'],ref['edge_orders'],ref['edge_displays']=token_list,atom_indices,coords,edges,edge_orders,edge_displays
    
    if tokenizer.atom_type_only:
        ref['symbol_labels'] = symbol_labels
        ref['superatom_indices'] = superatom_indices
        ref['superatom_bbox'] = superatom_bbox
        
    return image, ref

def process_indigo_data_for_train(image, smiles,graph,src_data,cur_id,transform,tokenizer,pseudo_coords=False):
    ref = {}
    edge_list,coords,symbols=graph['edge_list'],graph['coords'],graph['symbols']
            
    # transform
    image, coords = image_transform(transform,image, coords, renormalize=pseudo_coords)
    
    # tokens
    token_list,atom_indices,coords=tokenizer.get_token_list(smiles,coords)
    
    # indigo2uspto_dict={1: (1, 0), 2: (2, 0), 3: (4, 0), 4: (128, 0), 5: (1, 6), 6: (1, 3)}
    indigo2uspto_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 7, 6: 5}
    edge_dict = {(1, 0): 1, (2, 0): 2, (4, 0): 3, (128, 0): 4, (1, 3): 5, (1, 4): 6, (1, 6): 7, (1, 7): 8, (1, 9): 9, (1, 10): 10, (1, 11): 11, (1, 12): 12, (1, 1): 13, (1, 2): 14, (1, 5): 15, (1, 8): 16, (1, 13): 17, (1, 14): 18, (2, 1): 19, (2, 5): 20, (2, 8): 21, (8, 0): 22, (16, 0): 23, (32, 0): 24, (256, 0): 25, (512, 0): 26, (1024, 0): 27, (2048, 0): 28}
    edge_reverse_dict = {1: (1, 0), 2: (2, 0), 3: (4, 0), 4: (128, 0), 5: (1, 3), 6: (1, 4), 7: (1, 6), 8: (1, 7), 9: (1, 9), 10: (1, 10), 11: (1, 11), 12: (1, 12), 13: (1, 1), 14: (1, 2), 15: (1, 5), 16: (1, 8), 17: (1, 13), 18: (1, 14), 19: (2, 1), 20: (2, 5), 21: (2, 8), 22: (8, 0), 23: (16, 0), 24: (32, 0), 25: (256, 0), 26: (512, 0), 27: (1024, 0), 28: (2048, 0)}
    sym_dict = {5: 6, 6: 5, 8: 7, 7: 8, 10: 9, 9: 10, 12: 11, 11: 12}
    bond_display_sym_dict = {3: 4, 4: 3, 6: 7, 7: 6, 9: 10, 10: 9, 11: 12, 12: 11}
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 4
    AROM = 128
    bond_order_dict = {
        SINGLE: 1, 
        DOUBLE: 2, 
        TRIPLE: 3, 
        8: 4, 
        16: 5, 
        32: 6, 
        AROM: 1.5, 
        256: 2.5, 
        512: 3.5, 
        1024: 4.5, 
        2048: 5.5, 
    }
    bond_order_type_dict = {1: 1,
        2: 2,
        4: 3,
        8: 4,
        16: 5,
        32: 6,
        64: 7,
        128: 8,
        256: 9,
        512: 10,
        1024: 11,
        2048: 12,
        4096: 13,
        8192: 14,
        16384: 15,
        32768: 16,
        0: 0}
    
    # edges
    if edge_list is not None:
        n = len(atom_indices)
    
        edges = torch.zeros((n, n), dtype=torch.long)
        edge_orders = torch.zeros((n, n), dtype=torch.long)
        edge_displays = torch.zeros((n, n), dtype=torch.long)
        edges_valence = torch.zeros((n, n), dtype=torch.long)
        
        for u, v, t in edge_list:
            edge_type_reverse,edge_bond,edge_display,edge_display_reverse = -100,-100,-100,-100
            edge_type = indigo2uspto_dict[t] if t in indigo2uspto_dict.keys() else -100
            if edge_type!=-100:
                edge_bond,edge_display = edge_reverse_dict[edge_type][0],edge_reverse_dict[edge_type][1]
                
                if edge_bond in bond_order_dict:
                    this_valence = bond_order_dict[edge_bond] * 2
                else: # TODO
                    this_valence = -1
                    
                this_valence = bond_order_dict[edge_bond] * 2
                if edge_type in sym_dict:
                    edge_type_reverse = sym_dict[edge_type]
                    edge_display_reverse = bond_display_sym_dict[edge_display]
                else:
                    edge_type_reverse = edge_type
                    edge_display_reverse = edge_display
                edge_bond = bond_order_type_dict[edge_bond]
            try:
                edges[u, v] = edge_type
            except:
                logger.info(symbols)
                logger.info(len(symbols))
                logger.info(u)
                logger.info(v)
                
            edges[u, v] = edge_type
            edges[v, u] = edge_type_reverse
            
            edge_orders[u, v] = edge_bond
            edge_orders[v, u] = edge_bond
            
            edge_displays[u, v] = edge_display
            edge_displays[v, u] = edge_display_reverse
            
            edges_valence[u, v] = this_valence
            edges_valence[v, u] = this_valence
        
        edges_valence_tril = np.tril(edges_valence)
        
        atoms_valence_before = np.sum(edges_valence_tril, axis=1)
        atoms_valence_before[atoms_valence_before<-1]=15
        atoms_valence_before[atoms_valence_before>14]=15
        
        atoms_valence = np.sum(edges_valence.numpy(), axis=1)
        atoms_valence_after = atoms_valence - atoms_valence_before
        atoms_valence_after[atoms_valence_after<-1]=15
        atoms_valence_after[atoms_valence_after>14]=15
    else:
        if 'edges' in src_data.columns:
            edge_list = eval(src_data.loc[cur_id, 'edges'])
            n = len(atom_indices)
            edges = torch.zeros((n, n), dtype=torch.long)
            edge_orders = torch.zeros((n, n), dtype=torch.long)
            edge_displays = torch.zeros((n, n), dtype=torch.long)           
            edges_valence = torch.zeros((n, n), dtype=torch.long)
    
            for u, v, t in edge_list:
                edge_type_reverse,edge_bond,edge_display,edge_display_reverse = -100,-100,-100,-100
                edge_type = indigo2uspto_dict[t] if t in indigo2uspto_dict.keys() else -100
                if edge_type!=-100:
                    edge_bond,edge_display = edge_reverse_dict[edge_type][0],edge_reverse_dict[edge_type][1]
                    
                    if edge_bond in bond_order_dict:
                        this_valence = bond_order_dict[edge_bond] * 2
                    else: # TODO
                        this_valence = -1
                        
                    if edge_type in sym_dict:
                        edge_type_reverse = sym_dict[edge_type]
                        edge_display_reverse = bond_display_sym_dict[edge_display]
                    else:
                        edge_type_reverse = edge_type
                        edge_display_reverse = edge_display
                
                edges[u, v] = edge_type
                edges[v, u] = edge_type_reverse
                
                edge_orders[u, v] = edge_bond
                edge_orders[v, u] = edge_bond
                
                edge_displays[u, v] = edge_display
                edge_displays[v, u] = edge_display_reverse
                
                edges_valence[u, v] = this_valence
                edges_valence[v, u] = this_valence
            
            edges_valence_tril = np.tril(edges_valence)
            
            atoms_valence_before = np.sum(edges_valence_tril, axis=1)
            atoms_valence_before[atoms_valence_before<-1]=15
            atoms_valence_before[atoms_valence_before>14]=15
            
            atoms_valence = np.sum(edges_valence.numpy(), axis=1)
            atoms_valence_after = atoms_valence - atoms_valence_before
             
        else:
            edges = torch.ones(len(atom_indices), len(atom_indices), dtype=torch.long) * (-100)
            edge_orders = torch.ones(len(atom_indices), len(atom_indices), dtype=torch.long) * (-100)
            edge_displays = torch.ones(len(atom_indices), len(atom_indices), dtype=torch.long) * (-100)
            edges_valence = torch.ones(len(atom_indices), len(atom_indices), dtype=torch.long) * (-1)
            atoms_valence = torch.ones(len(atom_indices), len(atom_indices), dtype=torch.long) * (-1)
            atoms_valence_before = torch.ones(len(atom_indices), len(atom_indices), dtype=torch.long) * (-1)
            atoms_valence_after = torch.ones(len(atom_indices), len(atom_indices), dtype=torch.long) * (-1)
    
    ref['edges_valence'] = edges_valence
    ref['atoms_valence'] = atoms_valence
    ref['atoms_valence_before'] = atoms_valence_before
    ref['atoms_valence_after'] = atoms_valence_after
            
    ref['chartok_coords'],ref['atom_indices'],ref['coords'],ref['edges']=token_list,atom_indices,coords,edges
    ref['edge_orders'],ref['edge_displays'] = edge_orders,edge_displays
    # if 'rect_point_min' in graph:
    #     ref['rect_point_min'], ref['rect_point_max'] = rect_point_min[:len(atom_indices), :], rect_point_max[:len(atom_indices), :]
    for atom_info_type in ['rect_rtv_pos', 'is_R_or_super_atom', 'radical_electrons', 'num_Hs_total', 'num_Hs_implicit', 'valence', 'isotope', 'degree', 'charge', 'is_arom', 'atoms_valence', 'atoms_valence_before', 'atoms_valence_after']:
        if atom_info_type in graph:
            if atom_info_type == 'charge': 
                ref[atom_info_type] = np.array(graph[atom_info_type][:len(atom_indices)]) + 3
            else:
                ref[atom_info_type] = np.array(graph[atom_info_type][:len(atom_indices)])
            #print(type(ref[atom_info_type]), type(ref[atom_info_type][0]), ref[atom_info_type])
    
    for bond_info_type in ['edges_on_ring', 'edges_valence']:
        if bond_info_type in graph:
            ref[bond_info_type] = torch.tensor(graph[bond_info_type])[:len(atom_indices), :len(atom_indices)]
    
    return image, ref

def process_molgrapher_data_for_train(image, coords, molfile, transform, tokenizer, pseudo_coords=False, all_bonds_explicit=False):
    """
    """
    types_classes_bonds = {'SINGLE':1,'DOUBLE':2,'TRIPLE':3,'AROMATIC':4,"DASHED":5,"SOLID":6} 
    indigo2uspto_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 7, 6: 5}
    edge_dict = {(1, 0): 1, (2, 0): 2, (4, 0): 3, (128, 0): 4, (1, 3): 5, (1, 4): 6, (1, 6): 7, (1, 7): 8, (1, 9): 9, (1, 10): 10, (1, 11): 11, (1, 12): 12, (1, 1): 13, (1, 2): 14, (1, 5): 15, (1, 8): 16, (1, 13): 17, (1, 14): 18, (2, 1): 19, (2, 5): 20, (2, 8): 21, (8, 0): 22, (16, 0): 23, (32, 0): 24, (256, 0): 25, (512, 0): 26, (1024, 0): 27, (2048, 0): 28}
    edge_reverse_dict = {1: (1, 0), 2: (2, 0), 3: (4, 0), 4: (128, 0), 5: (1, 3), 6: (1, 4), 7: (1, 6), 8: (1, 7), 9: (1, 9), 10: (1, 10), 11: (1, 11), 12: (1, 12), 13: (1, 1), 14: (1, 2), 15: (1, 5), 16: (1, 8), 17: (1, 13), 18: (1, 14), 19: (2, 1), 20: (2, 5), 21: (2, 8), 22: (8, 0), 23: (16, 0), 24: (32, 0), 25: (256, 0), 26: (512, 0), 27: (1024, 0), 28: (2048, 0)}
    sym_dict = {5: 6, 6: 5, 8: 7, 7: 8, 10: 9, 9: 10, 12: 11, 11: 12}
    bond_display_sym_dict = {3: 4, 4: 3, 6: 7, 7: 6, 9: 10, 10: 9, 11: 12, 12: 11}
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 4
    AROM = 128
    bond_order_dict = {
        SINGLE: 1, 
        DOUBLE: 2, 
        TRIPLE: 3, 
        8: 4, 
        16: 5, 
        32: 6, 
        AROM: 1.5, 
        256: 2.5, 
        512: 3.5, 
        1024: 4.5, 
        2048: 5.5, 
    }
    bond_order_type_dict = {1: 1,
        2: 2,
        4: 3,
        8: 4,
        16: 5,
        32: 6,
        64: 7,
        128: 8,
        256: 9,
        512: 10,
        1024: 11,
        2048: 12,
        4096: 13,
        8192: 14,
        16384: 15,
        32768: 16,
        0: 0}
    ref = {}
    
    # molecule
    molecule = Chem.MolFromMolBlock(molfile, sanitize=False, removeHs=False)
    Chem.WedgeMolBonds(molecule, molecule.GetConformers()[0]) 
    
    # coords
    coords=[[coords[i] - 1, coords[i+1] - 1] for i in range(0, len(coords), 3)] 
    
    # s_time = time.time()
    # edges
    edge_list = []
    for bond_index, bond in enumerate(molecule.GetBonds()):
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        bond_type = str(bond.GetBondType())
        
        assert bond_type in types_classes_bonds.keys(), 'none type bond'
        
        # if bond_type == 'DOUBLE':  # display = 8
        #     assert not(bond.HasProp("_MolFileBondStereo") and bond.GetProp("_MolFileBondStereo") == "3"), 'cross double bond'

        # if bond_type == 'SINGLE':
        #     assert str(bond.GetBondDir()) != 'UNKNOWN', 'wavy bond'

        if str(bond.GetBondDir()) == "BEGINWEDGE":
            bond_type = "SOLID"  # 6
        
        if str(bond.GetBondDir()) == "BEGINDASH":
            bond_type = "DASHED"  # 5

        edge_type = types_classes_bonds[bond_type]
        edge_list.append([b,e,edge_type])
    
    # t_time=time.time() - s_time
    # logger.info(f'edge_list: {t_time}')
     
    # s_time = time.time()
    smiles, symbols, coords, edge_list = build_mol(molecule, coords, edge_list, molfile)
        
    # transform
    image, coords = image_transform(transform,image, coords, renormalize=pseudo_coords)
    # t_time=time.time() - s_time
    # logger.info(f'image_transform: {t_time}')
    # s_time = time.time()
    # tokens
    if all_bonds_explicit:
       smiles = Indigo.all_bonds_explicit_func(smiles)
    token_list,atom_indices,coords=tokenizer.get_token_list(smiles,coords)
    
    n = len(atom_indices)
    
    edges = torch.zeros((n, n), dtype=torch.long)
    edge_orders = torch.zeros((n, n), dtype=torch.long)
    edge_displays = torch.zeros((n, n), dtype=torch.long)
    edges_valence = torch.zeros((n, n), dtype=torch.long)
    
    for u, v, t in edge_list:
        edge_type_reverse,edge_bond,edge_display,edge_display_reverse = -100,-100,-100,-100
        edge_type = indigo2uspto_dict[t] if t in indigo2uspto_dict.keys() else -100
        if edge_type!=-100:
            edge_bond,edge_display = edge_reverse_dict[edge_type][0],edge_reverse_dict[edge_type][1]
            
            if edge_bond in bond_order_dict:
                this_valence = bond_order_dict[edge_bond] * 2
            else: # TODO
                this_valence = -1
                
            this_valence = bond_order_dict[edge_bond] * 2
            if edge_type in sym_dict:
                edge_type_reverse = sym_dict[edge_type]
                edge_display_reverse = bond_display_sym_dict[edge_display]
            else:
                edge_type_reverse = edge_type
                edge_display_reverse = edge_display
            
            edge_bond = bond_order_type_dict[edge_bond]
        
        edges[u, v] = edge_type
        edges[v, u] = edge_type_reverse
        
        edge_orders[u, v] = edge_bond
        edge_orders[v, u] = edge_bond
        
        edge_displays[u, v] = edge_display
        edge_displays[v, u] = edge_display_reverse
        
        edges_valence[u, v] = this_valence
        edges_valence[v, u] = this_valence

    # t_time=time.time() - s_time
    # logger.info(f'edges: {t_time}')
    edges_valence_tril = np.tril(edges_valence)
    
    atoms_valence_before = np.sum(edges_valence_tril, axis=1)
    atoms_valence_before[atoms_valence_before<-1]=15
    atoms_valence_before[atoms_valence_before>14]=15
    
    atoms_valence = np.sum(edges_valence.numpy(), axis=1)
    atoms_valence_after = atoms_valence - atoms_valence_before
    atoms_valence_after[atoms_valence_after<-1]=15
    atoms_valence_after[atoms_valence_after>14]=15
    
    ref['edges_valence'] = edges_valence
    ref['atoms_valence'] = atoms_valence
    ref['atoms_valence_before'] = atoms_valence_before
    ref['atoms_valence_after'] = atoms_valence_after

       
    ref['chartok_coords'],ref['atom_indices'],ref['coords'],ref['edges']=token_list,atom_indices,coords,edges
    ref['edge_orders'],ref['edge_displays'] = edge_orders,edge_displays
    
    return image, ref, smiles