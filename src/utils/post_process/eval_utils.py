import json
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import rdkit
from rdkit import Chem, DataStructs
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from SmilesPE.pretokenizer import atomwise_tokenizer
from rdkit.Chem.rdchem import GetPeriodicTable

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.post_process.molscribe_chemistry import _postprocess_smiles, _convert_graph_to_smiles, _keep_main_molecule, canonicalize_smiles
from src.utils import get_pylogger

import re
def remove_H(symbol):
    if len(re.sub("H[a-z]", "", symbol)) < len(symbol):
        pass
    else:
        if not(len(re.sub("H", "", symbol)) + 1 == len(symbol)):
            pass
        else:
            if symbol == 'H':
                pass
            else: # 
                if len(re.sub("H[0-9]*", "", symbol)) + 2 < len(symbol):
                    pass
                else:
                    symbol = re.sub('H', '', re.sub(r'H\d', '', symbol))
    return symbol

def coords2continuous(coords, coord_bins = 64):
    coords=np.array([[round(x*(coord_bins-1)),round(y*(coord_bins-1))] for x,y in coords])
    return coords

def _get_pairwise_L1_loss(preds, gts):
    dis=np.abs(preds.reshape(-1,1)-gts.reshape(1,-1))
    return dis

def _pairwise_distance(preds, gts):        
    dis=np.array([])
    if len(preds)>0:
        
        if np.max(preds) > 1:
            pass
        else:
            preds=coords2continuous(preds)
        gts=coords2continuous(gts)
        # print(preds)
        # print(gts)
        # print(preds)
        # print(gts)

        x_pred,y_pred=preds[:,0],preds[:,1]
        x_gts,y_gts=gts[:,0],gts[:,1]
        
        x_dis=_get_pairwise_L1_loss(x_pred,x_gts)
        y_dis=_get_pairwise_L1_loss(y_pred,y_gts)
        dis=(x_dis+y_dis)*0.5
    return dis

def get_match_id(_pred_coords,_gt_coords,pred_atom_scores):
    dis = _pairwise_distance(_pred_coords,_gt_coords)
    if dis.size > 0:
        row_indices, col_indices = linear_sum_assignment(dis)
        nearest_match_idx = list(zip(row_indices, col_indices))
        nearest_match_idx_dis = [dis[row_idx, col_idx] for row_idx, col_idx in zip(row_indices, col_indices)]
    else:
        nearest_match_idx = []
        nearest_match_idx_dis = []
    if len(nearest_match_idx_dis) > 0:
        dis_max = np.max(nearest_match_idx_dis)
    else:
        dis_max = 0
    return nearest_match_idx, dis_max, nearest_match_idx_dis

def eval_without_ocr_row(row):
    # pred
    node_symbols = eval(row['node_symbols'])
    node_coords = np.array(eval(row['node_coords']))
    #edges = np.array(eval(row['edges']))
    atom_scores = np.array(eval(row['atom_scores']))
    #edge_scores = np.array(eval(row['edge_scores']))
    # ground truth
    gt_symbols = eval(row['gt_symbols'])
    gt_coords = np.array(eval(row['gt_coords']))
    #gt_edges = np.array(eval(row['GT_gt_edges']))
    match_id_list,match_dis_max,nearest_match_idx_dis=get_match_id(node_coords,gt_coords,atom_scores)
    row['match_id_list'] = match_id_list
    row['match_dis_max'] = match_dis_max
    row['nearest_match_idx_dis'] = nearest_match_idx_dis
    
    if len(match_id_list) > 0:
        row_indices, col_indices = zip(*match_id_list)
        unassigned_rows = set(range(len(node_symbols))) - set(row_indices)
        unassigned_cols = set(range(len(gt_symbols))) - set(col_indices)
    else:
        unassigned_rows = set(range(len(node_symbols)))
        unassigned_cols = set(range(len(gt_symbols)))
        
    match_symbol_list = [(node_symbols[match_pair[0]], gt_symbols[match_pair[1]]) for match_pair in match_id_list]
    match_coord_list = [(node_coords[match_pair[0]].tolist(), gt_coords[match_pair[1]].tolist()) for match_pair in match_id_list]
    match_score_list = [atom_scores[match_pair[0]] for match_pair in match_id_list]
    for row_idx in unassigned_rows:
        match_symbol_list.append((node_symbols[row_idx], None))
        match_coord_list.append((node_coords[row_idx], None))
        match_score_list.append((atom_scores[row_idx]))
    for col in unassigned_cols:
        match_symbol_list.append((None, gt_symbols[col]))
        match_coord_list.append((None, gt_coords[col]))
        match_score_list.append(None)
    
    edge_dict = {(-100, -100): -100, (0, -1): 0, (1, 0): 1, (2, 0): 2, (4, 0): 3, (128, 0): 4, (1, 3): 5, (1, 4): 6, (1, 6): 7, (1, 7): 8, (1, 9): 9, (1, 10): 10, (1, 11): 11, (1, 12): 12, (1, 1): 13, (1, 2): 14, (1, 5): 15, (1, 8): 16, (1, 13): 17, (1, 14): 18, (2, 1): 19, (2, 5): 20, (2, 8): 21, (8, 0): 22, (16, 0): 23, (32, 0): 24, (256, 0): 25, (512, 0): 26, (1024, 0): 27, (2048, 0): 28, (4096, 0): 29, (8192, 0): 30, (16384, 0): 31, (32768, 0): 32, (128, 5): 33, (128, 6): 34, (128, 7): 35, (64, 0):36}
    sym_dict = {5: 6, 6: 5, 8: 7, 7: 8, 10: 9, 9: 10, 12: 11, 11: 12}
    edge_dict_reverse = {v: k for k, v in edge_dict.items()}
    edges = np.array(eval(row['edges']))
    gt_edges_sparse = eval(row['gt_edges'])
    gt_atom_num = gt_coords.shape[0]
    gt_edges = np.zeros((gt_atom_num, gt_atom_num), dtype = np.int32)
    for u, v, t in gt_edges_sparse:
        if u < gt_atom_num and v < gt_atom_num:
            if tuple(t) in edge_dict:
                edge_type = edge_dict[tuple(t)]
                if edge_type in sym_dict:
                    edge_type_reverse = sym_dict[edge_type]
                else:
                    edge_type_reverse = edge_type
            else: # TODO
                edge_type = -100
                edge_type_reverse = -100
            gt_edges[u, v] = edge_type
            gt_edges[v, u] = edge_type_reverse

    match_id_dict = {match_pair[0]: match_pair[1] for match_pair in match_id_list}
    for gt_others, row_idx in enumerate(unassigned_rows):
        match_id_dict[row_idx] = - gt_others - 1
    for pred_others, col_idx in enumerate(unassigned_cols):
        match_id_dict[-pred_others-1] = col_idx
    match_edge_order_list = []
    match_edge_display_list = []
    
    edge_modify_num = 0 # 加这一行
    for key, value in match_id_dict.items():
        for key1, value1 in match_id_dict.items():
            if key < key1:
                if key >= 0 and key1 >= 0:
                    pred_edge = edges[key, key1]
                else:
                    pred_edge = 0
                if value >= 0 and value1 >= 0:
                    gt_edge = gt_edges[value, value1]
                else:
                    gt_edge = 0
                # if pred_edge > 0 or gt_edge > 0:
                if 1:
                    pred_edge_order, pred_edge_display = edge_dict_reverse[pred_edge]
                    gt_edge_order, gt_edge_display = edge_dict_reverse[gt_edge]
                    if pred_edge_order != gt_edge_order or pred_edge_display != gt_edge_display: # 加这行
                        edge_modify_num += 1 # 加这行
                    match_edge_order_list.append((pred_edge_order, gt_edge_order))
                    match_edge_display_list.append((pred_edge_display, gt_edge_display))
    # print(match_edge_order_list)
    # print(match_edge_display_list)
    row['edge_modify_num'] = edge_modify_num # 加这行
    row['match_edge_order_list'] = match_edge_order_list
    row['match_edge_display_list'] = match_edge_display_list
    
    # print('row', row)
    # print(match_symbol_list)
    row['match_symbol_list'] = match_symbol_list
    row['match_score_list'] = match_score_list
    # print(match_coord_list)
    # print(match_score_list)
    row['gt_unmatched_node_num'] = len(gt_symbols)-len(match_id_list)
    row['pred_unmatched_node_num'] = len(node_symbols)-len(match_id_list)
    #print(row['gt_unmatched_node_num'], row['pred_unmatched_node_num'])
    return row

def eval_without_ocr(pred_file, gt_file, output_file):
    pred_df=pd.read_csv(pred_file)#.iloc[1045:1046]#, nrows=2)#.iloc[1045:1046]#
    gt_df=pd.read_csv(gt_file)#.iloc[1045:1046]#, nrows=2)#.iloc[1045:1046]#, nrows=30)
    # gt_df = gt_df.rename(columns=lambda x: 'GT_' + x)
    merged_df = pd.concat([pred_df, gt_df], axis=1)
    #merged_df['eval_without_ocr'] = merged_df.apply(lambda row: eval_without_ocr_row(row), axis=1)
    merged_df = merged_df.apply(eval_without_ocr_row, axis=1)
    merged_df.to_csv(output_file, index=False)
    return merged_df

def compute_occurrence(arr):
    arr = arr.reshape(-1)
    unique_values, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique_values, counts))

def symbol2class(symbol, labels = ['C', 'N', 'O', 'H', 'F', 'Cl', 'Br', 'P', 'S']):
    if symbol is None:
        return '[None]'
    if symbol.startswith('[') and (symbol.endswith(';]') or symbol.endswith(',]') or symbol.endswith('.]')):
        symbol = symbol[:-2] + symbol[-1]
    atom = Chem.AtomFromSmiles(symbol)
    if atom is None:
        return '[Super]'
    if atom.GetSymbol() in labels:
        return atom.GetSymbol()
    return '[Super]'

def symbol2ocr(symbol):
    if symbol is None:
        return '[None]'
    if symbol.startswith('[') and (symbol.endswith(';]') or symbol.endswith(',]') or symbol.endswith('.]')):
        symbol = symbol[:-2] + symbol[-1]
    atom = Chem.AtomFromSmiles(symbol)
    if atom is None:
        if symbol.startswith('[') and symbol.endswith(']'):
            return symbol[1:-1]
        else:
            return symbol
    else:
        return atom.GetSymbol()

def edges_adj_array2list(edges, edge_none = 0): # edge_none: 当edges的元素为这个值时，说明这里没有边
    edges_not_none_indices = np.argwhere(edges != edge_none)
    if edges_not_none_indices.size > 0:
        edges_not_none_values = edges[edges_not_none_indices[:, 0], edges_not_none_indices[:, 1]]
    else:
        edges_not_none_values = []
    return edges_not_none_indices, edges_not_none_values

def swap(x, y):
    return y, x

def atom_num2valence_dict():
    
    pt = GetPeriodicTable()
    valence_dict = {}
    for atomic_number in range(119):
        valence_list = pt.GetValenceList(atomic_number)
        valence_dict[atomic_number] = [valence for valence in valence_list]
        # print(atomic_number)
    return valence_dict

VALENCE_DICT = atom_num2valence_dict()

def build_mol(symbols, coords, edges):
    assert len(coords) > 0, '[Error] there is no molecules in this image'
    mol = Chem.RWMol()
    assert len(symbols) == len(coords), '[Error] the number of node names does not match the number of node coords'
    superatoms = []
    for i, symbol in enumerate(symbols):
        if symbol.startswith('[') and (symbol.endswith(';]') or symbol.endswith(',]') or symbol.endswith('.]')):
            symbol = symbol[:-2] + symbol[-1]
        atom = Chem.AtomFromSmiles(symbol)
        if atom is None: # H will be in here, [H] will not
            if symbol.startswith('[') and symbol.endswith(']'):
                symbol = symbol[1:-1]
            atom = Chem.Atom("*")
            Chem.SetAtomAlias(atom, symbol)
            superatoms.append((i, symbol)) # TODO write direction
        atom.SetIsAromatic(False) # TODO 让嘉鑫也加一下这句，在这里去芳香化
        idx = mol.AddAtom(atom)
        assert idx == i, '[Error] atom idx is strange when build mol using RDKit'
    bond_display_dict = {
        0: Chem.BondDir.NONE, 
        3: Chem.BondDir.BEGINDASH, 
        6: Chem.BondDir.BEGINWEDGE, 
        8: Chem.BondDir.UNKNOWN
    }
    bond_order_dict = {
        1: rdkit.Chem.rdchem.BondType.SINGLE, 
        2: rdkit.Chem.rdchem.BondType.DOUBLE, 
        4: rdkit.Chem.rdchem.BondType.TRIPLE, 
        8: rdkit.Chem.rdchem.BondType.QUADRUPLE, 
        16: rdkit.Chem.rdchem.BondType.QUINTUPLE, 
        32: rdkit.Chem.rdchem.BondType.HEXTUPLE, 
        64: rdkit.Chem.rdchem.BondType.OTHER, 
        128: rdkit.Chem.rdchem.BondType.AROMATIC, 
        256: rdkit.Chem.rdchem.BondType.TWOANDAHALF, 
        512: rdkit.Chem.rdchem.BondType.THREEANDAHALF, 
        1024: rdkit.Chem.rdchem.BondType.FOURANDAHALF, 
        2048: rdkit.Chem.rdchem.BondType.FIVEANDAHALF, 
        4096: rdkit.Chem.rdchem.BondType.DATIVE, 
        8192: rdkit.Chem.rdchem.BondType.IONIC, 
        16384: rdkit.Chem.rdchem.BondType.HYDROGEN, 
        32768: rdkit.Chem.rdchem.BondType.THREECENTER
    }
    new_edges = []
    for edge in edges:
        bond_begin = edge[0]
        bond_end = edge[1]
        bond_order = edge[2][0]
        bond_display = edge[2][1]
        if bond_display in [4, 7, 10, 12]: # if narraw at 'end', swap to 'begin'
            bond_display -= 1
            bond_begin, bond_end = swap(bond_begin, bond_end)
        #print('b', bond_begin, bond_end)
        mol.AddBond(bond_begin, bond_end, bond_order_dict[bond_order])
        #print('c')
        if bond_display in bond_display_dict: # and ((bond_order == 1) or not(bond_display == 8)):
            mol.GetBondBetweenAtoms(bond_begin, bond_end).SetBondDir(bond_display_dict[bond_display])
        new_edges.append([bond_begin, bond_end, [bond_order, bond_display]])
    for atom in mol.GetAtoms():
        explicit_valence = 0
        # print(help(atom))
        for bond in atom.GetBonds():
            # print(help(bond))
            explicit_valence += bond.GetValenceContrib(atom)
        explicit_valence += atom.GetNumExplicitHs()
        valence_list = VALENCE_DICT[atom.GetAtomicNum()]
        if atom.GetFormalCharge() != 0 and not((explicit_valence - atom.GetFormalCharge()) in valence_list):
            atom.SetFormalCharge(0)
            # print(explicit_valence)
            # print(valence_list)
            # print(atom.GetSymbol(), atom.GetAtomicNum())
    
    # try:
    #     Chem.SanitizeMol(mol) # TODO
    # except:
    #     #print(symbols, coords, edges)
    #     pass
        #print('[Warning] Sanitize failed')
    
    # conf = Chem.Conformer(len(coords))
    # conf.Set3D(True)
    # for i, (x, y) in enumerate(coords):
    #     conf.SetAtomPosition(i, (x, 1 - y, 0))
    # mol.AddConformer(conf)
    # Chem.SanitizeMol(mol)
    # Chem.AssignStereochemistryFrom3D(mol)
    # mol.RemoveAllConformers()
    conf = Chem.Conformer(len(coords))
    conf.Set3D(False)
    for i, (x, y) in enumerate(coords):
        conf.SetAtomPosition(i, (x, 1 - y, 0))
    mol.AddConformer(conf)
    molblock = Chem.MolToMolBlock(mol, kekulize = False, includeStereo = False).split('\n')
    for atom_line_idx in range(4, 4 + len(coords)):
        molblock[atom_line_idx] = 'C'.join(molblock[atom_line_idx].split('R'))
    #print(help(Chem.MolToMolBlock))
    #print('\n'.join(molblock))
    
    try:
        smiles = Chem.MolToSmiles(mol, canonical = True, kekuleSmiles = True)
    except:
        try:
            smiles = Chem.MolToSmiles(mol, canonical = True, kekuleSmiles = False)
        except:
            smiles = ''
    #print(smiles)
    atom_order = mol.GetProp('_smilesAtomOutputOrder')
    atom_order = eval(atom_order)
    reverse_map = np.argsort(atom_order)
    # print(len(atom_order), atom_order)
    # print(len(reverse_map), reverse_map)
    symbols = np.array(symbols)[atom_order].tolist()
    coords = np.array(coords)[atom_order].tolist()
    edges = [[int(reverse_map[start]), int(reverse_map[end]), etype] for (start, end, etype) in new_edges]
    pseudo_smiles = smiles
    if len(superatoms) > 0:
        superatoms = {int(reverse_map[atom_idx]): symb for atom_idx, symb in superatoms}
        tokens = atomwise_tokenizer(smiles)
        atom_idx = 0
        for i, t in enumerate(tokens):
            if t.isalpha() or t[0] == '[' or t == '*':
                if atom_idx in superatoms:
                    #print('t', t, atom_idx)
                    symb = superatoms[atom_idx]
                    tokens[i] = f"[{symb}]"
                atom_idx += 1
                #print(atom_idx, tokens[i])
        assert atom_idx == len(coords), ('[Error] the number of atoms in pseudo_smiles does not match the number of node coords', atom_idx, len(coords))
        pseudo_smiles = ''.join(tokens)
    return smiles, pseudo_smiles

def row2smiles(row):
    # edge_dict = {(1, 0): 1, (2, 0): 2, (4, 0): 3, (128, 0): 4, (1, 3): 5, (1, 4): 6, (1, 6): 7, (1, 7): 8, (1, 9): 9, (1, 10): 10, (1, 11): 11, (1, 12): 12, (1, 1): 13, (1, 2): 14, (1, 5): 15, (1, 8): 16, (1, 13): 17, (1, 14): 18, (2, 1): 19, (2, 5): 20, (2, 8): 21, (8, 0): 22, (16, 0): 23, (32, 0): 24, (256, 0): 25, (512, 0): 26, (1024, 0): 27, (2048, 0): 28}
    edge_dict = {(1, 0): 1, (2, 0): 2, (4, 0): 3, (128, 0): 4, (1, 3): 5, (1, 4): 6, (1, 6): 7, (1, 7): 8, (1, 9): 9, (1, 10): 10, (1, 11): 11, (1, 12): 12, (1, 1): 13, (1, 2): 14, (1, 5): 15, (1, 8): 16, (1, 13): 17, (1, 14): 18, (2, 1): 19, (2, 5): 20, (2, 8): 21, (8, 0): 22, (16, 0): 23, (32, 0): 24, (256, 0): 25, (512, 0): 26, (1024, 0): 27, (2048, 0): 28, (4096, 0): 29, (8192, 0): 30, (16384, 0): 31, (32768, 0): 32, (128, 5): 33, (128, 6): 34, (128, 7): 35,  (64, 0):36}
    edge_dict_reverse = {v: k for k, v in edge_dict.items()}
    #print(edge_dict_reverse)
    node_symbols = eval(row['node_symbols'])
    node_coords = eval(row['node_coords'])
    edges = np.array(eval(row['edges']))
    edges_not_none_indices, edges_not_none_values = edges_adj_array2list(edges)
    edges = []
    if len(edges_not_none_values) > 0:
        for (atom_begin, atom_end), edge_type in zip(edges_not_none_indices, edges_not_none_values):
            if atom_begin < atom_end:
                edges.append([int(atom_begin), int(atom_end), list(edge_dict_reverse[edge_type])])
    if len(node_symbols) > 0:
        try:
            row['smiles'], row['smiles_ocr'] = build_mol(node_symbols, node_coords, edges)
            row['success'] = True
            row['smiles_expand'], _, row['success_expand'] = _postprocess_smiles(row['smiles_ocr'])
            row['smiles_expand_canon'], row['success_expand_canon'] = canonicalize_smiles(row['smiles_expand'], ignore_chiral=True, ignore_cistrans=True)
        except:
            row['smiles'] = ''
            row['smiles_ocr'] = ''
            row['smiles_expand'] = ''
            row['smiles_expand_canon'] = ''
            row['success_expand'] = False
            row['success'] = False
    else:
        row['smiles'] = ''
        row['smiles_ocr'] = ''
        row['smiles_expand'] = ''
        row['smiles_expand_canon'] = ''
        row['success_expand'] = False
        row['success'] = False
    return row

def df2smiles(df):
    df = df.apply(row2smiles, axis = 1)
    print(df)
    return df['smiles'].tolist()

def row_base_expand_smiles(row):
    node_symbols = eval(row['node_symbols'])
    node_coords = eval(row['node_coords'])
    edges = np.array(eval(row['edges']))
    
    edge_dict_reverse = {1: (1, 0), 2: (2, 0), 3: (4, 0), 4: (128, 0), 5: (1, 6), 6: (1, 3)}
    edges_not_none_indices, edges_not_none_values = edges_adj_array2list(edges)
    edges_sparse = []
    if len(edges_not_none_values) > 0:
        for (atom_begin, atom_end), edge_type in zip(edges_not_none_indices, edges_not_none_values):
            if atom_begin < atom_end:
                edges_sparse.append([int(atom_begin), int(atom_end), list(edge_dict_reverse[edge_type])])
    if len(node_symbols) > 0:
        try:
            row['smiles'], row['smiles_ocr'] = build_mol(node_symbols, node_coords, edges_sparse)
            row['success'] = True
        except:
            row['smiles'] = ''
            row['smiles_ocr'] = ''
            row['success'] = False
    else:
        row['smiles'] = ''
        row['smiles_ocr'] = ''
        row['success'] = False

    row['smiles_expand'], _, row['success_expand'] = _convert_graph_to_smiles(node_coords, node_symbols, edges) # graph_smiles in MolScribe
    return row

def df_modify_row(df, row_func, col = None):
    df = df.apply(row_func, axis = 1)
    if col is None:
        return df
    else:
        return df[col].tolist()

def row_gt_expand_smiles(row):
    if 'gt_symbols' in row.index:
        gt_symbols = eval(row['gt_symbols'])
        gt_coords = eval(row['gt_coords'])
        gt_edges = eval(row['edges'])
        if len(gt_symbols) > 0:
            try:
                row['smiles'], row['smiles_ocr'] = build_mol(gt_symbols, gt_coords, gt_edges)
                row['success'] = True
                row['smiles_expand'], _, row['success_expand'] = _postprocess_smiles(row['smiles_ocr'])
            except:
                row['smiles'] = ''
                row['smiles_ocr'] = ''
                row['smiles_expand'] = ''
                row['success_expand'] = False
                row['success'] = False
        else:
            row['smiles'] = ''
            row['smiles_ocr'] = ''
            row['smiles_expand'] = ''
            row['success_expand'] = False
            row['success'] = False
    else:
        row['smiles_expand_canon'], row['success_expand_canon'] = canonicalize_smiles(row['SMILES'], ignore_cistrans=True, ignore_chiral=True)
    return row

def tanimoto_similarity(smiles1, smiles2):
    if smiles1 is None or smiles2 is None or smiles1 =='' or smiles2 =='':
        return 0
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
        return tanimoto
    except:
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            fp1 = Chem.RDKFingerprint(mol1)
            fp2 = Chem.RDKFingerprint(mol2)
            tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
            return tanimoto
        except:
            #print('[Warning] failed in tanimoto', smiles1, smiles2)
            return 0

def smiles_correct(pred_list, gt_list):
    smiles_corr = []
    tanimoto = []
    for pred, gt in zip(pred_list, gt_list):
        correct = (pred == gt)
        smiles_corr.append(correct)
        tanimoto.append(tanimoto_similarity(pred, gt))
    # print('smiles correct', np.sum(smiles_corr), (np.sum(smiles_corr) + 0.) / len(pred_list))
    # print('tanimoto average', np.mean(tanimoto))
    
    _smiles_correct =(np.sum(smiles_corr) + 0.) / len(pred_list)
    _tanimoto_average = np.mean(tanimoto)
    
    return _smiles_correct, _tanimoto_average

def gt_smiles(df):
    df = df.apply(row_gt_expand_smiles, axis = 1)
    return df['smiles'].tolist()

def symbol2ocr_gt(symbol):
    return symbol2ocr(symbol)
    if symbol is None:
        return '[None]'
    if len(symbol) == 0:
        symbol = 'C'
    else:
        symbol = ''.join(symbol)
    for special_char in ['\r\n', '\r', '\n']:
        symbol = symbol.replace(special_char, '')
    return symbol

def symbol2ocr_pred(symbol):
    return symbol2ocr(symbol)
    if symbol is None:
        return '[None]'
    if symbol in ['c', 'n', 's', 'se', 'si', 'b']: # 
        symbol = symbol[0].upper() + symbol[1:] # 
    if symbol.startswith('[') and symbol.endswith(']'):
        symbol = symbol[1:-1]
    for special_char in ['\x00', '\x07', '\x0b']:
        symbol = symbol.replace(special_char, '')
    return symbol

def df_statistic(df):
    len_df = len(df) + 0.
    cols = df.columns
    for col in ['gt_unmatched_node_num', 'pred_unmatched_node_num', 'match_dis_max']:
        print(col)
        occurs = compute_occurrence(np.array(df[col].tolist()))
        print(occurs)
        # freq = {k: v / len_df for k, v in occurs.items()}
        # print(freq)
        print('average', np.mean(df[col].tolist()))
    correct = np.sum(np.array(df['gt_unmatched_node_num'].tolist()) + np.array(df['pred_unmatched_node_num'].tolist()) == 0)
    indices = np.argwhere(np.array(df['gt_unmatched_node_num'].tolist()) + np.array(df['pred_unmatched_node_num'].tolist()) > 0)
    print('correct', correct, correct / len_df)
    #print('match_symbol_list', df['match_symbol_list'])
    # with_ocr
    ocr_exact_not = []
    strict_ocr_not = []
    col_match_symbol_list = df['match_symbol_list'].tolist()
    for match_symbol_list in col_match_symbol_list:
        if isinstance(match_symbol_list, str):
            match_symbol_list = eval(match_symbol_list)
        this_correct = True
        strict_ocr = True
        for match_symbol in match_symbol_list:
            if not(symbol2class(match_symbol[0]) == symbol2class(match_symbol[1])):
                this_correct = False
            if not(symbol2ocr_pred(match_symbol[0]) == symbol2ocr_gt(match_symbol[1])):
                strict_ocr = False
        ocr_exact_not.append(not(this_correct))
        strict_ocr_not.append(not(strict_ocr))
    #print(ocr_exact_not)
    ocr_indices = np.argwhere(ocr_exact_not)
    ocr_strict_indices = np.argwhere(strict_ocr_not)
    print('ocr class correct', int(len_df - ocr_indices.size), (len_df - ocr_indices.size) / len_df)
    print('ocr strict correct', int(len_df - ocr_strict_indices.size), (len_df - ocr_strict_indices.size) / len_df)
    return indices.reshape(-1), ocr_indices.reshape(-1), ocr_strict_indices.reshape(-1)

def get_scores(df,logger,return_df=False):
    df = df.astype(str)
    df = df.apply(eval_without_ocr_row, axis=1)
    
    scores ={}
    len_df = len(df) + 0.

    cols = df.columns
    for col in ['gt_unmatched_node_num', 'pred_unmatched_node_num', 'match_dis_max']:
        # print(col)
        occurs = compute_occurrence(np.array(df[col].tolist()))
        logger.info(col)
        logger.info(occurs)
        scores[f'{col}_average']=np.mean(df[col].tolist())
        
        correct_np = (np.array(df['gt_unmatched_node_num'].tolist()) + np.array(df['pred_unmatched_node_num'].tolist())) == 0 # add
        correct = np.sum(correct_np) # modify
        indices = np.argwhere(np.array(df['gt_unmatched_node_num'].tolist()) + np.array(df['pred_unmatched_node_num'].tolist()) > 0)

        scores['correct']=correct / len_df

        #print('match_symbol_list', df['match_symbol_list'])
        # with_ocr
        ocr_exact_not = []
        strict_ocr_not = []
        col_match_symbol_list = df['match_symbol_list'].tolist()
        for i, match_symbol_list in enumerate(col_match_symbol_list):
            if isinstance(match_symbol_list, str):
                match_symbol_list = eval(match_symbol_list)
            if correct_np[i]: # add
                this_correct = True
                strict_ocr = True
                for match_symbol in match_symbol_list:
                    symbol_pred = remove_H(symbol2ocr(match_symbol[0])) # 
                    symbol_gt = remove_H(symbol2ocr(match_symbol[1])) # 
                    if not(symbol2class(symbol_pred) == symbol2class(symbol_gt)): # 
                        this_correct = False # 
                    if not(symbol_pred == symbol_gt): # 
                        strict_ocr = False # 
            else: # add
                this_correct = False # add
                strict_ocr = False # add
            ocr_exact_not.append(not(this_correct))
            strict_ocr_not.append(not(strict_ocr))
        #print(ocr_exact_not)
        ocr_indices = np.argwhere(ocr_exact_not)
        ocr_strict_indices = np.argwhere(strict_ocr_not)

        scores['ocr_class_correct'] =  (len_df - ocr_indices.size) / len_df  # int(len_df - ocr_indices.size),
        scores['ocr_strict_correct'] = (len_df - ocr_strict_indices.size) / len_df  # int(len_df - ocr_strict_indices.size), 

        col_match_edge_order_list = df['match_edge_order_list'].tolist()
        col_match_edge_display_list = df['match_edge_display_list'].tolist()
        edge_order_not = []
        edge_display_not = []
        for i, (match_edge_order_list, match_edge_display_list) in enumerate(zip(col_match_edge_order_list, col_match_edge_display_list)): # modify
            if isinstance(match_edge_order_list, str):
                match_edge_order_list = eval(match_edge_order_list)
            if isinstance(match_edge_display_list, str):
                match_edge_display_list = eval(match_edge_display_list)
            if correct_np[i]: # add
                order_correct = True
                diplay_correct = True
                for match_edge_order, match_edge_display in zip(match_edge_order_list, match_edge_display_list):
                    if not(match_edge_order[0] == match_edge_order[1]):
                        order_correct = False
                    if not(match_edge_display[0] == match_edge_display[1]):
                        diplay_correct = False
            else: # add
                order_correct = False # add
                diplay_correct = False # add
            edge_order_not.append(not(order_correct))
            edge_display_not.append(not(diplay_correct))
        scores['edge_order_correct'] = (len_df - np.sum(edge_order_not)) / len_df
        scores['edge_display_correct'] = (len_df - np.sum(edge_display_not)) / len_df
        edge_not = np.logical_or(edge_order_not, edge_display_not)
        scores['edge_all_correct'] = (len_df - np.sum(edge_not)) / len_df
        scores['edge_modify_num'] = df['edge_modify_num'].mean()
        
        scores['atom_modify_num'] = atom_true_pred(df)
        
        ocr_class_or_edge_order_not = np.logical_or(ocr_exact_not, edge_order_not)
        scores['ocr_class_and_edge_order_correct']  = (len_df - np.sum(ocr_class_or_edge_order_not)) / len_df
        ocr_strict_or_edge_order_not = np.logical_or(strict_ocr_not, edge_order_not)
        scores['ocr_strict_and_edge_order_correct'] = (len_df - np.sum(ocr_strict_or_edge_order_not)) / len_df
        ocr_class_or_edge_not = np.logical_or(ocr_exact_not, edge_not)
        scores['ocr_class_and_edge_all_correct'] = (len_df - np.sum(ocr_class_or_edge_not)) / len_df
        ocr_strict_or_edge_not = np.logical_or(strict_ocr_not, edge_not)
        scores['ocr_strict_and_edge_all_correct'] = (len_df - np.sum(ocr_strict_or_edge_not)) / len_df

        is_ours = True
        if is_ours:
            modify_smiles_func = row2smiles
        else:
            modify_smiles_func = row_base_expand_smiles

        pred_df = df.copy()
        pred_df = df_modify_row(pred_df, modify_smiles_func)
        logger.info(f"for pred, [success number of converting to smiles] {np.sum(pred_df['success'].tolist())}, [success number of expanding smiles] {np.sum(pred_df['success_expand'].tolist())}")

        gt_df = df.copy()
        gt_df = df_modify_row(gt_df, row_gt_expand_smiles)
        logger.info(f"for gt, [success number of converting to smiles] {np.sum(gt_df['success'].tolist())}, [success number of expanding smiles] {np.sum(gt_df['success_expand'].tolist())}")

        # smiles_corr2, tanimoto2 = smiles_correct(pred_df['smiles'], gt_df['smiles'])
        # scores['smiles_corr2'] = smiles_corr2
        # scores['tanimoto2'] = tanimoto2

        # smiles_corr2ocr, tanimoto2ocr = smiles_correct(pred_df['smiles_ocr'], gt_df['SMILES'])
        # scores['smiles_corr2ocr'] = smiles_corr2ocr
        # scores['tanimoto2ocr'] = tanimoto2ocr

        # smiles_corr2expand, tanimoto2expand = smiles_correct(pred_df['smiles_expand'], gt_df['smiles_expand'])
        # scores['smiles_corr2expand'] = smiles_corr2expand
        # scores['tanimoto2expand'] = tanimoto2expand
    
    if return_df:
        return scores,df
    else:
        return scores   
    
def plot_cm(y_true1, y_pred1, labels, fig_labels, cm_img_fn):
    cm = confusion_matrix(y_true1, y_pred1, labels=labels)
    raw_cm = cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    
    #print("Confusion Matrix:\n", cm)

    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=fig_labels, yticklabels=fig_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    #plt.show()
    plt.savefig(cm_img_fn)

    
    for i, label in enumerate(labels):
        total = np.sum(cm[i, :])
        misclassified_as_label = np.sum(cm[:, i]) - cm[i, i]
        misclassified_from_label = total - cm[i, i]
        print(f"Class '{label}':")
        print(f"  - Misclassified as '{label}': {misclassified_as_label} times")
        print(f"  - Misclassified from '{label}': {misclassified_from_label} times")
        print(f"  - Misclassification Rate: {misclassified_from_label / total:.2f}")
    
    return raw_cm, cm

def print_f1(y_true, y_pred, labels):
    for average in ['macro', 'micro', 'weighted', None]:
        precision = precision_score(y_true, y_pred, labels=labels, average=average)
        recall = recall_score(y_true, y_pred, labels=labels, average=average)
        f1 = f1_score(y_true, y_pred, labels=labels, average=average)
        print('-' * 20 + ' metrics ' + '-' * 20)
        print(str(average) + ' precision', precision)
        print(str(average) + ' recall', recall)
        print(str(average) + ' f1', f1)

def atom_true_pred(df, debug=False):
    y_pred = []
    y_true = []
    col_match_symbol_list = df['match_symbol_list'].tolist()
    for match_symbol_list in col_match_symbol_list:
        if isinstance(match_symbol_list, str):
            match_symbol_list = eval(match_symbol_list)
        for match_symbol in match_symbol_list:
            y_pred.append(match_symbol[0])
            y_true.append(match_symbol[1])
    print(y_pred[:20], len(y_pred))
    print(y_true[:20], len(y_true))
    if debug:
        for pred, gt in zip(y_pred, y_true):
            if pred is None or gt is None:
                print(pred, gt)
    y_pred1 = [symbol2ocr_pred(symbol) for symbol in y_pred]
    y_true1 = [symbol2ocr_gt(symbol) for symbol in y_true]
    exact_symbol = []
    for pred, gt in zip(y_pred1, y_true1):
        exact_symbol.append(pred == gt)
    print('[strict ocr] exact_symbol / total_symbol', np.sum(exact_symbol) / len(y_pred1))
    print('[strict ocr] expected number of modification per image', (len(y_pred1) - np.sum(exact_symbol)) / len(col_match_symbol_list))
    # y_pred1 = [symbol2class(symbol, labels) for symbol in y_pred]
    # y_true1 = [symbol2class(symbol, labels) for symbol in y_true]
    # return y_pred1, y_true1, exact_symbol
    
    return (len(y_pred1) - np.sum(exact_symbol)) / len(col_match_symbol_list)

def load_pred_gt_df(key, dir_name = 'real', key2 = '', dir_name2 = 'uspto_edge_info_attn_epoch029'):
    df_gt = pd.read_csv('/workspace/summarization/Rgroup/eval/' + dir_name + '/' + key + '.csv')
    if key2 == '':
        df_pred = pd.read_csv('/workspace/summarization/Rgroup/eval/' + dir_name2 + '/' + key + key2 + '.csv')
    else:
        df_pred = pd.read_csv('/workspace/summarization/Rgroup/eval/' + dir_name2 + '/' + key + key2 + '.csv', sep = '\t')
    df_gt = df_modify_row(df_gt, row_gt_expand_smiles)
    df_pred = df_modify_row(df_pred, row2smiles)
    smiles_corr, tanimoto = smiles_correct(df_pred['smiles_expand_canon'].tolist(), df_gt['smiles_expand_canon'].tolist())
    return df_gt, df_pred, smiles_corr, tanimoto

if __name__=='__main__':
    # ============================================================
    # all
    # ============================================================
    pred_df = pd.read_csv('')
    output_file = ''
    log = get_pylogger(__name__)
    scores = get_scores(pred_df,log)
    with open(output_file, 'w') as f:
        json.dump(scores,f)