'''
Author: Jiaxin Zheng
Date: 2024-10-01 14:19:27
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 16:05:18
Description: 
'''
from SmilesPE.pretokenizer import atomwise_tokenizer
from rdkit import Chem
from rdkit.Chem.rdchem import GetPeriodicTable
import rdkit
import numpy as np
import re


def build_mol(mol, coords, edges, molfile): # property[i]: 0: charge, 1: isotope
    # [a.GetFormalCharge() for a in molecule.GetAtoms()]
    superatom_pre = {int(i)-1: symb for i, symb in re.findall(r'A\s+(\d+)\s+(\S+)\s', molfile)}
    
    for idx, atom in enumerate(mol.GetAtoms()):
        try:
            atom_num = atom.GetAtomicNum()
        except:
            atom_num = 0
        if atom_num == 0 and not(idx in superatom_pre):
            superatom_pre.append((idx, atom.GetSymbol()))
    superatoms = [(k, v) for k, v in superatom_pre.items()]
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    
    smiles = Chem.MolToSmiles(mol, canonical = False, kekuleSmiles = False)
    atom_order = mol.GetProp('_smilesAtomOutputOrder')
    atom_order = eval(atom_order)
    reverse_map = np.argsort(atom_order)
    
    symbols = np.array(symbols)[atom_order].tolist()
    coords = np.array(coords)[atom_order].tolist()
    # coords_bbox = np.array(coords_bbox)[atom_order].tolist()
    edges = [[int(reverse_map[start]), int(reverse_map[end]), etype] for (start, end, etype) in edges]
    pseudo_smiles = smiles
    if len(superatoms) > 0:
        superatoms = {int(reverse_map[atom_idx]): symb for atom_idx, symb in superatoms}

        tokens = atomwise_tokenizer(smiles)
        atom_idx = 0
        for i, t in enumerate(tokens):
            if t.isalpha() or t[0] == '[' or t == '*':
                #print(i, t)
                if atom_idx in superatoms:
                    #print('miaom', atom_idx, superatoms[atom_idx])
                    #print('t', t, atom_idx)
                    symb = superatoms[atom_idx]
                    tokens[i] = f"[{symb}]"
                atom_idx += 1
                #print(atom_idx, tokens[i])
        pseudo_smiles = ''.join(tokens)

    tokens = atomwise_tokenizer(pseudo_smiles)

    return pseudo_smiles, symbols, coords, edges
