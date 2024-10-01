'''
Author: Rong Ma
Date: 2023-11-29 17:56:51
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 15:57:01
Description: 
'''
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors

def swap(elem1, elem2):
    return elem2, elem1

def build_mol(symbols, coords, edges, return_mol_weight=False):
    assert len(coords) > 0, '[Error] there is no molecules in this image'
    mol = Chem.RWMol()
    assert len(symbols) == len(coords), '[Error] the number of node names does not match the number of node coords'
    for i, symbol in enumerate(symbols):
        if symbol.startswith('[') and (symbol.endswith(';]') or symbol.endswith(',]') or symbol.endswith('.]')):
            symbol = symbol[:-2] + symbol[-1]
        atom = Chem.AtomFromSmiles(symbol)
        if atom is None: # H will be in here
            if symbol.startswith('[') and symbol.endswith(']'):
                symbol = symbol[1:-1]
            atom = Chem.Atom("*")
            Chem.SetAtomAlias(atom, symbol)
        atom.SetIsAromatic(False)
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
    
    molblock = '\n'.join(molblock)
    mol_weight = 0
    
    try:
        smiles = Chem.MolToSmiles(mol, canonical = True, kekuleSmiles = True)
        mol_weight = Descriptors.MolWt(mol)
    except:
        try:
            smiles = Chem.MolToSmiles(mol, canonical = True, kekuleSmiles = False)
            mol_weight = Descriptors.MolWt(mol)
        except:
            smiles = ''
            mol_weight = 0
    # print(smiles)
    if return_mol_weight:
        return molblock,smiles,mol_weight
    else:
        return molblock,smiles