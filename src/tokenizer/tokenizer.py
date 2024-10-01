'''
Author: Jiaxin Zheng
Date: 2023-09-01 17:25:54
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 15:54:24
Description: 
'''
import os
import json
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from SmilesPE.pretokenizer import atomwise_tokenizer
from transformers import T5Tokenizer
from src.tokenizer.token import PAD,SOS,EOS,UNK,MASK,PAD_ID,SOS_ID,EOS_ID,UNK_ID,MASK_ID
from src.utils.data_utils.augment_utils import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy

class Tokenizer(object):

    def __init__(self, path=None,ckpt_path=''):
        self.stoi = {}
        self.itos = {}
        if ckpt_path!='':
            tokenizer = T5Tokenizer.from_pretrained(ckpt_path, model_max_length=480)
            self.stoi=tokenizer.get_vocab()
            self.itos = {item[1]: item[0] for item in self.stoi.items()}
        else:
            if path:
                self.load(path)

    def __len__(self):
        return len(self.stoi)

    @property
    def output_constraint(self):
        return False

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.stoi, f)

    def load(self, path):
        with open(path) as f:
            self.stoi = json.load(f)
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = [PAD, SOS, EOS, UNK] + list(vocab)
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID

    def text_to_sequence(self, text, tokenized=True):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        if tokenized:
            tokens = text.split(' ')
        else:
            tokens = atomwise_tokenizer(text)
        for s in tokens:
            if s not in self.stoi:
                s = '<unk>'
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions

    def sequence_to_smiles(self, sequence):
        return {'smiles': self.predict_caption(sequence)}


class AtomCharTokenizer(object):

    def __init__(self, path=None):
        self.stoi = {}
        self.itos = {}
        self.load(path)
        
    def __len__(self):
        return len(self.stoi)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.stoi, f)

    def load(self, path):
        with open(path) as f:
            self.stoi = json.load(f)
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def text_to_sequence(self, text, tokenized=True):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text:
            if s not in self.stoi:
                s = '<unk>'
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def texts_to_sequences(self, texts,return_tensor=True):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            if return_tensor:
                sequences.append(torch.Tensor(sequence))
            else:
                sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts
    
class NodeTokenizer(Tokenizer):

    def __init__(self, coord_bins=100, path=None, sep_xy=False, continuous_coords=False, debug=False,ckpt_path='',atom_type_only=False):
        super().__init__(path,ckpt_path)
        self.maxx = coord_bins
        self.maxy = coord_bins
        self.sep_xy = sep_xy
        self.special_tokens = [PAD, SOS, EOS, UNK, MASK]
        self.continuous_coords = continuous_coords
        self.debug = debug
        self.atom_type_only = atom_type_only
    def __len__(self):
        if self.sep_xy:
            return self.offset + self.maxx + self.maxy
        else:
            return self.offset + max(self.maxx, self.maxy)

    @property
    def offset(self):
        return len(self.stoi)

    @property
    def output_constraint(self):
        return not self.continuous_coords

    def len_symbols(self):
        return len(self.stoi)

    def fit_atom_symbols(self, atoms):
        vocab = self.special_tokens + list(set(atoms))
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID
        assert self.stoi[MASK] == MASK_ID
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def is_x(self, x):
        return self.offset <= x < self.offset + self.maxx

    def is_y(self, y):
        if self.sep_xy:
            return self.offset + self.maxx <= y
        return self.offset <= y

    def is_symbol(self, s):
        return len(self.special_tokens) <= s < self.offset or s == UNK_ID

    def is_atom(self, id):
        if self.is_symbol(id):
            return self.is_atom_token(self.itos[id])
        return False
    
    def is_atom_type(self, id):
        return id == 101 or id ==102

    def is_atom_token(self, token):
        return token[0].isalpha() or token.startswith("[") or token == '*' or token == UNK
    
    def is_script_token(self, token):
        return token in ['\x00', '\x07', '\x0b']
    
    def is_atom_token_start(self, token):
        return token == '(' or token == '-' or token == '=' or token == ':' or token == '#'
    
    def is_atom_token_end(self, token):
        return token == ')' or token.isdigit()
        
    def preprocess_tokens_from_atomwise_tokenizer(self,tokens):
        start_indices = [i for i in range(len(tokens)) if self.is_atom_token_start(tokens[i])]
        end_indices = [i for i in range(len(tokens)) if self.is_atom_token_end(tokens[i])]
        atom_indices=list(set(range(len(tokens)))-set(start_indices)-set(end_indices))
        
        for i in start_indices:
            if i+1<len(tokens):
                tokens[i+1]=tokens[i]+tokens[i+1]
        for i in reversed(end_indices):
            if i-1>=0:
                tokens[i-1]=tokens[i-1]+tokens[i]
        
        processed_tokens=[tokens[i] for i in atom_indices]
        return processed_tokens

    def x_to_id(self, x):
        return self.offset + round(x * (self.maxx - 1))

    def y_to_id(self, y):
        if self.sep_xy:
            return self.offset + self.maxx + round(y * (self.maxy - 1))
        return self.offset + round(y * (self.maxy - 1))

    def x_to_id_no_offset(self, x):
        return round(x * (self.maxx - 1))

    def y_to_id_no_offset(self, y):
        if self.sep_xy:
            return self.maxx + round(y * (self.maxy - 1))
        return round(y * (self.maxy - 1))
    
    def id_to_x(self, id):
        return (id - self.offset) / (self.maxx - 1)

    def id_to_y(self, id):
        if self.sep_xy:
            return (id - self.offset - self.maxx) / (self.maxy - 1)
        return (id - self.offset) / (self.maxy - 1)
    
    def id_to_x_no_offset(self, id):
        return (id) / (self.maxx - 1)

    def id_to_y_no_offset(self, id):
        if self.sep_xy:
            return (id - self.offset - self.maxx) / (self.maxy - 1)
        return (id) / (self.maxy - 1)
    
    def get_output_mask(self, id):
        mask = [False] * len(self)
        if self.continuous_coords:
            return mask
        if self.is_atom(id):
            return [True] * self.offset + [False] * self.maxx + [True] * self.maxy
        if self.is_x(id):
            return [True] * (self.offset + self.maxx) + [False] * self.maxy
        if self.is_y(id):
            return [False] * self.offset + [True] * (self.maxx + self.maxy)
        return mask

    def symbol_to_id(self, symbol):
        if symbol not in self.stoi:
            return UNK_ID
        return self.stoi[symbol]

    def symbols_to_labels(self, symbols):
        labels = []
        for symbol in symbols:
            labels.append(self.symbol_to_id(symbol))
        return labels

    def labels_to_symbols(self, labels):
        symbols = []
        for label in labels:
            symbols.append(self.itos[label])
        return symbols

    def nodes_to_grid(self, nodes):
        coords, symbols = nodes['coords'], nodes['symbols']
        grid = np.zeros((self.maxx, self.maxy), dtype=int)
        for [x, y], symbol in zip(coords, symbols):
            x = round(x * (self.maxx - 1))
            y = round(y * (self.maxy - 1))
            grid[x][y] = self.symbol_to_id(symbol)
        return grid

    def grid_to_nodes(self, grid):
        coords, symbols, indices = [], [], []
        for i in range(self.maxx):
            for j in range(self.maxy):
                if grid[i][j] != 0:
                    x = i / (self.maxx - 1)
                    y = j / (self.maxy - 1)
                    coords.append([x, y])
                    symbols.append(self.itos[grid[i][j]])
                    indices.append([i, j])
        return {'coords': coords, 'symbols': symbols, 'indices': indices}

    def nodes_to_sequence(self, nodes):
        coords, symbols = nodes['coords'], nodes['symbols']
        labels = [SOS_ID]
        for (x, y), symbol in zip(coords, symbols):
            assert 0 <= x <= 1
            assert 0 <= y <= 1
            labels.append(self.x_to_id(x))
            labels.append(self.y_to_id(y))
            labels.append(self.symbol_to_id(symbol))
        labels.append(EOS_ID)
        return labels

    def sequence_to_nodes(self, sequence):
        coords, symbols = [], []
        i = 0
        if sequence[0] == SOS_ID:
            i += 1
        while i + 2 < len(sequence):
            if sequence[i] == EOS_ID:
                break
            if self.is_x(sequence[i]) and self.is_y(sequence[i+1]) and self.is_symbol(sequence[i+2]):
                x = self.id_to_x(sequence[i])
                y = self.id_to_y(sequence[i+1])
                symbol = self.itos[sequence[i+2]]
                coords.append([x, y])
                symbols.append(symbol)
            i += 3
        return {'coords': coords, 'symbols': symbols}

    def smiles_to_sequence(self, smiles, coords=None, mask_ratio=0, atom_only=False):
        tokens = atomwise_tokenizer(smiles)
        labels = [SOS_ID]
        indices = []
        atom_idx = -1
        for token in tokens:
            if atom_only and not self.is_atom_token(token):
                continue
            if token in self.stoi:
                labels.append(self.stoi[token])
            else:
                if self.debug:
                    print(f'{token} not in vocab')
                labels.append(UNK_ID)
            if self.is_atom_token(token):
                atom_idx += 1
                if not self.continuous_coords:
                    if mask_ratio > 0 and random.random() < mask_ratio:
                        labels.append(MASK_ID)
                        labels.append(MASK_ID)
                    elif coords is not None:
                        if atom_idx < len(coords):
                            x, y = coords[atom_idx]
                            assert 0 <= x <= 1
                            assert 0 <= y <= 1
                        else:
                            x = random.random()
                            y = random.random()
                        labels.append(self.x_to_id(x))
                        labels.append(self.y_to_id(y))
                indices.append(len(labels) - 1)
        labels.append(EOS_ID)
        return labels, indices

    def sequence_to_smiles(self, sequence):
        has_coords = not self.continuous_coords
        smiles = ''
        coords, symbols, indices = [], [], []
        for i, label in enumerate(sequence):
            if label == EOS_ID or label == PAD_ID:
                break
            if self.is_x(label) or self.is_y(label):
                continue
            token = self.itos[label]
            smiles += token
            if self.is_atom_token(token):
                if has_coords:
                    if i+3 < len(sequence) and self.is_x(sequence[i+1]) and self.is_y(sequence[i+2]):
                        x = self.id_to_x(sequence[i+1])
                        y = self.id_to_y(sequence[i+2])
                        coords.append([x, y])
                        symbols.append(token)
                        indices.append(i+3)
                else:
                    if i+1 < len(sequence):
                        symbols.append(token)
                        indices.append(i+1)
        results = {'smiles': smiles, 'symbols': symbols, 'indices': indices}
        if has_coords:
            results['coords'] = coords
        return results
    
class CharTokenizer(NodeTokenizer):

    def __init__(self,path=None, 
                 sep_xy=False, coord_bins=100,continuous_coords=False,
                 max_len=100,mask_ratio=0,debug=False,
                 swap_token=False,atom_only=False,ckpt_path='',use_bbox=False,atom_type_only=False,is_cxcywh=False,remove_script_char=False):
        super().__init__(coord_bins, path, sep_xy, continuous_coords, debug,ckpt_path,atom_type_only)
        self.atom_only = atom_only
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.swap_token = swap_token
        self.use_bbox = use_bbox
        self.atom_type_only = atom_type_only
        self.is_cxcywh = is_cxcywh
        self.remove_script_char = remove_script_char

    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(list(text))
        if ' ' in vocab:
            vocab.remove(' ')
        vocab = [PAD, SOS, EOS, UNK] + list(vocab)
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID

    def text_to_sequence(self, text, tokenized=True):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        if tokenized:
            tokens = text.split(' ')
            assert all(len(s) == 1 for s in tokens)
        else:
            tokens = list(text)
        for s in tokens:
            if s not in self.stoi:
                s = '<unk>'
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def fit_atom_symbols(self, atoms):
        atoms = list(set(atoms))
        chars = []
        for atom in atoms:
            chars.extend(list(atom))
        vocab = self.special_tokens + chars
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID
        assert self.stoi[MASK] == MASK_ID
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def get_output_mask(self, id):
        ''' TO FIX '''
        mask = [False] * len(self)
        if self.continuous_coords:
            return mask
        if self.is_x(id):
            return [True] * (self.offset + self.maxx) + [False] * self.maxy
        if self.is_y(id):
            return [False] * self.offset + [True] * (self.maxx + self.maxy)
        return mask

    def nodes_to_sequence(self, nodes):
        coords, symbols = nodes['coords'], nodes['symbols']
        labels = [SOS_ID]
        for (x, y), symbol in zip(coords, symbols):
            assert 0 <= x <= 1
            assert 0 <= y <= 1
            labels.append(self.x_to_id(x))
            labels.append(self.y_to_id(y))
            for char in symbol:
                labels.append(self.symbol_to_id(char))
        labels.append(EOS_ID)
        return labels

    def sequence_to_nodes(self, sequence):
        coords, symbols = [], []
        i = 0
        if sequence[0] == SOS_ID:
            i += 1
        while i < len(sequence):
            if sequence[i] == EOS_ID:
                break
            if i+2 < len(sequence) and self.is_x(sequence[i]) and self.is_y(sequence[i+1]) and self.is_symbol(sequence[i+2]):
                x = self.id_to_x(sequence[i])
                y = self.id_to_y(sequence[i+1])
                for j in range(i+2, len(sequence)):
                    if not self.is_symbol(sequence[j]):
                        break
                symbol = ''.join(self.itos(sequence[k]) for k in range(i+2, j))
                coords.append([x, y])
                symbols.append(symbol)
                i = j
            else:
                i += 1
        return {'coords': coords, 'symbols': symbols}

    def smiles_to_sequence(self, smiles, coords=None, bbox=None, mask_ratio=0, atom_only=False):
        
        atom_only=self.atom_only
        if isinstance(smiles,list):
            tokens = smiles
        else:
            tokens = atomwise_tokenizer(smiles)
        labels = [SOS_ID]
        indices = []
        atom_idx = -1
        for token in tokens:
            if atom_only and not self.is_atom_token(token):
                continue

            if self.remove_script_char:
                token = ''.join([t for t in token if not self.is_script_token(t)])
            
            for c in token:
                if c in self.stoi:
                    labels.append(self.stoi[c])
                else:
                    if self.debug:
                        print(f'{c} not in vocab')
                    labels.append(UNK_ID)
            if self.is_atom_token(token):
                atom_idx += 1
                if not self.continuous_coords:
                    if mask_ratio > 0 and random.random() < mask_ratio:
                        if not self.swap_token:
                            labels.append(MASK_ID)
                            labels.append(MASK_ID)
                        else:
                            labels.insert(-1,MASK_ID)
                            labels.insert(-1,MASK_ID)
                    elif coords is not None:
                        if self.use_bbox:
                            x1, y1, x2, y2 = random.random(),random.random(),None,None
                            if atom_idx < len(coords):
                                x1, y1, x2, y2 = coords[atom_idx]
                                assert 0 <= x1 <= 1
                                assert 0 <= y1 <= 1
                                assert 0 <= x2 <= 1
                                assert 0 <= y2 <= 1
                                if x2-x1<=1e-6 and y2-y1<=1e-6 :
                                    x2, y2 = None, None
                                    
                            if not self.swap_token:
                                labels.append(self.x_to_id(x1))
                                labels.append(self.y_to_id(y1))
                                if x2 is not None and y2 is not None:
                                    labels.append(self.x_to_id(x2))
                                    labels.append(self.y_to_id(y2))
                            else:
                                labels.insert(-(len(token)),self.x_to_id(x1))
                                labels.insert(-(len(token)),self.y_to_id(y1))
                                if x2 is not None and y2 is not None:
                                    labels.insert(-(len(token)),self.x_to_id(x2))
                                    labels.insert(-(len(token)),self.y_to_id(y2))
                        else:
                            if atom_idx < len(coords):
                                x, y = coords[atom_idx]
                                assert 0 <= x <= 1
                                assert 0 <= y <= 1
                            else:
                                x = random.random()
                                y = random.random()
                            if not self.swap_token:
                                labels.append(self.x_to_id(x))
                                labels.append(self.y_to_id(y))
                            else:
                                labels.insert(-(len(token)),self.x_to_id(x))
                                labels.insert(-(len(token)),self.y_to_id(y))
                indices.append(len(labels) - 1)
        labels.append(EOS_ID)
        return labels, indices
    
    def sequence_to_smiles(self, sequence):
        has_coords = not self.continuous_coords
        smiles = ''
        coords, symbols, indices = [], [], []
        i = 0
        # print('Sequence Length: '+ str(len(sequence)))

        while i < len(sequence):
            label = sequence[i]
            
            if label == EOS_ID or label == PAD_ID:
                break
            if self.is_x(label) or self.is_y(label):
                i += 1
                continue
            
            # print(self.itos[label])
            
            if not self.is_atom(label):
                smiles += self.itos[label]
                i += 1
                continue
            
            # atom [i,j)
            if self.itos[label] == '[':
                j = i + 1
                while j < len(sequence):
                    if not self.is_symbol(sequence[j]):
                        break
                    if self.itos[sequence[j]] == ']':
                        j += 1
                        break
                    j += 1
            else:
                if i+1 < len(sequence) and (self.itos[label] == 'C' and self.is_symbol(sequence[i+1]) and self.itos[sequence[i+1]] == 'l' \
                        or self.itos[label] == 'B' and self.is_symbol(sequence[i+1]) and self.itos[sequence[i+1]] == 'r'):
                    j = i+2
                else:
                    j = i+1
            token = ''.join(self.itos[sequence[k]] for k in range(i, j))
            smiles += token
            if has_coords:
                if not self.swap_token:
                    if j+2 < len(sequence) and self.is_x(sequence[j]) and self.is_y(sequence[j+1]):
                        x = self.id_to_x(sequence[j])
                        y = self.id_to_y(sequence[j+1])
                        
                        # print('<'+str(x)+'>')
                        # print('<'+str(y)+'>')
                        
                        if self.use_bbox:
                            x2,y2=x,y
                            
                            if j+4 < len(sequence) and self.is_x(sequence[j+2]) and self.is_y(sequence[j+3]):
                                x2 = self.id_to_x(sequence[j+2])
                                y2 = self.id_to_y(sequence[j+3])
                        
                            coords.append([x, y, x2, y2])
                            symbols.append(token)
                            
                            if j+4 < len(sequence) and self.is_x(sequence[j+2]) and self.is_y(sequence[j+3]):
                                indices.append(j+4)
                                i = j+4
                            else:
                                indices.append(j+2)
                                i = j+2
                        else:
                            coords.append([x, y])
                            symbols.append(token)
                            indices.append(j+2)
                            i = j+2
                    else:
                        i = j
                else:
                    if i-2 >=0 and self.is_x(sequence[i-2]) and self.is_y(sequence[i-1]):
                        x = self.id_to_x(sequence[i-2])
                        y = self.id_to_y(sequence[i-1])
                        coords.append([x, y])
                        symbols.append(token)
                        indices.append(j-1)
                        i = j
                    else:
                        i = j
            else:
                if j < len(sequence):
                    symbols.append(token)
                    indices.append(j)
                i = j
        if not self.atom_only:
            results = {'smiles': smiles, 'symbols': symbols, 'indices': indices}
        else:
            results = {'symbols': symbols, 'indices': indices}
        if has_coords:
            results['coords'] = coords
        return results

    def sequence_to_smiles_coords(self, sequence):
        has_coords = not self.continuous_coords
        smiles = ''
        coords, symbols, indices = [], [], []
        i = 0
        while i < len(sequence):
            label = sequence[i]
            if label == EOS_ID or label == PAD_ID:
                break
            if self.is_x(label) or self.is_y(label):
                i += 1
                smiles += f',{label}'
                symbols.append(label)
                continue
            if not self.is_atom(label):
                smiles += self.itos[label]
                i += 1
                continue
            if self.itos[label] == '[':
                j = i + 1
                while j < len(sequence):
                    if not self.is_symbol(sequence[j]):
                        break
                    if self.itos[sequence[j]] == ']':
                        j += 1
                        break
                    j += 1
            else:
                if i+1 < len(sequence) and (self.itos[label] == 'C' and self.is_symbol(sequence[i+1]) and self.itos[sequence[i+1]] == 'l' \
                        or self.itos[label] == 'B' and self.is_symbol(sequence[i+1]) and self.itos[sequence[i+1]] == 'r'):
                    j = i+2
                else:
                    j = i+1
            token = ''.join(self.itos[sequence[k]] for k in range(i, j))
            smiles += token
            if has_coords:
                if not self.swap_token:
                    if j+2 < len(sequence) and self.is_x(sequence[j]) and self.is_y(sequence[j+1]):
                        x = self.id_to_x(sequence[j])
                        y = self.id_to_y(sequence[j+1])
                        coords.append([x, y])
                        symbols.append(token)
                        indices.append(j+2)
                        i = j+2
                    else:
                        i = j
                else:
                    if i-2 >=0 and self.is_x(sequence[i-2]) and self.is_y(sequence[i-1]):
                        x = self.id_to_x(sequence[i-2])
                        y = self.id_to_y(sequence[i-1])
                        coords.append([x, y])
                        symbols.append(token)
                        indices.append(j-1)
                        i = j
                    else:
                        i = j
            else:
                if j < len(sequence):
                    symbols.append(token)
                    indices.append(j)
                i = j
        if not self.atom_only:
            results = {'smiles': smiles, 'symbols': symbols, 'indices': indices}
        else:
            results = {'symbols': symbols, 'indices': indices}
        if has_coords:
            results['coords'] = coords
        return results
       
    def get_token_list(self,smiles,coords=None,mask_ratio=0,use_bbox=False):
        # if smiles is None or type(smiles) is not str:
        #     smiles = ""
        if smiles is None:
            smiles = ""

        label, indices = self.smiles_to_sequence(smiles, coords=coords, mask_ratio=mask_ratio)
        token_list=torch.LongTensor(label[:self.max_len])
        atom_indices = torch.LongTensor([i for i in indices if i < self.max_len])
        if self.continuous_coords:
            if coords is not None:
                coords = torch.tensor(coords)
            else:
                coords = torch.ones(len(indices), 2) * -1.
        return token_list,atom_indices,coords

class AtomTokenizer(NodeTokenizer):

    def __init__(self,path=None, 
                 sep_xy=False, coord_bins=100,continuous_coords=False,
                 max_len=100,mask_ratio=0,debug=False,
                 swap_token=False,atom_only=False,ckpt_path='',use_bbox=False,atom_type_only=True,atom_char_vocab_path='',atom_ocr = False, joint_keypoint_bbox = True, bbox_coords = False, is_cxcywh = False):
        super().__init__(coord_bins, path, sep_xy, continuous_coords, debug,ckpt_path,atom_type_only)
        self.atom_only = atom_only
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.swap_token = swap_token
        self.use_bbox = use_bbox
        self.atom_type_only = atom_type_only
        self.atom_ocr = atom_ocr
        self.joint_keypoint_bbox = joint_keypoint_bbox
        self.bbox_coords = bbox_coords
        self.is_cxcywh = is_cxcywh
        if self.atom_ocr:
            self.char_tokenizer = AtomCharTokenizer(atom_char_vocab_path)

    def smiles_to_sequence(self, smiles, coords=None, bbox=None, mask_ratio=0, atom_only=False):
        
        atom_only=self.atom_only
        if isinstance(smiles,list):
            tokens = smiles
        else:
            tokens = atomwise_tokenizer(smiles)
        labels = [SOS_ID]
        indices = []
        symbols = []
        superatom_indices = []
        superatom_bbox = []
        atom_idx = -1
        for token in tokens:
            if self.is_atom_token(token)==False:
                for c in token:
                    if c in self.stoi:
                        labels.append(self.stoi[c])
                    else:
                        if self.debug:
                            print(f'{c} not in vocab')
                        labels.append(UNK_ID)

            if self.is_atom_token(token):
                atom_idx += 1
                superatom_flag =False
                if not self.continuous_coords:
                    if mask_ratio > 0 and random.random() < mask_ratio:
                        if not self.swap_token:
                            labels.append(MASK_ID)
                            labels.append(MASK_ID)
                        else:
                            labels.insert(-1,MASK_ID)
                            labels.insert(-1,MASK_ID)
                    elif coords is not None:
                        if self.use_bbox:
                            x1, y1, x2, y2 = random.random(),random.random(),None,None
                            if atom_idx < len(coords):
                                x1, y1, x2, y2 = coords[atom_idx]
                                assert 0 <= x1 <= 1
                                assert 0 <= y1 <= 1
                                assert 0 <= x2 <= 1
                                assert 0 <= y2 <= 1
                                if x2-x1<=1e-6 and y2-y1<=1e-6 :
                                    # x2, y2 = None, None
                                    labels.append(self.stoi['[atom]'])
                                else:
                                    superatom_flag = True
                                    labels.append(self.stoi['[superatom]'])
                                    # superatom_bbox.append([self.x_to_id_no_offset(x1),self.y_to_id_no_offset(y1),self.x_to_id_no_offset(x2),self.y_to_id_no_offset(y2)])
                                    if self.is_cxcywh:
                                        superatom_bbox.append([*box_xyxy_to_cxcywh(x1,y1,x2,y2)])
                                    else:
                                        superatom_bbox.append([x1,y1,x2,y2])  # 0-1之间
                                    
                            if not self.swap_token:
                                labels.append(self.x_to_id(x1))
                                labels.append(self.y_to_id(y1))
                                if self.joint_keypoint_bbox:
                                    if superatom_flag or self.bbox_coords:
                                        labels.append(self.x_to_id(x2))
                                        labels.append(self.y_to_id(y2))
                            else:
                                labels.insert(-(len(token)),self.x_to_id(x1))
                                labels.insert(-(len(token)),self.y_to_id(y1))
                                if self.joint_keypoint_bbox:
                                    if superatom_flag or self.bbox_coords:
                                        labels.insert(-(len(token)),self.x_to_id(x2))
                                        labels.insert(-(len(token)),self.y_to_id(y2))
                        else:
                            if atom_idx < len(coords):
                                x, y = coords[atom_idx]
                                assert 0 <= x <= 1
                                assert 0 <= y <= 1
                            else:
                                x = random.random()
                                y = random.random()
                            if not self.swap_token:
                                labels.append(self.x_to_id(x))
                                labels.append(self.y_to_id(y))
                            else:
                                labels.insert(-(len(token)),self.x_to_id(x))
                                labels.insert(-(len(token)),self.y_to_id(y))
                if superatom_flag:
                    superatom_indices.append(len(labels) - 1)
                indices.append(len(labels) - 1)
        labels.append(EOS_ID)
        
        if self.atom_ocr:
            symbol_labels = self.char_tokenizer.texts_to_sequences(symbols,return_tensor=True)
            symbol_labels = pad_sequence(symbol_labels, batch_first=True, padding_value=PAD_ID)
        else:
            symbol_labels = torch.tensor([])
        return labels, indices, symbol_labels, superatom_indices ,superatom_bbox
    
    def sequence_to_smiles(self, sequence):
        has_coords = not self.continuous_coords
        smiles = ''
        coords, symbols, indices, superatom_indices = [], [], [], []
        i = 0
        while i < len(sequence):
            label = sequence[i]
            if label == EOS_ID or label == PAD_ID:
                break
            if self.is_x(label) or self.is_y(label):
                i += 1
                continue
            if not self.is_atom_type(label):
                smiles += self.itos[label]
                i += 1
                continue
            
            j = i+1   
            token = ''.join(self.itos[sequence[k]] for k in range(i, j))
            smiles += token
            
            superatom_flag = False
            if token =='[superatom]':
                superatom_flag = True
            if has_coords:
                if not self.swap_token:
                    if j+2 < len(sequence) and self.is_x(sequence[j]) and self.is_y(sequence[j+1]):
                        x = self.id_to_x(sequence[j])
                        y = self.id_to_y(sequence[j+1])
                        
                        if self.use_bbox:
                            x2,y2=x,y
                            
                            if j+4 < len(sequence) and self.is_x(sequence[j+2]) and self.is_y(sequence[j+3]):
                                x2 = self.id_to_x(sequence[j+2])
                                y2 = self.id_to_y(sequence[j+3])

                            if self.is_cxcywh and superatom_flag:  # TODO这个逻辑有点问题
                                coords.append([*box_cxcywh_to_xyxy(x, y, x2, y2)])
                            else:
                                coords.append([x, y, x2, y2])
                            symbols.append(token)
                            
                            if j+4 < len(sequence) and self.is_x(sequence[j+2]) and self.is_y(sequence[j+3]):
                                indices.append(j+4)
                                if superatom_flag:
                                    superatom_indices.append(j+4)
                                i = j+4
                            else:
                                indices.append(j+2)
                                if superatom_flag:
                                    superatom_indices.append(j+2)
                                    
                                i = j+2
                        else:
                            coords.append([x, y])
                            symbols.append(token)
                            indices.append(j+2)
                            if superatom_flag:
                                superatom_indices.append(j+2)
                            i = j+2
                    else:
                        i = j
                else:  # TODO
                    pass
            else:
                if j < len(sequence):
                    symbols.append(token)
                    indices.append(j)
                i = j
        if not self.atom_only:
            results = {'smiles': smiles, 'symbols': symbols, 'indices': indices, 'superatom_indices':superatom_indices}
        else:
            results = {'symbols': symbols, 'indices': indices, 'superatom_indices':superatom_indices}
        if has_coords:
            results['coords'] = coords
        return results
       
    def get_token_list(self,smiles,coords=None,mask_ratio=0,use_bbox=False):
        if smiles is None:
            smiles = ""

        label, indices, symbol_labels, superatom_indices, superatom_bbox = self.smiles_to_sequence(smiles, coords=coords, mask_ratio=mask_ratio)
        token_list=torch.LongTensor(label[:self.max_len])
        atom_indices = torch.LongTensor([i for i in indices if i < self.max_len])
        superatom_indices = torch.LongTensor([i for i in superatom_indices if i < self.max_len])
        superatom_bbox = superatom_bbox[:len(superatom_indices)]
        
        symbol_labels = symbol_labels.to(torch.int64)[:len(superatom_indices)]
        if self.continuous_coords:
            if coords is not None:
                coords = torch.tensor(coords)
            else:
                coords = torch.ones(len(indices), 2) * -1.
        return token_list,atom_indices,coords,symbol_labels,superatom_indices,superatom_bbox