import os
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
import pandas as pd

from lightning import LightningDataModule

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence

from src.tokenizer.token import PAD_ID

from src.utils import normalize_nodes,get_pylogger,get_data_line_num
from src.utils.data_utils.process import pad_images
from src.utils.data_utils.constants import REF_RANGES

log = get_pylogger(__name__)

class MolADataset(Dataset): 
    def __init__(self, transform,tokenizer,
                 split,df_len,file,
                 pseudo_coords,is_dynamic=True,re_gen=False,img_dir=None,smi_col='',remove_script_char=False):
        self.transform=transform
        self.tokenizer=tokenizer
        
        self.split=split
        self.pseudo_coords=pseudo_coords
        self.is_dynamic=is_dynamic
        self.re_gen=re_gen
        self.img_dir=img_dir  
        
        self.df_len=df_len
        self.file=file
        
        self.smi_col = smi_col
        self.remove_script_char = remove_script_char
        
        self.df = pd.concat([pd.read_csv(f) for f in self.file])
    
    def __del__(self):
        del self.df
        
    def __len__(self):
        return self.df_len
            
    def image_transform(self, image, coords=[], renormalize=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image, keypoints=coords)
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
    
    def is_script_token(self, token):
        return token in ['\x00', '\x07', '\x0b']
    
    def remove_script_in_token(self,token):
        token = ''.join([t for t in token if not self.is_script_token(t)])
        return token
    
    def getitem(self,idx):   
        try:             
            df = pd.concat([pd.read_csv(f) for f in self.file])
            
            coords_df = df if self.split=='train' and self.is_dynamic ==False else None
            
            smi_col = self.smi_col
            all_smiles = df[smi_col].values if smi_col in df.columns else None
            
            if smi_col=='smiles_tokens_exp':
                all_smiles = [eval(smi_list) for smi_list in all_smiles]        
                if self.remove_script_char:
                    all_smiles = [self.remove_script_in_token(token) for token in all_smiles]
            
            if 'file_id' in df.columns:
                file_paths = df['file_path'].values
        
                file_paths = [os.path.join(self.img_dir,'uspto/imgs', path) for path in df['file_id']]
                
            smiles=all_smiles[idx]
            
            ref={}
            file_path = file_paths[idx]
            image = cv2.imread(file_path)

            if image is None:
                image = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
                log.warning(f'{file_path} not found!')
            if coords_df is not None:
                h, w, _ = image.shape
                coords = np.array(eval(coords_df.loc[idx, 'node_coords']))
                if self.pseudo_coords:
                    coords = normalize_nodes(coords)
                coords[:, 0] = coords[:, 0] * w
                coords[:, 1] = coords[:, 1] * h
                image, coords = self.image_transform(image, coords, renormalize=self.pseudo_coords)
            else:
                image = self.image_transform(image)
                coords = None
            
            if self.split=='train':
                # token
                if coords is not None:
                    token_list,atom_indices,coords=self.tokenizer.get_token_list(smiles,coords,mask_ratio=0)
                else:
                    token_list,atom_indices,coords=self.tokenizer.get_token_list(smiles,coords,mask_ratio=1)
                # edge
                if 'edges' in df.columns:
                    edge_list = eval(df.loc[idx, 'edges'])
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
            
                del df
            
        except Exception as e:
            log.info(e)
        return idx, image,ref
    
    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            log.error(e)
    
class TrainDataset(Dataset):
    def __init__(self, split,data_dir,dynamic, non_dynamic, transform,tokenizer):
        dynamic_df,non_dynamic_df=None,None
        self.d_len,self.nd_len=0,0
        if len(dynamic.file)>0:
            file_list=[os.path.join(data_dir, file) for file in dynamic.file]
            df_len=get_data_line_num(file_list)
            self.d_len=df_len
            self.dynamic_dataset=MolADataset(transform,tokenizer,
                                               split,df_len,file_list,
                                               dynamic.pseudo_coords,dynamic.is_dynamic,
                                               smi_col=dynamic.smi_col,remove_script_char=dynamic.remove_script_char)
            
        
        if len(non_dynamic.file)>0:
            file_list=[os.path.join(data_dir, file) for file in non_dynamic.file]
            df_len=get_data_line_num(file_list)
            # df_len=non_dynamic.len
            self.nd_len=df_len
            self.non_dynamic_dataset=MolADataset(transform,tokenizer,
                                                   split,df_len,file_list,
                                                   non_dynamic.pseudo_coords,non_dynamic.is_dynamic,
                                                   img_dir=non_dynamic.img_dir,smi_col=non_dynamic.smi_col,remove_script_char=non_dynamic.remove_script_char)
    
    def __len__(self):
        return self.d_len + self.nd_len
    
    def __getitem__(self, idx):
        if self.d_len>0 and self.nd_len>0:
            if idx < len(self.dynamic_dataset):
                return self.dynamic_dataset[idx]
            else:
                return self.non_dynamic_dataset[idx - len(self.dynamic_dataset)]
        elif self.nd_len>0:
            return self.non_dynamic_dataset[idx]
        else:
            return self.dynamic_dataset[idx]
    
    def get_file_list(self):
        if self.d_len>0 and self.nd_len>0:
            file_list=self.dynamic_dataset.file+self.non_dynamic_dataset.file
        elif self.nd_len>0:
            file_list=self.non_dynamic_dataset.file
        else:
            file_list=self.dynamic_dataset.file
        return file_list

def bms_collate(batch):
    ids = []
    imgs = []
    try:
        batch = [ex for ex in batch if ex is not None and ex[1] is not None]
    except Exception as e:
        pass
    
    formats = list(batch[0][2].keys())

    seq_formats = [k for k in formats if
                   k in ['atomtok', 'inchi', 'nodes', 'atomtok_coords', 'chartok_coords', 'atom_indices', 'superatom_indices', 'symbol_labels']]
    
    refs = {key: [[], []] for key in seq_formats}
    for ex in batch:
        ids.append(ex[0])
        imgs.append(ex[1])
        ref = ex[2]
        for key in seq_formats:
            refs[key][0].append(ref[key])
            refs[key][1].append(torch.LongTensor([len(ref[key])]))

    # Sequence, token, symbol_labels, padding with 0
    for key in seq_formats:
        # this padding should work for atomtok_with_coords too, each of which has shape (length, 4)
        refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_ID)
        refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)

    for key in ['radical_electrons', 'num_Hs_total', 'num_Hs_implicit', 'valence', 'isotope', 'degree', 'charge', 'atoms_valence', 'atoms_valence_before', 'atoms_valence_after']: # REF_RANGES.keys(): edges 相关的已经是tensor了，所以，进这个循环会报错。一定不会出现其他值的key，就不进这个循环了。
        if key in formats:
            #print(key, np.isin(ex[2][key], list(REF_RANGES[key])).shape, ex[2][key].shape, np.sum(np.isin(ex[2][key], list(REF_RANGES[key])) == False), max(list(REF_RANGES[key])) + 1)
            ex[2][key][np.isin(ex[2][key], list(REF_RANGES[key])) == False] = max(list(REF_RANGES[key])) + 1 # isin的第二个参数不能是set，转成list可以

    # Atoms, superatom_bbox float, padding with -1.
    for coords_type in ['coords', 'rect_point_min', 'rect_point_max','superatom_bbox']:
        if coords_type in formats:
            refs[coords_type] = pad_sequence([torch.tensor(ex[2][coords_type]) if torch.tensor(ex[2][coords_type]).shape[0] > 0 else torch.tensor([[-1.,-1.,-1.,-1.]]) for ex in batch], batch_first=True, padding_value=-1.)
        #print(coords_type, refs[coords_type].shape)
    # Atoms, int, padding with -1
    for atom_info_type in ['rect_rtv_pos', 'is_R_or_super_atom', 'radical_electrons', 'num_Hs_total', 'num_Hs_implicit', 'isotope', 'degree', 'is_arom', \
                           'valence', 'charge', 'atoms_valence', 'atoms_valence_before', 'atoms_valence_after']:
        if atom_info_type in formats:
            refs[atom_info_type] = pad_sequence([torch.tensor(ex[2][atom_info_type]) for ex in batch], batch_first=True, padding_value=-1)
            #print(atom_info_type, refs[atom_info_type].shape, type(refs[atom_info_type][0][0].item()))
    # Edges, int, padding with -100
    for bond_info_type in ['edges', 'edges_on_ring', 'edges_valence', 'edge_orders', 'edge_displays']: # 非padding取值 >= 0
        if bond_info_type in formats:
            # edges_list = [torch.tensor(ex[2][bond_info_type]) for ex in batch]
            edges_list = [torch.tensor(ex[2][bond_info_type]) if isinstance(ex[2][bond_info_type], np.ndarray) else ex[2][bond_info_type] for ex in batch]
            max_len = max([len(edges) for edges in edges_list])
            refs[bond_info_type] = torch.stack(
                [F.pad(edges, (0, max_len - len(edges), 0, max_len - len(edges)), value=-100) for edges in edges_list],
                dim=0)
    
    return ids, pad_images(imgs), refs 
            
class MolADataModule(LightningDataModule):
    """`LightningDataModule` for the OCSR dataset.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers,
        pin_memory,
        train,
        val,
        test,
        indigo_redis,
        patent_redis,
        tokenizer
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed within a single process on CPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
           there is a barrier in between which ensures that all the processes proceed to
           `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            
            train=self.hparams.train
            val=self.hparams.val
            test=self.hparams.test
            indigo_redis=self.hparams.indigo_redis
            patent_redis=self.hparams.patent_redis
            tokenizer=self.hparams.tokenizer
            
            self.data_train=TrainDataset(train.split,train.data_dir,train.dynamic, train.non_dynamic, train.transform,tokenizer,indigo_redis,patent_redis)
            log.info(f"TrainDataset Load: # {len(self.data_train)}")
            self.data_val=TrainDataset(val.split,val.data_dir,val.dynamic, val.non_dynamic, val.transform,tokenizer,indigo_redis,patent_redis)
            log.info(f"ValDataset Load: # {len(self.data_val)}") 
            self.data_test=TrainDataset(test.split,test.data_dir,test.dynamic, test.non_dynamic, test.transform,tokenizer,indigo_redis,patent_redis)
            log.info(f"TestDataset Load: # {len(self.data_test)}") 

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=bms_collate
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=bms_collate
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=bms_collate
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = MolADataModule()
