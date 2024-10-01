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
from src.utils import Indigo,IndigoRenderer
from src.utils import normalize_nodes,get_pylogger,get_data_line_num
from src.utils.data_utils.process import pad_images

log = get_pylogger(__name__)

class FileDataset(Dataset):
    def __init__(self, transform,tokenizer,split,file,df,pseudo_coords,is_dynamic=True,re_gen=False,img_dir=None):
        self.transform=transform
        self.tokenizer=tokenizer
        
        self.split=split
        self.pseudo_coords=pseudo_coords
        self.is_dynamic=is_dynamic
        self.re_gen=re_gen
        self.img_dir=img_dir  
        
        # File
        self.file=file
        self.df=df
        self.coords_df = df if split=='train' and is_dynamic ==False else None
        if 'smiles' in self.df.columns:
            smiles_col='smiles' 
        elif 'SMILES' in self.df.columns:
            smiles_col='SMILES' 
        else:
            smiles_col='smiles_tokens_exp'
            
        # smiles_col='smiles' if 'smiles' in self.df.columns else 'SMILES'
        self.smiles = self.df[smiles_col].values if smiles_col in self.df.columns else None 
        
        # if smiles_col=='smiles_tokens_exp':
        #     self.smiles = [''.join(smi_list) for smi_list in self.smiles]

        if smiles_col=='smiles_tokens_exp':
            self.smiles = [eval(smi_list) for smi_list in self.smiles]
            
        if 'file_path' in self.df.columns:
            self.file_paths = self.df['file_path'].values
            self.file_paths = [os.path.join(img_dir, path) for path in self.df['file_path']]
        
    def __len__(self):
        return len(self.df)
            
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
      
    def get_file_item(self,idx,smiles):
        # TODO by Ma Rong: 增加原子信息类别
        ref={}
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        if image is None:
            image = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
            log.warning(f'{file_path} not found!')
        # else:
        #     log.info(image.shape)
        #     h, w = image.shape[:2]
        #     w_pad,h_pad=int(w*0.05),int(h*0.05)
        #     image=cv2.copyMakeBorder(image,h_pad,h_pad,w_pad,w_pad,cv2.BORDER_CONSTANT,value=(255, 255, 255))
        #     log.info(image.shape)
        if self.coords_df is not None:
            h, w, _ = image.shape
            coords = np.array(eval(self.coords_df.loc[idx, 'node_coords']))
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
            if 'edges' in self.df.columns:
                edge_list = eval(self.df.loc[idx, 'edges'])
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
        
        return image,ref
        
    def getitem(self,idx):
        smiles=self.smiles[idx]

        image, ref=self.get_file_item(idx,smiles)
            
        return idx,image,ref
    
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
            dynamic_df = pd.concat([pd.read_csv(os.path.join(data_dir, file)) for file in dynamic.file])
            self.dynamic_dataset=FileDataset(transform,tokenizer,split,file_list,dynamic_df,dynamic.pseudo_coords,dynamic.is_dynamic,re_gen=dynamic.re_gen)
            self.d_len=len(self.dynamic_dataset)
        
        if len(non_dynamic.file)>0:
            file_list=[os.path.join(data_dir, file) for file in non_dynamic.file]
            df_len=get_data_line_num(file_list)
            non_dynamic_df = pd.concat([pd.read_csv(os.path.join(data_dir, file)) for file in non_dynamic.file])
            self.non_dynamic_dataset=FileDataset(transform,tokenizer,split,file_list,non_dynamic_df,non_dynamic.pseudo_coords,non_dynamic.is_dynamic,img_dir=non_dynamic.img_dir)
            self.nd_len=len(self.non_dynamic_dataset)
    
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
        # TODO:这个地方要改进一些
        if self.d_len>0 and self.nd_len>0:
            file_list=self.dynamic_dataset.file+self.non_dynamic_dataset.file
        elif self.nd_len>0:
            file_list=self.non_dynamic_dataset.file
        else:
            file_list=self.dynamic_dataset.file
        return file_list
    
    def get_data_df(self):
        data_df=pd.DataFrame([])
        if self.d_len>0 and self.nd_len>0:
            data_df = pd.concat([self.dynamic_dataset.df[['SMILES','file_path']],self.non_dynamic_dataset.df[['file_path']]],axis=0)
        elif self.nd_len>0:
            data_df = self.non_dynamic_dataset.df[['SMILES','file_path']]
        else:
            data_df = self.dynamic_dataset.df[['SMILES','file_path']]
        return data_df

def bms_collate(batch):
    ids = []
    imgs = []
    try:
        batch = [ex for ex in batch if ex is not None and ex[1] is not None]  # TODO 这种情况下indigo会有返回None的情况
    except Exception as e:
        pass
        
    
    formats = list(batch[0][2].keys())
    seq_formats = [k for k in formats if
                   k in ['atomtok', 'inchi', 'nodes', 'atomtok_coords', 'chartok_coords', 'atom_indices', \
                       'num_Hs_total', 'num_Hs_implicit', 'valence', 'degree']]
    
    refs = {key: [[], []] for key in seq_formats}
    for ex in batch:
        ids.append(ex[0])
        imgs.append(ex[1])
        ref = ex[2]
        for key in seq_formats:
            refs[key][0].append(ref[key])
            refs[key][1].append(torch.LongTensor([len(ref[key])]))
    # Sequence
    for key in seq_formats:
        # this padding should work for atomtok_with_coords too, each of which has shape (length, 4)
        refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_ID)
        refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)

    # Coords
    if 'coords' in formats:
        refs['coords'] = pad_sequence([torch.tensor(ex[2]['coords']) for ex in batch], batch_first=True, padding_value=-1.)
    # Edges
    if 'edges' in formats:
        edges_list = [ex[2]['edges'] for ex in batch]
        max_len = max([len(edges) for edges in edges_list])
        refs['edges'] = torch.stack(
            [F.pad(edges, (0, max_len - len(edges), 0, max_len - len(edges)), value=-100) for edges in edges_list],
            dim=0)
    return ids, pad_images(imgs), refs
            
class OCSRDataModule(LightningDataModule):
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
            tokenizer=self.hparams.tokenizer
            
            self.data_train=TrainDataset(train.split,train.data_dir,train.dynamic, train.non_dynamic, train.transform,tokenizer)
            log.info(f"TrainDataset Load: # {len(self.data_train)}")
            self.data_val=TrainDataset(val.split,val.data_dir,val.dynamic, val.non_dynamic, val.transform,tokenizer)
            log.info(f"ValDataset Load: # {len(self.data_val)}") 
            self.data_test=TrainDataset(test.split,test.data_dir,test.dynamic, test.non_dynamic, test.transform,tokenizer)
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
            num_workers=0,
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
    _ = OCSRDataModule()
