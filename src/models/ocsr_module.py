'''
Author: Jiaxin Zheng
Date: 2023-09-01 15:03:01
LastEditors: Jiaxin Zheng
LastEditTime: 2024-01-29 11:38:12
Description: 
'''
import os
import json
import time
from typing import Any, Dict, Tuple
import pandas as pd

import cv2

import torch
from lightning import LightningModule
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.utilities import grad_norm
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torch.cuda.amp import GradScaler,autocast

from src.utils import get_data_df
from src.utils import LossMeter
from src.utils import get_pylogger

from src.utils import add_params_tensorboard_histogram,plot_filter
from src.utils.evaluate_utils import pred_dict2df
from src.utils.train_utils import EpochMeter, get_acc
from src.utils.post_process.eval_utils import get_scores

log = get_pylogger(__name__)

class OCSRLitModule(LightningModule):

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        loss:torch.nn.Module,
        encoder_optimizer: torch.optim.Optimizer = None,
        decoder_optimizer: torch.optim.Optimizer = None,
        encoder_scheduler: torch.optim.lr_scheduler = None,
        decoder_scheduler: torch.optim.lr_scheduler = None,
        aprop_cls_name: list =[],
        freeze_net=[],
        warmup=0.02,
        post_processer=None,
        evaluator=None,
        cache_threshold=2000*1024*1024,
        automatic_optimization= True,
        gradient_clip_val=1.0,
        encoder_updare_freq=1.0,
        fp16=True
    ) -> None:
        """Initialize a `OCSRLitModule`.

        :param encoder: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        self.save_hyperparameters(logger=False,ignore=['loss','encoder','decoder'])
        
        self.automatic_optimization=automatic_optimization
        if self.automatic_optimization==False:
            self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        
        self.encoder=encoder
        self.decoder=decoder

        self.criterion = loss
        
        self.post_processer=post_processer
        self.evaluator=evaluator

        self.val_pred={}
        self.test_pred={}
        self.last_time = time.time()

    def freeze(self):
        for freeze_net in self.hparams.freeze_net:
            if freeze_net=='encoder':
                for param in self.encoder.parameters():
                    param.requires_grad = False
            if freeze_net=='decoder':
                for param in self.decoder.parameters():
                    param.requires_grad = False

    def forward(self, images: torch.Tensor, refs: dict) -> torch.Tensor:
        """
        :param indices: A tensor of atom indices.
        :param images: A tensor of images.
        :param refs: refs.
        :return: dict results.
        """
        features, hiddens = self.encoder(images, refs)
        results = self.decoder(features, hiddens, refs)
        return results
    
    def inference(self, images: torch.Tensor, refs: dict) -> torch.Tensor:
        """
        :param indices: A tensor of atom indices.
        :param images: A tensor of images.
        :param refs: refs.
        :return: dict results.
        """
        # with torch.cuda.amp.autocast(enabled=self.hparams.fp16):
        #     with torch.no_grad():
        #         features, hiddens = self.encoder(images, refs)
        #         batch_preds = self.decoder.decode(features, hiddens, refs)
        with torch.no_grad():
            features, hiddens = self.encoder(images, refs)
            batch_preds = self.decoder.decode(features, hiddens, refs)
        return batch_preds

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        rank=self.trainer.global_rank
        
        epoch=self.current_epoch
        # if epoch%5==0 and rank==0:
        #     layer=self.encoder.transformer.patch_embed.proj
        #     filters, biases=layer.state_dict()['weight'],layer.state_dict()['bias']
        #     plot_filter(filters,self.loggers[0],is_grey=False,split=8,save_dir='train/filters',epoch=epoch)   

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        
        indices, images,refs = batch
        results = self.forward(images,refs)
        loss = self.criterion(results, refs)

        return loss, results, refs
    
    def model_inference_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        
        indices, images,refs = batch
        batch_preds = self.inference(images,refs)

        return indices,batch_preds

    # def _get_decoder_grad_norm(self):

    #     parameters = [param.grad for name, param in self.decoder.named_parameters() if param.grad is not None]
    #     return torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2).item()

    # def on_before_optimizer_step(self, optimizer):
    #     norms = grad_norm(self.encoder, norm_type=2)
    #     self.log_dict(norms)
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        batch_size = batch[1].size(0)
        data_time = time.time() - self.last_time
        s_time = time.time()

        if len(self.hparams.freeze_net)==0:
            encoder_optimizer,decoder_optimizer = self.optimizers()
            encoder_scheduler,decoder_scheduler = self.lr_schedulers()
            
            losses, preds, targets = self.model_step(batch)
            
            for atom_prop_i in self.hparams.aprop_cls_name:
                acc=get_acc(preds,targets,'atom_prop',atom_prop_i)
            
            if self.criterion.loss_weight is not None:
                loss_weight=self.criterion.loss_weight
                loss=sum([loss_weight[k]*v for k,v in losses.items()])
            else:
                loss = sum(losses.values())
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            self.manual_backward(loss)
            
            # clip gradients
            self.clip_gradients(encoder_optimizer, gradient_clip_val=self.hparams.gradient_clip_val, gradient_clip_algorithm="norm")
            self.clip_gradients(decoder_optimizer, gradient_clip_val=self.hparams.gradient_clip_val, gradient_clip_algorithm="norm")
            
            
            if (batch_idx+1)%self.hparams.encoder_updare_freq==0:
                encoder_optimizer.step()
                encoder_scheduler.step()
            
            decoder_optimizer.step()
            decoder_scheduler.step()

        else:
            # encoder_optimizer,decoder_optimizer = self.optimizers()
            # encoder_scheduler,decoder_scheduler = self.lr_schedulers()
            decoder_optimizer = self.optimizers()
            decoder_scheduler = self.lr_schedulers()
            
            # compute loss
            losses, preds, targets = self.model_step(batch)
            
            for atom_prop_i in self.hparams.aprop_cls_name:
                acc=get_acc(preds,targets,'atom_prop',atom_prop_i)
            
            if self.criterion.loss_weight is not None:
                loss_weight=self.criterion.loss_weight
                loss=sum([loss_weight[k]*v for k,v in losses.items()])
            else:
                loss = sum(losses.values())

            decoder_optimizer.zero_grad()
            self.manual_backward(loss)
            
            # clip gradients
            self.clip_gradients(decoder_optimizer, gradient_clip_val=self.hparams.gradient_clip_val, gradient_clip_algorithm="norm")
            
            decoder_optimizer.step()
            decoder_scheduler.step()


        gpu_time = time.time() - s_time
        
        # empty cache
        device=loss.device
        free,_=torch.cuda.mem_get_info(device)
        
        if free<self.hparams.cache_threshold:
            torch.cuda.empty_cache()
                    
        # log
        lr=self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        
        self.log("train/lr", lr, on_step=True, on_epoch=True, prog_bar=True)
        
        for atom_prop_i in self.hparams.aprop_cls_name:
            self.log(f"train/acc/{atom_prop_i}", acc, on_step=False, on_epoch=True, prog_bar=True)
            
        self.log("train/loss/all", loss, on_step=True, on_epoch=True, prog_bar=True)        
        for loss_i in losses.keys():
            self.log(f"train/loss/{loss_i}", losses[loss_i].item(), on_step=True, on_epoch=True, prog_bar=True)
            
        self.log("gpu_time",gpu_time,prog_bar=True,on_step=True)
        self.log("data_time",data_time,prog_bar=True,on_step=True)
        self.last_time = time.time()
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        if self.criterion.loss_weight is not None and self.criterion.loss_weight_freq>0:
            if (self.trainer.current_epoch+1)%self.criterion.loss_weight_freq==0:
                self.criterion.loss_weight['chartok_coords']=self.criterion.loss_weight['chartok_coords']*0.1
            
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        indices,batch_preds = self.model_inference_step(batch)
        
        for idx, preds in zip(indices, batch_preds):
            self.val_pred[idx] = preds 

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        predictions=list(dict(sorted(self.val_pred.items())).values())
        filepath_list=self.trainer.val_dataloaders.dataset.get_file_list()
        data_df=get_data_df(filepath_list)
        data_df.index=list(range(len(data_df)))
        predictions_index=sorted(self.val_pred.keys())
        data_df=data_df.iloc[predictions_index]
        data_df.index=list(range(len(data_df)))
        data_df=data_df.fillna('')
        
        pred_df = pred_dict2df(predictions)
        pred_df['image_id']=data_df['file_path'].apply(lambda x:x.split('/')[-1].split('.')[0])
        # pred_df['SMILES']=data_df['SMILES']
        # pred_df['gt_symbols']=data_df['gt_symbols']
        # pred_df['gt_coords']=data_df['gt_coords']
        # pred_df['gt_edges']=data_df['gt_edges']
        
        if 'smiles_tokens_exp' in data_df.columns:
            data_df['SMILES'] = data_df['smiles_tokens_exp'].apply(lambda x: ''.join(x))
            pred_df['SMILES']=data_df['SMILES']
        else:
            pred_df['SMILES']=data_df['SMILES']
            
        pred_df['gt_symbols']=data_df['symbols']
        pred_df['gt_coords']=data_df['coords']
        pred_df['gt_edges']=data_df['edges']    
        
        try:  # TODO
            pred_df['node_coords'] = pred_df['node_coords'].apply(lambda x_list: [[(x[0] + x[2]) * 0.5, (x[1] + x[3]) * 0.5] for x in x_list])
        except:
            pass
        
        scores = get_scores(pred_df,log)
        for k,v in scores.items():
            self.log(f"val/{k}", v, sync_dist=True, prog_bar=True)
            
        save_dir=os.path.join(self.trainer.log_dir,'test_results')
        if self.trainer.global_rank==0 and not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        epoch=self.current_epoch
        save_file_path = os.path.join(save_dir,f'{epoch}_scores_eval.json')
        with open(save_file_path,'w') as f:
            json.dump(scores,f)        

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        indices,batch_preds = self.model_inference_step(batch)
        
        for idx, preds in zip(indices, batch_preds):
            self.test_pred[idx] = preds

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        predictions=list(dict(sorted(self.test_pred.items())).values())
        filepath_list=self.trainer.test_dataloaders.dataset.get_file_list()
        data_df=get_data_df(filepath_list)
        data_df.index=list(range(len(data_df)))
        predictions_index=sorted(self.test_pred.keys())
        data_df=data_df.iloc[predictions_index]
        data_df.index=list(range(len(data_df)))
        
        scores = None
        pred_df = pred_dict2df(predictions)
        pred_df['image_id']=data_df['file_path'].apply(lambda x:x.split('/')[-1].split('.')[0])
        # print(data_df.columns)
        if 'smiles_tokens_exp' in data_df.columns:
            data_df['SMILES'] = data_df['smiles_tokens_exp'].apply(lambda x: ''.join(x))
            pred_df['SMILES']=data_df['SMILES']
        else:
            pred_df['SMILES']=data_df['SMILES']
        
        
        # pred_df['node_bboxes'] = pred_df['node_coords']
        try:  # TODO
            pred_df['node_coords'] = pred_df['node_coords'].apply(lambda x_list: [[(x[0] + x[2]) * 0.5, (x[1] + x[3]) * 0.5] for x in x_list])
        except:
            pass
        
        
        if 'symbols' in data_df.columns or 'gt_symbols' in data_df.columns:
            pred_df['gt_symbols']=data_df['symbols'] if 'symbols' in data_df.columns else data_df['gt_symbols']
            pred_df['gt_coords']=data_df['coords'] if 'coords' in data_df.columns else data_df['gt_coords']
            pred_df['gt_edges']=data_df['edges'] if 'edges' in data_df.columns else data_df['gt_edges']
                
            scores = get_scores(pred_df,log)
            for k,v in scores.items():
                self.log(f"val/{k}", v, sync_dist=True, prog_bar=True)
            
        save_dir=os.path.join(self.trainer.log_dir,'test_results')
        if self.trainer.global_rank==0 and not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        self.scores = scores
        self.pred_df = pred_df
           
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.
        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        encoder_optimizer=None
        decoder_optimizer=None
        
        train_steps_per_epoch=int(len(self.trainer.datamodule.train_dataloader())/self.trainer.num_devices)
        num_training_steps = train_steps_per_epoch*self.trainer.max_epochs
        num_warmup_steps=int(num_training_steps*self.hparams.warmup)
            
        if not isinstance(self.hparams.encoder_optimizer,str):
            encoder_optimizer = self.hparams.encoder_optimizer(params=self.encoder.parameters())
            encoder_optimizer = self.hparams.encoder_optimizer(params=self.encoder.parameters())
            try:
                encoder_scheduler = self.hparams.encoder_scheduler(optimizer=encoder_optimizer,num_training_steps=num_training_steps,num_warmup_steps=num_warmup_steps)
            except:
                encoder_scheduler = self.hparams.encoder_scheduler(optimizer=encoder_optimizer,num_warmup_steps=num_warmup_steps)
        if not isinstance(self.hparams.decoder_optimizer,str):
            decoder_optimizer = self.hparams.decoder_optimizer(params=self.decoder.parameters())
            try:
                decoder_scheduler = self.hparams.decoder_scheduler(optimizer=decoder_optimizer,num_training_steps=num_training_steps,num_warmup_steps=num_warmup_steps)
            except:
                decoder_scheduler = self.hparams.decoder_scheduler(optimizer=decoder_optimizer,num_warmup_steps=num_warmup_steps)     
            
        if encoder_optimizer is not None and decoder_optimizer is not None : #and self.hparams.freeze_net==[]:
            return {
                "optimizer": encoder_optimizer,
                "lr_scheduler": {
                    "scheduler": encoder_scheduler,
                    "name":"encoder",
                    # "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,}
                },{
                "optimizer": decoder_optimizer,
                "lr_scheduler": {
                    "scheduler": decoder_scheduler,
                    "name":"decoder",
                    # "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },}
        elif  encoder_optimizer is not None:  # 'encoder' not in self.hparams.freeze_net and
            return {"optimizer": encoder_optimizer,"lr_scheduler": {
                                                                        "scheduler": encoder_scheduler,
                                                                        # "monitor": "val/loss",
                                                                        "name":"encoder",
                                                                        "interval": "step",
                                                                        "frequency": 1,
                                                                    },}
        else:
            # if 'decoder' not in self.hparams.freeze_net:
            return {"optimizer": decoder_optimizer,"lr_scheduler": {
                                                                        "scheduler": decoder_scheduler,
                                                                        "name":"decoder",
                                                                        # "monitor": "val/loss",
                                                                        "interval": "step",
                                                                        "frequency": 1,
                                                                    },}

if __name__ == "__main__":
    _ = OCSRLitModule(None, None, None)
