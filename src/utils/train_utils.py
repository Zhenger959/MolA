'''
Author: Jiaxin Zheng
Date: 2023-08-31 14:25:23
LastEditors: Jiaxin Zheng
LastEditTime: 2024-10-01 16:02:21
Description: 
'''
import importlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import tensorboardX as tb
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import average_precision_score

def to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if type(data) is list:
        return [to_device(v, device) for v in data]
    if type(data) is dict:
        return {k: to_device(v, device) for k, v in data.items()}

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class APMeter(object):
    def __init__(self,ap_mode='calculate_ap_every_point',return_ap=True):
        self.ap_mode = ap_mode
        self.return_ap = return_ap
        self.reset()

    def reset(self):
        self.tp_list = []
        self.fp_list = []
        self.pred_num = 0.0
        self.gt_num = 0.0
        self.confidence = []

    def _sort_val_by_confidence(self):
        self.tp_list = np.array(self.tp_list)
        self.fp_list = np.array(self.fp_list)
        order_indices = np.argsort(-self.confidence)
        self.tp_list = self.tp_list[order_indices]
        self.fp_list = self.fp_list[order_indices]
    
    def update(self, tp_list, fp_list,  pred_num, gt_num, confidence=None):
        print(f'tp_list: {sum(tp_list)} ----{len(tp_list)}')
        if pred_num > 0:
            cumsum_TP=np.cumsum(tp_list)
            cumsum_FP=np.cumsum(fp_list)  
            precision_list=[TP/(TP+FP) if (TP+FP)>0 else 0.0 for TP,FP in zip(cumsum_TP,cumsum_FP)]
            recall_list=[TP/gt_num if gt_num>0 else 0.0 for TP in cumsum_TP]
            f1  = 2*precision_list[-1]*recall_list[-1]/(precision_list[-1]+recall_list[-1])if (precision_list[-1]+recall_list[-1]) > 0 else 0.0
            single_res = {
                    'precision':precision_list[-1],
                    'recall':recall_list[-1],
                    'f1':f1,
            } 
            if self.return_ap:
                ap = self.calculate_ap_every_point(precision_list,recall_list)[0]
                single_res['ap']=ap
                # print(f'ap: {ap} precision:{precision_list[-1]} recall:{recall_list[-1]}')
        else:
            single_res = {
                    'precision':0.0,
                    'recall':0.0,
                    'f1':0.0,
            } 
            if self.return_ap:
                single_res['ap']=0.0
                
        self.tp_list = np.concatenate((self.tp_list , tp_list))
        self.fp_list = np.concatenate((self.fp_list , fp_list))
        
        self.pred_num = self.pred_num + pred_num
        self.gt_num = self.gt_num + gt_num
        
        if self.return_ap and confidence is not None:
            if len(self.confidence) > 0 and len(confidence)>0:
                self.confidence = np.concatenate((self.confidence ,confidence))
            else:
                self.confidence = confidence
        
        print(f'update tp_list: {sum(self.tp_list)} ----{len(self.tp_list)}')
        return single_res
    
    # def calculate_ap_every_point(self,rec, prec):

    #     mrec,mpre = rec,prec
    #     mrec = np.append(mrec,1)
    #     mpre = np.append(mpre,0)
        
    #     for i in range(len(mpre) - 1, 0, -1):
    #         mpre[i - 1] = max(mpre[i - 1], mpre[i])
        
    #     ii = []
    #     for i in range(len(mrec) - 1):
    #         if mrec[1:][i] != mrec[0:-1][i]:
    #             ii.append(i + 1)
    #     ap = 0
    #     for i in ii:
    #         ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    #     return ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii
    
    def calculate_ap_every_point(self,rec, prec):
        
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]
      
    def get_metric_val(self,return_ap = True):
        res={}
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        ap = 0.0
        if len(self.confidence) > 0:
            self._sort_val_by_confidence()
        
        # 1. cumulative sum
        cumsum_TP=np.cumsum(self.tp_list)
        cumsum_FP=np.cumsum(self.fp_list)  
        
        # 2. get precision, recall and ap
        if self.pred_num>0:
            TP_val=cumsum_TP[-1]
            FP_val=cumsum_FP[-1]
            precision = TP_val/self.pred_num if self.pred_num > 0 else 0.0
            recall = TP_val/self.gt_num if self.gt_num > 0 else 0.0
            f1  = 2*precision*recall/(precision+recall)if (precision+recall) > 0 else 0.0
            
            if self.return_ap:
                precision_list=[TP/(TP+FP) if (TP+FP)>0 else 0.0 for TP,FP in zip(cumsum_TP,cumsum_FP)]
                recall_list=[TP/self.gt_num if self.gt_num>0 else 0.0 for TP in cumsum_TP]
                ap = self.calculate_ap_every_point(precision_list,recall_list)[0]
        res = {
                'precision':precision,
                'recall':recall,
                'f1':f1,
                
        } 
        if self.return_ap:
            res['ap']=ap
        return res    
        

class EpochMeter(AverageMeter):
    def __init__(self):
        super().__init__()
        self.epoch = AverageMeter()

    def update(self, val, n=1):
        super().update(val, n)
        self.epoch.update(val, n)

class LossMeter(EpochMeter):
    def __init__(self):
        self.subs = {}
        super().__init__()

    def reset(self):
        super().reset()
        for k in self.subs:
            self.subs[k].reset()

    def update(self, loss, losses, n=1):
        loss = loss.item()
        super().update(loss, n)
        losses = {k: v.item() for k, v in losses.items()}
        for k, v in losses.items():
            if k not in self.subs:
                self.subs[k] = EpochMeter()
            self.subs[k].update(v, n)

class MetricMeter(AverageMeter):
    def __init__(self):
        self.subs = {}
        super().__init__()

    def reset(self):
        super().reset()
        for k in self.subs:
            self.subs[k].reset()
    def update(self, metrics, n=1):
        
        for k, v in metrics.items():
            if k not in self.subs:
                self.subs[k] = AverageMeter()
            self.subs[k].update(v, n)
    def print_metrics(self):
        templ = "%-15s %15s"
        for k, v in self.subs.items():
            print(templ%(k.upper(),v.avg))
            
    def get_df(self):
        metric_list=[k.upper() for k in self.subs.keys()]
        val_list=[v.avg for v in self.subs.values()]
        df=pd.DataFrame([])
        df['Metric']=metric_list
        df['Val']=val_list
        return df
            
def get_acc(preds,targets,format_='atom_prop',atom_prop_i='charge',ignore_index=-1):
    # format_='atom_prop'
    # atom_prop_i='atoms_valence_before'
    
    predictions=preds[format_][0][format_][atom_prop_i]
    predictions=[np.argmax(x.detach().cpu(),axis=-1).flatten() for x in predictions]
    labels=targets[atom_prop_i].cpu()
    
    batch_size = labels.size(0)
    labels_list = [labels[i] for i in range(labels.shape[0])]
    
    pad_pl=pad_sequence(predictions+labels_list, batch_first=True, padding_value=-1)
    
    predictions_pad=pad_pl[:batch_size]
    labels_pad=pad_pl[batch_size:]
    
    mask = (labels_pad != -1)
    correct=(predictions_pad[mask] == labels_pad[mask]).sum().item()
    acc=correct/mask.sum().item()
    return acc

def add_params_tensorboard_histogram(model,global_step,logger,log):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            log.warning(f"{name}_grad is nan")
        logger.experiment.add_histogram(name, param, global_step)
        if param.requires_grad and param.grad is not None:
            logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)
 
def plot_filter(filters,writer,is_grey=True,split=8,save_dir='',epoch=0):
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    n_filters,n_channels,filter_size_1,filter_size_2=filters.shape
    m=int(n_filters/split)
    if is_grey:
        for r in range(m):
            ix=1
            fig  =  plt.figure(figsize=(8,5))
            for i in range(r*split,(r+1)*split):
                f = filters[i, :, :, :]
                # plot each channel separately
                for j in range(n_channels):
                    ax = plt.subplot(split, 3, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plot filter channel in grayscale
                    plt.imshow(f[j,:, :].cpu(), cmap='gray')
                    ix=ix+1
            plt.tight_layout()
            fig.canvas.draw()
            writer.add_image(f'{save_dir}/{str(r)}', tb.utils.figure_to_image(fig), epoch)

    else:
        for r in range(m):
            ix=1
            fig  =  plt.figure(figsize=(8,5))
            for i in range(r*split,(r+1)*split):
                f = filters[i, :, :, :]
                ax = plt.subplot(split, 1, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(f.cpu())
                ix=ix+1

            plt.tight_layout()
            fig.canvas.draw()
            writer.experiment.add_image(f'{save_dir}/{str(r)}', tb.utils.figure_to_image(fig), epoch)  
    
# def create_loss(hypes):
#     """Create the loss function based on the given loss name.
    
#     :param hypes:dict Configuration params for training.
#     :return: The loss function.
    
#     Examples: focal_loss.py:FocalLoss
#     """
#     loss_func_name = hypes['loss']['core_method']
#     target_loss_name = hypes['loss']['class']
#     loss_func_config = hypes['loss']['args']

#     loss_filename = "src.loss." + loss_func_name
#     loss_lib = importlib.import_module(loss_filename)
#     loss_func = None
    

#     for name, lfunc in loss_lib.__dict__.items():
#         if name.lower() == target_loss_name.lower():
#             loss_func = lfunc

#     if loss_func is None:
#         print('loss function not found in loss folder. Please make sure you '
#               'have a python file named %s and has a class '
#               'called %s ignoring upper/lower case' % (loss_filename,
#                                                        target_loss_name))
#         exit(0)

#     criterion = loss_func(loss_func_config)
#     return criterion