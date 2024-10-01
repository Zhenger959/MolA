import json
from tqdm import tqdm
import argparse
import numpy as np
import multiprocessing
import pandas as pd
import collections

import rdkit
from rdkit import Chem, DataStructs

rdkit.RDLogger.DisableLog('rdApp.*')
from SmilesPE.pretokenizer import atomwise_tokenizer

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.tokenizer.token import UNK
from src.utils.data_utils.process import image_transform
from src.utils.data_utils.augment_utils import get_transforms
from src.utils.train_utils import AverageMeter,MetricMeter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file', type=str, required=True)
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--pred_field', type=str, default='SMILES')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--tanimoto', action='store_true')
    parser.add_argument('--keep_main', action='store_true')
    args = parser.parse_args()
    return args

def replace_coords_to_superatom_bbox(coords,atom_indices,superatom_indices,superatom_bbox):
    if len(coords)==0:
        return []
    coords = np.array(coords) if not isinstance(coords,np.ndarray) else coords
    atom_indices = np.array(atom_indices) if not isinstance(atom_indices,np.ndarray) else atom_indices
    superatom_indices = np.array(superatom_indices) if not isinstance(superatom_indices,np.ndarray) else superatom_indices
    
    if coords.shape[1]==2:
        coords = np.concatenate([coords,coords],axis=1)
        
    superatom_list_indices = np.argwhere(atom_indices == superatom_indices[:, None])[:, 1]
    if len(superatom_list_indices)>0:
        coords[superatom_list_indices] = superatom_bbox
    return coords.tolist()

def pred_dict2df(predictions):
    pred_df = pd.DataFrame([])
    
    chartok_coords_key = []
    if len(predictions)>0:
        pred_key = list(predictions[0].keys())
        
    node_coords = [pred['chartok_coords']['coords'] for pred in predictions]
    node_symbols = [pred['chartok_coords']['symbols'] for pred in predictions]
    atom_scores=[pred['chartok_coords']['atom_scores'] for pred in predictions]
    x_coord_scores=[pred['chartok_coords']['x_coord_scores'] for pred in predictions]
    y_coord_scores=[pred['chartok_coords']['y_coord_scores'] for pred in predictions]
    coord_scores=[pred['chartok_coords']['coord_scores'] for pred in predictions]
    
    edges = [pred['edges']['edges'] if isinstance(pred['edges'],dict) else pred['edges'] for pred in predictions]
    edge_scores=[pred['edges']['edge_scores'] for pred in predictions]
    overall_score=[pred['edges']['overall_score'] for pred in predictions]
    
    pred_df['node_coords']=node_coords
    pred_df['node_symbols']=node_symbols
    pred_df['edges']=edges
    pred_df['atom_scores']=atom_scores
    pred_df['x_coord_scores']=x_coord_scores
    pred_df['y_coord_scores']=y_coord_scores
    pred_df['coord_scores']=coord_scores
    pred_df['edge_scores']=edge_scores
    pred_df['overall_score']=overall_score
    
    pred_df['node_bboxes'] = pred_df['node_coords']
    
    if 'atom_bbox' in pred_key:
        atom_indices =  [pred['chartok_coords']['indices'] for pred in predictions]
        superatom_indices = [pred['chartok_coords']['superatom_indices'] for pred in predictions]
        superatom_bbox = [pred['atom_bbox']['atom_bbox']['atom_bbox'].tolist() for pred in predictions]
        
        pred_df['atom_indices'] = atom_indices
        pred_df['superatom_indices'] = superatom_indices
        pred_df['superatom_bbox'] = superatom_bbox
        
        pred_df['node_bboxes'] = pred_df.apply(lambda x: replace_coords_to_superatom_bbox(x['node_coords'],x['atom_indices'],x['superatom_indices'],x['superatom_bbox']),axis=1)
    
    if len(predictions)>0 and 'edge_infos' in predictions[0].keys():
        # save_columns = save_columns +['edge_orders_scores','edge_displays_scores']
        edge_orders_scores=[pred['edges']['edge_orders_scores'] for pred in predictions]
        edge_displays_scores=[pred['edges']['edge_displays_scores'] for pred in predictions]
        pred_df['edge_orders_scores']=edge_orders_scores
        pred_df['edge_displays_scores']=edge_displays_scores
    return pred_df

def canonicalize_smiles(smiles, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True):
    if type(smiles) is not str or smiles == '':
        return '', False
    if ignore_cistrans:
        smiles = smiles.replace('/', '').replace('\\', '')  # smiles = smiles.replace('/', '-').replace('\\', '-')
    if replace_rgroup:
        tokens = atomwise_tokenizer(smiles)
        for j, token in enumerate(tokens):
            if token[0] == '[' and token[-1] == ']':
                symbol = token[1:-1]
                if symbol[0] == 'R' and symbol[1:].isdigit():
                    tokens[j] = f'[{symbol[1:]}*]'
                elif Chem.AtomFromSmiles(token) is None:
                    tokens[j] = '*'
        smiles = ''.join(tokens)
    try:
        canon_smiles = Chem.CanonSmiles(smiles, useChiral=(not ignore_chiral))
        success = True
    except:
        canon_smiles = smiles
        success = False
    return canon_smiles, success


def convert_smiles_to_canonsmiles(
        smiles_list, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True, num_workers=16):
    # with multiprocessing.Pool(num_workers) as p:
    #     results = p.starmap(canonicalize_smiles,
    #                         [(smiles, ignore_chiral, ignore_cistrans, replace_rgroup) for smiles in smiles_list],
    #                         chunksize=128)
    # canon_smiles, success = zip(*results)
    # return list(canon_smiles), np.mean(success)
    canon_smiles_list=[]
    success=[]
    for smiles, ignore_chiral, ignore_cistrans, replace_rgroup in [(smiles, ignore_chiral, ignore_cistrans, replace_rgroup) for smiles in smiles_list]:
        ps_alpha_ratio=float(sum(1 for char in smiles if char.isalpha()))/len(smiles) if type(smiles) is str and len(smiles)>0 else 0.0
        if smiles=='<invalid>' or ps_alpha_ratio<0.01:
            _canon_smiles=''
            _success=False
        else:
            _canon_smiles, _success=canonicalize_smiles(smiles, ignore_chiral, ignore_cistrans, replace_rgroup)
        canon_smiles_list.append(_canon_smiles)
        success.append(_success)
    return canon_smiles_list,np.mean(success)

def jaccard_similarity(label, pred):
    try:
        counter_a = collections.Counter(label)
        counter_b = collections.Counter(pred)
        intersection = list((counter_a & counter_b).elements())
        return len(intersection)/float(len(label)) if len(label)>0 else 0.0
    except:
        return 0.0

def compute_jaccard_similarities(gold_smiles, pred_smiles):
    similarities=[]
    for gs, ps in zip(gold_smiles, pred_smiles):
        ps_alpha_ratio=float(sum(1 for char in ps if char.isalpha()))/len(ps) if len(ps)>0 else 0.0
        if ps_alpha_ratio<0.01:
            t=0.0
        else:
            t=jaccard_similarity(gs,ps)
        similarities.append(t)
    return similarities

def tanimoto_similarity(smiles1, smiles2):
    try:
        if smiles1=='<invalid>' or smiles2=='<invalid>':
            return 0
        smiles1 = Chem.MolToSmiles(Chem.MolFromSmiles(smiles1))
        smiles2 = Chem.MolToSmiles(Chem.MolFromSmiles(smiles2))
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
        return tanimoto
    except:
        return 0

def compute_tanimoto_similarities(gold_smiles, pred_smiles, num_workers=16):
    # with multiprocessing.Pool(num_workers) as p:
    #     similarities = p.starmap(tanimoto_similarity, [(gs, ps) for gs, ps in zip(gold_smiles, pred_smiles)])
    # return similarities
    
    similarities=[]
    for gs, ps in zip(gold_smiles, pred_smiles):
        ps_alpha_ratio=float(sum(1 for char in ps if char.isalpha()))/len(ps) if len(ps)>0 else 0.0
        if ps_alpha_ratio<0.01:
            t=0
        else:
            t=tanimoto_similarity(gs,ps)
        similarities.append(t)
    return similarities

def is_atom_token(token):
    return token[0].isalpha() or token.startswith("[") or token == '*' or token == UNK
    
def get_atoms_list(smi):
    try:
        atoms_list=atomwise_tokenizer(smi)
    except:
        atoms_list=['']
    atoms_list=[x for x in atoms_list if is_atom_token(x)]
    return atoms_list

class SmilesEvaluator(object):
    def __init__(self, num_workers=16, tanimoto=False):
        self.num_workers = num_workers
        self.tanimoto=tanimoto

    def init_gold_smiles(self,gold_smiles):
        # Chem.MolToSmiles(mol,allBondsExplicit=True)
        self.gold_smiles = gold_smiles
        self.gold_smiles_cistrans, _ = convert_smiles_to_canonsmiles(gold_smiles,
                                                                     ignore_cistrans=True,
                                                                     num_workers=self.num_workers)
        self.gold_smiles_chiral, _ = convert_smiles_to_canonsmiles(gold_smiles,
                                                                   ignore_chiral=True, ignore_cistrans=True,
                                                                   num_workers=self.num_workers)
        self.gold_smiles_cistrans = self._replace_empty(self.gold_smiles_cistrans)
        self.gold_smiles_chiral = self._replace_empty(self.gold_smiles_chiral)
        
    def _replace_empty(self, smiles_list):
        """Replace empty SMILES in the gold, otherwise it will be considered correct if both pred and gold is empty."""
        return [smiles if smiles is not None and type(smiles) is str and smiles != "" else "<empty>"
                for smiles in smiles_list]

    def evaluate(self, pred_smiles, include_details=False):
        results = {}
        if self.tanimoto:
            results['tanimoto'] = np.mean(compute_tanimoto_similarities(self.gold_smiles, pred_smiles))
        # Ignore double bond cis/trans
        pred_smiles_cistrans, _ = convert_smiles_to_canonsmiles(pred_smiles,
                                                                ignore_cistrans=True,
                                                                num_workers=self.num_workers)
        results['canon_smiles'] = np.mean(np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans))
        
        # if self.tanimoto:
        #     results['canon_tanimoto'] = np.mean(compute_tanimoto_similarities(self.gold_smiles_cistrans, pred_smiles_cistrans))
            
        if include_details:
            results['canon_smiles_details'] = (np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans))
        # Ignore chirality (Graph exact match)
        pred_smiles_chiral, _ = convert_smiles_to_canonsmiles(pred_smiles,
                                                              ignore_chiral=True, ignore_cistrans=True,
                                                              num_workers=self.num_workers)
        results['graph'] = np.mean(np.array(self.gold_smiles_chiral) == np.array(pred_smiles_chiral))
        # Evaluate on molecules with chiral centers
        chiral = np.array([[g, p] for g, p in zip(self.gold_smiles_cistrans, pred_smiles_cistrans) if '@' in g])
        results['chiral'] = np.mean(chiral[:, 0] == chiral[:, 1]) if len(chiral) > 0 else -1
        return results,pred_smiles_cistrans,pred_smiles_chiral
    
    def evaluate_jaccard_similarity(self,pred_smiles_list,pred_smiles_cistrans,pred_smiles_chiral):
        results = {}
        results['tanimoto']=np.mean(compute_jaccard_similarities(self.gold_smiles,pred_smiles_list))
        results['canon_smiles']=np.mean(compute_jaccard_similarities(self.gold_smiles_cistrans,pred_smiles_cistrans))
        results['chiral']=np.mean(compute_jaccard_similarities(self.gold_smiles_chiral,pred_smiles_chiral))
        return results
        
    
    def get_all_scores(self,gold_smiles,pred_df):
        
        self.init_gold_smiles(gold_smiles)
        scores={}
        smiles_cistrans={}
        smiles_chiral={}

        if 'smiles' in pred_df.columns:
            s,pred_smiles_cistrans,pred_smiles_chiral=self.evaluate(pred_df['smiles'])
            r=self.evaluate_jaccard_similarity(pred_df['smiles'],pred_smiles_cistrans,pred_smiles_chiral)
            scores.update(s)
            
            scores['jaccard_tanimoto'] = r['tanimoto']
            scores['jaccard_canon_smiles'] = r['canon_smiles']
            scores['jaccard_chiral'] = r['chiral']
            
            smiles_cistrans['SMILES']=pred_smiles_cistrans
            smiles_chiral['SMILES']=pred_smiles_chiral
        if 'post_SMILES' in pred_df.columns:
            post_scores,pred_smiles_cistrans,pred_smiles_chiral = self.evaluate(pred_df['post_SMILES'])
            r=self.evaluate_jaccard_similarity(pred_df['post_SMILES'],pred_smiles_cistrans,pred_smiles_chiral)
            scores['post_jaccard_tanimoto'] = r['tanimoto']
            scores['post_jaccard_canon_smiles'] = r['canon_smiles']
            scores['post_jaccard_chiral'] = r['chiral']
            scores['post_smiles'] = post_scores['canon_smiles']
            scores['post_graph'] = post_scores['graph']
            scores['post_chiral'] = post_scores['chiral']
            scores['post_tanimoto'] = post_scores['tanimoto']
            # scores['post_canon_tanimoto'] = post_scores['canon_tanimoto']
            smiles_cistrans['post_SMILES']=pred_smiles_cistrans
            smiles_chiral['post_SMILES']=pred_smiles_chiral
        if 'graph_SMILES' in pred_df.columns:
            graph_scores,pred_smiles_cistrans,pred_smiles_chiral = self.evaluate(pred_df['graph_SMILES'])
            r=self.evaluate_jaccard_similarity(pred_df['graph_SMILES'],pred_smiles_cistrans,pred_smiles_chiral)
            scores['graph_jaccard_tanimoto'] = r['tanimoto']
            scores['graph_jaccard_canon_smiles'] = r['canon_smiles']
            scores['graph_jaccard_chiral'] = r['chiral']
            scores['graph_smiles'] = graph_scores['canon_smiles']
            scores['graph_graph'] = graph_scores['graph']
            scores['graph_chiral'] = graph_scores['chiral']
            scores['graph_tanimoto'] = graph_scores['tanimoto']
            # scores['graph_canon_tanimoto'] = graph_scores['canon_tanimoto']
            smiles_cistrans['graph_SMILES']=pred_smiles_cistrans
            smiles_chiral['graph_SMILES']=pred_smiles_chiral
        return scores,self.gold_smiles_cistrans,self.gold_smiles_chiral,smiles_cistrans,smiles_chiral
        

class  MolEvaluator(object):
    """
    """
    def __init__(self,continuous_coords,coord_bins=64,dis_methods='L1'):
        self.continuous_coords = continuous_coords
        self.maxx = coord_bins
        self.dis_methods=dis_methods
        
        # self.atoms_TP=AverageMeter()
        # self.atoms_FP=AverageMeter()
        # self.atoms_pred_num=AverageMeter()
        self.atoms_metric=MetricMeter()
        self.edges_metric=MetricMeter()
        
    def evaluate(self):
        pass
    
    def discrete_to_continuous_coords(self,coords):
        coords=np.array([[round(x*(self.maxx-1)),round(y*(self.maxx-1))] for x,y in coords])
        return coords
    
    def _get_pairwise_L1_loss(self,preds,gts):
        dis=np.abs(preds.reshape(-1,1)-gts.reshape(1,-1))
        return dis
         
    def _pairwise_distance(self,preds,gts):        
        if self.continuous_coords:
            pass
        else:
            dis=-1
            preds=self.discrete_to_continuous_coords(preds)
            gts=self.discrete_to_continuous_coords(gts)
            
            # x_pred,y_pred=np.array([x[0] for x in preds]),np.array([x[1] for x in preds])
            # x_gts,y_gts=np.array([x[0] for x in gts]),np.array([x[1] for x in gts])
            
            x_pred,y_pred=preds[:,0],preds[:,1]
            x_gts,y_gts=gts[:,0],gts[:,1]
            
            x_dis=self._get_pairwise_L1_loss(x_pred,x_gts)
            y_dis=self._get_pairwise_L1_loss(y_pred,y_gts)
            dis=(x_dis+y_dis)*0.5
            return dis
    
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


    def calculate_ap_11_point_interp(self,rec, prec, recall_vals=11):
        mrec = []
        # mrec.append(0)
        [mrec.append(e) for e in rec]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e) for e in prec]
        # mpre.append(0)
        recallValues = np.linspace(0, 1, recall_vals)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rhoInterp) / len(recallValues)
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)
        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]
        return [ap, rhoInterp, recallValues, None]
   
    def get_atom_score_by_discrete_coords(self,pred_symbols:list,gt_symbols:list,pred_coords:list,gt_coords:list,pred_atom_scores=None,thr=0):
        '''
            :param pred_symbols:list
            :param gt_symbols:list
            :param pred_coords:list
            :param gt_coords
        '''
        pred_coords=np.array(eval(pred_coords)) if isinstance(pred_coords,str) else np.array(pred_coords)
        gt_coords=np.array(eval(gt_coords)) if isinstance(gt_coords,str) else np.array(gt_coords)
        pred_symbols=np.array(eval(pred_symbols)) if isinstance(pred_symbols,str) else np.array(pred_symbols)
        gt_symbols=np.array(eval(gt_symbols)) if isinstance(gt_symbols,str) else np.array(gt_symbols)
        pred_atom_scores=np.array(eval(pred_atom_scores)) if isinstance(pred_atom_scores,str) else np.array(pred_atom_scores)
        res={}
        
        all_symbols=set(gt_symbols)
        # calculate the metrics of each symbol type    
        for symbol in all_symbols: 
            
            pred_indices=np.where(pred_symbols == symbol)[0]
            gt_indices=np.where(gt_symbols == symbol)[0]
            
            # 1. sort pred by atom scores
            if pred_atom_scores is not None:
                sort_indices=np.argsort(-pred_atom_scores[pred_symbols == symbol])
                pred_indices=pred_indices[sort_indices]
            
            _pred_symbols=pred_symbols[pred_indices]
            _gt_symbols=gt_symbols[gt_indices]
            
            _pred_coords=pred_coords[pred_indices]
            _gt_coords=gt_coords[gt_indices]
            
            pred_num=_pred_coords.shape[0]
            gt_num=_gt_coords.shape[0]
            
            match_idx=[]
            TP_list=[False]*pred_num
            FP_list=[True]*pred_num
            gt_num_list=[gt_num]*pred_num
            
            # 2. stat the TP&FP
            if min(pred_num,gt_num)>0:
                
                # 2.1 get the distance between pred adn gt
                dis=self._pairwise_distance(_pred_coords,_gt_coords)

                # 2.2 match and stat
                for i in range(min(pred_num,gt_num)):
                    
                    # 2.2.1 find the best match each row
                    pred_idx,gt_idx=np.unravel_index(np.argmin(dis), dis.shape)
                    
                    _dis=dis[pred_idx,gt_idx]
                    _pred_symbol=_pred_symbols[pred_idx]
                    _gt_symbol=_gt_symbols[gt_idx]
                    
                    # 2.2.2 marked
                    dis[pred_idx,:]=10000
                    dis[:,gt_idx]=10000
                    
                    # 2.2.3 set flag
                    
                    if _pred_symbol==_gt_symbol:
                        if thr==-1 or _dis<=thr :
                            TP_list[i]=True
                            FP_list[i]=False
                            match_idx.append((pred_indices[pred_idx],gt_indices[gt_idx]))
            
            # 3. cumulative sum
            cumsum_TP=np.cumsum(TP_list)
            cumsum_FP=np.cumsum(FP_list)        
            # FN=gt_num-TP
            
            # 4. get precision, recall and ap
            if pred_num>0:
                TP_val=cumsum_TP[-1]
                FP_val=cumsum_FP[-1]
                # acc=cumsum_TP[-1]/pred_num  # 我们不关心负例，所以acc和precision是等价的
                precision=np.array([TP/(TP+FP) if (TP+FP)>0 else 0.0 for TP,FP in zip(cumsum_TP,cumsum_FP)])
                recall=np.array([TP/gt_num if gt_num>0 else 0.0 for TP,gt_num in zip(cumsum_TP,gt_num_list)])
                [ap, mpre, mrec, ii] = self.calculate_ap_every_point(recall, precision)
                f1=2*precision[-1]*recall[-1]/(precision[-1]+recall[-1]) if (precision[-1]+recall[-1])>0 else 0.0
            else:
                TP_val,FP_val,precision,recall,f1,ap,match_idx=0.0,0.0,[0.0],[0.0],0.0,0.0,[]
            
            # self.atoms_TP.update(TP_val)
            # self.atoms_FP.update(FP_val)
            # self.atoms_pred_num.update(pred_num)
            
            
            res[symbol]={
                # 'acc':acc,
                'precision':precision[-1],
                'recall':recall[-1],
                'f1':f1,
                'AP':ap,
                'match_idx':match_idx
                }
        
        avg_res={
            'precision':0.0,
            'recall':0.0,
            'f1':0.0,
            'AP':0.0,
        }
        
        mAP,m_precision,m_recall,m_f1 = 0.0,0.0,0.0,0.0
        if len(all_symbols) > 0:
            
            for c, r in res.items():
                m_precision += r['precision']
                m_recall += r['recall']
                m_f1 += r['f1']
                mAP += r['AP']
            avg_res['precision'] = m_precision / len(all_symbols)
            avg_res['recall'] = m_recall / len(all_symbols)
            avg_res['f1'] = m_f1 / len(all_symbols)
            avg_res['AP'] = mAP / len(all_symbols)
        mAP = mAP / len(all_symbols)
        self.atoms_metric.update(avg_res)
        return res,avg_res,mAP
        
    def get_atom_score_by_continuous_coords():
        '''
        '''
        pass
    def get_edge_score(self,match_res,pred_edges,gt_edges,pred_edge_scores=None,gt_symbols=None):
        """
        
        """
        res={}
        
        
        pred_edges=np.array(eval(pred_edges)) if isinstance(pred_edges,str) else np.array(pred_edges)
        gt_edges=np.array(eval(gt_edges)) if isinstance(gt_edges,str) else np.array(gt_edges)
        pred_edge_num=float(pred_edges.shape[0]*pred_edges.shape[1])
        gt_edge_num=float(gt_edges.shape[0]*gt_edges.shape[1])
        
        # get gt&pred atom pairs
        pred2gt_atom={}
        for symbol in match_res.keys():
            match_idx=match_res[symbol]['match_idx']
            for pred_atom_idx,gt_atom_idx in match_idx:
                pred2gt_atom[gt_atom_idx]=pred_atom_idx
        gt_atom_idx_list=list(pred2gt_atom.keys())
        pred_atom_idx_list=list(pred2gt_atom.values())
        
        gt_con_edge_atom_idx_list=np.where(gt_edges.sum(axis=0)>0)[0]
        
        for gt_atom_idx in gt_con_edge_atom_idx_list:
            
            # acc=0.0
            precision=0.0
            recall=0.0
            f1=0.0
            
            if gt_atom_idx in gt_atom_idx_list:
                TP=0.0
                pred_atom_idx=pred2gt_atom[gt_atom_idx]
                pred_num=(pred_edges[pred_atom_idx,:]>0).sum()
                gt_num=(gt_edges[gt_atom_idx,:]>0).sum()
                
                gt_a2_list=np.where(gt_edges[gt_atom_idx,:]>0)[0]
                
                for gt_a2 in gt_a2_list:
                    if gt_a2 in gt_atom_idx_list:
                        pred_a2=pred2gt_atom[gt_a2]
                        if pred_edges[pred_atom_idx][pred_a2]==gt_edges[gt_atom_idx][gt_a2]:
                            TP=TP+1

                if pred_num>0:
                    FP=pred_num-TP
                    # acc=TP/pred_num
                    precision=TP/pred_num
                    recall=TP/gt_num
                    f1=2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        

            symbol=gt_symbols[gt_atom_idx] if gt_symbols is not None else gt_atom_idx
            res[symbol]={
                # 'acc':acc,
                'precision':precision,
                'recall':recall,
                'f1':f1    
            }
        
        avg_res={
            'precision':0.0,
            'recall':0.0,
            'f1':0.0,
        }
        
        m_precision,m_recall,m_f1 = 0.0,0.0,0.0
        if len(gt_con_edge_atom_idx_list) > 0:
            for c, r in res.items():
                m_precision += r['precision']
                m_recall += r['recall']
                m_f1 += r['f1']
            avg_res['precision'] = m_precision / len(gt_con_edge_atom_idx_list)
            avg_res['recall'] = m_recall / len(gt_con_edge_atom_idx_list)
            avg_res['f1'] = m_f1 / len(gt_con_edge_atom_idx_list)

        self.edges_metric.update(avg_res)
        
        return res,avg_res


def run_mol_evaluator(pred_file,gt_file):
    pred_df=pd.read_csv(pred_file)#,sep='\t')
    gt_df=pd.read_csv(gt_file)
    
    # Process df
    pred_symbols=pred_df['node_symbols'].values.tolist()
    pred_coords=pred_df['node_coords'].values.tolist()
    pred_edges=pred_df['edges'].values.tolist()
    pred_atom_scores=pred_df['atom_scores'].values.tolist()
    # pred_edge_scores=pred_df['edge_scores'].values.tolist()
    
    gt_symbols = gt_df['gt_symbols'].values.tolist()
    gt_coords = gt_df['gt_coords'].values.tolist()
    gt_edges = gt_df['gt_edges'].values.tolist()
    gt_file_path =  gt_df['file_path'].values.tolist()
    
    # Evaluate
    evaluator=MolEvaluator(continuous_coords=False)
    all_metric=pd.DataFrame([])
    for i in tqdm(range(gt_df.shape[0])): 
        thr=2 if len(gt_symbols[i])<20 else 0.0
            
        atoms_res,atoms_avg_res,mAP=evaluator.get_atom_score_by_discrete_coords(pred_symbols[i],gt_symbols[i],pred_coords[i],gt_coords[i],pred_atom_scores[i],thr)
        edges_res,edges_avg_res=evaluator.get_edge_score(atoms_res,pred_edges[i],gt_edges[i],None,gt_symbols[i])
        
        # print(atoms_avg_res)
        # print(edges_avg_res)
        metrics={}
        metrics['file_path']=gt_file_path[i]
        
        for k,v in atoms_avg_res.items():
            metrics[f'atoms_{k}']=v
        for k,v in edges_avg_res.items():
            metrics[f'edges_{k}']=v
        all_metric=all_metric.append(metrics,ignore_index=True)
        
    print(evaluator.atoms_metric)
    print(evaluator.edges_metric)
    return evaluator.atoms_metric,evaluator.edges_metric,all_metric

if __name__=='__main__':
    
    pred_file=''
    gt_file=''

    run_mol_evaluator(pred_file,gt_file)