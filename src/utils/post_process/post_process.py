'''
Author: Jiaxin Zheng
Date: 2023-09-04 14:57:22
LastEditors: Jiaxin Zheng
LastEditTime: 2023-09-04 18:17:02
Description: 
'''
import pandas as pd
from src.utils.post_process.chemistry import convert_graph_to_smiles,postprocess_smiles,keep_main_molecule
from src.utils import get_pylogger

log = get_pylogger(__name__)

class BasePostprocessor(object):
    
    def __init__(self,formats,molblock=False,keep_main_molecule=False):
        self.formats=formats
        self.molblock=molblock
        self.keep_main_molecule=keep_main_molecule 
    
    def post_process(self,data_df,predictions):
        pred_df=pd.DataFrame([])
        pred_df['image_id'] = [path.split('/')[-1].split('.')[0] for path in data_df['file_path']]
        
        for format_ in self.formats:
            format_preds = [preds[format_] for preds in predictions]
            df=pd.DataFrame(format_preds)
            pred_df=pd.concat([pred_df,df],axis=1)

        # Construct graph from predicted atoms and bonds (including verify chirality)
        if 'edges' in self.formats:
            
            smiles_list, molblock_list, r_success = convert_graph_to_smiles(
                pred_df['coords'], pred_df['symbols'], pred_df['edges'])
            log.info(f'Graph to SMILES success ratio: {r_success:.4f}')
        
            pred_df['graph_SMILES'] = smiles_list
            if self.molblock:
                pred_df['molblock'] = molblock_list
        
        # Postprocess the predicted SMILES (verify chirality, expand functional groups)
        if 'smiles' in pred_df.columns:
            if 'edges' in self.formats:
                post_smiles_list, _, r_success = postprocess_smiles(
                        pred_df['smiles'], pred_df['coords'], pred_df['symbols'], pred_df['edges'])
            else:
                smiles_list, _, r_success = postprocess_smiles(pred_df['smiles'])
            log.info(f'Postprocess SMILES success ratio: {r_success:.4f}')
            pred_df['post_SMILES'] = post_smiles_list
        
        # Keep the main molecule
        if self.keep_main_molecule:
            if 'graph_SMILES' in pred_df:
                pred_df['graph_SMILES'] = keep_main_molecule(pred_df['graph_SMILES'])
            if 'post_SMILES' in pred_df:
                pred_df['post_SMILES'] = keep_main_molecule(pred_df['post_SMILES'])
        
        return pred_df