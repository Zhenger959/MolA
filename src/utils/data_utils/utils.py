'''
Author: Jiaxin Zheng
Date: 2023-09-07 13:36:52
LastEditors: Jiaxin Zheng
LastEditTime: 2024-01-05 14:45:51
Description: 
'''
import pandas as pd
def get_data_line_num(filepath_list:list):
    num=0
    for filepath in filepath_list:
        with open(filepath) as f:
            for line in f:
                num=num+1
        f.close()
        num=num-1
    return num

def get_data_df_by_line(filepath_list,data_line=0):
    num_list=[get_data_line_num([file]) for file in filepath_list]
    file_num=0
    if data_line>0:
        cur_num=0
        for num in num_list:
            file_num=file_num+1
            cur_num=cur_num+num
            if cur_num>=data_line:
                break
    else:
        file_num=len(filepath_list)
    file_list=filepath_list[:file_num]
    if file_num==1:
        if data_line>0:
            data_df=pd.read_csv(file_list[0],nrows=data_line)[['SMILES','file_path']]
        else:
            data_df=pd.read_csv(file_list[0])[['SMILES','file_path']]
    else:
        data_df = pd.concat([pd.read_csv(f) for f in file_list[:-1]])[['SMILES','file_path']]
        last_len=data_line-len(data_df)
        df=pd.read_csv(file_list[-1],nrows=last_len)[['SMILES','file_path']]
        data_df=pd.concat([data_df,df])
    
    return data_df

def get_data_df(filepath_list):
    # num_list=[get_data_line_num([file]) for file in filepath_list]
    
    file_num=len(filepath_list)

    if file_num==1:    
        data_df=pd.read_csv(filepath_list[0])
        if 'smiles' in data_df.columns:
            col_name=['smiles','file_path','file_id']
        else:
            col_name=['SMILES','file_path','image_id']
        # data_df=data_df[col_name]
    else:
        data_df = pd.concat([pd.read_csv(f) for f in filepath_list])
        if 'smiles' in data_df.columns:
            col_name=['smiles','file_path','file_id']
        else:
            col_name=['SMILES','file_path','image_id']
        # data_df=data_df[col_name]
    
    if 'smiles' in col_name:
        data_df=data_df.rename(columns={'smiles':'SMILES','file_id':'image_id','file_path':'file_path'})
    return data_df
    