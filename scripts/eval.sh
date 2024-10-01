###
 # @Author: Jiaxin Zheng
 # @Date: 2023-11-13 14:23:40
 # @LastEditors: Jiaxin Zheng
 # @LastEditTime: 2024-10-01 19:46:43
 # @Description: 
### 


MODEL=paper_explicit_smiles

CKPT_PATH='ckpt/mola.ckpt'

CKPT=$(basename $CKPT_PATH | cut -d'.' -f1)


FILE_LIST=['uspto/test.csv']

# python src/eval.py ckpt_path=$CKPT_PATH task_name=test trainer.devices=1 data.test.non_dynamic.file=$FILE_LIST
python src/eval.py task_name=test trainer.devices=1 data.test.non_dynamic.file=$FILE_LIST