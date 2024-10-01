<!--
 * @Author: Jiaxin Zheng
 * @Date: 2024-10-01 19:19:48
 * @LastEditors: Jiaxin Zheng
 * @LastEditTime: 2024-10-01 19:56:53
 * @Description: 
-->

# MolA
Numerous pharmaceutical applications critically depend on the precise recognition of Optical Chemical Structures (OCSR) extracted from 2D image data. This study tackles the challenge of sub-optimal OCSR accuracy by shifting the focus from achieving perfect recognition to reducing the manual labor required for verification and correction. We introduce a novel set of metrics to quantify the labor cost associated with refining model predictions. Among these, the most demanding is the Perfectly-Matched Annotation Accuracy (PMAA), which demands exactitude not only in atom text but also in atom coordinates and bond types. A pivotal component of our research is the development and deployment of a comprehensively annotated training dataset consisting of 2,504,937 patent images. Complementarily, we prepare an independent test dataset, Annotated-USPTO, encompassing 1,325 unique patent images for unbiased evaluation. Our proposed model, MolA, prioritizes the reduction of manual intervention while maintaining robust OCSR performance. In comparative analysis, MolA significantly enhances the PMAA metric from 3.85% to 89.74% on the Annotated-USPTO dataset, surpassing the current state-of-the-art OCSR model, MolScribe. This study demonstrates that MolA, optimized with a high-quality training dataset, represents a substantial advancement in aligning OCSR technology with practical pharmaceutical applications.


# Quick Start
```
# clone project
git clone https://github.com/Zhenger959/MolA.git
cd MolA

# create conda environment
conda create -n mola python=3.9
conda activate mola

# install requirements
pip install -r requirements.txt
```