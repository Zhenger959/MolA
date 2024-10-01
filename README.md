<!--
 * @Author: Jiaxin Zheng
 * @Date: 2024-10-01 19:19:48
 * @LastEditors: Jiaxin Zheng
 * @LastEditTime: 2024-10-01 21:09:40
 * @Description: 
-->

# MolA: Optimizing Optical Chemical Structure Recognition with Advanced Metrics and a Comprehensively-Annotated Patent Dataset

This project addresses the limitations of achieving 100% accuracy in Optical Chemical Structure Recognition (OCSR) by focusing on reducing manual labor costs.

Key Contributions:
- **New Metrics**: Introduced Perfectly-Matched Annotation Accuracy (PMAA) to evaluate labor costs for verifying and correcting model outputs.
- **Large-Scale Dataset**: Compiled a comprehensive dataset of 2.5 million annotated patent images, alongside an independent test set, Annotated-USPTO, with 1,325 images.
- **MolA Model**: Our model significantly improves PMAA from 3.85% to 89.74%, surpassing MolScribe and minimizing manual intervention.

MolA effectively addresses issues such as atom misclassification and incorrect bond predictions, providing a robust solution for pharmaceutical OCSR applications by reducing operational burdens and enhancing accuracy.


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
Certainly! Here's the revised version with corrected grammar:

# Data

We provide the Annotated-USPTO dataset, which includes 1,325 images and their corresponding annotation file. Additionally, our large-scale training dataset and its annotation file are available. The corresponding TIFF files can be downloaded from the USPTO website.

# Training

We provide the training code so you can train your model:

```
bash scripts/train.sh
```
 

# Inference

To perform inference using our model, simply run the following command:
```
bash scripts/eval.sh
```