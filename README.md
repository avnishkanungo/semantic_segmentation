# Semantic Segmentation using Munet on MS COCO Dataset

This project implements semantic segmentation using the Munet architecture on the MS COCO dataset. The evaluation is conducted using Mean Squared Error (MSE) Loss, Dice Loss, and a combined Dice-MSE Loss. The implementation is done in Python using PyTorch, with references from the DLStudio library created by Professor Avinash Kak.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Implemenatation](#implementation)

## Introduction
Semantic segmentation is the task of classifying each pixel in an image into a predefined class. This project uses the Munet architecture to perform semantic segmentation on the MS COCO dataset. The model is evaluated using three different loss functions:
1. Mean Squared Error (MSE) Loss
2. Dice Loss
3. Combined Dice-MSE Loss

## Dataset

The MS COCO dataset is used for training and evaluation for the semantic segmentation. You can download the dataset from the official COCO website. Dataset not included as it would be too huge to include in the the github repo.

Inintially we have also used the Purdue5Shapes dataset to test the architecture out.

Once downloaded, ensure the dataset is organized as follows:

data/
└── coco/
    ├── train2017/
    ├── val2017/
    └── annotations/

## Implementation:

1. Initial test on the Purdue5Shapes dataset for checking the operation for the Combined Dice-MSE Loss function was carried out using a pre existing script(from [DLStudio Library](https://engineering.purdue.edu/kak/distDLS/#106) by Prof. Avinash Kak's) called sematic_segmentation.
2. Once the proper operation was confirmed the same was implmenated for the Coco Dataset referenceing the implementation of semantic_segmentation.py script.
3. Complete implemntation documentation can be found in the attached PDF file, with mathematical explanation of dice loss and details on the model architecture and code.






    
