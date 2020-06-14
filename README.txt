COVID-19 CT Classification for ECE228-G10

This is the final project code for UCSD ECE228, Group 10.

The dataset is introduced in this paper:
COVID-CT-Dataset: A CT Image Dataset about COVID-19

link:
https://arxiv.org/pdf/2003.13865.pdf

Before start running any scripts, make sure you have unzip the data.

You can do this with the following command:
cd COVID-CT/Images-procesed
unzip CT_COVID.zip
unzip CT_NonCOVID.zip

1. training
To start the training, run train.ipynb. On the first cell, you should be able to find the hints on the options you can set. Choose the one you like, and then just run all cells sequentially, and the training shall begins.

#################################
#################################

Before you start running pruning and tucker decomposition experiment, make sure you have already trained a model and save the statedict. A pretrained statedict is provided at checkpoint/CT/resnet18/ResNet18.pt.

2. pruning
To start pruning experiment, run 

python3 prune.py

3. tucker decomposition
To start tucker decomposition experiment, run 

python3 tucker.py

Members:
Zhifang Zeng
Yuting Jiang
Bo Zhou