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
To start training, run

python3 train.py

At the very begining of the train.py, you should be able to find hints that indicates the settings you can change.
By default, the train.py will train on resnet18, applying transfer learning.

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