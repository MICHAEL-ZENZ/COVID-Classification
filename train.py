
from enum import Enum

ModelOptions=Enum('ModelOptions',('densenet121','densenet169','resnet18','resnet50','vgg16'))

###################
# Here is the settings you can specify
pretrained=True
modelname = ModelOptions.resnet18
###################


from model import ResNet
from utils import CovidCTDataset,metrics

from torch.utils.data import DataLoader
from torchvision import models
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import torch.nn as nn
import os
import random
from torchvision import transforms
import numpy as np
from sklearn.metrics import roc_auc_score
import easydict
import matplotlib.pyplot as plt


args = easydict.EasyDict({
    'model_name':modelname.name,
    'checkpoint_path':'./checkpoint/CT',
    'batch_size':16,
    'lr':1e-4,
    'epoch':50,
    'root_dir':'./COVID-CT/Images-processed',

    'train_COV':'./COVID-CT/Data-split/COVID/trainCT_COVID.txt',
    'train_NonCOV':'./COVID-CT/Data-split/NonCOVID/trainCT_NonCOVID.txt',

    'val_COV':'./COVID-CT/Data-split/COVID/valCT_COVID.txt',
    'val_NonCOV':'./COVID-CT/Data-split/NonCOVID/valCT_NonCOVID.txt',

    'test_COV':'./COVID-CT/Data-split/COVID/testCT_COVID.txt',
    'test_NonCOV':'./COVID-CT/Data-split/NonCOVID/testCT_NonCOVID.txt',

    'pretrained':pretrained,
    'save_name':modelname.name+'.pt'
})

MODEL_DICT = {
    'densenet121': models.densenet121,
    'densenet169': models.densenet169,
    'resnet18': ResNet.resnet18,
    'resnet50': ResNet.resnet50,
    'vgg16': models.vgg16,
}


def train(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, device):
    model.train()
    loss_func = nn.CrossEntropyLoss()
    for index, batch in enumerate(tqdm(train_loader)):
        img, label = batch['img'].to(device), batch['label'].to(device)
        output = model(img)
        optimizer.zero_grad()
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()

def test(model, nb_classes, test_loader, device):
    model.eval()
    predlist = []
    targetlist = []
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    avg_val_loss = 0
    with torch.no_grad():
        for index, batch in enumerate(tqdm(test_loader)):
            img, label = batch['img'].to(device), batch['label'].to(device)
            output = model(img)
            _, preds = torch.max(output, 1)
            avg_val_loss += F.cross_entropy(output, label).item()/len(test_loader)
            for t, p in zip(label.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            y_score = F.softmax(output, dim=1)
            predlist = np.append(predlist, y_score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, label.long().cpu().numpy())

    AUC = roc_auc_score(targetlist, predlist)
    print(confusion_matrix)
    precision = metrics.Precision(confusion_matrix.cpu().numpy(), nb_classes)
    recall = metrics.Recall(confusion_matrix, nb_classes)
    f1 = metrics.F1(precision, recall)
    acc = metrics.Acc(confusion_matrix,nb_classes)
    tprecision=metrics.TPrecision(confusion_matrix)
    trecall=metrics.TRecall(confusion_matrix)
    return AUC, precision, recall, f1, acc, avg_val_loss,tprecision,trecall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device {}".format(device))
# Create checkpoint file
save_path = os.path.join(args.checkpoint_path, args.model_name)
if os.path.exists(save_path) == False:
    os.makedirs(save_path)

# Here perform the data augmentation

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
train_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale = (0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
val_trans = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

trainset = CovidCTDataset(root_dir=args.root_dir,
                            txt_COVID=args.train_COV,
                            txt_NonCOVID=args.train_NonCOV,
                            transform=train_trans
                            )
valset = CovidCTDataset(root_dir=args.root_dir,
                        txt_COVID=args.val_COV,
                        txt_NonCOVID=args.val_NonCOV,
                            transform=val_trans
                            )

testset = CovidCTDataset(root_dir=args.root_dir,
                            txt_COVID=args.test_COV,
                            txt_NonCOVID=args.test_NonCOV,
                            transform=val_trans
                            )

train_loader = DataLoader(trainset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            shuffle=True)
val_loader = DataLoader(valset, batch_size=args.batch_size)
test_loader = DataLoader(testset, batch_size=args.batch_size)

PRINT_INTERVAL = 10
nb_classes = 2
seg_num_class = 2
print(args.model_name,trainset.classes)


model = MODEL_DICT[args.model_name](num_classes=nb_classes, pretrained=args.pretrained)
# model = Deeplabv3.DeeplabVV3(model, num_class=seg_num_class)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model).to(device)
    print("Using %d GPUs"%torch.cuda.device_count())
elif torch.cuda.device_count()==1:
    model.to(device)
    print("Using 1 GPU")
elif torch.cuda.is_available():
    print("WARNING: GPU detected but cannot use")
else:
    print("Using CPU")


optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
sheduler = torch. optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

maxAcc=0
save = os.path.join(save_path,'{}'.format(args.save_name))

train_precisions=[]
train_recalls=[]
train_accs=[]
train_AUCs=[]

test_precisions=[]
test_recalls=[]
test_accs=[]
test_AUCs=[]

pltInterval=5
for epoch in range(args.epoch):
    train(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, device)

    AUC, precision, recall, f1, acc, mean_loss, tprecision, trecall\
       = test(model, nb_classes, train_loader, device)

    if epoch%pltInterval==0:
      train_precisions.append(tprecision)
      train_recalls.append(trecall)
      train_accs.append(acc)
      train_AUCs.append(AUC)

    AUC, precision, recall, f1, acc, mean_loss, tprecision, trecall\
       = test(model, nb_classes, val_loader, device)
    maxAcc=max(maxAcc,acc)
    print('Precision {}\tRecall {}\nF1 {}\nAUC {}\tAcc {}\tMean Loss {}'.format(precision, recall, f1, AUC, acc, mean_loss))
    
    if epoch%pltInterval==0:
      test_precisions.append(tprecision)
      test_recalls.append(trecall)
      test_accs.append(acc)
      test_AUCs.append(AUC)
      
    if maxAcc == acc:
        torch.save(model.state_dict(), save)
        print("saved")

    sheduler.step(epoch)

print('...........Testing..........')
model.load_state_dict(torch.load(save))
AUC, precision, recall, f1, acc, mean_loss, tprecision, trecall = test(model, nb_classes, test_loader, device)

print('Precision {}\tRecall {}\nF1 {}\nAUC {}\tAcc {}\tMean Loss {}'.format(precision, recall, f1, AUC, acc,
                                                                            mean_loss))

plt.figure(figsize=(15,12))
plt.subplot(221)
plt.plot(train_accs)
plt.plot(train_AUCs)
plt.title('train accuracy & AUC vs epoch')
plt.ylabel('accuracy')
plt.xlabel('10 epoch')
plt.legend(['Accuracy', 'AUC'], loc='upper left')


plt.subplot(222)
plt.plot(train_recalls)
plt.plot(train_precisions)
plt.title('train precision & recall vs epoch')
plt.ylabel('percentage')
plt.xlabel('10 epoch')
plt.legend(['recall', 'precision'], loc='upper left')

plt.subplot(223)
plt.plot(test_accs)
plt.plot(test_AUCs)
plt.title('test accuracy & AUC vs epoch')
plt.ylabel('accuracy')
plt.xlabel('10 epoch')
plt.legend(['Accuracy', 'AUC'], loc='upper left')


plt.subplot(224)
plt.plot(test_recalls)
plt.plot(test_precisions)
plt.title('test accuracy & AUC vs epoch')
plt.ylabel('percentage')
plt.xlabel('10 epoch')
plt.legend(['recall', 'precision'], loc='upper left')

plt.show()