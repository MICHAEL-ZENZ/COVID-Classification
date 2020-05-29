from model import Densenet, Inceptionv3, ResNet, VGG, SimpleCNN, Efficientnet
from utils import CovidCTDatasetRotate,metrics
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
import time

import os

from torchvision import transforms
import numpy as np
from sklearn.metrics import roc_auc_score

class RotationModel(nn.Module):
    def __init__(self,basemodel):
        super(RotationModel, self).__init__()
        if isinstance(basemodel,ResNet.ResNet):
            features = [basemodel.conv1, basemodel.bn1, basemodel.relu, basemodel.maxpool,
                        basemodel.layer1, basemodel.layer2, basemodel.layer3, basemodel.layer4]
            self.features = nn.Sequential(*features)
        elif isinstance(basemodel,Densenet.DenseNet):
            self.features = nn.Sequential(basemodel.features,
                                         nn.ReLU(inplace=True))

        self.fc = basemodel.fc
        self.in_features = self.fc.weight.shape[1]
        self.rotation_fc = nn.Linear(self.in_features,4)

    def forward(self, x):
        feature = self.features(x)
        feature = F.adaptive_avg_pool2d(feature, (1, 1))
        feature = torch.flatten(feature, 1)
        pred = self.fc(feature)
        r_pred = self.rotation_fc(feature)
        return pred, r_pred

MODEL_DICT = {
        'densenet121':  Densenet.densenet121,
        'densenet161':  Densenet.densenet161,
        'densenet169':  Densenet.densenet169,
        'resnet18':     ResNet.resnet18,
        'resnet50':     ResNet.resnet50,
        'wide_resnet101':ResNet.wide_resnet101_2,
        'vgg16':        VGG.vgg16,
        'CNN':          SimpleCNN.CNN,
        'Linear':       SimpleCNN.Linear,
        'SimpleCNN':    SimpleCNN.SimpleCNN,
        'efficientnet-b7': Efficientnet.efficientnetb7,
        'efficientnet-b1': Efficientnet.efficientnetb1,
        'efficientnet-b0': Efficientnet.efficientnetb0
    }

def train(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, device):
    model.train()

    for index, batch in enumerate(tqdm(train_loader)):
        img, label, rotation = batch['img'].to(device), batch['label'].to(device), batch['rotate'].to(device)
        output, r_output = model(img)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, label) + F.cross_entropy(r_output, rotation)
        loss.backward()
        optimizer.step()
        if (index + 1) % PRINT_INTERVAL == 0:
            tqdm.write('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                       % (epoch + 1, args.epoch, index + 1, len(train_loader), loss.item()))

def test(model, nb_classes, test_loader, device):
    model.eval()
    predlist = []
    targetlist = []
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    avg_val_loss = 0
    with torch.no_grad():
        for index, batch in enumerate(tqdm(test_loader)):
            img, label, rotation = batch['img'].to(device), batch['label'].to(device), batch['rotate'].to(device)
            output, r_output = model(img)
            _, preds = torch.max(output, 1)
            loss = F.cross_entropy(output, label) + F.cross_entropy(r_output, rotation)
            avg_val_loss += loss.item()/len(test_loader)
            for t, p in zip(label.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            y_score = F.softmax(output, dim=1)
            predlist = np.append(predlist, y_score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, label.long().cpu().numpy())

    AUC = roc_auc_score(targetlist, predlist)
    print(confusion_matrix)
    precision = metrics.Precision(confusion_matrix.cpu().numpy(), nb_classes)
    recall = metrics.Recall(confusion_matrix, nb_classes)
    f1 = metrics.f1_score(precision, recall)
    acc = metrics.Acc(confusion_matrix, nb_classes)

    return AUC, precision, recall, f1, acc, avg_val_loss

def main():
    parser = argparse.ArgumentParser(description='COVID-19 CT Classification.')
    parser.add_argument('--model-name',  type=str, default='resnet50')
    parser.add_argument('--checkpoint-path',type = str, default='./checkpoint/CT')
    parser.add_argument('--batch-size', type = int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch',type = int ,default=50)
    parser.add_argument('--root-dir',type=str,default='../data/4_4_data_crop')

    parser.add_argument('--train-COV',type=str,default='../data/4_4_txt_crop/trainCT_COVID.txt')
    parser.add_argument('--train-NonCOV',type=str,default='../data/4_4_txt_crop/trainCT_NonCOVID.txt')

    parser.add_argument('--val-COV',type=str,default='../data/4_4_txt_crop/valCT_COVID.txt')
    parser.add_argument('--val-NonCOV',type=str,default='../data/4_4_txt_crop/valCT_NonCOVID.txt')

    parser.add_argument('--test-COV',type=str,default='../data/4_4_txt_crop/testCT_COVID.txt')
    parser.add_argument('--test-NonCOV',type=str,default='../data/4_4_txt_crop/testCT_NonCOVID.txt')
    parser.add_argument('--pretrained',type=bool, default=True)
    parser.add_argument('--save-name',type=str, default='resnet50_ssl480_rotate.pt')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device {}".format(device))
    # Create checkpoint file
    save_path = os.path.join(args.checkpoint_path, args.model_name)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)

    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])
    test_trans = transforms.Compose(
                                 [transforms.Resize((480,480)),
                                  transforms.ToTensor(),
                                  normalize]
                             )
    trainset = CovidCTDatasetRotate(root_dir=args.root_dir,
                              txt_COVID=args.train_COV,
                              txt_NonCOVID=args.train_NonCOV,
                              transform=transforms.Compose(
                                    [transforms.Resize((500,500)),
                                     transforms.RandomResizedCrop((480,480),
                                                                  scale=(0.5, 1.25)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize]
                                ),
                              rotate=True)
    valset = CovidCTDatasetRotate(root_dir=args.root_dir,
                            txt_COVID=args.val_COV,
                            txt_NonCOVID=args.val_NonCOV,
                            transform=test_trans,
                            rotate=False
                             )

    testset = CovidCTDatasetRotate(root_dir=args.root_dir,
                             txt_COVID=args.test_COV,
                             txt_NonCOVID=args.test_NonCOV,
                             transform=test_trans,
                             rotate=False
                               )

    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size)
    test_loader = DataLoader(testset,batch_size=args.batch_size)

    PRINT_INTERVAL = 5
    nb_classes = 2
    
    print(args.model_name,trainset.classes)


    basemodel = MODEL_DICT[args.model_name](num_classes=nb_classes, pretrained=args.pretrained)
    model = RotationModel(basemodel).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    sheduler = torch. optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    accs = []
    save = os.path.join(save_path,'{}'.format(args.save_name))

    for epoch in range(args.epoch):
        train(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, device)

        AUC, precision, recall, f1, acc, mean_loss = test(model, nb_classes, val_loader, device)
        accs.append(acc)
        print('Precision {}\tRecall {}\nF1 {}\nAUC {}\tAcc {}\tMean Loss {}'.format(precision, recall, f1, AUC, acc, mean_loss))

        if np.max(accs) == acc:
            torch.save(model.state_dict(), save)
            print("saved")
        sheduler.step(epoch)
    print('...........Testing..........')
    model.load_state_dict(torch.load(save))
    AUC, precision, recall, f1, acc, mean_loss = test(model, nb_classes, test_loader, device)

    print('Precision {}\tRecall {}\nF1 {}\nAUC {}\tAcc {}\tMean Loss {}'.format(precision, recall, f1, AUC, acc,
                                                                                mean_loss))


if __name__ == '__main__':
    print("Start training")
    main()