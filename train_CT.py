from model import Densenet, Inceptionv3, ResNet, VGG, SimpleCNN, Efficientnet, ResNeSt, Ensemble,SeResNet, Deeplabv3
from utils import CovidCTDataset,metrics, SimCLR_loss, LabelSmoothSoftmaxCE
from utils import autoaugment as auto
from torch.utils.data import DataLoader
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

MODEL_DICT = {
    'densenet121': Densenet.densenet121,
    'densenet161': Densenet.densenet161,
    'densenet169': Densenet.densenet169,
    'densenet201': Densenet.densenet201,
    'resnet18': ResNet.resnet18,
    'resnet50': ResNet.resnet50,
    'resnet101': ResNet.resnet101,
    'resnet152': ResNet.resnet152,
    'seresnet50': SeResNet.se_resnet50,
    'seresnet101': SeResNet.se_resnet101,
    'seresnet152': SeResNet.se_resnet152,
    'resnext101': ResNet.resnext101_32x8d,
    'resnest50': ResNeSt.resnest50,
    'resnest200': ResNeSt.resnest200,
    'wide_resnet101': ResNet.wide_resnet101_2,
    'wide_resnet50': ResNet.wide_resnet50_2,
    'vgg16': VGG.vgg16,
    'CNN': SimpleCNN.CNN,
    'Linear': SimpleCNN.Linear,
    'SimpleCNN': SimpleCNN.SimpleCNN,
    'efficientnet-b7': Efficientnet.efficientnetb7,
    'efficientnet-b1': Efficientnet.efficientnetb1,
    'efficientnet-b0': Efficientnet.efficientnetb0
}
def train(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, device):
    model.train()
    # LOSS_FUNC = LabelSmoothSoftmaxCE()
    LOSS_FUNC = nn.CrossEntropyLoss()
    for index, batch in enumerate(tqdm(train_loader)):
        img, label = batch['img'].to(device), batch['label'].to(device)
        output = model(img)
        optimizer.zero_grad()
        loss = LOSS_FUNC(output, label)
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
    f1 = metrics.f1_score(precision, recall)
    acc = metrics.Acc(confusion_matrix,nb_classes)

    return AUC, precision, recall, f1, acc, avg_val_loss

def test_mask(model, nb_classes, test_loader, device):
    model.eval()
    predlist = []
    targetlist = []
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    avg_val_loss = 0
    with torch.no_grad():
        for index, batch in enumerate(tqdm(test_loader)):
            img, label = batch['img'].to(device), batch['label'].to(device)
            cls_pred, mask_pred = model(img)
            _, preds = torch.max(cls_pred, 1)

            avg_val_loss += F.cross_entropy(cls_pred, label).item()/len(test_loader)
            for t, p in zip(label.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            y_score = F.softmax(cls_pred, dim=1)
            predlist = np.append(predlist, y_score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, label.long().cpu().numpy())

    AUC = roc_auc_score(targetlist, predlist)
    print(confusion_matrix)
    precision = metrics.Precision(confusion_matrix.cpu().numpy(), nb_classes)
    recall = metrics.Recall(confusion_matrix, nb_classes)
    f1 = metrics.f1_score(precision, recall)
    acc = metrics.Acc(confusion_matrix,nb_classes)

    return AUC, precision, recall, f1, acc, avg_val_loss

def main():
    parser = argparse.ArgumentParser(description='COVID-19 CT Classification.')
    parser.add_argument('--model-name',  type=str, default='resnet50')
    parser.add_argument('--checkpoint-path',type = str, default='./checkpoint/CT')
    parser.add_argument('--batch-size', type = int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch',type = int ,default=50)
    parser.add_argument('--root-dir',type=str,default='./COVID-CT/Images-processed')

    parser.add_argument('--train-COV',type=str,default='./COVID-CT/Data-split/COVID/trainCT_COVID.txt')
    parser.add_argument('--train-NonCOV',type=str,default='./COVID-CT/Data-split/NonCOVID/trainCT_NonCOVID.txt')

    parser.add_argument('--val-COV',type=str,default='./COVID-CT/Data-split/COVID/valCT_COVID.txt')
    parser.add_argument('--val-NonCOV',type=str,default='./COVID-CT/Data-split/NonCOVID/valCT_NonCOVID.txt')

    parser.add_argument('--test-COV',type=str,default='./COVID-CT/Data-split/COVID/testCT_COVID.txt')
    parser.add_argument('--test-NonCOV',type=str,default='./COVID-CT/Data-split/NonCOVID/testCT_NonCOVID.txt')

    parser.add_argument('--pretrained',type=bool, default=True)
    parser.add_argument('--save-name',type=str, default='densenet169_480_newdata_pretrained.pt')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device {}".format(device))
    # Create checkpoint file
    save_path = os.path.join(args.checkpoint_path, args.model_name)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)

    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])
    test_trans = transforms.Compose(
                                 [
                                  transforms.Resize((480,480)),
                                  transforms.ToTensor(),
                                  normalize
                                 ]
                             )
    trainset = CovidCTDataset(root_dir=args.root_dir,
                              txt_COVID=args.train_COV,
                              txt_NonCOVID=args.train_NonCOV,
                                transform=transforms.Compose(
                                    [transforms.RandomResizedCrop((480,480),scale=(0.8,1.2)),
                                     transforms.RandomHorizontalFlip(),
                                     auto.ImageNetPolicy(),
                                     transforms.ToTensor(),
                                     normalize
                                     ]
                                ))
    valset = CovidCTDataset(root_dir=args.root_dir,
                            txt_COVID=args.val_COV,
                            txt_NonCOVID=args.val_NonCOV,
                             transform=test_trans
                             )

    testset = CovidCTDataset(root_dir=args.root_dir,
                             txt_COVID=args.test_COV,
                             txt_NonCOVID=args.test_NonCOV,
                               transform=test_trans
                               )

    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              num_workers=8,
                              shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size)
    test_loader = DataLoader(testset,batch_size=args.batch_size)

    PRINT_INTERVAL = 10
    nb_classes = 2
    seg_num_class = 2
    print(args.model_name,trainset.classes)


    model = MODEL_DICT[args.model_name](num_classes=nb_classes, pretrained=args.pretrained)
    # model = Deeplabv3.DeeplabVV3(model, num_class=seg_num_class)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model).to(device)
    elif torch.cuda.is_available():
        print("GPU detected but cannot use")


    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    sheduler = torch. optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # sheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_max=10)
    # sheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=15)
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
def main_ensemble(models_config):


    parser = argparse.ArgumentParser(description='COVID-19 CT Classification.')

    parser.add_argument('--checkpoint-path',type = str, default='./checkpoint/CT')
    parser.add_argument('--batch-size', type = int, default=16)

    parser.add_argument('--root-dir',type=str,default='../data/dataset_5_5/dataset_4_26_with_seg/4_4_data_crop')

    parser.add_argument('--train-COV',type=str,default='../data/dataset_5_5/train_COVID.txt')
    parser.add_argument('--train-NonCOV',type=str,default='../data/dataset_5_5/train_NonCOVID.txt')

    parser.add_argument('--val-COV',type=str,default='../data/dataset_5_5/val_COVID.txt')
    parser.add_argument('--val-NonCOV',type=str,default='../data/dataset_5_5/val_NonCOVID.txt')

    parser.add_argument('--test-COV',type=str,default='../data/dataset_5_5/test_COVID.txt')
    parser.add_argument('--test-NonCOV',type=str,default='../data/dataset_5_5/test_NonCOVID.txt')


    parser.add_argument('--multiscale', type=bool ,default=False)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device {}".format(device))
    # Create checkpoint file



    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])
    test_trans = transforms.Compose(
        [transforms.Resize((480, 480)),
         transforms.ToTensor(),
         normalize]
    )

    valset = CovidCTDataset(root_dir=args.root_dir,
                            txt_COVID=args.val_COV,
                            txt_NonCOVID=args.val_NonCOV,
                            transform=test_trans
                            )

    testset = CovidCTDataset(root_dir=args.root_dir,
                             txt_COVID=args.test_COV,
                             txt_NonCOVID=args.test_NonCOV,
                             transform=test_trans
                             )


    val_loader = DataLoader(valset, batch_size=args.batch_size)
    test_loader = DataLoader(testset, batch_size=args.batch_size)

    nb_classes = 2
    num_model = len(models_config)
    print(testset.classes)
    model_list = []
    weight = []
    for i in range(num_model):
        print(models_config[i])
        weight.append(models_config[i][2])
        model = MODEL_DICT[models_config[i][0]](num_classes=nb_classes).to(device)
        if models_config[i][3]:
            model = nn.DataParallel(model).to(device)
        save_path = os.path.join(args.checkpoint_path, models_config[i][0])
        save = os.path.join(save_path, models_config[i][1])
        model.load_state_dict(torch.load(save))
        if args.multiscale:
            model = Ensemble.Multiscale(model)
        model_list.append(model)
    ensemble_model = Ensemble.EnsembleNet(model_list,weight).to(device)
    AUC, precision, recall, f1, acc, mean_loss = test(ensemble_model, nb_classes, test_loader, device)

    print('Precision {}\tRecall {}\nF1 {}\nAUC {}\tAcc {}\tMean Loss {}'.format(precision, recall, f1, AUC, acc,
                                                                                mean_loss))
    return acc
def search():
    models_config_list = []
    accs = []
    weight = [0, 1, 3, 5]
    for i in range(100):
        weight_index = np.random.randint(0,4,6)
        models_config = (
            # model name, model path, weight, data_parallel
            ('resnet152', 'resnet152_4_4_crop_480_b16_pretrained.pt', weight[weight_index[0]], True),
            ('resnet152', 'resnet152_4_4_crop_480_b16w1.2_pretrained.pt', weight[weight_index[1]], True),
            ('resnext101', 'resnext101_4_4_crop_480_pretrained.pt', weight[weight_index[2]], True),
            ('densenet169', 'densenet169-480-moco-soft-COVID.pt', weight[weight_index[3]], True),
            ('densenet169', 'densenet169_4_4_crop_480_b16_pretrained.pt', weight[weight_index[4]], True),
            ('densenet169', 'densenet169_soft_480_pretrained.pt', weight[weight_index[5]], True),
        )
        if models_config in models_config_list:
            continue
        models_config_list.append(models_config)

        acc = main_ensemble(models_config)
        accs.append(acc)
    best = np.argmax(accs)
    print("-----------------------Best model-----------------------")
    main_ensemble(models_config_list[best])
if __name__ == '__main__':
    print("Start training")
    # search()
    models_config = (
        # model name, model path, weight, data_parallel
        ('resnet152', 'resnet152_4_4_crop_480_b16_pretrained.pt', 1, True),
        ('resnet152', 'resnet152_4_4_crop_480_b16w1.2_pretrained.pt', 1, True),
        ('resnext101', 'resnext101_4_4_crop_480_pretrained.pt', 1, True),
        ('densenet169', 'densenet169-480-moco-soft-COVID.pt', 1, True),
        ('densenet169', 'densenet169_4_4_crop_480_b16_pretrained.pt', 1, True),
        ('densenet169', 'densenet169_soft_480_pretrained.pt', 1, True),
    )
    # main_ensemble(models_config)
    main()