from model import Densenet, Inceptionv3, ResNet, VGG, SimpleCNN, Efficientnet, ResNeSt, Ensemble,SeResNet, Deeplabv3
from utils import CovidCTSegTaskDataset,metrics, LabelSmoothSoftmaxCE, RandomCrop, Resize,RandomRescale, RandomFlip,RandomRotation, RandomColor
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

class MaskModel(nn.Module):
    def __init__(self,basemodel):
        super(MaskModel, self).__init__()
        if isinstance(basemodel,ResNet.ResNet):
            conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2,
                                                 padding=3, bias=False)
            features = [conv1, basemodel.bn1, basemodel.relu, basemodel.maxpool,
                        basemodel.layer1, basemodel.layer2, basemodel.layer3, basemodel.layer4]
            self.features = nn.Sequential(*features)
        elif isinstance(basemodel,Densenet.DenseNet):
            basemodel.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)
            self.features = nn.Sequential(basemodel.features,
                                         nn.ReLU(inplace=True))


        self.fc = basemodel.fc

    def forward(self, x, mask):
        x = torch.cat([x,mask],dim=1)
        feature = self.features(x)
        feature = F.adaptive_avg_pool2d(feature, (1, 1))
        feature = torch.flatten(feature, 1)
        pred = self.fc(feature)
        return pred

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

def train_lungmask(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, device):
    model.train()
    # LOSS_FUNC = LabelSmoothSoftmaxCE()
    LOSS_FUNC = nn.CrossEntropyLoss()
    SEG_FUNC = nn.CrossEntropyLoss()
    for index, batch in enumerate(tqdm(train_loader)):
        img, mask, label = batch['img'].to(device), batch['mask'].to(device), batch['label'].to(device)
        cls_pred, mask_pred = model(img)
        optimizer.zero_grad()
        loss = LOSS_FUNC(cls_pred, label) + SEG_FUNC(mask_pred,mask.long())
        loss.backward()
        optimizer.step()
        if (index + 1) % PRINT_INTERVAL == 0:
            tqdm.write('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                       % (epoch + 1, args.epoch, index + 1, len(train_loader), loss.item()))

def train_covidgmask(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, device):
    model.train()
    # LOSS_FUNC = LabelSmoothSoftmaxCE()
    LOSS_FUNC = nn.CrossEntropyLoss()
    SEG_FUNC = nn.CrossEntropyLoss()
    for index, batch in enumerate(tqdm(train_loader)):

        img, mask, label, seg = batch['img'].to(device), batch['covid'].to(device), batch['label'].to(device), batch['seg']
        cls_pred, mask_pred = model(img)
        optimizer.zero_grad()
        mask = mask[seg]
        mask_pred = mask_pred[seg]
        loss = LOSS_FUNC(cls_pred, label) + SEG_FUNC(mask_pred, mask.long())
        loss.backward()
        optimizer.step()
        if (index + 1) % PRINT_INTERVAL == 0:
            tqdm.write('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                       % (epoch + 1, args.epoch, index + 1, len(train_loader), loss.item()))

def train_maskinput(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, device):
    model.train()
    # LOSS_FUNC = LabelSmoothSoftmaxCE()
    LOSS_FUNC = nn.CrossEntropyLoss()
    for index, batch in enumerate(tqdm(train_loader)):
        img, mask, label = batch['img'].to(device), batch['mask'].to(device), batch['label'].to(device)
        output = model(img,mask.unsqueeze(1).float())
        optimizer.zero_grad()
        loss = LOSS_FUNC(output, label)
        loss.backward()
        optimizer.step()
        if (index + 1) % PRINT_INTERVAL == 0:
            tqdm.write('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                       % (epoch + 1, args.epoch, index + 1, len(train_loader), loss.item()))


def train_maskinput_covidout(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, device):
    model.train()
    # LOSS_FUNC = LabelSmoothSoftmaxCE()
    SEG_FUNC = nn.CrossEntropyLoss()
    LOSS_FUNC = nn.CrossEntropyLoss()
    for index, batch in enumerate(tqdm(train_loader)):
        img, covid, mask, label, seg = batch['img'].to(device), batch['covid'].to(device), batch['mask'].to(device), batch['label'].to(device), batch['seg']
        cls_pred, mask_pred = model(img,mask.unsqueeze(1).float())
        optimizer.zero_grad()
        if True in seg:
            covid = covid[seg]
            mask_pred = mask_pred[seg]
            loss = LOSS_FUNC(cls_pred, label) + SEG_FUNC(mask_pred, covid.long())
        else:
            loss = LOSS_FUNC(cls_pred, label)
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

def test_maskin(model, nb_classes, test_loader, device):
    model.eval()
    predlist = []
    targetlist = []
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    avg_val_loss = 0
    with torch.no_grad():
        for index, batch in enumerate(tqdm(test_loader)):
            img, mask, label = batch['img'].to(device), batch['mask'].to(device), batch['label'].to(device)
            output = model(img,mask.unsqueeze(1).float())
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

def test_maskin_covidout(model, nb_classes, test_loader, device):
    model.eval()
    predlist = []
    targetlist = []
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    avg_val_loss = 0
    with torch.no_grad():
        for index, batch in enumerate(tqdm(test_loader)):
            img, mask, label = batch['img'].to(device), batch['mask'].to(device), batch['label'].to(device)
            cls_pred, mask_pred = model(img,mask.unsqueeze(1).float())
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
    parser.add_argument('--model-name',  type=str, default='densenet169')
    parser.add_argument('--checkpoint-path',type = str, default='./checkpoint/CT')
    parser.add_argument('--batch-size', type = int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch',type = int ,default=50)


    parser.add_argument('--root-dir',type=str,default='../data/dataset_5_5.5/data/4_4_data_crop')
    parser.add_argument('--mask-dir',type=str,default='../data/dataset_5_5.5/data/med_seg_lungmask')
    parser.add_argument('--covidmask-dir',type=str,default='../data/dataset_5_5.5/data/med_seg_mask')

    parser.add_argument('--train-COV',type=str,default='../data/dataset_5_5.5/new_split/train_COVID.txt')
    parser.add_argument('--train-NonCOV',type=str,default='../data/dataset_5_5.5/new_split/train_NonCOVID.txt')

    parser.add_argument('--val-COV',type=str,default='../data/dataset_5_5.5/new_split/val_COVID.txt')
    parser.add_argument('--val-NonCOV',type=str,default='../data/dataset_5_5.5/new_split/val_NonCOVID.txt')

    parser.add_argument('--test-COV',type=str,default='../data/dataset_5_5.5/new_split/test_COVID.txt')
    parser.add_argument('--test-NonCOV',type=str,default='../data/dataset_5_5.5/new_split/test_NonCOVID.txt')

    parser.add_argument('--pretrained',type=bool, default=True)
    parser.add_argument('--save-name',type=str, default='densenet169_480_deeplabv3_mask input|covid output_newdata_pretrained.pt')
    parser.add_argument('--exp',type= str, default='mask input|covid output')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device {}".format(device), args.save_name)
    # Create checkpoint file
    save_path = os.path.join(args.checkpoint_path, args.model_name)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)

    TRAIN_DICT = {
        'no mask input|covid output':train_covidgmask,
        'no mask input|lung output':train_lungmask,
        'mask input|no segmentation':train_maskinput,
        'mask input|covid output':train_maskinput_covidout
    }
    TEST_DICT = {
        'no mask input|covid output':test,
        'no mask input|lung output':test,
        'mask input|no segmentation':test_maskin,
        'mask input|covid output': test_maskin_covidout
    }
    MODEL={
        'no mask input|covid output': Deeplabv3.DeeplabVV3,
        'no mask input|lung output': Deeplabv3.DeeplabVV3,
        'mask input|no segmentation': MaskModel,
        'mask input|covid output': Deeplabv3.DeeplabV3maskin
    }
    TRAIN = TRAIN_DICT[args.exp]
    TEST = TEST_DICT[args.exp]
    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])


    trainset = CovidCTSegTaskDataset(root_dir=args.root_dir,
                                 mask_dir=args.mask_dir,
                                covid_mask_dir=args.covidmask_dir,
                                txt_COVID=args.train_COV,
                                txt_NonCOVID=args.train_NonCOV,
                                 size_transform=transforms.Compose(
                                    [
                                     Resize((480, 480)),
                                     RandomRescale(0.6,1.2),
                                     RandomRotation(),
                                     RandomCrop((480,480)),
                                     RandomFlip()
                                    ]
                                ),
                                 img_transform=transforms.Compose(
                                     [
                                     transforms.ToTensor(),
                                     normalize]
                                 )
                                 )
    valset = CovidCTSegTaskDataset(root_dir=args.root_dir,
                               mask_dir=args.mask_dir,
                               covid_mask_dir=args.covidmask_dir,
                               txt_COVID=args.val_COV,
                               txt_NonCOVID=args.val_NonCOV,
                               size_transform=transforms.Compose(
                                   [Resize((480, 480))]
                               ),
                               img_transform=transforms.Compose(
                                   [transforms.ToTensor(),
                                    normalize]
                               )
                             )

    testset = CovidCTSegTaskDataset(root_dir=args.root_dir,
                                mask_dir=args.mask_dir,
                                txt_COVID=args.test_COV,
                                txt_NonCOVID=args.test_NonCOV,
                                covid_mask_dir=args.covidmask_dir,
                                size_transform=transforms.Compose(
                                    [Resize((480, 480))]
                                ),
                                img_transform=transforms.Compose(
                                    [transforms.ToTensor(),
                                     normalize]
                                )
                               )

    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              num_workers=8,
                              shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size)
    test_loader = DataLoader(testset,batch_size=args.batch_size)

    PRINT_INTERVAL = 5
    nb_classes = 2
    seg_num_class = 2

    print(args.model_name,trainset.classes)
    print(args.exp)

    model = MODEL_DICT[args.model_name](num_classes=nb_classes, pretrained=args.pretrained)
    # model = MODEL[args.exp](model)
    model = MaskModel(model)
    # model = Deeplabv3.DeeplabV3maskin(model,num_class=seg_num_class)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model).to(device)


    # optimizer = torch.optim.SGD(model.parameters(),lr = args.lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # sheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_max=10)
    # sheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=15)
    accs = []
    losses = []
    save = os.path.join(save_path,'{}'.format(args.save_name))

    for epoch in range(args.epoch):
        TRAIN(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, device)

        AUC, precision, recall, f1, acc, mean_loss = TEST(model, nb_classes, test_loader, device)
        accs.append(acc)
        losses.append(mean_loss)
        print('Precision {}\tRecall {}\nF1 {}\nAUC {}\tAcc {}\tMean Loss {}'.format(precision, recall, f1, AUC, acc, mean_loss))

        # if np.min(losses) == mean_loss:
        if np.max(accs) == acc:
            torch.save(model.state_dict(), save)
            print("saved")
        sheduler.step(epoch)
    print('...........Testing..........')
    model.load_state_dict(torch.load(save))
    AUC, precision, recall, f1, acc, mean_loss = TEST(model, nb_classes, test_loader, device)

    print('Precision {}\tRecall {}\nF1 {}\nAUC {}\tAcc {}\tMean Loss {}'.format(precision, recall, f1, AUC, acc,
                                                                                mean_loss))

def main_ensemble(models_config):


    parser = argparse.ArgumentParser(description='COVID-19 CT Classification.')

    parser.add_argument('--checkpoint-path',type = str, default='./checkpoint/CT')
    parser.add_argument('--batch-size', type = int, default=16)

    parser.add_argument('--root-dir',type=str,default='../data/dataset_4_26_with_seg/4_4_data_crop')
    parser.add_argument('--mask-dir',type=str,default='../data/dataset_4_26_with_seg/4_4_data_crop_mask')

    parser.add_argument('--train-COV',type=str,default='../data/4_4_txt_crop/trainCT_COVID.txt')
    parser.add_argument('--train-NonCOV',type=str,default='../data/4_4_txt_crop/trainCT_NonCOVID.txt')

    parser.add_argument('--val-COV',type=str,default='../data/4_4_txt_crop/valCT_COVID.txt')
    parser.add_argument('--val-NonCOV',type=str,default='../data/4_4_txt_crop/valCT_NonCOVID.txt')

    parser.add_argument('--test-COV',type=str,default='../data/4_4_txt_crop/testCT_COVID.txt')
    parser.add_argument('--test-NonCOV',type=str,default='../data/4_4_txt_crop/testCT_NonCOVID.txt')


    parser.add_argument('--multiscale', type=bool ,default=False)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device {}".format(device))
    # Create checkpoint file



    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])

    valset = CovidCTSegTaskDataset(root_dir=args.root_dir,
                               mask_dir=args.mask_dir,
                               txt_COVID=args.val_COV,
                               txt_NonCOVID=args.val_NonCOV,
                               size_transform=transforms.Compose(
                                   [Resize((480, 480))]
                               ),
                               img_transform=transforms.Compose(
                                   [transforms.ToTensor(),
                                    normalize]
                               )
                               )

    testset = CovidCTSegTaskDataset(root_dir=args.root_dir,
                                mask_dir=args.mask_dir,
                                txt_COVID=args.test_COV,
                                txt_NonCOVID=args.test_NonCOV,
                                size_transform=transforms.Compose(
                                    [Resize((480, 480))]
                                ),
                                img_transform=transforms.Compose(
                                    [transforms.ToTensor(),
                                     normalize]
                                )
                                )


    val_loader = DataLoader(valset, batch_size=args.batch_size)
    test_loader = DataLoader(testset, batch_size=args.batch_size)

    PRINT_INTERVAL = 5
    nb_classes = 2
    seg_num_class = 2


    num_model = len(models_config)
    print(testset.classes)
    model_list = []
    weight = []
    for i in range(num_model):
        print(models_config[i])
        weight.append(models_config[i][2])
        model = MODEL_DICT[models_config[i][0]](num_classes=nb_classes).to(device)
        model = Deeplabv3.DeeplabV3(model, num_class=seg_num_class)
        if models_config[i][3]:
            model = nn.DataParallel(model).to(device)
        save_path = os.path.join(args.checkpoint_path, models_config[i][0])
        save = os.path.join(save_path, models_config[i][1])
        model.load_state_dict(torch.load(save))
        if args.multiscale:
            model = Ensemble.MultiscaleSeg(model)
        model_list.append(model)
    ensemble_model = Ensemble.EnsembleSegNet(model_list,weight).to(device)
    AUC, precision, recall, f1, acc, mean_loss = test(ensemble_model, nb_classes, test_loader, device)

    print('Precision {}\tRecall {}\nF1 {}\nAUC {}\tAcc {}\tMean Loss {}'.format(precision, recall, f1, AUC, acc,
                                                                                mean_loss))
    return acc
if __name__ == '__main__':
    print("Start training")
    models_config = (
        # model name, model path, weight, data_parallel
        # ('resnet152', 'resnet152_480_masksoft_pretrained.pt', 1, True),
        ('wide_resnet101', 'wide_resnet101_480_mask_pretrained.pt', 2, True),
        ('resnext101', 'resnext101_480_mask_pretrained.pt', 1, True),
        ('densenet169', 'densenet169_480_masksoft_pretrained.pt', 1, True),
        ('densenet201', 'densenet201_480_mask_pretrained.pt', 1, True),
    )
    # main_ensemble(models_config)
    main()