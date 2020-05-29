from model import Densenet, Inceptionv3, ResNet, VGG, SimpleCNN, Efficientnet, ResNeSt, Ensemble,SeResNet
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
    LOSS_FUNC = LabelSmoothSoftmaxCE()
    # LOSS_FUNC = nn.CrossEntropyLoss()
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
    acc = metrics.Acc(confusion_matrix, nb_classes)

    return AUC, precision, recall, f1, acc, avg_val_loss

def main():
    parser = argparse.ArgumentParser(description='COVID-19 CT Classification.')
    parser.add_argument('--model-name',  type=str, default='densenet169')
    parser.add_argument('--checkpoint-path',type = str, default='./checkpoint/CT')
    parser.add_argument('--batch-size', type = int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch',type = int ,default=100)
    parser.add_argument('--root-dir',type=str,default='../data/4_4_data_crop')

    parser.add_argument('--train-COV',type=str,default='../data/4_4_txt_crop/trainCT_COVID.txt')
    parser.add_argument('--train-NonCOV',type=str,default='../data/4_4_txt_crop/trainCT_NonCOVID.txt')

    parser.add_argument('--val-COV',type=str,default='../data/4_4_txt_crop/valCT_COVID.txt')
    parser.add_argument('--val-NonCOV',type=str,default='../data/4_4_txt_crop/valCT_NonCOVID.txt')

    parser.add_argument('--test-COV',type=str,default='../data/4_4_txt_crop/testCT_COVID.txt')
    parser.add_argument('--test-NonCOV',type=str,default='../data/4_4_txt_crop/testCT_NonCOVID.txt')
    parser.add_argument('--pretrained', default='../moco/COVID_DenseNet169_512_imagenet_combine/checkpoint_0120.pth.tar', type=str,
                    help='path to moco pretrained checkpoint')
    parser.add_argument('--save-name',type=str, default='densenet169-moco-COVID.pt')

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
                                 [transforms.Resize((224,224)),
                                  transforms.ToTensor(),
                                  normalize]
                             )
    trainset = CovidCTDataset(root_dir=args.root_dir,
                              txt_COVID=args.train_COV,
                              txt_NonCOVID=args.train_NonCOV,
                                transform=transforms.Compose(
                                    [transforms.Resize((256,256)),
                                     transforms.RandomResizedCrop((224,224),
                                                                  scale=(0.5, 1.25)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                     transforms.ToTensor(),
                                     normalize]
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
                              shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size)
    test_loader = DataLoader(testset,batch_size=args.batch_size)

    PRINT_INTERVAL = 5
    nb_classes = 2

    print(args.model_name,trainset.classes)


    model = MODEL_DICT[args.model_name](num_classes=nb_classes,pretrained = True).to(device)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))


    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(parameters, lr=args.lr,momentum=0.9)
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
def main_finetuneall():
    parser = argparse.ArgumentParser(description='COVID-19 CT Classification.')
    parser.add_argument('--model-name',  type=str, default='densenet169')
    parser.add_argument('--checkpoint-path',type = str, default='./checkpoint/CT')
    parser.add_argument('--batch-size', type = int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch',type = int ,default=80)
    parser.add_argument('--root-dir',type=str,default='../data/4_4_data_crop')

    parser.add_argument('--train-COV',type=str,default='../data/4_4_txt_crop/trainCT_COVID.txt')
    parser.add_argument('--train-NonCOV',type=str,default='../data/4_4_txt_crop/trainCT_NonCOVID.txt')

    parser.add_argument('--val-COV',type=str,default='../data/4_4_txt_crop/valCT_COVID.txt')
    parser.add_argument('--val-NonCOV',type=str,default='../data/4_4_txt_crop/valCT_NonCOVID.txt')

    parser.add_argument('--test-COV',type=str,default='../data/4_4_txt_crop/testCT_COVID.txt')
    parser.add_argument('--test-NonCOV',type=str,default='../data/4_4_txt_crop/testCT_NonCOVID.txt')
    parser.add_argument('--pretrained', default='../moco/COVID_DenseNet169_512_imagenet/checkpoint_0640.pth.tar', type=str,
                    help='path to moco pretrained checkpoint')
    parser.add_argument('--save-name',type=str, default='densenet169-480-moco-soft-COVID.pt')

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
                                 [transforms.Resize((480,480)),
                                  transforms.ToTensor(),
                                  normalize]
                             )
    trainset = CovidCTDataset(root_dir=args.root_dir,
                              txt_COVID=args.train_COV,
                              txt_NonCOVID=args.train_NonCOV,
                                transform=transforms.Compose(
                                    [transforms.RandomResizedCrop((480,480),
                                                                  scale=(0.8, 1.2)),
                                     transforms.RandomHorizontalFlip(),
                                     auto.ImageNetPolicy(),
                                     transforms.ToTensor(),
                                     normalize]
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
                              shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size)
    test_loader = DataLoader(testset,batch_size=args.batch_size)

    PRINT_INTERVAL = 5
    nb_classes = 2
    print(args.model_name,trainset.classes)


    model = MODEL_DICT[args.model_name](num_classes=nb_classes,pretrained = True).to(device)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    print("UnFreeze Model")
    for name, param in model.named_parameters():
        param.requires_grad = True
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    accs = []
    AUCS = []
    mean_losses = []
    save = os.path.join(save_path,'{}'.format(args.save_name))

    for epoch in range(args.epoch):
        train(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, device)

        AUC, precision, recall, f1, acc, mean_loss = test(model, nb_classes, val_loader, device)
        accs.append(acc)
        AUCS.append(AUC)
        mean_losses.append(mean_loss)
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
    # main()
    main_finetuneall()