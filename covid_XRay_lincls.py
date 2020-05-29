from model import Densenet, Inceptionv3, ResNet, VGG, SimpleCNN, Efficientnet, COVIDNet
from utils import CovidXRayDataset,metrics
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import os
from torchvision import transforms
import numpy as np
import  torch.utils.data.sampler as Sampler

MODEL_DICT = {
    'densenet121': Densenet.densenet121,
    'densenet161': Densenet.densenet161,
    'densenet169': Densenet.densenet169,
    'resnet18': ResNet.resnet18,
    'resnet50': ResNet.resnet50,
    'wide_resnet101': ResNet.wide_resnet101_2,
    'vgg16': VGG.vgg16,
    'COVIDNet_small': COVIDNet.covid_net_small,
    'COVIDNet_large': COVIDNet.covid_net_large,
    'CNN': SimpleCNN.CNN,
    'Linear': SimpleCNN.Linear,
    'SimpleCNN': SimpleCNN.SimpleCNN,
    'efficientnet-b7': Efficientnet.efficientnetb7,
    'efficientnet-b1': Efficientnet.efficientnetb1,
    'efficientnet-b0': Efficientnet.efficientnetb0
}

def train(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, device):
    model.train()
    weight = torch.Tensor(args.class_weight).to(device)
    for index, batch in enumerate(tqdm(train_loader)):
        img, label = batch['img'].to(device), batch['label'].to(device)
        output = model(img)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, label, weight=weight)
        loss.backward()
        optimizer.step()
        if (index + 1) % PRINT_INTERVAL == 0:
            tqdm.write('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                       % (epoch + 1, args.epoch, index + 1, len(train_loader), loss.item()))

def test(model, nb_classes, test_loader, args, device):
    model.eval()
    # predlist = []
    # targetlist = []
    weight = torch.Tensor(args.class_weight).to(device)
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    avg_val_loss = 0
    with torch.no_grad():
        for index, batch in enumerate(tqdm(test_loader)):
            img, label = batch['img'].to(device), batch['label'].to(device)
            output = model(img)
            _, preds = torch.max(output, 1)
            avg_val_loss += F.cross_entropy(output, label, weight=weight).item()/len(test_loader)
            for t, p in zip(label.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            # y_score = F.softmax(output, dim=1)
            # predlist = np.append(predlist, y_score.cpu().numpy()[:, 1])
            # targetlist = np.append(targetlist, label.long().cpu().numpy())

    # AUC = roc_auc_score(targetlist, predlist)
    print(confusion_matrix)
    precision = metrics.Precision(confusion_matrix.cpu().numpy(), nb_classes)
    recall = metrics.Recall(confusion_matrix, nb_classes)
    f1 = metrics.f1_score(precision, recall)
    acc = metrics.Acc(confusion_matrix, nb_classes)

    return precision, recall, f1, acc, avg_val_loss

def main():
    parser = argparse.ArgumentParser(description='COVID-19 X-ray Classification.')
    parser.add_argument('--model-name',  type=str, default='COVIDNet_large')
    parser.add_argument('--checkpoint-path',type = str, default='./checkpoint/X-ray')
    parser.add_argument('--batch-size', type = int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch',type = int ,default=50)
    parser.add_argument('--train-txt',type=str, default='train_split_v3.txt')
    parser.add_argument('--test-txt',type=str, default='test_split_v3.txt')
    parser.add_argument('--train-dir',type=str,default='../COVID-Net/data/train/')
    parser.add_argument('--test-dir',type=str,default='../COVID-Net/data/test/')
    parser.add_argument('--pretrained', default='../moco/COVID_COVIDNet_large_1024_rand_Xray/checkpoint_0120.pth.tar', type=str,
                    help='path to moco pretrained checkpoint')
    parser.add_argument('--save-name',type=str, default='COVIDNet_large_moco_weight_randinit120.pt')
    parser.add_argument('--class-weight',type=list, default=[1.,1.,25.])
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device {}\tModel {}".format(device,args.model_name))
    # Create checkpoint file
    save_path = os.path.join(args.checkpoint_path, args.model_name)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)

    trainset = CovidXRayDataset(txt_path=args.train_txt,
                                root_dir=args.train_dir,
                                transform=transforms.Compose(
                                    [transforms.Resize((256,256)),
                                     transforms.RandomResizedCrop((224,224),
                                                                  scale=(0.5, 1.25)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5),
                                                          (0.5, 0.5, 0.5))]
                                ))

    testset = CovidXRayDataset(txt_path=args.test_txt,
                               root_dir=args.test_dir,
                               transform=transforms.Compose(
                                   [transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),
                                                         (0.5, 0.5, 0.5))]
                               )
                               )

    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(testset,batch_size=args.batch_size)

    PRINT_INTERVAL = 50
    nb_classes = 3
    model = MODEL_DICT[args.model_name](num_classes=nb_classes, pretrained=True).to(device)

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
            print(msg)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # sheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.2)
    sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    accs = []
    save = os.path.join(save_path, '{}'.format(args.save_name))

    for epoch in range(args.epoch):
        train(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, device)

        precision, recall, f1, acc, mean_loss = test(model, nb_classes, test_loader, args, device)
        accs.append(acc)
        print('Precision {}\tRecall {}\nF1 {}\tAcc {}\tMean Loss {}'.format(precision, recall, f1, acc,
                                                                                    mean_loss))

        if np.max(accs) == acc:
            torch.save(model.state_dict(), save)
            print("saved")
        sheduler.step(epoch)
    print('...........Testing..........')
    print(save)
    model.load_state_dict(torch.load(save))
    precision, recall, f1, acc, mean_loss = test(model, nb_classes, test_loader, args, device)

    print('Precision {}\tRecall {}\nF1 {}\nAUC {}\tMean Loss {}'.format(precision, recall, f1, acc,
                                                                                mean_loss))
if __name__ == '__main__':
    print("Start training")
    main()