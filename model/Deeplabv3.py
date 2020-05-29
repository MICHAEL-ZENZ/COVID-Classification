import torchimport torch.nn.functional as Fimport torch.nn as nnfrom .ResNet import ResNetfrom .Densenet import DenseNetclass DeeplabV3maskin(nn.Module):    def __init__(self, backbone, num_class, num_seg=2):        super(DeeplabV3maskin, self).__init__()        if isinstance(backbone,ResNet):            conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2,                              padding=3, bias=False)            features = [conv1, backbone.bn1, backbone.relu, backbone.maxpool,                        backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]            self.backbone = nn.Sequential(*features)        elif isinstance(backbone,DenseNet):            backbone.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2,                                                 padding=3, bias=False)            self.backbone = nn.Sequential(backbone.features,                                         nn.ReLU(inplace=True))        self.inplane = backbone.fc.in_features        self.ASPP = ASPP(self.inplane, [12, 24, 36])        self.classifier = nn.Sequential(            nn.Conv2d(256, 256, 3, padding=1, bias=False),            nn.BatchNorm2d(256),            nn.ReLU(),            nn.Conv2d(256, 256, 3, padding=1, bias=False),            nn.BatchNorm2d(256),            nn.ReLU(),            nn.Conv2d(256, num_seg, 1)        )        self.fc = nn.Linear(256 + num_seg, num_class)    def forward(self, x, mask):        input_shape = x.shape[-2:]        # contract: features is a dict of tensors        x = torch.cat([x, mask], dim=1)        features = self.backbone(x)        features = self.ASPP(features)        features = F.interpolate(features, scale_factor=4, mode='bilinear', align_corners=False)        mask_features = self.classifier(features)        mask_pred = F.interpolate(mask_features, size=input_shape, mode='bilinear', align_corners=False)        cls_features = torch.cat([features,mask_features],1)        cls_features = F.adaptive_avg_pool2d(cls_features, (1, 1))        cls_features = torch.flatten(cls_features, 1)        cls_pred = self.fc(cls_features)        return cls_pred,mask_predclass DeeplabVV3(nn.Module):    def __init__(self, backbone, num_class, num_seg=2):        super(DeeplabVV3, self).__init__()        if isinstance(backbone,ResNet):            features = [backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,                        backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]            self.backbone = nn.Sequential(*features)        elif isinstance(backbone,DenseNet):            self.backbone = nn.Sequential(backbone.features,                                         nn.ReLU(inplace=True))        self.inplane = backbone.fc.in_features        self.ASPP = ASPP(self.inplane, [12, 24, 36])        self.classifier = nn.Sequential(            nn.Conv2d(256, 256, 3, padding=1, bias=False),            nn.BatchNorm2d(256),            nn.ReLU(),            nn.Conv2d(256, 256, 3, padding=1, bias=False),            nn.BatchNorm2d(256),            nn.ReLU(),            nn.Conv2d(256, num_seg, 1)        )        self.fc = nn.Linear(256 + num_seg, num_class)    def forward(self, x):        input_shape = x.shape[-2:]        # contract: features is a dict of tensors        features = self.backbone(x)        features = self.ASPP(features)        features = F.interpolate(features, scale_factor=4, mode='bilinear', align_corners=False)        mask_features = self.classifier(features)        mask_pred = F.interpolate(mask_features, size=input_shape, mode='bilinear', align_corners=False)        cls_features = torch.cat([features,mask_features],1)        cls_features = F.adaptive_avg_pool2d(cls_features, (1, 1))        cls_features = torch.flatten(cls_features, 1)        cls_pred = self.fc(cls_features)        return cls_pred,mask_predclass DeeplabV3(nn.Module):    def __init__(self, backbone, num_class):        super(DeeplabV3, self).__init__()        if isinstance(backbone,ResNet):            features = [backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,                        backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]            self.backbone = nn.Sequential(*features)        elif isinstance(backbone,DenseNet):            self.backbone = nn.Sequential(backbone.features,                                         nn.ReLU(inplace=True))        self.fc = backbone.fc        self.inplane = self.fc.in_features        self.classifier = nn.Sequential(            ASPP(self.inplane, [12, 24, 36]),            nn.Conv2d(256, 256, 3, padding=1, bias=False),            nn.BatchNorm2d(256),            nn.ReLU(),            nn.Conv2d(256, num_class, 1)        )    def forward(self, x):        input_shape = x.shape[-2:]        # contract: features is a dict of tensors        features = self.backbone(x)        mask_x = self.classifier(features)        mask_pred = F.interpolate(mask_x, size=input_shape, mode='bilinear', align_corners=False)        cls_x = F.adaptive_avg_pool2d(features, (1, 1))        cls_x = torch.flatten(cls_x, 1)        cls_pred = self.fc(cls_x)        return cls_pred,mask_predclass ASPPConv(nn.Sequential):    def __init__(self, in_channels, out_channels, dilation):        modules = [            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),            nn.BatchNorm2d(out_channels),            nn.ReLU()        ]        super(ASPPConv, self).__init__(*modules)class ASPPPooling(nn.Sequential):    def __init__(self, in_channels, out_channels):        super(ASPPPooling, self).__init__(            nn.AdaptiveAvgPool2d(1),            nn.Conv2d(in_channels, out_channels, 1, bias=False),            nn.BatchNorm2d(out_channels),            nn.ReLU()        )    def forward(self, x):        size = x.shape[-2:]        x = super(ASPPPooling, self).forward(x)        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)class ASPP(nn.Module):    def __init__(self, in_channels, atrous_rates):        super(ASPP, self).__init__()        out_channels = 256        modules = []        modules.append(nn.Sequential(            nn.Conv2d(in_channels, out_channels, 1, bias=False),            nn.BatchNorm2d(out_channels),            nn.ReLU()))        rate1, rate2, rate3 = tuple(atrous_rates)        modules.append(ASPPConv(in_channels, out_channels, rate1))        modules.append(ASPPConv(in_channels, out_channels, rate2))        modules.append(ASPPConv(in_channels, out_channels, rate3))        modules.append(ASPPPooling(in_channels, out_channels))        self.convs = nn.ModuleList(modules)        self.project = nn.Sequential(            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),            nn.BatchNorm2d(out_channels),            nn.ReLU(),            nn.Dropout(0.5))    def forward(self, x):        res = []        for conv in self.convs:            res.append(conv(x))        res = torch.cat(res, dim=1)        return self.project(res)