import torch
import torch.nn as nn

from .backbones.Gghostnet import g_ghost_regnetx_032
from .backbones.WeightNet_shufflenetv2 import shufflenet_v2_x1_0
from .backbones.ghostnetv2 import ghostnetv2
from .backbones.mobilenetv3 import MobileNetV3_Large
from .backbones.pyramidTNT import ptnt_ti_patch16_192
from .backbones.repvgg import create_RepVGG_A0
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from loss.arcface import ArcFace
from .backbones.shufflenetv2 import shufflenetv2


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.LAST_STRIDE
        model_path = cfg.PRETRAIN_PATH
        self.cos_layer = cfg.COS_LAYER
        model_name = cfg.MODEL_NAME
        pretrain_choice = cfg.PRETRAIN_CHOICE

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])

        elif model_name == 'repvggA0':
            self.in_planes = 1280
            self.base = create_RepVGG_A0()

        elif model_name == 'mobilenetv3':
            self.in_planes = 960
            self.base = MobileNetV3_Large()

        elif model_name == 'ghostnetv2':
            self.in_planes = 960
            self.base = ghostnetv2()

        elif model_name == 'shufflenetv2':
            self.in_planes = 1024
            self.base = shufflenetv2()

        elif model_name == 'Gghostregnet':
            self.in_planes = 1008
            self.base = g_ghost_regnetx_032()

        elif model_name == 'pyramidTNT-Ti':
            self.in_planes = 320
            self.base = ptnt_ti_patch16_192()

        elif model_name == 'WeightNet_shufflenetv2':
            self.in_planes = 1024
            self.base = shufflenet_v2_x1_0()

        else:
            print('unsupported backbone! only support resnet50, repvggA0, mobilenetv3, ghostnetv2, shufflenetv2, '
                  'but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        if self.cos_layer:
            print('using cosine layer')
            self.arcface = ArcFace(self.in_planes, self.num_classes, s=30.0, m=0.50)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        global_feat = self.base(x)
       # print(global_feat.shape)
        #if not (global_feat.shape[2] == 1 and global_feat.shape[3] == 1):
        global_feat = self.gap(global_feat)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
