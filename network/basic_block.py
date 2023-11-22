#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jun Cen
@file: basic_block.py
@time: 2023/2/20'''
import torch
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet34, resnet50, resnet101
from utils.lovasz_loss import lovasz_softmax


class SparseBasicBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, indice_key):
        super(SparseBasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(out_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        identity = self.layers_in(x)
        output = self.layers(x)
        return output.replace_feature(F.leaky_relu(output.features + identity.features, 0.1))


class UpsampleNet_2(nn.Module):
    def __init__(self, dataset, scale=[4, 8, 16, 32]):
        super(UpsampleNet_2, self).__init__()
    
        self.up0 = nn.Upsample(scale_factor=scale[0], mode='bilinear')
        self.up1 = nn.Upsample(scale_factor=scale[1], mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=scale[2], mode='bilinear')
        self.up3 = nn.Upsample(scale_factor=scale[3], mode='bilinear')

        if dataset != 'nuScenes':
            self.up = nn.Upsample(size=[360, 1200], mode='bilinear')
            self.down0 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
            self.down1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False)
            self.down2 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0, bias=False)
            self.down3 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.up = nn.Upsample(size=[240, 400], mode='bilinear')
            self.down0 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
            self.down1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
            self.down2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
            self.down3 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x, data_dict):
        fea_down0 = self.down0(x[0])
        fea_down1 = self.down1(x[1])
        fea_down2 = self.down2(x[2])
        fea_down3 = self.down3(x[3])

        fea_up0 = self.up(fea_down0)
        fea_up1 = self.up(fea_down1)
        fea_up2 = self.up(fea_down2)
        fea_up3 = self.up(fea_down3)

        data_dict['img_scale2'] = fea_up0
        data_dict['img_scale4'] = fea_up1
        data_dict['img_scale8'] = fea_up2
        data_dict['img_scale16'] = fea_up3

        return data_dict


class ResNetFCN(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, config=None):
        super(ResNetFCN, self).__init__()

        if backbone == "resnet34":
            net = resnet34(pretrained)
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))
        self.hiden_size = config['model_params']['hiden_size']
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(128, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )

    def forward(self, data_dict):
        x = data_dict['img']
        h, w = x.shape[2], x.shape[3]
        # if h % 16 != 0 or w % 16 != 0:
        #     assert False, "invalid input size: {}".format(x.shape)

        # Encoder
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(self.maxpool(conv1_out))
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        # Deconv
        layer1_out = self.deconv_layer1(layer1_out)
        layer2_out = self.deconv_layer2(layer2_out)
        layer3_out = self.deconv_layer3(layer3_out)
        layer4_out = self.deconv_layer4(layer4_out)

        data_dict['img_scale2'] = layer1_out
        data_dict['img_scale4'] = layer2_out
        data_dict['img_scale8'] = layer3_out
        data_dict['img_scale16'] = layer4_out

        process_keys = [k for k in data_dict.keys() if k.find('img_scale') != -1]
        img_indices = data_dict['img_indices']

        temp = {k: [] for k in process_keys}

        for i in range(x.shape[0]):
            for k in process_keys:
                temp[k].append(data_dict[k].permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])

        for k in process_keys:
            data_dict[k] = torch.cat(temp[k], 0)

        return data_dict

class ResNetFCN_2(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, config=None):
        super(ResNetFCN_2, self).__init__()

        if backbone == "resnet34":
            net = resnet34(pretrained)
        elif backbone == 'resnet50':
            net = resnet50(pretrained)
        elif backbone == 'resnet101':
            net = resnet101(pretrained)
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))
        self.hiden_size = config['model_params']['hiden_size']
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(128, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )

    def forward(self, data_dict):
        x = data_dict['img']
        h, w = x.shape[2], x.shape[3]
        # if h % 16 != 0 or w % 16 != 0:
        #     assert False, "invalid input size: {}".format(x.shape)

        # Encoder
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(self.maxpool(conv1_out))
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        return layer1_out, layer2_out, layer3_out, layer4_out


class Lovasz_loss(nn.Module):
    def __init__(self, ignore=None):
        super(Lovasz_loss, self).__init__()
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, ignore=self.ignore)