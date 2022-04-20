# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio)

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION)

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            x = self.pooler(x, proposals)
            '''
            if oneStage:
                sup_feature_ROI = (torch.mul(sup_features[0], x) + torch.mul(sup_features[1], x) + torch.mul(sup_features[2], x) + torch.mul(sup_features[3], x) +
                                   torch.mul(sup_features[4], x) + torch.mul(sup_features[5], x) + torch.mul(sup_features[6], x) + torch.mul(sup_features[7], x) +
                                   torch.mul(sup_features[8], x) + torch.mul(sup_features[9], x) + torch.mul(sup_features[10], x) + torch.mul(sup_features[11], x) +
                                   torch.mul(sup_features[12], x) + torch.mul(sup_features[13], x) + torch.mul(sup_features[14], x))/int(len(sup_features))
            else:
                sup_feature_ROI = (torch.mul(sup_features[0], x) + torch.mul(sup_features[1], x) + torch.mul(sup_features[2], x) + torch.mul(sup_features[3], x) +
                                   torch.mul(sup_features[4], x) + torch.mul(sup_features[5], x) + torch.mul(sup_features[6], x) + torch.mul(sup_features[7], x) +
                                   torch.mul(sup_features[8], x) + torch.mul(sup_features[9], x) + torch.mul(sup_features[10], x) + torch.mul(sup_features[11], x) +
                                   torch.mul(sup_features[12], x) + torch.mul(sup_features[13], x) + torch.mul(sup_features[14], x) + torch.mul(sup_features[15], x) +
                                   torch.mul(sup_features[16], x) + torch.mul(sup_features[17], x) + torch.mul(sup_features[18], x) + torch.mul(sup_features[19], x)) / int(len(sup_features))
            
            '''

            x = x.view(x.size(0), -1)

            xr = F.relu(self.fc6r(x))
            xr = F.relu(self.fc7r(xr))  # [1024, 1024]

            xc = F.relu(self.fc6c(x))  # [1024, 1024]
            xc = F.relu(self.fc7c(xc))

            return xc, xr

        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorARRM")
class FPN2MLPFeatureExtractorARRM(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorARRM, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.sup_GAP       = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.sup_GAP_key   = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        self.sup_GAP_value = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)

        self.query_value = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        self.query_key   = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)

        self.sup_value   = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)
        self.sup_key     = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1)

        self.catConv     = torch.nn.Conv2d(256*3, 256, kernel_size=3, padding=1, dilation=1, stride=1)

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            x = self.pooler(x, proposals)

            B, C, H, W = x.size()

            query_value  = self.query_value(x)                                              # [B,   C/2, H, W]
            query_key    = self.query_key(x).permute(0, 2, 3, 1).reshape(-1, 256)           # [BHW, C/2]

            sup_GAP       = self.sup_GAP(sup_features)                                       # [N,   C/2, 1, 1]
            sup_GAP_value = self.sup_GAP_key(sup_GAP).reshape(-1, 256)  # [N, C/2]
            sup_GAP_key   = self.sup_GAP_value(sup_GAP).permute(1, 0, 2, 3).reshape(256, -1)  # [C/2, N]


            sup_value = self.sup_value(sup_features).permute(0, 2, 3, 1).reshape(-1, 256)   # [NHW, C/2]
            sup_key = self.sup_key(sup_features).permute(1, 0, 2, 3).reshape(256, -1)       # [C/2, HWN]

            similarity = torch.mm(query_key, sup_key)           # [BHW, HWN]
            similarity = F.softmax(similarity, dim=-1)          # [BHW, HWN]

            similarity_chl = torch.mm(query_key, sup_GAP_key)  # [BHW, N]
            similarity_chl = F.softmax(similarity_chl, dim=-1)   # [BHW, N]

            sup_value_similarity     = torch.mm(similarity, sup_value).reshape(B, H, W, 256).permute(0, 3, 1, 2)      # [B, C/2, H, W]
            sup_value_similarity_chl = torch.mm(similarity_chl, sup_GAP_value).reshape(B, H, W, 256).permute(0, 3, 1, 2)  # [B, C/2, H, W]

            fuseFeature = F.relu(self.catConv(torch.cat((query_value, sup_value_similarity, sup_value_similarity_chl), dim=1)))             # [B, C, H, W]
            fuseFeature = fuseFeature.reshape(fuseFeature.size(0), -1)

            xr = F.relu(self.fc6r(fuseFeature))
            xr = F.relu(self.fc7r(xr))

            xc = F.relu(self.fc6c(fuseFeature))
            xc = F.relu(self.fc7c(xc))
            return xc, xr
        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification.
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x

def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR]
    return func(cfg, in_channels)