# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework.
"""
import math

from scipy import misc
import cv2
import scipy.misc
import torch
from torch import nn
from maskrcnn_benchmark.structures.image_list import to_image_list
from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from maskrcnn_benchmark.modeling.make_layers import make_fc
from torch.nn import functional as F


class GeneralizedRCNN(nn.Module):

    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.sup_fc1 = make_fc(16384, 1024, use_gn)
        self.sup_fc2 = make_fc(1024, 1024, use_gn)

        self.light_sup_branch = nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, dilation=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, stride=2),

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, dilation=1, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, stride=2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, dilation=1, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),

            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, dilation=1, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2, stride=2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, dilation=1, stride=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),

            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, stride=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.sup_weight = torch.nn.Parameter(torch.FloatTensor(num_classes - 1, 1024))
        nn.init.kaiming_uniform_(self.sup_weight, a=math.sqrt(5))

    def forward(self, images, targets=None, sup=None, supTarget=None, oneStage=None, gamma=None, margin=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        Batch = to_image_list(sup).tensors.shape[0]
        H = to_image_list(sup).tensors.shape[3]
        W = to_image_list(sup).tensors.shape[4]
        sup_imgs = to_image_list(sup).tensors.reshape(Batch, -1, H, W)  # [15, 3, 192, 192]
        sup_feats = self.light_sup_branch(sup_imgs)  # [20, 256, 8, 8]

        if self.training:
            each_SUP = sup_feats.reshape(sup_feats.size(0), -1)
            each_SUP = F.relu(self.sup_fc1(each_SUP))
            each_SUP = F.relu(self.sup_fc2(each_SUP))  # torch.Size([15, 1024])

            cosine = F.linear(F.normalize(each_SUP), F.normalize(self.sup_weight))

            cosine *= 3.0
            sup_loss = F.cross_entropy(cosine, supTarget)
            sup_losses = {"sup_Losses": sup_loss}

        images = to_image_list(images)
        features = self.backbone(images.tensors)

        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, sup_feats, supTarget, oneStage)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(sup_losses)
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result