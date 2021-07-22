import math
import json

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from boxlist import BoxList
from loss import PoseLoss
from postprocess import PostProcessor

from utils import load_bbox_3d

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale


def init_conv_kaiming(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, a=1)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_conv_std(module, std=0.01):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=std)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class FPN(nn.Module):
    def __init__(self, in_channels, out_channel, top_blocks=None):
        super().__init__()

        self.inner_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()

        for i, in_channel in enumerate(in_channels, 1):
            if in_channel == 0:
                self.inner_convs.append(None)
                self.out_convs.append(None)

                continue

            inner_conv = nn.Conv2d(in_channel, out_channel, 1)
            feat_conv = nn.Conv2d(out_channel, out_channel, 3, padding=1)

            self.inner_convs.append(inner_conv)
            self.out_convs.append(feat_conv)

        self.apply(init_conv_kaiming)

        self.top_blocks = top_blocks

    def forward(self, inputs):
        inner = self.inner_convs[-1](inputs[-1])
        outs = [self.out_convs[-1](inner)]

        for feat, inner_conv, out_conv in zip(
            inputs[:-1][::-1], self.inner_convs[:-1][::-1], self.out_convs[:-1][::-1]
        ):
            if inner_conv is None:
                continue

            upsample = F.interpolate(inner, scale_factor=2, mode='nearest')
            inner_feat = inner_conv(feat)
            inner = inner_feat + upsample
            outs.insert(0, out_conv(inner))

        if self.top_blocks is not None:
            top_outs = self.top_blocks(outs[-1], inputs[-1])
            outs.extend(top_outs)

        return outs


class FPNTopP6P7(nn.Module):
    def __init__(self, in_channel, out_channel, use_p5=True):
        super().__init__()

        self.p6 = nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)

        self.apply(init_conv_kaiming)

        self.use_p5 = use_p5

    def forward(self, f5, p5):
        input = p5 if self.use_p5 else f5

        p6 = self.p6(input)
        p7 = self.p7(F.relu(p6))

        return p6, p7


class TargetCoder(object):
    def __init__(self, anchor_sizes, anchor_strides):
        self.anchor_sizes = anchor_sizes
        self.anchor_strides = anchor_strides

    def encode(self, gt_K, gt_3Ds, gt_Rs, gt_Ts, anchors):
        TO_REMOVE = 1  #
        anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

        # 2D reprojection from pose
        gt_K = gt_K.repeat(anchors.shape[0], 1, 1)
        ptn = torch.bmm(gt_K, torch.bmm(gt_Rs, gt_3Ds.transpose(1, 2)) + gt_Ts)
        ptx = ptn[:,0,:] / ptn[:,2,:]
        pty = ptn[:,1,:] / ptn[:,2,:]

        dx = (ptx - anchors_cx.view(-1, 1)) / anchors_w.view(-1, 1)
        dy = (pty - anchors_cy.view(-1, 1)) / anchors_h.view(-1, 1)

        targets = torch.cat((dx, dy), dim=1)

        return targets

    def decode(self, preds, anchors):
        TO_REMOVE = 1  #
        anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

        ptx = preds[:, :8] * anchors_w.view(-1, 1) + anchors_cx.view(-1, 1)
        pty = preds[:, 8:] * anchors_h.view(-1, 1) + anchors_cy.view(-1, 1)

        pred_xy = torch.cat((ptx, pty), dim=1)
        return pred_xy

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32),
        straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()

        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [
                generate_anchors(anchor_stride, sizes, aspect_ratios).float()
            ]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")

            cell_anchors = [
                generate_anchors(
                    anchor_stride,
                    size if isinstance(size, (tuple, list)) else (size,),
                    aspect_ratios
                ).float()
                for anchor_stride, size in zip(anchor_strides, sizes)
            ]
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.cell_anchors
        ):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        boxlist.add_field("visibility", inds_inside)

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(
                    anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors

def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float),
    )

def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 0.5
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return torch.from_numpy(anchors)

def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors

def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def make_anchor_generator(anchor_sizes, anchor_strides):
    aspect_ratios = [1.0]
    straddle_thresh = 0
    octave = 2.0
    scales_per_octave = 1

    assert len(anchor_strides) == len(anchor_sizes), "Only support FPN now"
    new_anchor_sizes = []
    for size in anchor_sizes:
        per_layer_anchor_sizes = []
        for scale_per_octave in range(scales_per_octave):
            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
            per_layer_anchor_sizes.append(octave_scale * size)
        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))

    anchor_generator = AnchorGenerator(
        tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh
    )
    return anchor_generator


class PoseHead(nn.Module):
    def __init__(self, in_channel, n_class, n_conv, prior):
        super(PoseHead, self).__init__()
        num_classes = n_class - 1
        num_anchors = 1

        cls_tower = []
        pose_tower = []
        for i in range(n_conv):
            conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channel,
                    in_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channel))
            # cls_tower.append(nn.BatchNorm2d(in_channel))
            cls_tower.append(nn.ReLU())
            pose_tower.append(
                conv_func(
                    in_channel,
                    in_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            pose_tower.append(nn.GroupNorm(32, in_channel))
            # cls_tower.append(nn.BatchNorm2d(in_channel))
            pose_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('pose_tower', nn.Sequential(*pose_tower))
        self.cls_logits = nn.Conv2d(
            in_channel, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.pose_pred = nn.Conv2d(
            in_channel, num_anchors * num_classes * 16, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.pose_tower,
                        self.cls_logits, self.pose_pred]:
                        # self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = prior
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        pose_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            pose_tower = self.pose_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            pose_pred = self.scales[l](self.pose_pred(pose_tower))
            pose_reg.append(pose_pred)

        return logits, pose_reg


class PoseModule(nn.Module):
    def __init__(self, cfg, backbone):
        super(PoseModule, self).__init__()

        n_class = cfg['DATASETS']['N_CLASS']
        bbox_json = cfg['DATASETS']['BBOX_FILE']
        diameters = cfg['DATASETS']['MESH_DIAMETERS']

        n_conv = cfg['MODEL']['N_CONV']
        prior = cfg['MODEL']['PRIOR']
        use_higher_levels = cfg['MODEL']['USE_HIGHER_LEVELS']
        feat_channels = cfg['MODEL']['FEAT_CHANNELS']
        out_channel = cfg['MODEL']['OUT_CHANNEL']
        anchor_sizes = cfg['MODEL']['ANCHOR_SIZES']
        anchor_strides = cfg['MODEL']['ANCHOR_STRIDES']

        internal_K = cfg['INPUT']['INTERNAL_K']

        positive_num = cfg['SOLVER']['POSITIVE_NUM']
        positive_lambda = cfg['SOLVER']['POSITIVE_LAMBDA']
        loss_weight_cls = cfg['SOLVER']['LOSS_WEIGHT_CLS']
        loss_weight_reg = cfg['SOLVER']['LOSS_WEIGHT_REG']
        focal_gamma = cfg['SOLVER']['FOCAL_GAMMA']
        focal_alpha = cfg['SOLVER']['FOCAL_ALPHA']

        inference_th = cfg['TEST']['CONFIDENCE_TH']

        self.backbone = backbone
        if use_higher_levels:
            fpn_top = FPNTopP6P7(feat_channels[-1], out_channel)
            self.fpn = FPN(feat_channels, out_channel, fpn_top)
        else:
            self.fpn = FPN(feat_channels, out_channel, None)

        self.head = PoseHead(out_channel, n_class, n_conv, prior)
        target_coder = TargetCoder(anchor_sizes, anchor_strides)
        self.loss_evaluator = PoseLoss(
            focal_gamma, focal_alpha, anchor_sizes, anchor_strides, positive_num, positive_lambda,
            loss_weight_cls, loss_weight_reg, internal_K, diameters, target_coder
            )
        self.post_processor = PostProcessor(inference_th, n_class, target_coder, positive_num, positive_lambda)
        self.anchor_generator = make_anchor_generator(anchor_sizes, anchor_strides)

    def forward(self, images, targets):
        features = self.backbone(images.tensors)
        features = self.fpn(features)
        # features = [features[-1]] # disable FPN and pick up only the deepest features

        pred_cls, pred_reg = self.head(features)
        anchors = self.anchor_generator(images, features)
 
        if self.training:
            return self._forward_train(pred_cls, pred_reg, targets, anchors)
        else:
            return self._forward_test(pred_cls, pred_reg, targets, anchors)

    def _forward_train(self, pred_cls, pred_reg, targets, anchors):
        loss_cls, loss_reg = self.loss_evaluator(
            pred_cls, pred_reg, targets, anchors
        )
        losses = {
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
        }
        return None, losses

    def _forward_test(self, pred_cls, pred_reg, targets, anchors):
        pred = self.post_processor(pred_cls, pred_reg, targets, anchors)
        return pred, {}

