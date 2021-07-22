import torch
from torch import nn
import cv2
import numpy as np

from loss import permute_and_flatten

class PostProcessor(nn.Module):
    def __init__(
        self, inference_th, num_classes, box_coder, positive_num, positive_lambda):
        super(PostProcessor, self).__init__()
        self.inference_th = inference_th
        self.num_classes = num_classes
        self.positive_num = positive_num
        self.positive_lambda = positive_lambda
        self.box_coder = box_coder

    def forward_for_single_feature_map(self, sCls, sReg, sAnchors):
        N, _, H, W = sCls.shape
        C = sReg.size(1) // 16
        A = 1

        # put in the same format as anchors
        sCls = permute_and_flatten(sCls, N, A, C, H, W)
        sCls = sCls.sigmoid()

        sReg = permute_and_flatten(sReg, N, A, C*16, H, W)
        sReg = sReg.reshape(N, -1, C*16)

        candidate_inds = sCls > self.inference_th
        pre_ransac_top_n = candidate_inds.view(N, -1).sum(1)

        results = []
        for per_sCls, per_sReg, per_pre_ransac_top_n, per_candidate_inds, per_anchors \
                in zip(sCls, sReg, pre_ransac_top_n, candidate_inds, sAnchors):

            per_sCls = per_sCls[per_candidate_inds]
            per_sCls, top_k_indices = per_sCls.topk(per_pre_ransac_top_n, sorted=False)
            if len(per_sCls) == 0:
                results.append(None)
                continue

            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            detections = self.box_coder.decode(
                per_sReg.view(-1, C, 16)[per_box_loc, per_class],
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )

            results.append([detections, per_class + 1, torch.sqrt(per_sCls)])

        return results

    def forward(self, pred_cls, pred_reg, targets, anchors):
        sampled_boxes = []
        anchors = list(zip(*anchors))
        for layerIdx, (o, b, a) in enumerate(zip(pred_cls, pred_reg, anchors)):
            sampled_boxes.append(self.forward_for_single_feature_map(o, b, a))
        pred_inter_list = list(zip(*sampled_boxes))
        return self.select_over_all_levels(pred_inter_list, targets)

    def select_over_all_levels(self, pred_inter_list, targets):
        num_images = len(pred_inter_list)
        results = []
        for i in range(num_images):
            result = self.pose_infer_ml(pred_inter_list[i], targets[i])
            results.append(result)
        return results

    def pose_infer_ml(self, preds, target):
        K = target.K
        keypoints_3d = target.keypoints_3d
        # 
        # extract valid preds from multiple layers
        preds_mgd = [p for p in preds if p is not None]
        if len(preds_mgd) == 0:
            return []
        # merge labels from multi layers
        _, labels, _ = list(zip(*preds_mgd))
        candi_labels = torch.unique(torch.cat(labels, dim=0))
        # 
        results = []
        for lb in candi_labels:
            clsId = lb - 1
            # 
            # fetch only desired cells
            # 
            validCntPerLayer = [0]*len(preds)
            # get the reprojected box size of maximum confidence
            boxSize = 0
            boxConf = 0
            detects = [[]] * len(preds)
            scores = [[]] * len(preds)
            for i in range(len(preds)):
                item = preds[i]
                if item is not None: 
                    det, lbl, scs = item
                    mask = (lbl == lb) # choose the current label only
                    det = det[mask]
                    scs = scs[mask]
                    detects[i] = det
                    scores[i] = scs
                    # 
                    validCntPerLayer[i] = len(scs)
                    if len(scs) > 0:
                        idx = torch.argmax(scs)
                        if scs[idx] > boxConf:
                            boxConf = scs[idx]
                            kpts = det[idx].view(2, -1)
                            size = max(kpts[0].max()-kpts[0].min(), kpts[1].max()-kpts[1].min())
                            if size > boxSize:
                                boxSize = size
                                
            # compute the desired cell numbers for each layer
            dk = torch.log2(boxSize / torch.FloatTensor(self.box_coder.anchor_sizes).type_as(boxSize))
            nk = torch.exp(-self.positive_lambda * (dk * dk))
            nk = self.positive_num * nk / nk.sum(0, keepdim=True)
            nk = (nk + 0.5).int()

            # extract most confident cells
            detection_per_lb = []
            scores_per_lb = []
            for i in range(len(preds)):
                pkNum = min(validCntPerLayer[i], nk[i])
                if pkNum > 0:
                    scs, indexes = scores[i].topk(pkNum)
                    detection_per_lb.append(detects[i][indexes])
                    scores_per_lb.append(scs)
            if len(scores_per_lb) == 0:
                continue
            detection_per_lb = torch.cat(detection_per_lb)
            scores_per_lb = torch.cat(scores_per_lb)

            # PnP solver
            xy3d = keypoints_3d[clsId].repeat(len(scores_per_lb), 1, 1)
            xy2d = detection_per_lb.view(len(scores_per_lb), 2, -1).transpose(1, 2).contiguous()
            
            # CPU is more effective here
            K = K.to('cpu')
            xy3d = xy3d.to('cpu')
            xy2d = xy2d.to('cpu')

            xy3d_np = xy3d.view(-1,3).numpy()
            xy2d_np = xy2d.view(-1,2).numpy()
            K_np = K.numpy()

            retval, rot, trans, inliers = cv2.solvePnPRansac(xy3d_np, xy2d_np, K_np, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=5.0)

            if retval:
                # print('%d/%d' % (len(inliers), len(xy2d_np)))
                R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
                T = trans.reshape(-1, 1)

                if np.isnan(R.sum()) or np.isnan(T.sum()):
                    continue

                results.append([float(scores_per_lb.max()), int(clsId), R, T])
        return results
