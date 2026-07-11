import torch
import torch.nn as nn
from ultralytics.utils.metrics import bbox_iou

class DynamicAssigner(nn.Module):
    def __init__(self, top_k=9, alpha=0.8, beta=6.0):
        super().__init__()
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta

    def forward(self, pd_scores, pd_bboxes, gt_bboxes, anchor_bboxes):
        iou = bbox_iou(gt_bboxes, anchor_bboxes, xywh=True, CIoU=False)
        align_metric = pd_scores[:, None] ** self.alpha * iou ** self.beta
        pos_mask = align_metric > 0.5
        return pos_mask
