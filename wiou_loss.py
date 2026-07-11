import torch
import torch.nn as nn

class WIoULoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        inter = (torch.min(pred[:, 2], target[:, 2]) - torch.max(pred[:, 0], target[:, 0])).clamp(0) * \
                (torch.min(pred[:, 3], target[:, 3]) - torch.max(pred[:, 1], target[:, 1])).clamp(0)
        area1 = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        area2 = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = area1 + area2 - inter
        iou = inter / (union + 1e-7)
        center_dist = ((pred[:, 0] + pred[:, 2])/2 - (target[:, 0] + target[:, 2])/2)**2 + \
                      ((pred[:, 1] + pred[:, 3])/2 - (target[:, 1] + target[:, 3])/2)**2
        enclosing_diag = ((torch.max(pred[:, 2], target[:, 2]) - torch.min(pred[:, 0], target[:, 0]))**2 +
                          (torch.max(pred[:, 3], target[:, 3]) - torch.min(pred[:, 1], target[:, 1]))**2)
        r_wiou = torch.exp(-center_dist / (enclosing_diag + 1e-7))
        wiou_v1 = 1 - iou + r_wiou
        beta = (wiou_v1.detach() / (wiou_v1.mean() + 1e-7)).clamp(0, 10)
        delta = 1.9
        alpha = 1.9
        gamma = beta / (alpha * beta + delta)
        loss = gamma * wiou_v1
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
