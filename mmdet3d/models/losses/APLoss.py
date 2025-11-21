import torch
import torch.nn as nn
from mmdet3d.registry import MODELS

# losses/APLoss.py
@MODELS.register_module()
class MyPostLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = float(weight)

    def forward(self, batched_scores, batched_bbox_preds):
        loss_val = 0.0
        for batch in batched_scores:
            for cls_scores in batch:
                loss_val = loss_val + (cls_scores**2).mean()
        return {'loss_post': self.weight * loss_val}
