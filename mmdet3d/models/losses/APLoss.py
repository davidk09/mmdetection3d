import torch
import torch.nn as nn
from mmdet3d.registry import MODELS

# losses/APLoss.py
@MODELS.register_module()
class MyPostLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = float(weight)

    def forward(self, lvl_scores, bbox_preds):
        loss_val = 0.0
        for level in lvl_scores:
            for batch in level:
                for cls_scores in batch:
                    loss_val = loss_val + (cls_scores**2).mean()
        return {'loss_post': self.weight * loss_val}
