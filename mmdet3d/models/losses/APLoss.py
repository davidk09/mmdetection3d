import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
import torch.nn.functional as F

# losses/APLoss.py
@MODELS.register_module()
class MyPostLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = float(weight)

    def forward(self, batched_scores, batched_assignments):

        #loss_val = torch.zeros((), device=batched_scores[0][0].device, dtype=batched_scores[0][0].dtype)
        batched_loss = []
        for b, batch in enumerate(batched_scores):
            loss_cls = []
            for c, rescores_cls in enumerate(batch): # per cls lists
                targets = batched_assignments[b][c].float()
                loss_bc = F.binary_cross_entropy_with_logits(
                    rescores_cls, targets, reduction='mean'
                )
                loss_cls.append(loss_bc)
            batched_loss.append(torch.stack(loss_cls).mean())
                
        loss_val = torch.stack(batched_loss).mean()
        return {'loss_post': self.weight * loss_val}
