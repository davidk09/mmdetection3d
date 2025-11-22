import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
import torch.nn.functional as F


class APLoss_Customs(nn.Module):
    def __init__(self, positive_label= 1.0, negative_label= 0.0):
        super(APLoss_Customs, self).__init__()
        self.gamma = 0.5
        self.eps = 1e-6
        self.positive_label = positive_label
        self.negative_label = negative_label

    def set_delta(self, delta: float):
        self.delta = float(delta)

    def forward(self, logits, targets):

        target_mask = (targets == self.positive_label)

        order = torch.argsort(logits,descending=True)

        sorted_logits = logits[order]
        sorted_targets = target_mask[order]

        device = logits.device
        ap = torch.zeros((), device=device, dtype=logits.dtype)

        tps = torch.tensor(1.0, device=device, dtype=logits.dtype)

        for i in torch.where(sorted_targets)[0]:
            fps = sorted_logits[:i][(~sorted_targets)[:i]] # fps until current tp
            fp_measure = (torch.abs(fps - sorted_logits[i]) + self.eps)**self.gamma

            #assert (fps - sorted_logits[i] >= 0).all().item()

            fp_sum = torch.sum(fp_measure)

            prec = tps/(tps+fp_sum)

            ap = ap + prec

            tps = tps + 1.0
        
        num_pos = sorted_targets.sum()
        if num_pos.item() > 0:
            ap = ap / num_pos

        return (1 - ap)



# losses/APLoss.py
@MODELS.register_module()
class MyPostLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.ap_loss = APLoss_Customs()

    def forward(self, batched_scores, batched_assignments):

        #loss_val = torch.zeros((), device=batched_scores[0][0].device, dtype=batched_scores[0][0].dtype)
        batched_loss = []
        for b, batch in enumerate(batched_scores):
            loss_cls = []
            for c, rescores_cls in enumerate(batch): # per cls lists
                targets = batched_assignments[b][c].float()
                loss_bc = self.ap_loss(rescores_cls,targets)
                loss_cls.append(loss_bc)
            batched_loss.append(torch.stack(loss_cls).mean())
                
        loss_val = torch.stack(batched_loss).mean()
        return {'loss_post': self.weight * loss_val}
