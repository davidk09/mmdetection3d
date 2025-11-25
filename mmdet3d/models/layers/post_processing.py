from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmdet.structures.bbox import bbox_overlaps  # differentiable xyxy IoU


@MODELS.register_module()
class MyPostHead(nn.Module):
    def __init__(self, nms_pre: int = 200):
        super().__init__()
        self.nms_pre = int(nms_pre)  # 0 = disabled

    # per-class update (your formula)
    @staticmethod
    def forward_feat_class(
        cls_scores_vec: torch.Tensor,  # [N]
        iou_mat: torch.Tensor,         # [N, N]
        pp_params_c: torch.Tensor      # [N, 3]  (for class c)
    ) -> torch.Tensor:                 # -> [N]
        p0 = pp_params_c[:, 0]                     # [N]
        p1 = pp_params_c[:, 1]                     # [N]
        inter = p0[:, None] * p1[None, :]          # [N, N]
        weight = iou_mat +  F.softplus(inter)       # [N, N]
        return  cls_scores_vec - (weight @ torch.sigmoid(cls_scores_vec))  # [N]

    def forward(
    self,
    scores: torch.Tensor,            # [N, C]
    bbox_preds: LiDARInstance3DBoxes,
    pp_params: torch.Tensor,         # [N, C, 3]
    num_classes: int
    ) -> Tuple[List[torch.Tensor], List[LiDARInstance3DBoxes]]:

        cls_rescores = []
        cls_reboxes = []
        
        for c in range(num_classes):

            cls_scores = scores[:,c]
            cls_params = pp_params[:,c]
            cls_boxes = bbox_preds
        
            bev   = cls_boxes.nearest_bev
            iou   = bbox_overlaps(bev, bev, mode='iou', is_aligned=False)  # [N, N]

            new_scores = self.forward_feat_class(cls_scores,iou,cls_params)

            cls_rescores.append(new_scores)
            cls_reboxes.append(cls_boxes)


        return cls_rescores,cls_reboxes
