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
        scores: torch.Tensor,  # [N]
        boxes: torch.Tensor,
        eval_iou: torch.Tensor,         # [N, N]
        pp_params_c: torch.Tensor      # [N, 3]  (for class c)
    ) -> torch.Tensor:                 # -> [N]
        iou_gate = 0.03
        gate_steepness = 70.0

        order = torch.argsort(scores, descending=False)

        boxes = boxes[order]
        scores = scores[order]

        bbox_sup_iou_params = pp_params_c[:,0]
        bbox_sup_iou_params_feature = pp_params_c[:,1]

        bbox_sup_func_params = pp_params_c[:,2]
        bbox_sup_func_params_feature = pp_params_c[:,3]

        #print(f"feat: {bbox_sup_func_params_feature.shape} , param: {bbox_sup_func_params.T.shape}")

        model_supp_ma = torch.outer(bbox_sup_func_params_feature, bbox_sup_func_params)

        iou_supp_ma = torch.outer(bbox_sup_iou_params_feature,bbox_sup_iou_params)

        #print(f"model_supp_ma: {model_supp_ma.shape} , iou_supp_ma: {iou_supp_ma.shape}")

        supp_ma =   eval_iou * iou_supp_ma + model_supp_ma

        gate = torch.sigmoid((eval_iou - iou_gate) * gate_steepness)

        supp_ma =  F.softplus(supp_ma,beta=0.5) * gate

        mask = torch.triu(torch.ones_like(supp_ma, dtype=torch.bool), diagonal=1)

        supp_ma = supp_ma * mask

        row_logits = scores.unsqueeze(0).expand_as(supp_ma)

        row_logits = torch.sigmoid(row_logits) * gate * mask

        logits = row_logits.masked_fill(~mask, float('-inf'))
        
        weights  = row_logits
        weights = torch.softmax(logits, dim=1)

        weights = torch.cat([weights[:-1], torch.zeros_like(weights[-1:])], dim=0)

        supp_ma = torch.sum(supp_ma * weights, dim=1)

        scores =  scores - supp_ma

        return scores, boxes
    

    @staticmethod
    def iou2d(bboxes1, bboxes2, metric=0):
        '''
        bboxes1: (n, 4), (x1, y1, x2, y2)
        bboxes2: (m, 4), (x1, y1, x2, y2)
        return: (n, m)
        '''

        bboxes1 = bboxes1[:, :4]
        bboxes2 = bboxes2[:, :4]

        bboxes_x1 = torch.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :]) # (n, m)
        bboxes_y1 = torch.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :]) # (n, m)
        bboxes_x2 = torch.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
        bboxes_y2 = torch.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])

        bboxes_w = torch.clamp(bboxes_x2 - bboxes_x1, min=0)
        bboxes_h = torch.clamp(bboxes_y2 - bboxes_y1, min=0)

        iou_area = bboxes_w * bboxes_h # (n, m)
        
        bboxes1_wh = bboxes1[:, 2:] - bboxes1[:, :2]
        area1 = bboxes1_wh[:, 0] * bboxes1_wh[:, 1] # (n, )
        bboxes2_wh = bboxes2[:, 2:] - bboxes2[:, :2]
        area2 = bboxes2_wh[:, 0] * bboxes2_wh[:, 1] # (m, )
        if metric == 0:
            iou = iou_area / (area1[:, None] + area2[None, :] - iou_area + 1e-8)
        elif metric == 1:
            iou = iou_area / (area1[:, None] + 1e-8)
        return iou


    #commit msg

    def forward(
    self,
    scores: torch.Tensor,            # [N, C]
    bbox_lidar: LiDARInstance3DBoxes,
    pp_params: torch.Tensor,         # [N, C, 3]
    bboxes_pred: torch.Tensor,
    num_classes: int
    ) -> Tuple[List[torch.Tensor], List[LiDARInstance3DBoxes]]:

        cls_rescores = []
        cls_reboxes = []
        
        for c in range(num_classes):

            cls_scores = scores[:,c]
            cls_params = pp_params[:,c]
            
            
            iou = self.iou2d(bbox_lidar, bbox_lidar)  # [N, N]

            new_scores, cls_boxes = self.forward_feat_class(cls_scores,bboxes_pred, iou,cls_params)

            cls_rescores.append(new_scores)
            cls_reboxes.append(cls_boxes)


        return cls_rescores,cls_reboxes
