from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmdet.structures.bbox import bbox_overlaps  # differentiable xyxy IoU

@MODELS.register_module()
class MyPostHead(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def flatten_scores(sc, B, A, C, H, W):
        # [B, A*C, H, W] -> [B, H, W, A, C] -> [B, N, C]
        return sc.view(B, A, C, H, W).permute(0, 3, 4, 1, 2).reshape(B, H*W*A, C)

    @staticmethod
    def flatten_bbox(bx, B, A, box_dim, H, W):
        # [B, A*box_dim, H, W] -> [B, N, box_dim]
        return bx.view(B, A, box_dim, H, W).permute(0, 3, 4, 1, 2).reshape(B, H*W*A, box_dim)

    @staticmethod
    def flatten_pp(pp, B, C, H, W):
        # current head: [B, C*3, H, W] -> [B, N, C, 3] (broadcast across anchors)
        # We’ll broadcast over A later; first make [B, H, W, C, 3].
        return pp.view(B, C, 3, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, C, 3]



    @staticmethod
    def forward_feat_class(
        cls_scores_vec: torch.Tensor,  # [N]
        iou_mat: torch.Tensor,         # [N, N]
        pp_params_c: torch.Tensor      # [N, 3]  (for this class only)
    ) -> torch.Tensor:                 # -> [N]
        # interaction term from pp params (p0 ⊗ p1)
        p0 = pp_params_c[:, 0]                     # [N]
        p1 = pp_params_c[:, 1]                     # [N]
        inter = p0[:, None] * p1[None, :]          # [N, N]

        # attention over source proposals (dim=1)
        weight = torch.softmax(iou_mat + inter, dim=1)  # [N, N]

        # mix scores (softmax over proposals for stability)
        cls_scores_vec = weight @ torch.softmax(cls_scores_vec, dim=0)  # [N]
        return cls_scores_vec



    def forward(
        self,
        batched_scores:       List[torch.Tensor],  # per level: [B, A*C, H, W]
        batched_bbox_preds:   List[torch.Tensor],  # per level: [B, A*box_dim, H, W]
        batched_pp_params:    Optional[List[torch.Tensor]],  # per level: [B, A*C*3, H, W]
        batched_decoded:      List[torch.Tensor],  # per level: [B, H*W*A, 7(+C)]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
               Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:


        B = batched_scores[0].size(0)
        _, AC, H0, W0 = batched_scores[0].shape
        N0 = batched_decoded[0].size(1)  # = H0*W0*A
        A  = N0 // (H0 * W0)
        C  = AC // A
        box_dim = batched_bbox_preds[0].size(1) // A

        
        flat_scores = []
        flat_bbox   = []
        flat_pp     = []

        for sc, bx, pp, dec in zip(batched_scores, batched_bbox_preds, batched_pp_params, batched_decoded):
            B, _, H, W = sc.shape
            # [B, N, C] / [B, N, box_dim]
            sc_f = self.flatten_scores(sc, B, A, C, H, W)
            bx_f = self.flatten_bbox(bx, B, A, box_dim, H, W)
            # pp: [B, H, W, C, 3] -> tile over A -> [B, H, W, A, C, 3] -> [B, N, C, 3]
            pp_hw_c3 = self.flatten_pp(pp, B, C, H, W)                 # [B,H,W,C,3]
            pp_hw_ac3 = pp_hw_c3.unsqueeze(3).expand(B, H, W, A, C, 3)
            pp_f = pp_hw_ac3.reshape(B, H*W*A, C, 3)
            flat_scores.append(sc_f)
            flat_bbox.append(bx_f)
            flat_pp.append(pp_f)

        scores_cat = torch.cat(flat_scores, dim=1)          # [B, N, C]
        bbox_cat   = torch.cat(flat_bbox,   dim=1)          # [B, N, box_dim]
        pp_cat     = torch.cat(flat_pp,     dim=1)          # [B, N, C, 3]
        decoded_cat = torch.cat([d[..., :7] for d in batched_decoded], dim=1)
        B, N, _ = decoded_cat.shape

        new_scores = torch.empty_like(scores_cat)
        for b in range(B):
            boxes7 = decoded_cat[b]                                # [N, 7]
            boxes  = LiDARInstance3DBoxes(boxes7, box_dim=7)
            bev    = boxes.nearest_bev                             # [N, 4] xyxy
            iou    = bbox_overlaps(bev, bev, mode='iou', is_aligned=False)  # [N, N]

            # explicit class loop
            for c in range(C):
                cls_vec   = new_scores[b, :, c]        # [N]
                pp_params = pp_cat[b, :, c, :]         # [N, 3]
                new_scores[b, :, c] = self.forward_feat_class(cls_vec, iou, pp_params)



        out_scores: List[torch.Tensor] = []
        offset = 0
        for sc in batched_scores:
            _, _, H, W = sc.shape
            Nl = H * W * A
            part = new_scores[:, offset:offset+Nl, :]            # [B, Nl, C]
            sc_lvl = part.view(B, H, W, A, C).permute(0, 3, 4, 1, 2).reshape(B, A*C, H, W)
            out_scores.append(sc_lvl)
            offset += Nl


        # TODO: use iou_mats + pp_params for your learned post-processing.
        # For now: pass through unchanged so training/inference behave identically.
        return out_scores, batched_bbox_preds, batched_pp_params
