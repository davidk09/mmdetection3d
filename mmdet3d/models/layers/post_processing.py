from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmdet.structures.bbox import bbox_overlaps  # differentiable xyxy IoU


@MODELS.register_module()
class MyPostHead(nn.Module):
    def __init__(self, nms_pre: int = 0):
        super().__init__()
        self.nms_pre = int(nms_pre)  # 0 = disabled

    # [B, A*C, H, W] -> [B, N, C]
    @staticmethod
    def flatten_scores(sc, B, A, C, H, W):
        return sc.view(B, A, C, H, W).permute(0, 3, 4, 1, 2).reshape(B, H * W * A, C)

    # [B, A*box_dim, H, W] -> [B, N, box_dim]  (kept for completeness; not used here)
    @staticmethod
    def flatten_bbox(bx, B, A, box_dim, H, W):
        return bx.view(B, A, box_dim, H, W).permute(0, 3, 4, 1, 2).reshape(B, H * W * A, box_dim)

    # [B, C*3, H, W] -> [B, H, W, C, 3]
    @staticmethod
    def flatten_pp(pp, B, C, H, W):
        return pp.view(B, C, 3, H, W).permute(0, 3, 4, 1, 2)

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
        weight = torch.softmax(iou_mat + inter, dim=1)      # [N, N]
        return weight @ torch.softmax(cls_scores_vec, dim=0)  # [N]

    def forward(
        self,
        batched_scores:       List[torch.Tensor],  # per level: [B, A*C, H, W]
        batched_bbox_preds:   List[torch.Tensor],  # per level: [B, A*box_dim, H, W]
        dir_cls:              Optional[List[torch.Tensor]],
        batched_pp_params:    Optional[List[torch.Tensor]],  # per level: [B, C*3, H, W]
        anchors:              List[torch.Tensor],            # unused here
        batched_decoded:      List[torch.Tensor],            # per level: [B, H*W*A, 7(+...)]
        metas:                List[dict],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
               Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:

        # infer dims from level 0
        B = batched_scores[0].size(0)
        _, AC, H0, W0 = batched_scores[0].shape
        N0 = batched_decoded[0].size(1)   # = H0*W0*A
        A  = N0 // (H0 * W0)
        C  = AC // A
        box_dim = batched_bbox_preds[0].size(1) // A  # not used below

        # flatten each level
        flat_scores, flat_pp = [], []
        level_sizes = []
        for sc, pp in zip(batched_scores, batched_pp_params):
            B2, _, H, W = sc.shape
            assert B2 == B
            level_sizes.append((H, W))
            sc_f = self.flatten_scores(sc, B, A, C, H, W)            # [B, Nl, C]
            pp_hw_c3 = self.flatten_pp(pp, B, C, H, W)               # [B, H, W, C, 3]
            # broadcast pp over anchors: [B, H, W, A, C, 3] -> [B, Nl, C, 3]
            pp_hw_ac3 = pp_hw_c3.unsqueeze(3).expand(B, H, W, A, C, 3)
            pp_f = pp_hw_ac3.reshape(B, H * W * A, C, 3)
            flat_scores.append(sc_f)
            flat_pp.append(pp_f)

        # concat across levels
        scores_cat = torch.cat(flat_scores, dim=1)                 # [B, N, C]
        pp_cat     = torch.cat(flat_pp,     dim=1)                 # [B, N, C, 3]
        decoded_cat = torch.cat([d[..., :7] for d in batched_decoded], dim=1)  # [B, N, 7]
        _, N, _ = decoded_cat.shape

        # optional pre-filter (nms_pre) to cap N
        if self.nms_pre > 0 and N > self.nms_pre:
            # keep top-K by max class prob (per batch)
            with torch.no_grad():
                conf = torch.softmax(scores_cat, dim=-1).amax(dim=-1)  # [B, N]
                topk_vals, topk_idx = torch.topk(conf, k=self.nms_pre, dim=1, sorted=False)
            # gather
            batch_idx = torch.arange(B, device=scores_cat.device)[:, None]
            scores_cat = scores_cat[batch_idx, topk_idx]          # [B, K, C]
            pp_cat     = pp_cat[batch_idx, topk_idx]              # [B, K, C, 3]
            decoded_cat= decoded_cat[batch_idx, topk_idx]         # [B, K, 7]
            N = self.nms_pre

        # per-batch IoU + explicit per-class loop
        new_scores = scores_cat.clone()                           # [B, N, C]
        for b in range(B):
            boxes = LiDARInstance3DBoxes(decoded_cat[b], box_dim=7)
            bev   = boxes.nearest_bev                             # [N, 4] xyxy
            iou   = bbox_overlaps(bev, bev, mode='iou', is_aligned=False)  # [N, N]
            for c in range(C):
                cls_vec   = scores_cat[b, :, c]       # read from original scores
                pp_params = pp_cat[b, :, c, :]        # [N, 3]
                new_scores[b, :, c] = self.forward_feat_class(cls_vec, iou, pp_params)

        # un-concat to per-level shapes [B, A*C, H, W]
        out_scores: List[torch.Tensor] = []
        offset = 0
        for (H, W), sc in zip(level_sizes, batched_scores):
            Nl = H * W * A
            part = new_scores[:, offset:offset + Nl, :]                     # [B, Nl, C]
            sc_lvl = part.view(B, H, W, A, C).permute(0, 3, 4, 1, 2).reshape(B, A * C, H, W)
            out_scores.append(sc_lvl)
            offset += Nl

        # pass through others unchanged to keep the rest of the pipeline working
        return out_scores, batched_bbox_preds, dir_cls, batched_pp_params
