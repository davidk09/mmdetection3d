from typing import List, Optional, Tuple
import torch
import torch.nn as nn
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
        weight = torch.softmax(iou_mat + inter, dim=1)      # [N, N]
        return weight @ torch.softmax(cls_scores_vec, dim=0)  # [N]

    def forward(
        self,
        batched_scores:       List[torch.Tensor],  # per level: [B, A*C, H, W]
        batched_bbox_preds:   List[torch.Tensor],  # per level: [B, A*box_dim, H, W]
        dir_cls:              Optional[List[torch.Tensor]], # per level: [B, A*2, H, W]
        batched_pp_params:    Optional[List[torch.Tensor]],  # per level: [B, A*C*3, H, W]
        batched_decoded:      List[torch.Tensor],            # per level: [B, H*W*A, 7(+...)]
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
        level_rescores = []
        level_reboxes = []

        for sc, box, pp in zip(batched_scores,batched_decoded, batched_pp_params):
            B2, _, H, W = sc.shape
            #assert B2 == B
            #level_sizes.append((H, W))
            batched_rescores = []
            batched_reboxes = []

            for b in range(B2):
                sc_flat = sc[b].view(A,C,H,W).permute(2,3,0,1).reshape(H*W*A,C)
                box_flat = box[b] #.view(A,box_dim,H,W).permute(2,3,0,1).reshape(H*W*A,box_dim)
                pp_flat = pp[b].view(A,C,3,H,W).permute(3,4,0,1,2).reshape(H*W*A,C,3)
                
                cls_rescores = []
                cls_reboxes = []
                for c in range(C):
                    scores_cls = sc_flat[:,c]
                    boxes_cls = box_flat
                    param_cls = pp_flat[:,c]

                    scores_cls_scored = torch.sigmoid(scores_cls)

                    _, topk_idx = torch.topk(scores_cls_scored, k=self.nms_pre)

                    scores_survive = scores_cls[topk_idx]
                    boxes_survive = boxes_cls[topk_idx]
                    param_survive = param_cls[topk_idx]

                    boxes_lidar = LiDARInstance3DBoxes(boxes_survive, box_dim=box_dim)
                    bev   = boxes_lidar.nearest_bev
                    iou   = bbox_overlaps(bev, bev, mode='iou', is_aligned=False)  # [N, N]

                    new_scores = self.forward_feat_class(scores_survive,iou,param_survive)
                    cls_rescores.append(new_scores)
                    cls_reboxes.append(boxes_survive)

                batched_rescores.append(cls_rescores)
                batched_reboxes.append(cls_reboxes)

            level_rescores.append(batched_rescores)
            level_reboxes.append(batched_reboxes)

        

        return level_rescores,level_reboxes
