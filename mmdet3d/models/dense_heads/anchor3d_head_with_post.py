import torch
import torch.nn as nn
from mmdet3d.models.dense_heads.anchor3d_head import Anchor3DHead
from mmdet3d.registry import MODELS

from mmengine.data import InstanceData
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes

import os
from mmengine.logging import MMLogger

@MODELS.register_module()
class Anchor3DHeadWithPostPP(Anchor3DHead):
    def __init__(self, post=None, loss_post=None, **kwargs):
        super().__init__(**kwargs)
        # add 3 params per class
        self.conv_pp = nn.Conv2d(self.feat_channels, self.num_anchors * self.num_classes * 3, 1)
        self.post = MODELS.build(post) if post else None
        self.loss_post = MODELS.build(loss_post) if loss_post else None
        self._last_pp_params = None


    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.conv_pp.weight, mean=0, std=1e-3)
        nn.init.constant_(self.conv_pp.bias, 0)

    # per level
    def forward_single(self, x: torch.Tensor):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_pred = self.conv_dir_cls(x) if self.use_direction_classifier else None
        pp_params = self.conv_pp(x)  # [B, C*3, H, W]
        return cls_score, bbox_pred, dir_cls_pred, pp_params


    def forward(self, feats):
        cls_scores, bbox_preds, dir_cls_preds, pp_params = [], [], [], []
        for f in feats:
            cs, bp, dp, pp = self.forward_single(f)
            cls_scores.append(cs)
            bbox_preds.append(bp)
            dir_cls_preds.append(dp)
            pp_params.append(pp)
        self._last_pp_params = pp_params
        # IMPORTANT: keep stock signature (no pp in outputs)
        return cls_scores, bbox_preds, dir_cls_preds

    # helper: (B,C,H,W)->(B,HW,C)
    def _flat(self, x):
        return x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1))

    def _unwrap_outs(self, outs):
        """Return (cls_scores, bbox_preds, dir_cls_preds, pp_params or None)."""
        # handle extra one-tuple wrapping
        if isinstance(outs, (tuple, list)) and len(outs) == 1 and isinstance(outs[0], (tuple, list)):
            outs = outs[0]

        if not isinstance(outs, (tuple, list)):
            raise TypeError(f"Head outs must be tuple/list, got {type(outs)}")

        if len(outs) == 4:
            cls_scores, bbox_preds, dir_cls_preds, pp_params = outs
        elif len(outs) == 3:
            cls_scores, bbox_preds, dir_cls_preds = outs
            pp_params = None
        else:
            raise ValueError(f"Unexpected outs length {len(outs)} (expected 3 or 4)")

        return cls_scores, bbox_preds, dir_cls_preds, pp_params 


    # ---------- DEBUG HELPERS ----------
    @staticmethod
    def _shape_list(tlist):
        if tlist is None:
            return None
        return [tuple(t.shape) if t is not None else None for t in tlist]

    def _log_received(self, where, outs, batch_data_samples):
        logger = MMLogger.get_current_instance()
        msg = f"[{where}] outs type={type(outs)}"
        if isinstance(outs, (tuple, list)):
            msg += f", len={len(outs)}"
        logger.info(msg)
        if isinstance(outs, (tuple, list)) and len(outs) >= 1:
            logger.info(f"[{where}] outs[0] type={type(outs[0])} "
                        f"(if tuple/list expect 3 tensors lists)")
        # try to peek shapes if it looks like the triple
        try:
            cls_scores, bbox_preds, dir_cls_preds = outs
            logger.info(f"[{where}] cls_scores shapes: {self._shape_list(cls_scores)}")
            logger.info(f"[{where}] bbox_preds  shapes: {self._shape_list(bbox_preds)}")
            logger.info(f"[{where}] dir_cls_preds shapes: {self._shape_list(dir_cls_preds)}")
        except Exception:
            pass
        logger.info(f"[{where}] batch_data_samples type={type(batch_data_samples)}")

    
    def loss(self, x, batch_data_samples, **kwargs):
        """MMDet3D 1.x-style loss entrypoint.

        Args:
            x: Tuple of feature maps from the neck.
            batch_data_samples: list[Det3DDataSample]
        """
        # For now, just use the standard Anchor3DHead loss
        return super().loss(x, batch_data_samples, **kwargs)

    def loss_by_feat(self,
                 cls_scores,
                 bbox_preds,
                 dir_cls_preds,
                 batch_gt_instances_3d,
                 batch_input_metas,
                 batch_gt_instances_ignore=None):

        # 1) original PointPillars losses on raw outputs
        base_losses = super().loss_by_feat(
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            batch_gt_instances_3d,
            batch_input_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )

        # 2) our extra loss from post-processed scores
        if self.post is not None and self.loss_post is not None:
            pp_params = getattr(self, '_last_pp_params', None)

            featmap_sizes = [t.shape[-2:] for t in cls_scores]
            device = cls_scores[0].device
            anchors = self.prior_generator.grid_anchors(
                featmap_sizes, device=device)
            flat_bbox = [b.permute(0, 2, 3, 1).reshape(
                b.size(0), -1, self.box_code_size) for b in bbox_preds]
            decoded = [self.bbox_coder.decode(a, fb)
                    for a, fb in zip(anchors, flat_bbox)]

            # post: returns [pp_scores], [pp_boxes] for training
            pp_scores, pp_boxes = self.post(
                cls_scores, bbox_preds, dir_cls_preds,
                pp_params, decoded
            )

            extra_losses = self.loss_post(pp_scores, pp_boxes)
            base_losses.update(extra_losses)

        return base_losses



    # inference: same post, then delegate to base predict
    def predict_by_feat(self,
                    cls_scores,
                    bbox_preds,
                    dir_cls_preds,
                    batch_data_samples=None,
                    **kwargs):

        pp_params = getattr(self, '_last_pp_params', None)

        if self.post is not None:
            featmap_sizes = [t.shape[-2:] for t in cls_scores]
            device = cls_scores[0].device
            anchors = self.prior_generator.grid_anchors(
                featmap_sizes, device=device)
            flat_bbox = [b.permute(0, 2, 3, 1)
               .reshape(b.size(0), -1, self.box_code_size)
             for b in bbox_preds]
            decoded = [self.bbox_coder.decode(a, fb)
                    for a, fb in zip(anchors, flat_bbox)]

            batched_scores, batched_boxes = self.post(
                cls_scores, bbox_preds, dir_cls_preds, pp_params, decoded
            )

            C = len(batched_scores[0])

            final_dicts = []

            for b in range(len(batched_scores)):
                
                per_img_scores = []
                per_img_boxes = []
                per_img_labels = []

                for c in range(C):

                    batch_cls_score = batched_scores[b][c]
                    batch_cls_box = batched_boxes[b][c]

                    per_img_scores.append(batch_cls_score)
                    per_img_boxes.append(batch_cls_box)
                    per_img_labels.append(torch.full((batch_cls_score.numel(),), c, dtype=torch.long, device=batch_cls_score.device))


                scores_3d = torch.cat(per_img_scores, dim=0)
                boxes_3d  = torch.cat(per_img_boxes,  dim=0)
                labels_3d = torch.cat(per_img_labels, dim=0)

                inst = InstanceData()
                inst.bboxes_3d = LiDARInstance3DBoxes(boxes_3d, box_dim=7)
                inst.scores_3d = scores_3d
                inst.labels_3d = labels_3d
                
                final_dicts.append(inst)

        return final_dicts
