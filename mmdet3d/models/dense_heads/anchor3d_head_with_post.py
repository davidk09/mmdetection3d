import torch
import torch.nn as nn
from mmdet3d.models.dense_heads.anchor3d_head import Anchor3DHead
from mmdet3d.registry import MODELS

from mmengine.structures import InstanceData
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmdet.structures.bbox import bbox_overlaps 
from mmdet.models.utils import select_single_mlvl
from mmdet3d.structures.det3d_data_sample import SampleList

from mmdet3d.structures import limit_period, xywhr2xyxyr

import os
from mmengine.logging import MMLogger


# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Tuple

import numpy as np
from mmdet.models.utils import multi_apply
from mmdet.utils.memory import cast_tensor_type
from mmengine.runner import amp
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.task_modules import PseudoSampler
from mmdet3d.models.test_time_augs import merge_aug_bboxes_3d
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.utils.typing_utils import (ConfigType, InstanceList,
                                        OptConfigType, OptInstanceList)
from .base_3d_dense_head import Base3DDenseHead
from .train_mixins import AnchorTrainMixin

@MODELS.register_module()
class Anchor3DHeadWithPostPP(Anchor3DHead):
    def __init__(self, post=None, loss_post=None, **kwargs):
        super().__init__(**kwargs)
        # add 3 params per class
        self.parameter_per_box = 4

        self.conv_pp = nn.Conv2d(self.feat_channels, self.num_anchors * self.num_classes * self.parameter_per_box, 1)
        self.post = MODELS.build(post) if post else None
        self.loss_post = MODELS.build(loss_post) if loss_post else None
        self._last_pp_params = None
        self.target_assignment_thres = 0.1
        self.cls_min_iou =  {0 : 0.7} #{0: 0.5, 1: 0.5, 2: 0.7}
        self.nms_pre=200


    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.conv_pp.weight, mean=0, std=1e-3)
        nn.init.constant_(self.conv_pp.bias, 0)

    # per level
    def forward_single(self, x: torch.Tensor):
        """Per-level forward."""
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_pred = self.conv_dir_cls(x) if self.use_direction_classifier else None
        pp_params = self.conv_pp(x)  # [B, A*C*3, H, W]
        return cls_score, bbox_pred, dir_cls_pred, pp_params

    def forward(self, feats):
        """Now returns 4 lists (over FPN levels)."""
        cls_scores, bbox_preds, dir_cls_preds, pp_params = multi_apply(
            self.forward_single, feats
        )
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

    
    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList, **kwargs):
        #this is copied from Base3DDenseHead, which is what is used in Anchor3DHead as loss(..) function
        #we overwrite this to include pp_parms in forward to loss, it can be handeled exactly like score or bbox parameters
        # the loss_by_feat is overwritten by Anchor3DHead and can be seen there 



        cls_scores, bbox_preds, dir_cls_preds, pp_params = self(x)

        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None)
            )

        # -> NOTE: pass pp_params to our custom loss_by_feat
        loss_inputs = (cls_scores, bbox_preds, dir_cls_preds, pp_params,
                       batch_gt_instances_3d, batch_input_metas,
                       batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses


    # def loss(self, x, batch_data_samples, **kwargs):
    #     """MMDet3D 1.x-style loss entrypoint.

    #     Args:
    #         x: Tuple of feature maps from the neck.
    #         batch_data_samples: list[Det3DDataSample]
    #     """
    #     # For now, just use the standard Anchor3DHead loss
    #     return super().loss(x, batch_data_samples, **kwargs)

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        dir_cls_preds: List[Tensor],
        pp_params: List[Tensor],
        batch_gt_instances_3d: InstanceList,
        batch_input_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d``
                and ``labels_3d`` attributes.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss_dir (list[torch.Tensor]): Direction classification
                    losses.
        """



        # 1) original PointPillars losses on raw outputs
        base_losses = super().loss_by_feat(
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            batch_gt_instances_3d,
            batch_input_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )


        #taken from Base3DDenseHead predict_by_feat, because this is inherited by 
        #anchor3d_head and predict actually decodes the boxes and not only predicts deltas,
        # such that they could be used for evaluation / target assignment -> we do this too. 
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_anchors(
            featmap_sizes, device=cls_scores[0].device)
        mlvl_priors = [
            prior.reshape(-1, self.box_code_size) for prior in mlvl_priors
        ]

        batched_rescores, batched_reboxes = [], []

        for input_id in range(len(batch_input_metas)):
            input_meta = batch_input_metas[input_id]
            cls_score_list = select_single_mlvl(cls_scores, input_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, input_id)
            dir_cls_pred_list = select_single_mlvl(dir_cls_preds, input_id)
            pp_params_list = select_single_mlvl(pp_params, input_id)
            mlvl_bboxes = []
            mlvl_scores = []
            mlvl_dir_scores = []
            mlvl_params = []
            for cls_score, bbox_pred, dir_cls_pred, priors, params in zip(
            cls_score_list, bbox_pred_list, dir_cls_pred_list,
            mlvl_priors,pp_params_list):
                assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
                
                dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
                dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

                cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)        

                cls_params = params.permute(1, 2, 0).reshape(-1, self.num_classes, self.parameter_per_box)        

                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.box_code_size)

                if  cls_score.shape[0] > self.nms_pre:
                    max_scores, _ = cls_score.max(dim=1)
                    _, topk_inds = max_scores.topk(self.nms_pre)
                    priors = priors[topk_inds, :]
                    bbox_pred = bbox_pred[topk_inds, :]
                    cls_score = cls_score[topk_inds, :]
                    dir_cls_score = dir_cls_score[topk_inds, :]
                    cls_params = cls_params[topk_inds, :]


                bboxes = self.bbox_coder.decode(priors, bbox_pred)

                mlvl_bboxes.append(bboxes)
                mlvl_scores.append(cls_score)
                mlvl_params.append(cls_params)
                mlvl_dir_scores.append(dir_cls_score)
            
            mlvl_bboxes = torch.cat(mlvl_bboxes)
            mlvl_scores = torch.cat(mlvl_scores)
            mlvl_dir_scores = torch.cat(mlvl_dir_scores)
            mlvl_params = torch.cat(mlvl_params)
            dir_rot = limit_period(mlvl_bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            mlvl_bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * mlvl_dir_scores.to(bboxes.dtype))

            lidar_bboxes = input_meta['box_type_3d'](mlvl_bboxes, box_dim=self.box_code_size)
            
            # mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            # mlvl_bboxes, box_dim=self.box_code_size).bev)

            #everything up until here mostly follows from Base3DDenseHead precdict_by_feat
            


            cls_rescores, cls_reboxes = self.post(
                    mlvl_scores, lidar_bboxes, 
                    mlvl_params, mlvl_bboxes , self.num_classes
                )
            cls_reboxes = [input_meta['box_type_3d'](boxes, box_dim=self.box_code_size) for boxes in cls_reboxes]
            batched_rescores.append(cls_rescores)
            batched_reboxes.append(cls_reboxes)


        batched_assignments = []
        for b in range(len(batch_gt_instances_3d)):
            
            gt_labels   = batch_gt_instances_3d[b].labels_3d     # (N_gt,)
            cls_assignments = []
            for c in range(len(batched_rescores[0])):
                scores_cls = batched_rescores[b][c]
                boxes_cls = batched_reboxes[b][c]
                #do target assignment
                gt_mask_c = (gt_labels == c)

                gt_boxes_3d = batch_gt_instances_3d[b].bboxes_3d[gt_mask_c]     # LiDARInstance3DBoxes

                if isinstance(gt_boxes_3d, torch.Tensor):
                    gt_boxes_3d = LiDARInstance3DBoxes(gt_boxes_3d, box_dim=gt_boxes_3d.shape[-1])

                bev_gt_c   = gt_boxes_3d.bev     # GT input format to bboxes should be correct, see anchor target assign pipeli

                bev_pred_c   = boxes_cls.bev

                eval_iou = bbox_overlaps(bev_pred_c, bev_gt_c,mode='iou', is_aligned=False)

                assigned = torch.zeros((len(scores_cls),), dtype=torch.bool, device=scores_cls.device)
                for k in range(bev_gt_c.size(0)):
                    best_match = -1
                    best_score = float('-inf')
                    #count_away = 0
                    for j, score in enumerate(scores_cls):
                        if torch.sigmoid(score) < self.target_assignment_thres:
                            #count_away += 1.0
                            continue
                        if (not assigned[j].item()) and float(eval_iou[j, k]) > self.cls_min_iou[c] and float(score) > best_score:
                            best_score = float(score)
                            best_match = j
                    if best_match != -1:
                        assigned[best_match] = True
                    #print(f"Of {scores_cls.shape[0]}, {count_away} were not close enough, mean logit: {scores_cls.mean()}, mean score: {torch.sigmoid(scores_cls).mean()}")
                cls_assignments.append(assigned)
                

            batched_assignments.append(cls_assignments)

        extra_losses = self.loss_post(batched_rescores, batched_assignments)
        base_losses.update(extra_losses)

        return base_losses


    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        cls_scores, bbox_preds, dir_cls_preds, pp_params = self(x)
        predictions = self.predict_by_feat(
            cls_scores, bbox_preds, dir_cls_preds, pp_params,
            batch_input_metas=batch_input_metas, rescale=rescale)
        return predictions



# inference: same post, then delegate to base predict
    def predict_by_feat(self,
                    cls_scores,
                    bbox_preds,
                    dir_cls_preds,
                    pp_params,   # <--- new
                    batch_input_metas=None,
                    cfg=None,
                    rescale: bool = False,
                    **kwargs):


        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_anchors(
            featmap_sizes, device=cls_scores[0].device)
        mlvl_priors = [
            prior.reshape(-1, self.box_code_size) for prior in mlvl_priors
        ]

        final_dicts = []

        for input_id in range(len(batch_input_metas)):
            input_meta = batch_input_metas[input_id]
            cls_score_list = select_single_mlvl(cls_scores, input_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, input_id)
            pp_params_list = select_single_mlvl(pp_params, input_id)
            dir_cls_pred_list = select_single_mlvl(dir_cls_preds, input_id)
            
            mlvl_bboxes = []
            mlvl_scores = []
            mlvl_dir_scores = []
            mlvl_params = []

            for cls_score, bbox_pred, dir_cls_pred, priors, params in zip(
            cls_score_list, bbox_pred_list, dir_cls_pred_list,
            mlvl_priors,pp_params_list):
                assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
                
                dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
                dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

                cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)        

                cls_params = params.permute(1, 2, 0).reshape(-1, self.num_classes, self.parameter_per_box)        

                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.box_code_size)

                if  cls_score.shape[0] > self.nms_pre:
                    max_scores, _ = cls_score.max(dim=1)
                    _, topk_inds = max_scores.topk(self.nms_pre)
                    priors = priors[topk_inds, :]
                    bbox_pred = bbox_pred[topk_inds, :]
                    cls_score = cls_score[topk_inds, :]
                    dir_cls_score = dir_cls_score[topk_inds, :]
                    cls_params = cls_params[topk_inds, :]


                bboxes = self.bbox_coder.decode(priors, bbox_pred)

                mlvl_bboxes.append(bboxes)
                mlvl_scores.append(cls_score)
                mlvl_params.append(cls_params)
                mlvl_dir_scores.append(dir_cls_score)
            
            mlvl_bboxes = torch.cat(mlvl_bboxes)
            
            mlvl_scores = torch.cat(mlvl_scores)
            #mlvl_dir_scores = torch.cat(mlvl_dir_scores)
            mlvl_params = torch.cat(mlvl_params)

            dir_rot = limit_period(mlvl_bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            mlvl_bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * mlvl_dir_scores.to(bboxes.dtype))

            lidar_bboxes = input_meta['box_type_3d'](mlvl_bboxes, box_dim=self.box_code_size)

            cls_rescores, cls_reboxes = self.post(
                    mlvl_scores, lidar_bboxes, 
                    mlvl_params, mlvl_bboxes ,self.num_classes
                )


            per_img_scores = []
            per_img_boxes = []
            per_img_labels = []

            for c in range(self.num_classes):

                batch_cls_score = cls_rescores[c]
                batch_cls_box = cls_reboxes[c]

                per_img_scores.append(batch_cls_score)
                per_img_boxes.append(batch_cls_box)
                per_img_labels.append(torch.full((batch_cls_score.numel(),), c, dtype=torch.long, device=batch_cls_score.device))


            scores_3d = torch.cat(per_img_scores, dim=0)
            boxes_3d  = torch.cat(per_img_boxes,  dim=0)
            labels_3d = torch.cat(per_img_labels, dim=0)

            inst = InstanceData()
            inst.bboxes_3d = input_meta['box_type_3d'](boxes_3d, box_dim=self.box_code_size)
            inst.scores_3d = scores_3d.sigmoid()
            inst.labels_3d = labels_3d
            
            final_dicts.append(inst)

        return final_dicts


    