import torch
import torch.nn as nn
from mmdet3d.models.dense_heads.anchor3d_head import Anchor3DHead
from mmdet3d.registry import MODELS

@MODELS.register_module()
class Anchor3DHeadWithPostPP(Anchor3DHead):
    def __init__(self, post=None, loss_post=None, **kwargs):
        super().__init__(**kwargs)
        # add 3 params per class
        self.conv_pp = nn.Conv2d(self.feat_channels, self.num_anchors * self.num_classes * 3, 1)
        self.post = MODELS.build(post) if post else None
        self.loss_post = MODELS.build(loss_post) if loss_post else None

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

    # multi-level
    def forward(self, feats):
        outs = [self.forward_single(f) for f in feats]
        cls_scores, bbox_preds, dir_cls_preds, pp_params = zip(*outs)
        return list(cls_scores), list(bbox_preds), list(dir_cls_preds), list(pp_params)

    # helper: (B,C,H,W)->(B,HW,C)
    def _flat(self, x):
        return x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1))

    # training: decode + post, then delegate to base loss
    def loss(self, *outs, batch_data_samples, **kwargs):
        cls_scores, bbox_preds, dir_cls_preds, pp_params = outs

        base_loss = super().loss(
            cls_scores, bbox_preds, dir_cls_preds,
            batch_data_samples=batch_data_samples, **kwargs
        )


        featmap_sizes = [t.shape[-2:] for t in cls_scores]
        device = cls_scores[0].device
        anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)
        flat_bbox = [self._flat(b) for b in bbox_preds]
        decoded   = [self.bbox_coder.decode(a, fb) for a, fb in zip(anchors, flat_bbox)]
        cls_scores, bbox_preds, dir_cls_preds, pp_params = self.post(
            cls_scores, bbox_preds, dir_cls_preds, pp_params, decoded
        )

        post_loss = self.loss_post(
            cls_scores, bbox_preds
        )

        if isinstance(post_loss, dict):
            base_loss.update(post_loss)                 # expects e.g. {"loss_post": tensor}
        else:
            base_loss['loss_post'] = post_loss          # tensor -> keyed

        return base_loss

    # inference: same post, then delegate to base predict
    def predict_by_feat(self, *outs, batch_data_samples=None, **kwargs):
        cls_scores, bbox_preds, dir_cls_preds, pp_params = outs

        if self.post is not None:
            featmap_sizes = [t.shape[-2:] for t in cls_scores]
            device = cls_scores[0].device
            anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)
            flat_bbox = [self._flat(b) for b in bbox_preds]
            decoded   = [self.bbox_coder.decode(a, fb) for a, fb in zip(anchors, flat_bbox)]
            cls_scores, bbox_preds, dir_cls_preds, pp_params = self.post(
                cls_scores, bbox_preds, dir_cls_preds, pp_params, decoded
            )

        return super().predict_by_feat(
            cls_scores, bbox_preds, dir_cls_preds,
            batch_data_samples=batch_data_samples, **kwargs
        )
