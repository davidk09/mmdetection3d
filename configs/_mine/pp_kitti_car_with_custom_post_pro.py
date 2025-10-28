# configs/_mine/pp_kitti_car_with_custom_post_pro.py
_base_ = '../pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'

custom_imports = dict(
    imports=[
        'mmdet3d.models.dense_heads.anchor3d_head_with_post',
        'mmdet3d.models.layers.post_processing',
        'mmdet3d.models.losses.APLoss',
    ],
    allow_failed_imports=False,
)

model = dict(
    bbox_head=dict(
        # NOTE: no _delete_=True
        type='Anchor3DHeadWithPostPP',     # swap the class
        post=dict(type='MyPostHead', nms_pre=2048),
        loss_post=dict(type='MyPostLoss', weight=0.1),
        # keep everything else (anchor_generator, bbox_coder, losses, channels, num_classes) from base
    )
)
