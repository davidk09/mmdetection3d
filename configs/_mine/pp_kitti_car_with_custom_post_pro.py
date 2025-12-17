# configs/_mine/pp_kitti_car_with_custom_post_pro.py
_base_ = '../pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'

custom_imports = dict(
    imports=[
        'mmdet3d.models.dense_heads.anchor3d_head_with_post',
        'mmdet3d.models.layers.post_processing',
        'mmdet3d.models.losses.APLoss_OG',
    ],
    allow_failed_imports=False,
)

model = dict(
    bbox_head=dict(
        # only override what you must
        type='Anchor3DHeadWithPostPP',

        # your extras
        post=dict(type='MyPostHead', nms_pre=200),
        loss_post=dict(type='BatchedAPLoss', weight=1.0),
    )
)