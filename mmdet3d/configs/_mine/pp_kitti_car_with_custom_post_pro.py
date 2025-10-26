_base_ = '../pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'

model = dict(
    bbox_head=dict(
        _delete_=True,
        type='Anchor3DHeadWithPostPP',
        num_classes=3,
        in_channels=384, feat_channels=384,
        use_direction_classifier=True,
        post=dict(type='MyPostHead'),  # your simple identity post layer
        # (rest of the head args same as your current PointPillars head)
    )
)
