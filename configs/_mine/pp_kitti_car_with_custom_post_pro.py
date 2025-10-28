# configs/_mine/pp_kitti_car_with_custom_post_pro.py
_base_ = '../pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'

# --- disable ground planes cleanly: re-declare train_pipeline ---
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            backend_args=None,
            classes=['Car'],
            data_root='data/kitti/',
            info_path='data/kitti/kitti_dbinfos_train.pkl',
            points_loader=dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=None),
            prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
            rate=1.0,
            sample_groups=dict(Car=15),
        ),
        use_ground_plane=False,   # <-- key change
    ),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    dict(type='ObjectRangeFilter', point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_labels_3d', 'gt_bboxes_3d']),
]

train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='KittiDataset',
            data_root='data/kitti/',
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced'),
            metainfo=dict(classes=['Car']),
            modality=dict(use_camera=False, use_lidar=True),
            pipeline=train_pipeline,
            test_mode=False,
            backend_args=None,
            box_type_3d='LiDAR',
        ),
    ),
)

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
        type='Anchor3DHeadWithPostPP',
        post=dict(type='MyPostHead', nms_pre=2048),
        loss_post=dict(type='MyPostLoss', weight=0.1),
        # all other fields inherited from base bbox_head
    )
)
