point_cloud_range = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'CustomNuScenesOfflineLocalMapDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='CustomPointToMultiViewDepth',
        downsample=1,
        grid_config=dict(
            x=[-30.0, -30.0, 0.15],
            y=[-15.0, -15.0, 0.15],
            z=[-10, 10, 20],
            depth=[1.0, 35.0, 0.5])),
    dict(type='PadMultiViewImageDepth', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        with_gt=False,
        with_label=False,
        class_names=['divider', 'ped_crossing', 'boundary']),
    dict(type='CustomCollect3D', keys=['img', 'gt_depth'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                with_gt=False,
                with_label=False,
                class_names=['divider', 'ped_crossing', 'boundary']),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CustomNuScenesOfflineLocalMapDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_map_infos_maptrv2_train_newsplit.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='CustomPointToMultiViewDepth',
                downsample=1,
                grid_config=dict(
                    x=[-30.0, -30.0, 0.15],
                    y=[-15.0, -15.0, 0.15],
                    z=[-10, 10, 20],
                    depth=[1.0, 35.0, 0.5])),
            dict(type='PadMultiViewImageDepth', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                with_gt=False,
                with_label=False,
                class_names=['divider', 'ped_crossing', 'boundary']),
            dict(type='CustomCollect3D', keys=['img', 'gt_depth'])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        aux_seg=dict(
            use_aux_seg=True,
            bev_seg=False,
            pv_seg=True,
            seg_classes=1,
            feat_down_sample=32,
            pv_thickness=1),
        use_valid_flag=True,
        bev_size=(200, 100),
        pc_range=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
        fixed_ptsnum_per_line=20,
        eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        map_classes=['divider', 'ped_crossing', 'boundary'],
        queue_length=1,
        dn_enabled=True),
    val=dict(
        type='CustomNuScenesOfflineLocalMapDataset',
        ann_file='data/nuscenes/nuscenes_map_infos_maptrv2_val_newsplit.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        with_gt=False,
                        with_label=False,
                        class_names=['divider', 'ped_crossing', 'boundary']),
                    dict(type='CustomCollect3D', keys=['img'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        data_root='data/nuscenes/',
        map_ann_file='data/nuscenes/nuscenes_map_anns_val_newsplit.json',
        bev_size=(200, 100),
        pc_range=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
        fixed_ptsnum_per_line=20,
        eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        map_classes=['divider', 'ped_crossing', 'boundary'],
        samples_per_gpu=1),
    test=dict(
        type='CustomNuScenesOfflineLocalMapDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_map_infos_maptrv2_val_newsplit.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        with_gt=False,
                        with_label=False,
                        class_names=['divider', 'ped_crossing', 'boundary']),
                    dict(type='CustomCollect3D', keys=['img'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        map_ann_file='data/nuscenes/nuscenes_map_anns_val_newsplit.json',
        bev_size=(200, 100),
        pc_range=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
        fixed_ptsnum_per_line=20,
        eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
        map_classes=['divider', 'ped_crossing', 'boundary']),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=25,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
        dict(
            type='NormalizeMultiviewImage',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1600, 900),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(type='PadMultiViewImage', size_divisor=32),
                dict(
                    type='DefaultFormatBundle3D',
                    with_gt=False,
                    with_label=False,
                    class_names=['divider', 'ped_crossing', 'boundary']),
                dict(type='CustomCollect3D', keys=['img'])
            ])
    ],
    metric='chamfer',
    save_best='NuscMap_chamfer/mAP',
    rule='greater')
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mask2map_newsplit', name='mask2map_newsplit_phase1'))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/M2M_nusc_r50_full_1Phase_12n12ep'
load_from = None
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
voxel_size = [0.15, 0.15, 20.0]
dbound = [1.0, 35.0, 0.5]
grid_config = dict(
    x=[-30.0, -30.0, 0.15],
    y=[-15.0, -15.0, 0.15],
    z=[-10, 10, 20],
    depth=[1.0, 35.0, 0.5])
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
map_classes = ['divider', 'ped_crossing', 'boundary']
num_vec = 50
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag = True
num_map_classes = 3
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 1
bev_h_ = 200
bev_w_ = 100
queue_length = 1
dn_enabled = True
aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=False,
    pv_seg=True,
    seg_classes=1,
    feat_down_sample=32,
    pv_thickness=1)
model = dict(
    type='Mask2Map',
    use_grid_mask=True,
    video_test_mode=False,
    pretrained=dict(img='ckpts/sparsebev_resnet.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=1,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='Mask2MapHead_1Phase',
        bev_h=200,
        bev_w=100,
        num_vec_one2one=50,
        num_pts_per_vec=20,
        num_pts_per_gt_vec=20,
        dir_interval=1,
        num_classes=3,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        dn_enabled=True,
        dn_weight=1,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        aux_seg=dict(
            use_aux_seg=True,
            bev_seg=False,
            pv_seg=True,
            seg_classes=1,
            feat_down_sample=32,
            pv_thickness=1),
        pc_range=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
        transformer=dict(
            type='Mask2Map_Transformer_1Phase',
            rotate_prev_bev=True,
            use_can_bus=True,
            dn_enabled=True,
            dn_group_num=5,
            dn_noise_scale=0.01,
            mask_noise_scale=0.2,
            num_vec_one2one=50,
            embed_dims=256,
            bev_encoder=None,
            bev_neck=dict(
                type='MSDeformAttnPixelDecoder',
                num_outs=4,
                in_channels=[256, 256, 256, 256],
                strides=[1, 2, 4, 8],
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU'),
                encoder=dict(
                    num_layers=6,
                    layer_cfg=dict(
                        self_attn_cfg=dict(
                            embed_dims=256,
                            num_heads=8,
                            num_levels=3,
                            num_points=4,
                            dropout=0.0,
                            batch_first=True),
                        ffn_cfg=dict(
                            embed_dims=256,
                            feedforward_channels=1024,
                            num_fcs=2,
                            ffn_drop=0.0,
                            act_cfg=dict(type='ReLU', inplace=True)))),
                positional_encoding=dict(num_feats=128, normalize=True)),
            encoder=dict(
                type='LSSTransform',
                in_channels=256,
                out_channels=256,
                feat_down_sample=32,
                pc_range=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
                voxel_size=[0.15, 0.15, 20.0],
                dbound=[1.0, 35.0, 0.5],
                downsample=2,
                loss_depth_weight=3.0,
                depthnet_cfg=dict(
                    use_dcn=False, with_cp=False, aspp_mid_channels=96),
                grid_config=dict(
                    x=[-30.0, -30.0, 0.15],
                    y=[-15.0, -15.0, 0.15],
                    z=[-10, 10, 20],
                    depth=[1.0, 35.0, 0.5])),
            segm_decoder=dict(
                type='Mask2FormerTransformerDecoder',
                return_intermediate=True,
                num_layers=3,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.0,
                        batch_first=True),
                    cross_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_seg=dict(type='SimpleLoss', pos_weight=4.0, loss_weight=1.0),
        loss_pv_seg=dict(type='SimpleLoss', pos_weight=1.0, loss_weight=2.0),
        loss_segm_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0, 1.0, 1.0, 0.1]),
        loss_segm_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_segm_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.15, 0.15, 20.0],
            point_cloud_range=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
            out_size_factor=4,
            assigner=dict(
                type='MapTRAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='OrderedPtsL1Cost', weight=5),
                pc_range=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]),
            assigner_segm=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=2.0),
                mask_cost=dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dice_cost=dict(
                    type='DiceCost', weight=5.0, pred_act=True, eps=1.0)))))
optimizer = dict(
    type='AdamW',
    lr=0.0006,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=12)
fp16 = dict(loss_scale=512.0)
find_unused_parameters = True
gpu_ids = range(0, 2)
