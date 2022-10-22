# 多个数据在Chanel方向进行Concatenate
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=5,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='ASPPHead',
        in_channels=64,
        in_index=4,
        channels=16,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',use_sigmoid=False, loss_weight=1.0)),#, alpha=0.2,beta=0.8
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),#,alpha=0.2,beta=0.8
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(85, 85)))

div = [1,1,1,100,0.001]  # 量纲统一
mean = [68.250244, 64.657715, 35.8221, 40.245823, -3.22352]
std = [45.502422, 38.297836, 38.109894, 4.4519363, 5.37972]
img_scale = (512, 512) ; ratio_range=(0.5, 2.0)
crop_size = (256, 256)
pad_size = (256, 256)

dataset_type = 'ManyInputChanelConcatDatsets'
data_root = 'data/'
img_norm_cfg = dict(
    mean=mean, std=std, to_rgb=False)#mean=[59.2766, 61.479, 34.658], std=[39.5417, 30.968, 31.036]

train_pipeline = [
    dict(type='LoadImageFromFile_ConcatChannel',to_float32=True,imdecode_backend='tifffile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=img_scale, ratio_range=ratio_range),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion',Saturation=True,Hue=True,channel=[0,1,2]),
    dict(type='Div',div=div),
    dict(
        type='Normalize',
        mean=mean, std=std,
        to_rgb=False),
    dict(type='Pad', size=pad_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile_ConcatChannel',to_float32=True,imdecode_backend='tifffile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Div', div=div),
            dict(
                type='Normalize',
                mean=mean, std=std,
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

train1=dict(
    type='ManyInputChanelConcatDatsets',
    data_root=r'data/',
    ChannelCat_dir=['my_dataset_DEM/img_dir/train', 'my_dataset_Velocity/img_dir/train'],

    img_dir='my_dataset_2022_3q/img_dir/train',
    ann_dir='my_dataset_2022_3q/ann_dir/train',

    pipeline=[
        dict(type='LoadImageFromFile_ConcatChannel', to_float32=True, imdecode_backend='tifffile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='Resize', img_scale=img_scale, ratio_range=ratio_range),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion',Saturation=True,Hue=True,channel=[0,1,2]),
        dict(type='Div', div=div),
        dict(
            type='Normalize',
            mean=mean,
            std=std,
            to_rgb=False),
        dict(type='Pad', size=pad_size, pad_val=0, seg_pad_val=0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])

train2=dict(
    type='ManyInputChanelConcatDatsets',
    data_root=r'data/',
    ChannelCat_dir=['my_dataset_DEM/img_dir/train', 'my_dataset_Velocity/img_dir/train'],

    img_dir='my_dataset_2021_08/img_dir/train',
    ann_dir='my_dataset_2021_08/ann_dir/train',

    pipeline=[
        dict(type='LoadImageFromFile_ConcatChannel', to_float32=True, imdecode_backend='tifffile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='Resize', img_scale=img_scale, ratio_range=ratio_range),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion',Saturation=True,Hue=True,channel=[0,1,2]),
        dict(type='Div', div=div),
        dict(
            type='Normalize',
            mean=mean,
            std=std,
            to_rgb=False),
        dict(type='Pad', size=pad_size, pad_val=0, seg_pad_val=0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])

train3=dict(
    type='ManyInputChanelConcatDatsets',
    data_root=r'data/',
    ChannelCat_dir=['my_dataset_DEM/img_dir/train', 'my_dataset_Velocity/img_dir/train'],

    img_dir='my_dataset_2016_12/img_dir/train',
    ann_dir='my_dataset_2016_12/ann_dir/train',

    pipeline=[
        dict(type='LoadImageFromFile_ConcatChannel', to_float32=True, imdecode_backend='tifffile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='Resize', img_scale=img_scale, ratio_range=ratio_range),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion',Saturation=True,Hue=True,channel=[0,1,2]),
        dict(type='Div', div=div),
        dict(
            type='Normalize',
            mean=mean,
            std=std,
            to_rgb=False),
        dict(type='Pad', size=pad_size, pad_val=0, seg_pad_val=0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=[train1,train2,train3],
        #     img_dir=['my_dataset_2016_12/img_dir/train','my_dataset_2021_08/img_dir/train'
        # 'my_dataset_2022_3q/img_dir/train'],
        # ann_dir=['my_dataset_2016_12/ann_dir/train','my_dataset_2021_08/ann_dir/train'
        # 'my_dataset_2022_3q/ann_dir/train'],#'my_dataset_2022_3q/ann_dir/train',
    val=dict(
        type='ManyInputChanelConcatDatsets',
        data_root=r'data/',
        ChannelCat_dir=['my_dataset_DEM/img_dir/val', 'my_dataset_Velocity/img_dir/val'],
        img_dir='my_dataset_2022_3q/img_dir/val',
        ann_dir='my_dataset_2022_3q/ann_dir/val',
        pipeline=[
            dict(type='LoadImageFromFile_ConcatChannel',to_float32=True,imdecode_backend='tifffile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=img_scale,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Div', div=div),
                    dict(
                        type='Normalize',
                        mean=mean, std=mean,
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='ManyInputChanelConcatDatsets',
        data_root=r'data/',
        ChannelCat_dir=['my_dataset_DEM/img_dir/val', 'my_dataset_Velocity/img_dir/val'],
        img_dir='my_dataset_2022_3q/img_dir/val',
        ann_dir='my_dataset_2022_3q/ann_dir/val',
        pipeline=[
            dict(type='LoadImageFromFile_ConcatChannel',to_float32=True,imdecode_backend='tifffile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=img_scale,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Div', div=div),
                    dict(
                        type='Normalize',
                        mean=mean, std=std,
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))


log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')

log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
#evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
work_dir = '.\mmseg_log\2022-3qtest'
gpu_ids = [0]
auto_resume = False
