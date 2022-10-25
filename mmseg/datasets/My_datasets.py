# Copyright (c) OpenMMLab. All rights reserved.

# from .builder import DATASETS
from mmseg.datasets.builder import DATASETS
# from .custom import CustomDataset
from mmseg.datasets.custom import CustomDataset
import os.path as osp

@DATASETS.register_module()
class ManyInputChanelConcatDatsets(CustomDataset):
    """Chase_db1 ManyInputChanelConcatDatsets.
    使用该数据集构造方案，需要每个文件夹里的文件名称相同
    需要每个文件放置在data_root根目录下
    img_dir作为基文件，所有的ChannelCat_dir按顺序进行Channel Concate
    """
    # 0，1
    CLASSES = ('background', 'vessel')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self,ChannelCat_dir:list=None, **kwargs)->classmethod:
        self.ChannelCat_dir = ChannelCat_dir
        super(ManyInputChanelConcatDatsets, self).__init__(
            # img_suffix='.png',
            # seg_map_suffix='_1stHO.png',
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)


    # 给results增加数据路径
    # 这里验证一下self.img_dir是否继承自父类
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.Listimg_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

if __name__=='__main__':
    # 测试时候注释 @DATASETS.register_module()
    test = ManyInputChanelConcatDatsets(
    data_root = r'D:\BaiduSyncdisk\09_Code\python-script\gitclone\mmsegmentation\data/',
    ChannelCat_dir = ['my_dataset_2016_12/img_dir/train', 'my_dataset_2016_12/img_dir/train'],
    img_dir = 'my_dataset_2016_12/img_dir/train',
    ann_dir = 'my_dataset_2016_12/ann_dir/train',
    classes = None,
    palette = None,
    pipeline = [
        dict(type='LoadImageFromFile_ConcatChannel', to_float32=True, imdecode_backend='tifffile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        # dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[59.2766, 61.479, 34.658, 59.2766, 61.479, 34.658, 59.2766, 61.479, 34.658, 59.2766, 61.479, 34.658],
            std=[39.5417, 30.968, 31.036, 39.5417, 30.968, 31.036, 39.5417, 30.968, 31.036, 39.5417, 30.968, 31.036],
            to_rgb=False),
        dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])

    test.__getitem__(1)
# 需要重写loading中的数据载入，将img_prefix增加为字典，循环字典合并多个数据