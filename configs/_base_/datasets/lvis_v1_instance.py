# dataset settings
_base_ = 'coco_instance.py'
dataset_type = 'LVISV1Dataset'
data_root = '/home/grantorshadow/Shaunak/mmdetection/data/lvis_v1/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        _delete_=True,
        type='MultiImageMixClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/lvis_v1_val_small.json',
            img_prefix=data_root,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True)],
            filter_empty_gt=False)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val_small.json',
        img_prefix=data_root),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val_small.json',
        img_prefix=data_root))
evaluation = dict(metric=['bbox', 'segm'])
