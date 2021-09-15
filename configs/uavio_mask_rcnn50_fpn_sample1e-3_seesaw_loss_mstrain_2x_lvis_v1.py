_base_ = [
	'/home/grantorshadow/Shaunak/mmdetection/configs/_base_/models/mask_rcnn_r50_fpn.py',
	'/home/grantorshadow/Shaunak/mmdetection/configs/_base_/schedules/schedule_2x.py',
	'/home/grantorshadow/Shaunak/mmdetection/configs/_base_/default_runtime.py',
	'/home/grantorshadow/Shaunak/mmdetection/configs/_base_/datasets/coco_instance.py'
]

# dataset settings
dataset_type = 'LVISV1Dataset'
data_root = '/home/grantorshadow/Shaunak/mmdetection/data/lvis_v1/'

model = dict(
	roi_head=dict(
		bbox_head=dict(
			num_classes=1203,
			cls_predictor_cfg=dict(type='NormedLinear', tempearture=20),
			loss_cls=dict(
				type='SeesawLoss',
				p=0.8,
				q=2.0,
				num_classes=1203,
				loss_weight=1.0)),
		mask_head=dict(num_classes=1203)),
	test_cfg=dict(
		rcnn=dict(
			score_thr=0.0001,
			# LVIS allows up to 300
			max_per_img=300)))
img_norm_cfg = dict(
	mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
	dict(type='CopyPaste'),
	dict(
		type='Resize',
		img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
				   (1333, 768), (1333, 800)],
		multiscale_mode='value',
		keep_ratio=True),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=32),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(
	samples_per_gpu=1,
	workers_per_gpu=0,
	train=dict(
		_delete_=True,
		type='ClassBalancedDataset',  # TODO remove pipeline arg if used for class balanced / keep for multiimagemixeddataset
		oversample_thr=1e-3,
		dataset=dict(
			type=dataset_type,
			ann_file=data_root + 'annotations/lvis_v1_val.json',
			img_prefix=data_root),
		pipeline=[
			dict(type='LoadImageFromFile'),
			dict(type='LoadAnnotations', with_bbox=True, with_mask=True)]
		),
	val=dict(
		type=dataset_type,
		ann_file=data_root + 'annotations/lvis_v1_val.json',
		img_prefix=data_root),
	test=dict(
		type=dataset_type,
		ann_file=data_root + 'annotations/lvis_v1_val.json',
		img_prefix=data_root))
evaluation = dict(metric=['bbox', 'segm'])

# data = dict(train=dict(dataset=dict(pipeline=train_pipeline))) # TODO keep if used in balanced class remove if used for multiclass
# evaluation = dict(interval=12, metric=['bbox', 'segm'])
