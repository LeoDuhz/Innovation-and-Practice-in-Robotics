import mmcv
import matplotlib.pyplot as plt
import os.path as osp
import os
from PIL import Image
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv import Config
from mmseg.apis import set_random_seed
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from numpy.core.records import array
import torch
import numpy as np

def mkdir_or_exist(path):
    exist_flag = osp.exists(path)
    if not exist_flag:
        os.makedirs(path)

#dirs config
data_root = './data/linemod/01'
img_dir = 'images'
ann_dir = 'annotations'

classes = ('others', 'ape')
palette = [[0,0,0], [255,255,255]]

#split train/val dataset randomly
split_dir = 'splits'
mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(data_root, ann_dir), suffix='.png')]
# print('filename:', filename_list)
with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
  # select first 4/5 as train set
  train_length = int(len(filename_list)*4/5)
  f.writelines(line + '\n' for line in filename_list[:train_length])
with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
  # select last 1/5 as train set
  f.writelines(line + '\n' for line in filename_list[train_length:])


#construct dataset class
@DATASETS.register_module()
class LineModDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        print('img_dir:', self.img_dir)
        assert osp.exists(self.img_dir) and self.split is not None

cfg = Config.fromfile('./fast_scnn.py')
# Since we use ony one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head[0].norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head[1].norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 2
cfg.model.auxiliary_head[0].num_classes = 2
cfg.model.auxiliary_head[1].num_classes = 2

cfg.dataset_type = 'LineModDataset'
cfg.data_root = data_root

cfg.data.samples_per_gpu = 8
cfg.data.workers_per_gpu=8

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.crop_size = (512, 512)
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(640, 480), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 480),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
# cfg.data.train.img_dir = img_dir
# cfg.data.train.ann_dir = ann_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = 'splits/train.txt'

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
# cfg.data.val.img_dir = img_dir
# cfg.data.val.ann_dir = ann_dir
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = 'splits/val.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
# cfg.data.test.img_dir = img_dir
# cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = 'splits/val.txt'

# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/'

cfg.runner.max_iters = 10000
cfg.log_config.interval = 10
cfg.evaluation.interval = 200
cfg.checkpoint_config.interval = 200

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# # Build the detector
# model = build_segmentor(
#     cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# # Add an attribute for visualization convenience
# model.CLASSES = datasets[0].CLASSES

# # Create work_dir
# mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# train_segmentor(model, datasets, cfg, distributed=False, validate=True, 
#                 meta=dict())

checkpoint_file = './ape/iter_8000.pth'
config_file = './fast_scnn.py'
model = init_segmentor(cfg, checkpoint_file, device='cuda:0')
# model = build_segmentor(
#     cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

tmp_dir = 'data/linemod/01/images'
save_dir = 'data/linemod/01/our_mask_new'
mkdir_or_exist(save_dir)
for img_name in sorted(os.listdir(tmp_dir))[:]:
    img = mmcv.imread(osp.join(tmp_dir, img_name))

    # model.cfg = cfg
    result = inference_segmentor(model, img)
    # plt.figure(figsize=(8, 6))
    img_array = np.asarray(img)
    # print(len(img_array), len(img_array[0]))
    seg_mat = np.zeros((len(img_array), len(img_array[0])))
    for i in range(len(result[0])):
        for j in range(len(result[0][0])):
            seg_mat[i][j] = result[0][i][j]
    # seg_mat[0][0] = list(seg_mat[0][0])
    # seg_mat[0][0] = seg_mat[0][0][0]
    # print(seg_mat)
    seg_img = Image.fromarray(seg_mat).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    # plt.figure(figsize=(8, 6))
    # im = plt.imshow(np.array(seg_img.convert('RGB')))
    seg_img.save(osp.join(save_dir, img_name))

# plt.show()
# show_result_pyplot(model, img, result, palette)
