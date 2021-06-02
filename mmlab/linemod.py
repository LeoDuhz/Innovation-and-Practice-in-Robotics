import mmcv
import matplotlib.pyplot as plt
import os.path as osp
from PIL import Image
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv import Config

#dirs config
data_root = 'linemod'
img_dir = 'images'
ann_dir = 'annotations'

classes = ('ape', 'others')
palette = [[255,255,255], [0,0,0]]

#TODO:split train/val dataset randomly

#construct dataset class
@DATASETS.register_module()
class LineModDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exits(self.img_dir) and self.split is not None


    