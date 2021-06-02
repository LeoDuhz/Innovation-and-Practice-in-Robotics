import mmcv
import matplotlib.pyplot as plt
import os.path as osp
from PIL import Image
import numpy as np
# from mmseg.datasets.builder import DATASETS
# from mmseg.datasets.custom import CustomDataset
# from mmcv import Config
# from mmseg.apis import set_random_seed
import matplotlib.patches as mpatches

classes = ('others', 'ape')
palette = [[0,0,0], [255,255,255]]
seg_map = Image.open('/home/zjunlict/dhz/NeuroRobotics/REDE/datasets/linemod/data/MilkBro/annotations/training/0000.png')
seg_img = seg_map.convert('P')
seg_img.putpalette(np.array(palette, dtype=np.uint8))
# img = Image.open('/home/zjunlict/dhz/NeuroRobotics/REDE/datasets/linemod/data/MilkBro/annotations/training/0000.png')
plt.figure(figsize=(8, 6))
im = plt.imshow(np.array(seg_img.convert('RGB')))

# create a patch (proxy artist) for every color 
patches = [mpatches.Patch(color=np.array(palette[i])/255., 
                          label=classes[i]) for i in range(2)]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
           fontsize='large')

plt.show()