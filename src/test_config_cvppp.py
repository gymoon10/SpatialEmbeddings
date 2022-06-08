"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms

CVPPP_DIR = 'E:/CVPPP2017_LSC_training'

args = dict(

    cuda=True,
    display=True,

    save=True,
    save_dir='C:/Users/Moon/Desktop/SpatialEmbedding/src/masks/',
    save_dir1='C:/Users/Moon/Desktop/SpatialEmbedding/src/inference/',  # additional
    checkpoint_path='C:/Users/Moon/Desktop/SpatialEmbedding/src/exp/10/best_iou_model.pth',

    dataset={
        'name': 'cvppp2',
        'kwargs': {
            'root_dir': CVPPP_DIR,
            'type_': 'val',
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        }
    },

    model={
        # follow the train_config_cvppp
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [3, 1],
        }
    }
)


def get_args():
    return copy.deepcopy(args)
