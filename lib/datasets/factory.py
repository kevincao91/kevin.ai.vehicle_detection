# --------------------------------------------------------
# PyTorch Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Kevin Cao, based on code from Jianwei Yang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_car import pascal_voc_car

import numpy as np

# Set up voc_face_<year>_<split>
for year in ['2007', '2010']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_car_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc_car(split, year))

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
