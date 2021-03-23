
""" Configuration file."""

import json
import numpy as np
import os.path as osp
from enum import Enum
from easydict import EasyDict as edict


class phase(Enum):
    TRAIN = 'train'
    VAL = 'valid'
    TESTDEV = 'test-dev'
    TRAINVAL = 'trainval'


__C = edict()

# Public access to configuration settings
cfg = __C

# Number of CPU cores used to parallelize evaluation.
__C.N_JOBS = 32

# Paths to dataset folders
__C.PATH = edict()

__C.PHASE = phase.TRAIN

# Multiobject evaluation (Set to False only when evaluating DAVIS 2016)
__C.MULTIOBJECT = True

# Root folder of project
__C.PATH.ROOT = osp.abspath('.')

# Data folder
__C.PATH.DATA = osp.abspath('/Your_path/YouTube-VOS')

# Path to input images
__C.PATH.SEQUENCES_TRAIN = osp.join(__C.PATH.DATA, phase.TRAIN.value,
                                    "JPEGImages")
__C.PATH.SEQUENCES_VAL = osp.join(__C.PATH.DATA, phase.TRAIN.value,
                                  "JPEGImages")
__C.PATH.SEQUENCES_TRAINVAL = osp.join(__C.PATH.DATA, phase.TRAIN.value,
                                       "JPEGImages")
__C.PATH.SEQUENCES_TEST = osp.join(__C.PATH.DATA, phase.VAL.value,
                                   "JPEGImages")

# Path to annotations
__C.PATH.ANNOTATIONS_TRAIN = osp.join(__C.PATH.DATA, phase.TRAIN.value,
                                      "Annotations")
__C.PATH.ANNOTATIONS_VAL = osp.join(__C.PATH.DATA, phase.TRAIN.value,
                                    "Annotations")
__C.PATH.ANNOTATIONS_TRAINVAL = osp.join(__C.PATH.DATA, phase.TRAIN.value,
                                         "Annotations")
__C.PATH.ANNOTATIONS_TEST = osp.join(__C.PATH.DATA, phase.VAL.value,
                                     "Annotations")

__C.PATH.FLOW = osp.join(__C.PATH.DATA, phase.TRAIN.value, 'YouTube-flow/')
__C.PATH.HED = osp.join(__C.PATH.DATA, phase.TRAIN.value, 'YouTubeVOS_2018-hed/')
__C.PATH.ANNOTATIONS_TRAIN_CTR = osp.join(__C.PATH.DATA, phase.TRAIN.value,
                                           "Annotations_ctr")

__C.PATH.PALETTE = osp.abspath(osp.join(__C.PATH.ROOT, 'libs/dataset/palette.txt'))

# Paths to files
__C.FILES = edict()

# Path to property file, holding information on evaluation sequences.
__C.FILES.DB_INFO_TRAIN = osp.abspath(
    osp.join(__C.PATH.DATA, phase.TRAIN.value, "meta.json"))
# __C.FILES.DB_INFO_VAL = osp.abspath(
#     osp.join(__C.PATH.DATA, phase.TRAIN.value, "train-val-meta.json"))
# __C.FILES.DB_INFO_TRAINVAL = osp.abspath(
#     osp.join(__C.PATH.DATA, phase.TRAIN.value, "meta.json"))
# __C.FILES.DB_INFO_TEST = osp.abspath(
#     osp.join(__C.PATH.DATA, phase.VAL.value, "meta.json"))

# Measures and Statistics
__C.EVAL = edict()

# Metrics: J: region similarity, F: contour accuracy, T: temporal stability
__C.EVAL.METRICS = ['J', 'F']

# Statistics computed for each of the metrics listed above
__C.EVAL.STATISTICS = ['mean', 'recall', 'decay']


def db_read_sequences_train():
    """ Read list of sequences. """

    json_data = open(__C.FILES.DB_INFO_TRAIN)
    data = json.load(json_data)
    sequences = data['videos'].keys()

    # return list(sequences)
    return sequences


# def db_read_sequences_val():
#     """ Read list of sequences. """
#
#     json_data = open(__C.FILES.DB_INFO_VAL)
#     data = json.load(json_data)
#     sequences = data['videos'].keys()
#
#     # return list(sequences)
#     return sequences


# def db_read_sequences_trainval():
#     """ Read list of sequences. """
#
#     json_data = open(__C.FILES.DB_INFO_TRAINVAL)
#     data = json.load(json_data)
#     sequences = data['videos'].keys()
#
#     # return list(sequences)
#     return sequences


# def db_read_sequences_test():
#     """ Read list of sequences. """
#
#     json_data = open(__C.FILES.DB_INFO_TEST)
#     data = json.load(json_data)
#     sequences = data['videos'].keys()
#
#     # return list(sequences)
#     return sequences


# Load all sequences
__C.SEQUENCES_TRAIN = db_read_sequences_train()
# __C.SEQUENCES_VAL = db_read_sequences_val()
# __C.SEQUENCES_TRAINVAL = db_read_sequences_trainval()
# __C.SEQUENCES_TEST = db_read_sequences_test()

__C.palette = np.loadtxt(__C.PATH.PALETTE, dtype=np.uint8).reshape(-1, 3)
