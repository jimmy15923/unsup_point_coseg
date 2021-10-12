# Unsupervised object co-segmentation/config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.EXP = CN()
_C.EXP.NAME = "test"
_C.EXP.DATASET = "scanobjectnn"
_C.EXP.PRETRAIN_MODEL_PATH = './work_dirs/pretrain_modelnet40_aug_cls/model.t7'
_C.EXP.DATA_PATH = '/home/jimmy15923/mnt/data/scanobjectnn/h5_files/main_split'

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 48
_C.TRAIN.N_EPOCHS = 2000
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.SCHEDULER = 'cos'
_C.TRAIN.LR = 0.007
_C.TRAIN.TEMP = 0.07
_C.TRAIN.N_POS_PAIRS = 1000
_C.TRAIN.N_NEG_PAIRS = 5000
_C.TRAIN.N_POINTS = 2048

_C.MODEL = CN()
_C.MODEL.N_FG = 512
_C.MODEL.N_BG = 512
_C.MODEL.ALPHA = 30
_C.MODEL.LAMDA = 1.0
_C.MODEL.FG_GAMMA = 1.0
_C.MODEL.BG_GAMMA = 1.0
_C.MODEL.GROUP_SIZE = 7
_C.MODEL.REPULSION = 1.0

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()