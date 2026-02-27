from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'transformer'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224_TransOSS'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.USE_AMP = True

# Modality Information Embeddings
_C.MODEL.MIE_COE = 3.0
_C.MODEL.MIE = False

# Ship Size Embeddings
_C.MODEL.SSE = False

# Pretrain with images pairs
_C.MODEL.PAIR = False

_C.MODEL.RGB_CHANNELS = [0, 1, 2]
_C.MODEL.SAR_CHANNELS = [10, 11]

# -----------------------------------------------------------------------------
# iBOT pretraining
# -----------------------------------------------------------------------------
_C.IBOT = CN()
_C.IBOT.OUT_DIM = 8192
_C.IBOT.PATCH_OUT_DIM = 8192
_C.IBOT.SHARED_HEAD = False
_C.IBOT.SHARED_HEAD_TEACHER = True
_C.IBOT.NORM_LAST_LAYER = True
_C.IBOT.NORM_IN_HEAD = ""
_C.IBOT.ACT_IN_HEAD = "gelu"
_C.IBOT.HEAD_NLAYERS = 3
_C.IBOT.HEAD_HIDDEN_DIM = 2048
_C.IBOT.HEAD_BOTTLENECK_DIM = 256
_C.IBOT.GLOBAL_CROPS_NUMBER = 2
_C.IBOT.LOCAL_CROPS_NUMBER = 10
_C.IBOT.GLOBAL_CROPS_SCALE = (0.32, 1.0)
_C.IBOT.LOCAL_CROPS_SCALE = (0.05, 0.32)
_C.IBOT.GLOBAL_SIZE = 224
_C.IBOT.LOCAL_SIZE = 96
_C.IBOT.PRED_RATIO = (0.0, 0.7)
_C.IBOT.PRED_RATIO_VAR = (0.0, 0.05)
_C.IBOT.PRED_SHAPE = "rand"
_C.IBOT.PRED_START_EPOCH = 0
_C.IBOT.WARMUP_TEACHER_TEMP = 0.04
_C.IBOT.TEACHER_TEMP = 0.06
_C.IBOT.WARMUP_TEACHER_PATCH_TEMP = 0.04
_C.IBOT.TEACHER_PATCH_TEMP = 0.06
_C.IBOT.WARMUP_TEACHER_TEMP_EPOCHS = 30
_C.IBOT.MOMENTUM_TEACHER = 0.996
_C.IBOT.LAMBDA1 = 1.0
_C.IBOT.LAMBDA2 = 1.0
_C.IBOT.CLIP_GRAD = 0.5
_C.IBOT.FREEZE_LAST_LAYER_EPOCHS = 1
_C.IBOT.SAMPLING_SUBSET = True
_C.IBOT.SYNC_CHANNEL_COUNT = True
_C.IBOT.MODALITY_PURE_SAMPLING = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5

_C.INPUT.PIXEL_MEAN_RGB = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD_RGB = [0.229, 0.224, 0.225]
_C.INPUT.PIXEL_MEAN_SAR = [0.340619, 0.340619, 0.340619]
_C.INPUT.PIXEL_STD_SAR = [0.276758, 0.276758, 0.276758]

# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('HOSS')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('/data/ship')
# Evaluation mode for datasets: 'rgb_sar', 'sar_rgb', 'rgb_mixed', 'sar_mixed' or 'all'
_C.DATASETS.EVAL_MODE = 'all'


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Whether to force deterministic algorithms/kernels (slower but reproducible)
_C.SOLVER.DETERMINISTIC = True
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# LR scheduler type: 'cosine', 'step', or 'wsd'
_C.SOLVER.SCHEDULER_TYPE = 'cosine'
# WSD scheduler: fraction of total epochs for the decay phase
_C.SOLVER.WSD_DECAY_PCT = 0.1
# WSD scheduler: target total epochs for which extra checkpoints are saved
# e.g. (20, 40, 80, 100) -> checkpoints at floor((1-WSD_DECAY_PCT)*T) for each T
_C.SOLVER.WSD_DECAY_TARGETS = ()

# epoch number of saving checkpoints (-1 to disable periodic saving)
_C.SOLVER.CHECKPOINT_PERIOD = 10
# save latest checkpoint after every epoch
_C.SOLVER.SAVE_LATEST_EVERY_EPOCH = True
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# Whether to compute/log gradient norm diagnostics
_C.SOLVER.LOG_GRAD_NORM = True
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64
# Number of gradient accumulation micro-steps per optimizer update
_C.SOLVER.GRAD_ACCUM_STEPS = 1
_C.SOLVER.TRACK_VALIDATION_METRICS = True
_C.SOLVER.TRACK_VALIDATION_METRICS_OPTISAR = True
_C.SOLVER.PRETRAIN_TRACK_VALIDATION_DIR = ''
_C.SOLVER.RESUME_FROM = ""
_C.SOLVER.USE_MULTI_PRETRAIN = False

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = True
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""

# ---------------------------------------------------------------------------- #
# Weights & Biases
# ---------------------------------------------------------------------------- #
_C.WANDB = CN()
_C.WANDB.PROJECT = "vessel-reidentification"
_C.WANDB.NAME = ""
_C.WANDB.MODE = "online"
_C.WANDB.ALLOW_NO_KEY_FALLBACK = True
