from .default import DefaultConfig


# class Config(DefaultConfig):
#     """
#     mAP 85.8, Rank1 94.1, @epoch 175
#     """
#     def __init__(self):
#         super(Config, self).__init__()
#         self.CFG_NAME = 'baseline'
#         self.DATA_DIR = '/nfs/public/datasets/person_reid/Market-1501-v15.09.15'
#         self.PRETRAIN_CHOICE = 'imagenet'
#         self.PRETRAIN_PATH = '/nfs/public/pretrained_models/resnet50-19c8e357.pth'
#
#         self.LOSS_TYPE = 'triplet+softmax+center'
#         self.TEST_WEIGHT = './output/resnet50_175.pth'
#         self.FLIP_FEATS = 'on'


class Config(DefaultConfig):
    """
    mAP 86.2, Rank1 94.4, @epoch 185
    """

    def __init__(self):
        super(Config, self).__init__()
        self.CFG_NAME = 'baseline'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = 'F:\行人重识别-论文\person-reid-tiny-baseline-master\pretrained_model\snetv2_1.0x_wnet1x_M2G2.model'
        self.MODEL_NAME =  "WeightNet_shufflenetv2"  # backbone name, option: 'resnet50',
        if self.MODEL_NAME =="pyramidTNT-Ti":
            self.INPUT_SIZE = [192,192]
        self.LOSS_TYPE = 'triplet+softmax'
        self.TEST_WEIGHT = ''
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False

# class Config(DefaultConfig):
#     def __init__(self):
#         super(Config, self).__init__()
#         self.CFG_NAME = 'baseline'
#         self.DATA_DIR = '/nfs/public/datasets/person_reid/Market-1501-v15.09.15'
#         self.PRETRAIN_CHOICE = 'imagenet'
#         self.PRETRAIN_PATH = '/nfs/public/pretrained_models/resnet50-19c8e357.pth'
#         self.COS_LAYER = True
#         self.LOSS_TYPE = 'softmax'
#         self.TEST_WEIGHT = './output/resnet50_185.pth'
#         self.FLIP_FEATS = 'off'
#         self.RERANKING = True
