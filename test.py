import os
from torch.backends import cudnn

from config import Config
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
from utils.metrics import evaluate
import torch.nn as nn
if __name__ == "__main__":
    cfg = Config()
    log_dir = cfg.LOG_DIR
    cfg.LOG_NAME = cfg.LOG_NAME + "_test"
    cfg.PRETRAIN_CHOICE = "no"
    logger = setup_logger('{}.test'.format(cfg.PROJECT_NAME), log_dir,cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, test_loader, num_query, num_classes = make_dataloader(cfg)
    loss_fn = nn.CrossEntropyLoss()
    model = make_model(cfg, num_classes)
    model.load_param(cfg.TEST_WEIGHT)
    logger.info("-----------val dataloader set-----------")
    evaluate(logger,model, val_loader, loss_fn, cfg.NCROP,cfg.LOG_NAME)
    logger.info("-----------test dataloader set-----------")
    evaluate(logger,model, test_loader, loss_fn,cfg.NCROP,cfg.LOG_NAME)
