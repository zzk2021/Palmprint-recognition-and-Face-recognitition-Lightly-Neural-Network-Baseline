import os

from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .Market1501 import Market1501
from .bases import ImageDataset
from .ck import CK
from .fer2013 import load_data, prepare_data, CustomDataset
from .ref_db import TxtImage
from .sampler import RandomIdentitySampler

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths

def make_dataloader(cfg):
    mu, st = 0.5, 0.5
    if cfg.NCROP:
        train_transforms = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(cfg.INPUT_SIZE),
            T.RandomResizedCrop(48, scale=(0.8, 1.2)),
            T.RandomApply([T.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            T.RandomApply(
                [T.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.RandomRotation(10)], p=0.5),
            T.FiveCrop(40),
            T.Lambda(lambda crops: torch.stack(
                [T.ToTensor()(crop) for crop in crops])),
            T.Lambda(lambda tensors: torch.stack(
                [T.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            T.Lambda(lambda tensors: torch.stack(
                [T.RandomErasing()(t) for t in tensors])),
        ])

        val_transforms = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(cfg.INPUT_SIZE),
            T.FiveCrop(40),
            T.Lambda(lambda crops: torch.stack(
                [T.ToTensor()(crop) for crop in crops])),
             T.Lambda(lambda tensors: torch.stack(
                [T.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
        ])

    else:
        train_transforms = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(cfg.INPUT_SIZE),
            T.RandomResizedCrop(48, scale=(0.8, 1.2)),
            T.RandomApply([T.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            T.RandomApply(
                [T.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.RandomRotation(10)], p=0.5),
            #T.FiveCrop(40),
           # T.Lambda(lambda crops: torch.stack(
            #    [T.ToTensor()(crop) for crop in crops])),
            #T.Lambda(lambda tensors: torch.stack(
            #    [T.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            #T.Lambda(lambda tensors: torch.stack(
            #    [T.RandomErasing()(t) for t in tensors])),
            T.ToTensor(),
            T.Normalize(mean=(mu,), std=(st,)),
            T.RandomErasing()
        ])

        val_transforms = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(cfg.INPUT_SIZE),
            #T.FiveCrop(40),
            #T.Lambda(lambda crops: torch.stack(
            #    [T.ToTensor()(crop) for crop in crops])),
            #T.Lambda(lambda tensors: torch.stack(
            #    [T.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            T.ToTensor(),
            T.Normalize(mean=(mu,), std=(st,)),
        ])

    num_workers = cfg.DATALOADER_NUM_WORKERS
    num_classes = 0

    if cfg.DATASET_NAME == "fer2013":
        num_classes = 7
        fer2013, emotion_mapping = load_data(cfg.DATA_DIR)
        xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
        xval, yval = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
        xtest, ytest = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

        train = CustomDataset(xtrain, ytrain, train_transforms)
        val = CustomDataset(xval, yval, val_transforms)
        test = CustomDataset(xtest, ytest, val_transforms)
        #train_loader = DataLoader(train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.DATALOADER_NUM_WORKERS)
        val_loader = DataLoader(val, batch_size=cfg.TEST_IMS_PER_BATCH, shuffle=True, num_workers=cfg.DATALOADER_NUM_WORKERS)
        test_loader = DataLoader(test, batch_size=cfg.TEST_IMS_PER_BATCH, shuffle=True, num_workers=cfg.DATALOADER_NUM_WORKERS)

    elif cfg.DATASET_NAME == "ref_db":

        TestlabelPath = os.path.join(cfg.DATA_DIR, '/basic/alignByMyself/test.txt')
        TrainlabelPath = os.path.join(cfg.DATA_DIR, '/basic/alignByMyself/train.txt')
        ImageRoot = cfg.DATA_DIR + '/basic/Image/aligned'  # aligned

        with open(TestlabelPath, 'r') as testf:
            Testlabels = testf.readlines()
        TestDataset = TxtImage(Testlabels, ImageRoot, val_transforms, index=9)
        val_loader = DataLoader(TestDataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.DATALOADER_NUM_WORKERS)

        with open(TrainlabelPath, 'r') as trainf:
            Trainlabels = trainf.readlines()
        train = TxtImage(Trainlabels, ImageRoot, train_transforms, index=11)

    elif cfg.DATASET_NAME == "ck+":
        train = CK(split='Training', transform=train_transforms)
        TestDataset = CK(split='Testing', transform=val_transforms)
        val_loader = DataLoader(TestDataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.DATALOADER_NUM_WORKERS)

    else:
        train_set = ImageDataset(dataset.train, train_transforms)

    if cfg.SAMPLER == 'triplet':
        print('using triplet sampler')
        train_loader = DataLoader(train,
                                  batch_size=cfg.BATCH_SIZE,
                                  num_workers=num_workers,
                                  sampler=RandomIdentitySampler(dataset.train, cfg.BATCH_SIZE, cfg.NUM_IMG_PER_ID),
                                  collate_fn=train_collate_fn  # customized batch sampler
                                  )

    elif cfg.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(train,
                                  batch_size=cfg.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  sampler=None,
                                  collate_fn=train_collate_fn,  # customized batch sampler
                                  drop_last=True
                                  )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    return train_loader, val_loader, test_loader, 0, num_classes
