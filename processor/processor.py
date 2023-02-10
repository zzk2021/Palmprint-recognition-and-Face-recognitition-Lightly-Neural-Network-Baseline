import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torchvision.utils as vutils
from utils.meter import AverageMeter
from utils.metrics import R1_mAP, evaluate
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast

def mixup_criterion(criterion, cls_score, global_feat, y_a, y_b, lam):
    return lam * criterion(cls_score, global_feat, y_a) + (1 - lam) * criterion(cls_score, global_feat, y_b)

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    device = true_labels.device
    true_labels = torch.nn.functional.one_hot(
        true_labels, classes).detach().cpu()
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(
            size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)

        true_dist.scatter_(1, torch.LongTensor(
            index.unsqueeze(1)), confidence)
    return true_dist.to(device)

def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def do_train_fer2013(cfg, model, train_loader, loss_fn, optimizer, val_loader, scheduler, device="cuda:0"):
    logger = logging.getLogger('{}.train'.format(cfg.PROJECT_NAME))
    logger.info('start training')
    log_period = cfg.LOG_PERIOD
    checkpoint_period = cfg.CHECKPOINT_PERIOD
    eval_period = cfg.EVAL_PERIOD
    scaler = GradScaler()
    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        count = 0
        correct = 0
        train_loss = 0
        start_t = time.time()
        model.train()
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            with autocast():

                if cfg.NCROP:
                    bs, ncrops, c, h, w = images.shape
                    images = images.view(-1, c, h, w)
                    labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)

                if cfg.MIXUP:
                    images, labels_a, labels_b, lam = mixup_data(
                        images, labels, cfg.MIXUP_ALPHA)
                    images, labels_a, labels_b = map(
                        Variable, (images, labels_a, labels_b))

                cls_score, global_feat = model(images)

                if cfg.MIXUP:
                    # mixup
                    loss = mixup_criterion(
                        loss_fn, cls_score, global_feat, labels_a, labels_b, lam)
                else:
                    # normal CE
                    loss = loss_fn(cls_score, global_feat, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss
            _, preds = torch.max(cls_score, 1)
            correct += torch.sum(preds == labels.data).item()
            count += labels.shape[0]

        scheduler.step()
        epoch_time = time.time() - start_t
        logger.info("Epoch: %d\t Epoch_time: %.4f \t train loss:%.4f \t training acc:%.4f \t",
                    epoch, epoch_time, train_loss / count, correct / count)
        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            evaluate(logger, model, val_loader, nn.CrossEntropyLoss(),cfg.NCROP)

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query):
    log_period = cfg.LOG_PERIOD
    checkpoint_period = cfg.CHECKPOINT_PERIOD
    eval_period = cfg.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.MAX_EPOCHS

    logger = logging.getLogger('{}.train'.format(cfg.PROJECT_NAME))
    logger.info('start training')

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM)
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        for n_iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            if 'center' in cfg.LOSS_TYPE:
                optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)

            score, feat = model(img, target)
            loss = loss_fn(score, feat, target)

            loss.backward()
            optimizer.step()
            if 'center' in cfg.LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            acc = (score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
        scheduler.step()
        end_time = time.time()
        time_per_batch = (end_time - start_time)
        logger.info("Epoch {} done. Time: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            evaluate(logger, model, val_loader, nn.CrossEntropyLoss())

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger('{}.test'.format(cfg.PROJECT_NAME))
    logger.info("Enter inferencing")
    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM, \
                       method=cfg.TEST_METHOD, reranking=cfg.RERANKING)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, pid, camid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)

            if cfg.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, distmat, pids, camids, qfeats, gfeats = evaluator.compute()

    np.save(os.path.join(cfg.LOG_DIR, cfg.DIST_MAT) , distmat)
    np.save(os.path.join(cfg.LOG_DIR, cfg.PIDS), pids)
    np.save(os.path.join(cfg.LOG_DIR, cfg.CAMIDS), camids)
    np.save(os.path.join(cfg.LOG_DIR, cfg.IMG_PATH), img_path_list[num_query:])
    torch.save(qfeats, os.path.join(cfg.LOG_DIR, cfg.Q_FEATS))
    torch.save(gfeats, os.path.join(cfg.LOG_DIR, cfg.G_FEATS))

    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
