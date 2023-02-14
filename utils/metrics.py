import os
import time

import numpy as np
import torch
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix, accuracy_score, \
    ConfusionMatrixDisplay
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def correct_count(output, target, topk=(1,)):
    """Computes the top k corrrect count for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res

@autocast()
def evaluate(logger, net, dataloader, loss_fn, Ncrop=True, logname="model", device="cuda:0"):
    net = net.eval()
    loss_tr, n_samples = 0.0, 0.0

    y_pred = []
    y_gt = []

    correct_count1 = 0
    correct_count2 = 0
    epoch_time = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            if Ncrop:
                # fuse crops and batchsize
                bs, ncrops, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)

                # forward
                start = time.time()
                outputs = net(inputs)
                end = time.time()
                epoch_time += ( end - start)
                # combine results across the crops
                outputs = outputs.view(bs, ncrops, -1)
                outputs = torch.sum(outputs, dim=1) / ncrops
            else:
                start = time.time()
                outputs = net(inputs)
                end = time.time()
                epoch_time += ( end - start)

            #loss = loss_fn(outputs, labels)

            # calculate performance metrics
            #loss_tr += loss.item()

            # accuracy
            counts = correct_count(outputs, labels, topk=(1, 2))
            correct_count1 += counts[0].item()
            correct_count2 += counts[1].item()

            _, preds = torch.max(outputs.data, 1)
            n_samples += labels.size(0)

            y_pred.extend(pred.item() for pred in preds)
            y_gt.extend(y.item() for y in labels)

    acc1 = 100 * correct_count1 / n_samples
    acc2 = 100 * correct_count2 / n_samples
    epoch_time = epoch_time / n_samples
    fps = 1 / epoch_time
    #loss = loss_tr / n_samples
    logger.info("----------------validation--------------------")
    logger.info("validation. Avg time per image: %2.6f" % epoch_time)
    logger.info("validation. Avg FPS: %2.6f" % fps)
    logger.info("validation. Top 1 Accuracy: %2.6f %%" % acc1)
    logger.info("validation. Top 2 Accuracy: %2.6f %%" % acc2)
   # logger.info("validation. Loss: %2.6f" % loss)
    logger.info("validation. Precision: %2.6f" % precision_score(y_gt, y_pred, average='micro'))
    logger.info("validation. Recall: %2.6f" % recall_score(y_gt, y_pred, average='micro'))
    logger.info("validation. F1 Score: %2.6f" % f1_score(y_gt, y_pred, average='micro'))
    logger.info("validation. Acc: %2.6f"%accuracy_score(y_gt, y_pred))
    cm = confusion_matrix(y_true=y_gt, y_pred=y_pred, normalize='true')
    # 画出混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,2,3,4,5,6,7])
    disp.plot()
    if not os.path.exists("./log/%s/"%logname):
        os.mkdir("./log/%s"%logname)
    plt.savefig('./log/%s/ConfusionMatrix.png'%logname, bbox_inches='tight')

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

# FAR FRR
# test 总image 为1245张，136类
# 类内匹配总次数 约 1245
# 类间匹配总次数 约 136*1245 - 1245 = 135*1245

def Normalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    for i in range(len(data)):
        for j in range((len(data[0]))):
            data[i][j] = ( data[i][j]-m)/(mx-mn)
    return data

def eval_far_frr_palm(distmat, q_pids, g_pids, pid_array):
    """Evaluation with PolyU metric
        Key: compute FAR FRR in different Threshold
        """
    list_a = []
    mask_T = [i/100 for i in range(1,100,1)]
    distmat = Normalize(distmat)
    for T in mask_T:
        mask = (distmat >= T)
        distmat_ = np.multiply(distmat, mask)
        indices = np.argsort(-distmat_, axis=1)
        FAR = 0
        FRR = 0
        TP = 0
        FP = 0
        TN_FP = 0
        TPR = 0
        FPR = 0
        for i in range(len(indices)):
            for j in range(len(g_pids[indices[i]][mask[i][indices[i]]])):
                if not int(q_pids[i, np.newaxis]) == int(g_pids[indices[i]][mask[i][indices[i]]][j]):
                    FAR += 1
                    FP += 1
                    TN_FP += pid_array[int(g_pids[indices[i]][mask[i][indices[i]]][j])]
                else:
                    TP += 1
            TPR += TP / pid_array[int(q_pids[i, np.newaxis])]
            if FP==0:
                pass
            else:
                FPR += FP / TN_FP
            FRR += (pid_array[int(q_pids[i, np.newaxis])] - TP)
            TP = 0
            FP = 0
        TPR /= len(indices)
        FPR /= len(indices)
        print("Threshold:{} FRR:{} FAR:{} TPR:{} FPR:{}".format(T, FRR / 1245, FAR / (135 * 1245),TPR,FPR))
        list_a.append([T, FRR / 1245, FAR / (135 * 1245),TPR,FPR])
    return list_a

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def eval_func_palm(distmat, q_pids, g_pids, q_camids, g_camids ,max_rank=10):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(-distmat, axis=1)
    list_inpid=[]

    for i in range(len(indices)):
        list_inpid.append([int(q_pids[i, np.newaxis]),g_pids[indices[i]]])

    #print(indices)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()

        tmp_cmc = orig_cmc.cumsum()

        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]

        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc

        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

class R1_mAP():
    def __init__(self, num_query, max_rank=50, feat_norm=True, method='euclidean', reranking=False, pid_array=None):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.method = method
        self.reranking=reranking
        self.pid_array = pid_array

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        if self.method == 'euclidean':
            print('=> Computing DistMat with euclidean distance')
            distmat = euclidean_distance(qf, gf)
        elif self.method == 'cosine':
            print('=> Computing DistMat with cosine similarity')
            distmat = cosine_similarity(qf, gf)

        cmc, mAP = eval_func_palm(distmat.cpu().numpy(), q_pids, g_pids, q_camids, g_camids)
        far_frr_line = eval_far_frr_palm(distmat.cpu().numpy(), q_pids, g_pids, self.pid_array)
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf
