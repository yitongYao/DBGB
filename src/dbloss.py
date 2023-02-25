import torch
from torch import nn
import random
import math
import numpy as np
def get_inv_propensity(n,count_vector:list, a=0.55, b=1.5,othernum=0):
    n = n
    number = np.append(count_vector,[0]*(othernum-len(count_vector)))
    c = (np.log(n) - 1) * ((b + 1) ** a)
    return 1.0 + c * (number + b) ** (-a)
def dbloss_mask(logits,labels, freq, maskNum=0, headnum = 0,wl =1,train_num=10000):
    weight = torch.ones(labels.shape).cuda()
    alpha = 0.1
    n = train_num
    freq = get_inv_propensity(n, freq)
    freq = torch.Tensor(freq).cuda()
    div = wl+freq*alpha
    weight = 1/div* (1 - labels) + 1 * labels
    if maskNum > 0.0:
        for i in range(labels.shape[0]):
            idxs = torch.nonzero(labels[i,:headnum]).squeeze(1).detach().cpu().tolist()
            if len(idxs) > 1:
                num = math.ceil(len(idxs)*maskNum)
                L1 = random.sample(range(0, len(idxs)), num)
                for j in L1:
                    weight[i][idxs[j]] = 0
    cls_criterion = nn.BCEWithLogitsLoss(weight=weight)
    cls_score = logits*(1-labels)*div+logits*labels
    loss = cls_criterion(cls_score, labels.float())
    return loss