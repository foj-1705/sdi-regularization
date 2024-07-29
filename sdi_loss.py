# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step_pos(x):
    return torch.where(x >= 0.0, 1.0, 0.0)

def PM(logit, target):
    eye = torch.eye(10).cuda()
    probs_GT = (logit.softmax(1) * eye[target]).sum(1).detach()
    top2_probs = logit.softmax(1).topk(2, largest = True)
    GT_in_top2_ind = (top2_probs[1] == target.view(-1,1)).float().sum(1) == 1
    probs_2nd = torch.where(GT_in_top2_ind, top2_probs[0].sum(1) - probs_GT, top2_probs[0][:,0]).detach()
    margin = probs_GT - probs_2nd
    return margin


import torch.nn.functional as F

def min_variance_loss(logits, label):

    direc = step_pos(PM(logits, label)).detach()
    #print(direc.item())
    logits_ =  logits.softmax(1)

    eye = torch.eye(logits_.shape[1]).cuda()

    logits_y = (logits_ * eye[label.data]).sum(1)

    logits_y = logits_y.view(-1, 1)

    diff =   logits_ -  logits_y  #torch.mean(logits_, dim=1)            #logits_y

    var  = (torch.sum(torch.square(diff), dim=1)) /(logits_.shape[1] - 1)

    std =  torch.sqrt(var)

    std_loss =  direc*std  #torch.clamp(std, min=-0.1)                         #std  #F.relu(std)

    return  std_loss.mean()
