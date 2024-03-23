# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def min_variance_loss(logits, label):
    
    logits_ =  logits.softmax(1)
    
    eye = torch.eye(logits_.shape[1]).cuda()
    
    logits_y = (logits_ * eye[label.data]).sum(1)
    
    logits_y = logits_y.view(-1, 1)
    
    diff =   logits_ -  logits_y           
    
    
    var  = (torch.sum(torch.square(diff), dim=1)) /(logits_.shape[1])
    
    std =  torch.sqrt(var)
    
    std_loss =  std  
     
    return  std_loss.mean()
