# -*- coding: utf-8 -*-

from sdi_loss import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



def perturb_example(model,
              x_natural,
              y,
              #training_version,
              #beta = 0.4,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              distance='l_inf'):
    kl = nn.KLDivLoss(size_average=False)
    model.eval()
    
    criterion_kl = nn.KLDivLoss(size_average = False)
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                                       
                loss_ce =  -min_variance_loss(model(x_adv), y)
                #F.cross_entropy(model(x_adv), y)
            
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv
