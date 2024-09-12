# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F


def cross_entropy_loss(pred, target):
    """Cross entropy loss."""
    B, L, C = target.shape
    loss = torch.nn.functional.cross_entropy(
        pred.reshape(-1, C),
        target.reshape(-1, C).argmax(-1),
        reduction="none"
    ).reshape(B, L)

    return loss


def log_sum_exp(x):
    """Numerically stable log_sum_exp implementation that prevents
    overflow.
    """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def myp(name,value): # LZC
    print(name + ':', value.shape)
    # print(name + ':', value.shape, type(value))
def dmll(pred, target, log_scale_min=-7.0, num_classes=256):
    # pred:   torch.Size([26, 1, 30])
    # target: torch.Size([26, 1, 1 ])
    """Discretized mixture of logistic distributions loss
    Note that it is assumed that input is scaled to [-1, 1].

    Code adapted
    from https://github.com/idiap/linear-transformer-experiments/blob/0a540938ec95e1ec5b159ceabe0463d748ba626c/image-generation/utils.py#L31

    Arguments
    ----------
        pred (Tensor): Predicted output (B x L x T)
        target (Tensor): Target (B x L x 1).
        log_scale_min (float): Log scale minimum value
        num_classes (int): Number of classes

    Returns:
    --------
        Tensor: loss
    """

    B, L, C = target.shape
    nr_mix = pred.shape[-1] // 3 # 通过3个参数描述一个L函数，10个L函数预测一个值

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = pred[:, :, :nr_mix]     # 方差variance？
    means = pred[:, :, nr_mix:2 * nr_mix] # 均值mean
    log_scales = torch.clamp(             # 权重weight
        pred[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min
    )
    # logit_probs: torch.Size([26, 1, 10])
    # means:       torch.Size([26, 1, 10])
    # log_scales:  torch.Size([26, 1, 10])

    centered_y = target - means
    # myp('centered_y',centered_y)
    inv_stdv = torch.exp(-log_scales)
    # myp('inv_stdv',inv_stdv)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    # myp('plus_in',plus_in)
    cdf_plus = torch.sigmoid(plus_in)
    # myp('cdf_plus',cdf_plus)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    # myp('min_in',min_in)
    cdf_min = torch.sigmoid(min_in)
    # myp('cdf_min',cdf_min)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(torch.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # myp('log_cdf_plus',log_cdf_plus)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - torch.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)
    # myp('log_one_minus_cdf_min',log_one_minus_cdf_min)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min
    # myp('cdf_delta',cdf_delta)

    mid_in = inv_stdv * centered_y
    # myp('mid_in',mid_in)
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
    # myp('log_pdf_mid',log_pdf_mid)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = (cdf_delta > 1e-5).float()
    # myp('inner_inner_cond',inner_inner_cond)

    inner_inner_out = inner_inner_cond * \
        torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
        (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    # myp('inner_inner_out',inner_inner_out)
    inner_cond = (target > 0.999).float()
    # myp('inner_cond',inner_cond)
    inner_out = inner_cond * log_one_minus_cdf_min + \
        (1. - inner_cond) * inner_inner_out
    # myp('inner_out',inner_out)
    cond = (target < -0.999).float()
    # myp('cond',cond)
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    # myp('log_probs',log_probs)

    log_probs = log_probs + F.log_softmax(logit_probs, -1)
    # myp('log_probs',log_probs)
    # exit(0)
    return -log_sum_exp(log_probs)
