# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import torch
import torch.nn as nn


class FixedPositionalEncoding(nn.Module):
    def __init__(self, proj_dims, val=0.1):
        super().__init__()
        ll = proj_dims//2
        exb = 2 * torch.linspace(0, ll-1, ll) / proj_dims
        '''
        exb:
        [0.0000, 0.0312, 0.0625, 0.0938, 0.1250, 0.1562, 0.1875, 0.2188, 0.2500,
        0.2812, 0.3125, 0.3438, 0.3750, 0.4062, 0.4375, 0.4688, 0.5000, 0.5312,
        0.5625, 0.5938, 0.6250, 0.6562, 0.6875, 0.7188, 0.7500, 0.7812, 0.8125,
        0.8438, 0.8750, 0.9062, 0.9375, 0.9688]
        '''
        self.sigma = 1.0 / torch.pow(val, exb).view(1, -1)
        '''
        torch.pow(0.1, exb): 
        [1.0000, 0.9306, 0.8660, 0.8058, 0.7499, 0.6978, 0.6494, 0.6043, 0.5623,
        0.5233, 0.4870, 0.4532, 0.4217, 0.3924, 0.3652, 0.3398, 0.3162, 0.2943,
        0.2738, 0.2548, 0.2371, 0.2207, 0.2054, 0.1911, 0.1778, 0.1655, 0.1540,
        0.1433, 0.1334, 0.1241, 0.1155, 0.1075]
        
        torch.pow(val, exb).view(1, -1) :
        [[1.0000, 0.9306, 0.8660, 0.8058, 0.7499, 0.6978, 0.6494, 0.6043, 0.5623,
         0.5233, 0.4870, 0.4532, 0.4217, 0.3924, 0.3652, 0.3398, 0.3162, 0.2943,
         0.2738, 0.2548, 0.2371, 0.2207, 0.2054, 0.1911, 0.1778, 0.1655, 0.1540,
         0.1433, 0.1334, 0.1241, 0.1155, 0.1075]]
        
        1.0 / torch.pow(val, exb).view(1, -1):
        [[1.0000, 1.0746, 1.1548, 1.2409, 1.3335, 1.4330, 1.5399, 1.6548, 1.7783,
         1.9110, 2.0535, 2.2067, 2.3714, 2.5483, 2.7384, 2.9427, 3.1623, 3.3982,
         3.6517, 3.9242, 4.2170, 4.5316, 4.8697, 5.2330, 5.6234, 6.0430, 6.4938,
         6.9783, 7.4989, 8.0584, 8.6596, 9.3057]]
        '''
        self.sigma = 2 * torch.pi * self.sigma
        '''
        [[ 6.2832,  6.7520,  7.2557,  7.7970,  8.3788,  9.0039,  9.6756, 10.3975,
         11.1733, 12.0069, 12.9027, 13.8653, 14.8998, 16.0114, 17.2060, 18.4897,
         19.8692, 21.3516, 22.9446, 24.6564, 26.4960, 28.4728, 30.5971, 32.8799,
         35.3329, 37.9691, 40.8019, 43.8460, 47.1172, 50.6326, 54.4101, 58.4696]]
        '''

    def forward(self, x):
        '''
        x : [26, 8, 1]
        self.sigma : [1, 32]
        x * self.sigma : [26, 8, 32]
        torch.sin(x * self.sigma) : [26, 8, 32]
        torch.Size([26, 8, 64])
        '''
        return torch.cat([
            torch.sin(x * self.sigma.to(x.device)),
            torch.cos(x * self.sigma.to(x.device))
        ], dim=-1)


def sample_from_dmll(pred, num_classes=256):
    """Sample from mixture of logistics.

    Arguments
    ---------
        pred: NxC where C is 3*number of logistics
    """
    assert len(pred.shape) == 2

    N = pred.size(0)
    nr_mix = pred.size(1) // 3

    probs = torch.softmax(pred[:, :nr_mix], dim=-1)
    means = pred[:, nr_mix:2 * nr_mix]
    scales = torch.nn.functional.elu(pred[:, 2*nr_mix:3*nr_mix]) + 1.0001

    indices = torch.multinomial(probs, 1).squeeze()
    batch_indices = torch.arange(N, device=probs.device)
    mu = means[batch_indices, indices]
    s = scales[batch_indices, indices]
    u = torch.rand(N, device=probs.device)
    preds = mu + s*(torch.log(u) - torch.log(1-u))

    return torch.clamp(preds, min=-1, max=1)[:, None]


def optimizer_factory(config, parameters):
    """Based on the input arguments create a suitable optimizer object."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)

    if optimizer == "SGD":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    elif optimizer == "Adam":
        return torch.optim.Adam(parameters, lr=lr)
    else:
        raise NotImplementedError()
