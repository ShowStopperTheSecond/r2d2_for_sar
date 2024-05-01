# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.sampler import FullSampler

# import matplotlib.pyplot as plt
# import matplotlib

'''

GTK3Agg
GTK3Cairo
GTK4Agg
GTK4Cairo
MacOSX
nbAgg
QtAgg
QtCairo
Qt5Agg
Qt5Cairo
TkAgg
TkCairo
WebAgg
WX
WXAgg
WXCairo
agg
cairo
pdf
pgf
ps
svg
template

'''
# matplotlib.use('Qt5Agg')
# matplotlib.use("TkAgg")
# matplotlib.use("nbAgg")

class CosimLoss (nn.Module):
    """ Try to make the repeatability repeatable from one image to the other.
    """
    def __init__(self, N=16):
        nn.Module.__init__(self)
        self.name = f'cosim{N}'
        self.patches = nn.Unfold(N, padding=0, stride=N//2)

    def extract_patches(self, sal):
        patches = self.patches(sal).transpose(1,2) # flatten
        patches = F.normalize(patches, p=2, dim=2) # norm
        return patches
        
    def forward(self, repeatability, aflow, **kw):
        B,two,H,W = aflow.shape
        assert two == 2

        # normalize
        sali1, sali2 = repeatability
        grid = FullSampler._aflow_to_grid(aflow)
        sali2 = F.grid_sample(sali2, grid, mode='bilinear', padding_mode='border')

        patches1 = self.extract_patches(sali1)
        patches2 = self.extract_patches(sali2)
        cosim = (patches1 * patches2).sum(dim=2)
        return 1 - cosim.mean()


class ReprojectionLocLoss(nn.Module):
    """ Try to make the repeatability repeatable from one image to the other.
    """

    def __init__(self, N=17):
        nn.Module.__init__(self)
        self.name = f'ReprojectionLocLoss'
        self.mode = 'bilinear'
        self.padding = 'zeros'
        self.ksize= N
        self.max_filter = torch.nn.MaxPool2d(kernel_size=N, stride=1, padding=N // 2)
        self.rep_thr = 0

    def extract_patches(self, sal):
        patches = self.patches(sal).transpose(1, 2)  # flatten
        patches = F.normalize(patches, p=2, dim=2)  # norm
        return patches

    @staticmethod
    def _aflow_to_grid(aflow):
        H, W = aflow.shape[2:]
        grid = aflow.permute(0, 2, 3, 1).clone()
        grid[:, :, :, 0] *= 2 / (W - 1)
        grid[:, :, :, 1] *= 2 / (H - 1)
        grid -= 1
        grid[torch.isnan(grid)] = 9e9  # invalids
        return grid


    def _warp(self, confs, aflow):
        if isinstance(aflow, tuple): return aflow  # result was precomputed
        conf1, conf2 = confs if confs else (None, None)
        B, two, H, W = aflow.shape
        assert conf1.shape == conf2.shape == (B, 1, H, W) if confs else True
        # warp img2 to img1
        grid = self._aflow_to_grid(aflow)
        conf2to1 = F.grid_sample(conf2, grid, mode=self.mode, padding_mode=self.padding) \
            if confs else None
        return  conf2to1




    def nms(self, repeatability, **kw):
        # assert len(reliability) == len(repeatability) == 1
        # reliability, repeatability = reliability[0], repeatability[0]
        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)

        return maxima.nonzero()

    def forward(self, repeatability, aflow, **kw):
        B, two, H, W = aflow.shape
        assert two == 2

        # normalize
        sali1, sali2 = repeatability
        sali2to1 = self._warp(repeatability, aflow)
        x = sali2to1.detach().cpu().numpy()[0, 0]
        y = sali1.detach().cpu().numpy()[0, 0]


        locs1 = self.nms(sali1)
        locs2 = self.nms(sali2to1)
        diff = locs1[None,:, :] - locs2[:, None, :]
        dists = torch.sum(diff**2, dim=-1)
        mask = diff[:, :, 0]==0
        plt.imshow(dists*mask)
        # fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        # ax[0].imshow(x[4:-4, 4:-4])
        # ax[1].imshow(y)
        # ax[2].imshow(x+y)
        plt.show()

        return 1




class SharpenPeak(nn.Module):
    """ Try to make the repeatability repeatable from one image to the other.
    """

    def __init__(self, N=17):
        nn.Module.__init__(self)
        self.name = 'SharpenPeak'
        self.mode = 'bilinear'
        self.padding = 'zeros'
        self.ksize= N
        self.max_filter = torch.nn.MaxPool2d(kernel_size=N, stride=1, padding=N // 2)
        self.rep_thr = 0


    def nms(self, repeatability, **kw):
        # assert len(reliability) == len(repeatability) == 1
        # reliability, repeatability = reliability[0], repeatability[0]
        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)

        return maxima

    def forward(self, repeatability, aflow, **kw):
        B, two, H, W = aflow.shape
        assert two == 2

        # normalize
        sali1, sali2 = repeatability
        locsMaxima1 = self.nms(sali1).float()
        locsMaxima2 = self.nms(sali2).float()

        return  torch.mean((locsMaxima1 - sali1)**2 + (locsMaxima2-sali1)**2 )


class PeakyLoss (nn.Module):
    """ Try to make the repeatability locally peaky.

    Mechanism: we maximize, for each pixel, the difference between the local mean
               and the local max.
    """
    def __init__(self, N=16):
        nn.Module.__init__(self)
        self.name = f'peaky{N}'
        assert N % 2 == 0, 'N must be pair'
        self.preproc = nn.AvgPool2d(3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(N+1, stride=1, padding=N//2)
        self.avgpool = nn.AvgPool2d(N+1, stride=1, padding=N//2)

    def forward_one(self, sali):
        sali = self.preproc(sali) # remove super high frequency
        return 1 - (self.maxpool(sali) - self.avgpool(sali)).mean()

    def forward(self, repeatability, **kw):
        sali1, sali2 = repeatability
        return (self.forward_one(sali1) + self.forward_one(sali2)) /2





