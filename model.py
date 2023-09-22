# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 2020

'''
import matplotlib
#matplotlib.use('agg')
import numpy as np
import sys
import torch
import torch.nn as nn
import os
import pathlib
from collections import defaultdict
import rgb_ted
from util import ImageProcessing
from torch.autograd import Variable
import math
from math import exp
import torch.nn.functional as F
from scipy.io import loadmat
import copy
from torchvision.transforms.functional import to_tensor

np.set_printoptions(threshold=sys.maxsize)


class CURLLoss(nn.Module):

    def __init__(self, ssim_window_size=5, alpha=0.5):
        """Initialisation of the DeepLPF loss function

        :param ssim_window_size: size of averaging window for SSIM
        :param alpha: interpolation paramater for L1 and SSIM parts of the loss
        :returns: N/A
        :rtype: N/A

        """
        super(CURLLoss, self).__init__()
        self.alpha = alpha
        self.ssim_window_size = ssim_window_size

    def create_window(self, window_size, num_channel):
        """Window creation function for SSIM metric. Gaussian weights are applied to the window.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param window_size: size of the window to compute statistics
        :param num_channel: number of channels
        :returns: Tensor of shape Cx1xWindow_sizexWindow_size
        :rtype: Tensor

        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(
            num_channel, 1, window_size, window_size).contiguous())
        return window

    def gaussian(self, window_size, sigma):
        """
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        :param window_size: size of the SSIM sampling window e.g. 11
        :param sigma: Gaussian variance
        :returns: 1xWindow_size Tensor of Gaussian weights
        :rtype: Tensor

        """
        gauss = torch.Tensor(
            [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def compute_ssim(self, img1, img2):
        """Computes the structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        (_, num_channel, _, _) = img1.size()
        window = self.create_window(self.ssim_window_size, num_channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
            window = window.type_as(img1)

        mu1 = F.conv2d(
            img1, window, padding=self.ssim_window_size // 2, groups=num_channel)
        mu2 = F.conv2d(
            img2, window, padding=self.ssim_window_size // 2, groups=num_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map1 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        ssim_map2 = ((mu1_sq.cuda() + mu2_sq.cuda() + C1) *
                     (sigma1_sq.cuda() + sigma2_sq.cuda() + C2))
        ssim_map = ssim_map1.cuda() / ssim_map2.cuda()

        v1 = 2.0 * sigma12.cuda() + C2
        v2 = sigma1_sq.cuda() + sigma2_sq.cuda() + C2
        cs = torch.mean(v1 / v2)

        return ssim_map.mean(), cs


    def compute_msssim(self, img1, img2):
        """Computes the multi scale structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        if img1.shape[2]!=img2.shape[2]:
                img1=img1.transpose(2,3)

        if img1.shape != img2.shape:
            raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
        if img1.ndim != 4:
            raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        ssims = []
        mcs = []
        for _ in range(levels):
            ssim, cs = self.compute_ssim(img1, img2)

            # Relu normalize (not compliant with original definition)
            ssims.append(ssim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        ssims = torch.stack(ssims)
        mcs = torch.stack(mcs)

        # Simple normalize (not compliant with original definition)
        # TODO: remove support for normalize == True (kept for backward support)
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = ssims ** weights

        # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        output = torch.prod(pow1[:-1] * pow2[-1])
        return output

    def forward(self, predicted_img_batch, target_img_batch, gradient_regulariser):
        """Forward function for the CURL loss

        :param predicted_img_batch_high_res: 
        :param predicted_img_batch_high_res_rgb: 
        :param target_img_batch: Tensor of shape BxCxWxH
        :returns: value of loss function
        :rtype: float

        """
        num_images = target_img_batch.shape[0]
        target_img_batch = target_img_batch

        ssim_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        l1_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        cosine_rgb_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        hsv_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        rgb_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))

        for i in range(0, num_images):

            target_img = target_img_batch[i, :, :, :].cuda()
            predicted_img = predicted_img_batch[i, :, :, :].cuda().to(torch.float32)

            predicted_img_lab = torch.clamp(
                ImageProcessing.rgb_to_lab(predicted_img.squeeze(0)), 0, 1)
            target_img_lab = torch.clamp(
                ImageProcessing.rgb_to_lab(target_img.squeeze(0)), 0, 1)

            target_img_hsv = torch.clamp(ImageProcessing.rgb_to_hsv(
                target_img), 0, 1)
            predicted_img_hsv = torch.clamp(ImageProcessing.rgb_to_hsv(
                predicted_img.squeeze(0)), 0, 1)

            predicted_img_hue = (predicted_img_hsv[0, :, :]*2*math.pi)
            predicted_img_val = predicted_img_hsv[2, :, :]
            predicted_img_sat = predicted_img_hsv[1, :, :]
            target_img_hue = (target_img_hsv[0, :, :]*2*math.pi)
            target_img_val = target_img_hsv[2, :, :]
            target_img_sat = target_img_hsv[1, :, :]

            target_img_L_ssim = target_img_lab[0, :, :].unsqueeze(0)
            predicted_img_L_ssim = predicted_img_lab[0, :, :].unsqueeze(0)
            target_img_L_ssim = target_img_L_ssim.unsqueeze(0)
            predicted_img_L_ssim = predicted_img_L_ssim.unsqueeze(0)

            ssim_value = self.compute_msssim(
                predicted_img_L_ssim, target_img_L_ssim)

            ssim_loss_value += (1.0 - ssim_value)

            predicted_img_1 = predicted_img_val * \
                predicted_img_sat*torch.cos(predicted_img_hue)
            predicted_img_2 = predicted_img_val * \
                predicted_img_sat*torch.sin(predicted_img_hue)

            target_img_1 = target_img_val * \
                target_img_sat*torch.cos(target_img_hue)
            target_img_2 = target_img_val * \
                target_img_sat*torch.sin(target_img_hue)

            predicted_img_hsv = torch.stack(
                (predicted_img_1, predicted_img_2, predicted_img_val), 2)
            target_img_hsv = torch.stack((target_img_1, target_img_2, target_img_val), 2)

            l1_loss_value += F.l1_loss(predicted_img_lab, target_img_lab)
            rgb_loss_value += F.l1_loss(predicted_img, target_img)
            hsv_loss_value += F.l1_loss(predicted_img_hsv, target_img_hsv)

            cosine_rgb_loss_value += (1-torch.mean(
                torch.nn.functional.cosine_similarity(predicted_img, target_img, dim=0)))

        l1_loss_value = l1_loss_value/num_images
        rgb_loss_value = rgb_loss_value/num_images
        ssim_loss_value = ssim_loss_value/num_images
        cosine_rgb_loss_value = cosine_rgb_loss_value/num_images
        hsv_loss_value = hsv_loss_value/num_images

        curl_loss = (rgb_loss_value + cosine_rgb_loss_value + l1_loss_value +
                     hsv_loss_value + 10*ssim_loss_value + 1e-6*gradient_regulariser)/6

        return curl_loss

class CURLLayer(nn.Module):

    import torch.nn.functional as F

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialisation of class

        :param num_in_channels: number of input channels
        :param num_out_channels: number of output channels
        :returns: N/A
        :rtype: N/A

        """
        super(CURLLayer, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.make_init_network()

    def make_init_network(self):
        """ Initialise the CURL block layers

        :returns: N/A
        :rtype: N/A

        """
        self.lab_layer1 = ConvBlock(64, 64)
        self.lab_layer2 = MaxPoolBlock()
        self.lab_layer3 = ConvBlock(64, 64)
        self.lab_layer4 = MaxPoolBlock()
        self.lab_layer5 = ConvBlock(64, 64)
        self.lab_layer6 = MaxPoolBlock()
        self.lab_layer7 = ConvBlock(64, 64)
        self.lab_layer8 = GlobalPoolingBlock(2)

        self.fc_lab = torch.nn.Linear(64, 48)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.rgb_layer1 = ConvBlock(64, 64)
        self.rgb_layer2 = MaxPoolBlock()
        self.rgb_layer3 = ConvBlock(64, 64)
        self.rgb_layer4 = MaxPoolBlock()
        self.rgb_layer5 = ConvBlock(64, 64)
        self.rgb_layer6 = MaxPoolBlock()
        self.rgb_layer7 = ConvBlock(64, 64)
        self.rgb_layer8 = GlobalPoolingBlock(2)

        self.fc_rgb = torch.nn.Linear(64, 48)

        self.hsv_layer1 = ConvBlock(64, 64)
        self.hsv_layer2 = MaxPoolBlock()
        self.hsv_layer3 = ConvBlock(64, 64)
        self.hsv_layer4 = MaxPoolBlock()
        self.hsv_layer5 = ConvBlock(64, 64)
        self.hsv_layer6 = MaxPoolBlock()
        self.hsv_layer7 = ConvBlock(64, 64)
        self.hsv_layer8 = GlobalPoolingBlock(2)

        self.fc_hsv = torch.nn.Linear(64, 64)

    def forward(self, x):
        """Forward function for the CURL layer

        :param x: forward the data x through the network
        :returns: Tensor representing the predicted image
        :rtype: Tensor

        """

        '''
        This function is where the magic happens :)
        '''
        x.contiguous()  # remove memory holes

        feat = x[:, 3:64, :, :]
        img = x[:, 0:3, :, :]

        torch.cuda.empty_cache()
        shape = x.shape

        img_clamped = torch.clamp(img, 0, 1)
        img_lab = torch.clamp(ImageProcessing.rgb_to_lab(
            img_clamped.squeeze(0)), 0, 1)

        feat_lab = torch.cat((feat, img_lab.unsqueeze(0)), 1)

        x = self.lab_layer1(feat_lab)
        del feat_lab
        x = self.lab_layer2(x)
        x = self.lab_layer3(x)
        x = self.lab_layer4(x)
        x = self.lab_layer5(x)
        x = self.lab_layer6(x)
        x = self.lab_layer7(x)
        x = self.lab_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout1(x)
        L = self.fc_lab(x)

        img_lab, gradient_regulariser_lab = ImageProcessing.adjust_lab(
            img_lab.squeeze(0), L[0, 0:48])
        img_rgb = ImageProcessing.lab_to_rgb(img_lab.squeeze(0))
        img_rgb = torch.clamp(img_rgb, 0, 1)

        feat_rgb = torch.cat((feat, img_rgb.unsqueeze(0)), 1)

        x = self.rgb_layer1(feat_rgb)
        x = self.rgb_layer2(x)
        x = self.rgb_layer3(x)
        x = self.rgb_layer4(x)
        x = self.rgb_layer5(x)
        x = self.rgb_layer6(x)
        x = self.rgb_layer7(x)
        x = self.rgb_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout2(x)
        R = self.fc_rgb(x)

        img_rgb, gradient_regulariser_rgb = ImageProcessing.adjust_rgb(
            img_rgb.squeeze(0), R[0, 0:48])
        img_rgb = torch.clamp(img_rgb, 0, 1)

        img_hsv = ImageProcessing.rgb_to_hsv(img_rgb.squeeze(0))
        img_hsv = torch.clamp(img_hsv, 0, 1)
        feat_hsv = torch.cat((feat, img_hsv.unsqueeze(0)), 1)

        x = self.hsv_layer1(feat_hsv)
        del feat_hsv
        x = self.hsv_layer2(x)
        x = self.hsv_layer3(x)
        x = self.hsv_layer4(x)
        x = self.hsv_layer5(x)
        x = self.hsv_layer6(x)
        x = self.hsv_layer7(x)
        x = self.hsv_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout3(x)
        H = self.fc_hsv(x)

        img_hsv, gradient_regulariser_hsv = ImageProcessing.adjust_hsv(
            img_hsv, H[0, 0:64])
        img_hsv = torch.clamp(img_hsv, 0, 1)

        img_residual = torch.clamp(ImageProcessing.hsv_to_rgb(
           img_hsv.squeeze(0)), 0, 1)

        img = torch.clamp(img + img_residual.unsqueeze(0), 0, 1)

        gradient_regulariser = gradient_regulariser_rgb + \
            gradient_regulariser_lab+gradient_regulariser_hsv

        return img, gradient_regulariser


class CURLLayer_new(nn.Module):

    import torch.nn.functional as F

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialisation of class

        :param num_in_channels: number of input channels
        :param num_out_channels: number of output channels
        :returns: N/A
        :rtype: N/A

        """
        super(CURLLayer_new, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.make_init_network()

    def make_init_network(self):
        """ Initialise the CURL block layers

        :returns: N/A
        :rtype: N/A

        """
        self.lab_layer1 = ConvBlock(64, 64)
        self.lab_layer2 = MaxPoolBlock()
        self.lab_layer3 = ConvBlock(64, 64)
        self.lab_layer4 = MaxPoolBlock()
        self.lab_layer5 = ConvBlock(64, 64)
        self.lab_layer6 = MaxPoolBlock()
        self.lab_layer7 = ConvBlock(64, 64)
        self.lab_layer8 = GlobalPoolingBlock(2)
        self.dropout1 = nn.Dropout(0.5)

        self.fc_lab = torch.nn.Linear(64, 48)

        self.rgb_layer1 = ConvBlock(64, 64)
        self.rgb_layer2 = MaxPoolBlock()
        self.rgb_layer3 = ConvBlock(64, 64)
        self.rgb_layer4 = MaxPoolBlock()
        self.rgb_layer5 = ConvBlock(64, 64)
        self.rgb_layer6 = MaxPoolBlock()
        self.rgb_layer7 = ConvBlock(64, 64)
        self.rgb_layer8 = GlobalPoolingBlock(2)
        self.dropout2 = nn.Dropout(0.5)

        self.fc_rgb = torch.nn.Linear(64, 48)

        self.hsv_layer1 = ConvBlock(64, 64)
        self.hsv_layer2 = MaxPoolBlock()
        self.hsv_layer3 = ConvBlock(64, 64)
        self.hsv_layer4 = MaxPoolBlock()
        self.hsv_layer5 = ConvBlock(64, 64)
        self.hsv_layer6 = MaxPoolBlock()
        self.hsv_layer7 = ConvBlock(64, 64)
        self.hsv_layer8 = GlobalPoolingBlock(2)
        self.dropout3 = nn.Dropout(0.5)

        self.fc_hsv = torch.nn.Linear(64, 64)

    def forward(self, x, cn_x):
        """Forward function for the CURL layer

        :param x: forward the data x through the network 
        :returns: Tensor representing the predicted image
        :rtype: Tensor
        W, H, 64
        """

        '''
        This function is where the magic happens :)
        '''
        x.contiguous()  # remove memory holes
        feat = x[:, 3:64, :, :]
        img = x[:, 0:3, :, :]

        torch.cuda.empty_cache()
        shape = x.shape

        img_clamped = torch.clamp(img, 0, 1)
        img_lab = torch.clamp(ImageProcessing.rgb_to_lab(
            img_clamped.squeeze(0)), 0, 1)

        feat_lab = torch.cat((feat, img_lab.unsqueeze(0)), 1)

        x = self.lab_layer1(feat_lab)
        del feat_lab
        x = self.lab_layer2(x)
        x = self.lab_layer3(x)
        x = self.lab_layer4(x)
        x = self.lab_layer5(x)
        x = self.lab_layer6(x)
        x = self.lab_layer7(x)
        x = self.lab_layer8(x)

        x = x.view(x.size()[0], -1)
        x = self.dropout1(x)
        L = self.fc_lab(x)

        img_lab, gradient_regulariser_lab = ImageProcessing.adjust_lab(
            img_lab.squeeze(0), L[0, 0:48])
        img_rgb = ImageProcessing.lab_to_rgb(img_lab.squeeze(0))
        img_rgb = torch.clamp(img_rgb, 0, 1)
        feat_rgb = torch.cat((feat, img_rgb.unsqueeze(0)), 1)

        x = self.rgb_layer1(feat_rgb)
        x = self.rgb_layer2(x)
        x = self.rgb_layer3(x)
        x = self.rgb_layer4(x)
        x = self.rgb_layer5(x)
        x = self.rgb_layer6(x)
        x = self.rgb_layer7(x)
        x = self.rgb_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout2(x)
        R = self.fc_rgb(x)

        img_rgb, gradient_regulariser_rgb = ImageProcessing.adjust_rgb(
            img_rgb.squeeze(0), R[0, 0:48])
        img_rgb = torch.clamp(img_rgb, 0, 1)

        img_hsv = ImageProcessing.rgb_to_hsv(img_rgb.squeeze(0))
        img_hsv = torch.clamp(img_hsv, 0, 1)
        feat_hsv = torch.cat((feat, img_hsv.unsqueeze(0)), 1)

        x = self.hsv_layer1(feat_hsv)
        del feat_hsv
        x = self.hsv_layer2(x)
        x = self.hsv_layer3(x)
        x = self.hsv_layer4(x)
        x = self.hsv_layer5(x)
        x = self.hsv_layer6(x)
        x = self.hsv_layer7(x)
        x = self.hsv_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout3(x)
        H = self.fc_hsv(x)

        img_hsv, gradient_regulariser_hsv = ImageProcessing.adjust_hsv(
            img_hsv, H[0, 0:64])
        img_hsv = torch.clamp(img_hsv, 0, 1)

        img_residual = torch.clamp(ImageProcessing.hsv_to_rgb(
           img_hsv.squeeze(0)), 0, 1)

        img = torch.clamp(img + img_residual.unsqueeze(0), 0, 1)

        cn_x = torch.clamp(cn_x, 0, 1)
        cn_lab = torch.clamp(ImageProcessing.rgb_to_lab(
            cn_x.squeeze(0)), 0, 1)
        cn_lab, gradient_regulariser_lab = ImageProcessing.adjust_lab(
            cn_lab.squeeze(0), L[0, 0:48])
        cn_rgb = ImageProcessing.lab_to_rgb(cn_lab.squeeze(0))
        cn_rgb = torch.clamp(cn_rgb, 0, 1)
        cn_rgb, gradient_regulariser_rgb = ImageProcessing.adjust_rgb(
            cn_rgb.squeeze(0), R[0, 0:48])
        cn_rgb = torch.clamp(cn_rgb, 0, 1)
        cn_hsv = ImageProcessing.rgb_to_hsv(cn_rgb.squeeze(0))
        cn_hsv = torch.clamp(cn_hsv, 0, 1)

        cn_hsv, gradient_regulariser_hsv = ImageProcessing.adjust_hsv(
            cn_hsv, H[0, 0:64])
        cn_hsv = torch.clamp(cn_hsv, 0, 1)

        cn_residual = torch.clamp(ImageProcessing.hsv_to_rgb(
            cn_hsv.squeeze(0)), 0, 1)

        gradient_regulariser = gradient_regulariser_rgb + \
            gradient_regulariser_lab+gradient_regulariser_hsv

        cn = torch.clamp(cn_x + cn_residual.unsqueeze(0), 0, 1)

        return cn, gradient_regulariser


class Block(nn.Module):

    def __init__(self):
        """Initialisation for a lower-level DeepLPF conv block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

    def conv3x3(self, in_channels, out_channels, stride=1):
        """Represents a convolution of shape 3x3

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: the convolution stride
        :returns: convolution function with the specified parameterisation
        :rtype: function

        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=True)


class ConvBlock(Block, nn.Module):

    def __init__(self, num_in_channels, num_out_channels, stride=1):
        """Initialise function for the higher level convolution block

        :param in_channels:
        :param out_channels:
        :param stride:
        :param padding:
        :returns:
        :rtype:

        """
        super(Block, self).__init__()
        self.conv = self.conv3x3(num_in_channels, num_out_channels, stride=2)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        """ Forward function for the higher level convolution block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.lrelu(self.conv(x))
        return img_out


class MaxPoolBlock(Block, nn.Module):

    def __init__(self):
        """Initialise function for the max pooling block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """ Forward function for the max pooling block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.max_pool(x)
        return img_out


class GlobalPoolingBlock(Block, nn.Module):

    def __init__(self, receptive_field):
        """Implementation of the global pooling block. Takes the average over a 2D receptive field.
        :param receptive_field:
        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Forward function for the high-level global pooling block

        :param x: Tensor of shape BxCxAxA
        :returns: Tensor of shape BxCx1x1, where B is the batch size
        :rtype: Tensor

        """
        out = self.avg_pool(x)
        return out

class CURLNet_original(nn.Module):

    def __init__(self):
        """Initialisation function

        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(CURLNet_original, self).__init__()
        self.tednet = rgb_ted.TEDModel()
        self.curllayer = CURLLayer()

    def forward(self, img):
        """Neural network forward function

        :param img: forward the data img through the network
        :returns: residual image
        :rtype: numpy ndarray

        """
        feat = self.tednet(img)
        img, gradient_regulariser = self.curllayer(feat)
        return img, gradient_regulariser

class CURLNet(nn.Module):

    def __init__(self):
        """Initialisation function

        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(CURLNet, self).__init__()
        self.tednet = rgb_ted.TEDModel()
        checkpoint = torch.load(
            "/home/dserrano/Workspace/CURL/log_2023-09-12_17-15-10/curl_validpsnr_22.405160786330203_validloss_0.07452978938817978_testpsnr_23.671673845595638_testloss_0.06112174317240715_epoch_70_model.pt")
        model_state_dict = checkpoint['model_state_dict']
        model_state_dict_ted = {k: v for k, v in model_state_dict.items() if not k.startswith('curl')}
        model_state_dict_ted = {k.replace('tednet.', ''): v for k, v in model_state_dict_ted.items()}

        self.tednet.load_state_dict(model_state_dict_ted)
        self.tednet.eval()

        self.color_naming = ColorNaming()

        model_state_dict_curl = {k: v for k, v in model_state_dict.items() if not k.startswith('ted')}
        model_state_dict_curl = {k.replace('curllayer.', ''): v for k, v in model_state_dict_curl.items()}

        self.curl_orange = CURLLayer()
        #self.curl_orange.load_state_dict(model_state_dict_curl)
        self.curl_orange = self.curl_orange.cuda()

        self.curl_achromatic = CURLLayer()
        # self.curl_achromatic.load_state_dict(model_state_dict_curl)
        self.curl_achromatic = self.curl_achromatic.cuda()

        self.curl_pink = CURLLayer()
        # self.curl_pink.load_state_dict(model_state_dict_curl)
        self.curl_pink = self.curl_pink.cuda()

        self.curl_red = CURLLayer()
        # self.curl_red.load_state_dict(model_state_dict_curl)
        self.curl_red = self.curl_red.cuda()

        self.curl_green = CURLLayer()
        # self.curl_green.load_state_dict(model_state_dict_curl)
        self.curl_green = self.curl_green.cuda()

        self.curl_blue = CURLLayer()
        # self.curl_blue.load_state_dict(model_state_dict_curl)
        self.curl_blue = self.curl_blue.cuda()

    def forward(self, img):
        """Neural network forward function

        :param img: forward the data img through the network
        :returns: residual image
        :rtype: numpy ndarray

        """
        x = self.tednet(img)
        x.contiguous()
        img = x[:, 0:3, :, :].detach().cpu()
        feat = x[:, 3:64, :, :]
        img = torch.clamp(img, 0, 1)
        x_color, x_probs = self.color_naming(img)

        x_color = x_color.cuda()
        x_probs = x_probs.cuda()

        x_0, gradient_0 = self.curl_orange(torch.cat((feat, x_color[0]), dim=1))
        x_1, gradient_1 = self.curl_achromatic(torch.cat((feat, x_color[1]), dim=1))
        x_2, gradient_2 = self.curl_pink(torch.cat((feat, x_color[2]), dim=1))
        x_3, gradient_3 = self.curl_red(torch.cat((feat, x_color[3]), dim=1))
        x_4, gradient_4 = self.curl_green(torch.cat((feat, x_color[4]), dim=1))
        x_5, gradient_5 = self.curl_blue(torch.cat((feat, x_color[5]), dim=1))

        x = torch.stack([x_0, x_1, x_2, x_3, x_4, x_5], dim=0)
        x = torch.sum(x*x_probs.unsqueeze(2), dim=0)

        return x, gradient_0 + gradient_1 + gradient_2 + gradient_3 + gradient_4 + gradient_5

class CURLNet_new(nn.Module):

    def __init__(self):
        """Initialisation function

        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(CURLNet_new, self).__init__()
        self.tednet = rgb_ted.TEDModel()
        checkpoint = torch.load(
            "/hhome/dserrano/Workspace/Workspace/curl_validpsnr_22.405160786330203_validloss_0.07452978938817978_testpsnr_23.671673845595638_testloss_0.06112174317240715_epoch_70_model.pt")
        model_state_dict = checkpoint['model_state_dict']
        model_state_dict_ted = {k: v for k, v in model_state_dict.items() if not k.startswith('curl')}
        model_state_dict_ted = {k.replace('tednet.', ''): v for k, v in model_state_dict_ted.items()}

        self.tednet.load_state_dict(model_state_dict_ted)
        self.tednet.eval()

        self.color_naming = ColorNaming()

        model_state_dict_curl = {k: v for k, v in model_state_dict.items() if not k.startswith('ted')}
        model_state_dict_curl = {k.replace('curllayer.', ''): v for k, v in model_state_dict_curl.items()}

        model_state_dict_achromatic = {k: v for k, v in model_state_dict_curl.items() if k.startswith('curl_achromatic')}
        model_state_dict_achromatic = {k.replace('curl_achromatic.', ''): v for k, v in model_state_dict_achromatic.items()}

        model_state_dict_orange = {k: v for k, v in model_state_dict_curl.items() if k.startswith('curl_orange')}
        model_state_dict_orange = {k.replace('curl_orange.', ''): v for k, v in model_state_dict_orange.items()}

        model_state_dict_pink = {k: v for k, v in model_state_dict_curl.items() if k.startswith('curl_pink')}
        model_state_dict_pink = {k.replace('curl_pink.', ''): v for k, v in model_state_dict_pink.items()}

        model_state_dict_red = {k: v for k, v in model_state_dict_curl.items() if k.startswith('curl_red')}
        model_state_dict_red = {k.replace('curl_red.', ''): v for k, v in model_state_dict_red.items()}

        model_state_dict_blue = {k: v for k, v in model_state_dict_curl.items() if k.startswith('curl_blue')}
        model_state_dict_blue = {k.replace('curl_blue.', ''): v for k, v in model_state_dict_blue.items()}

        model_state_dict_green = {k: v for k, v in model_state_dict_curl.items() if k.startswith('curl_green')}
        model_state_dict_green = {k.replace('curl_green.', ''): v for k, v in model_state_dict_green.items()}

        self.curl_black = CURLLayer()
        self.curl_black.load_state_dict(model_state_dict_achromatic)
        self.curl_black = self.curl_black.cuda()

        self.curl_blue = CURLLayer()
        self.curl_blue.load_state_dict(model_state_dict_blue)
        self.curl_blue = self.curl_blue.cuda()

        self.curl_brown = CURLLayer()
        self.curl_brown.load_state_dict(model_state_dict_orange)
        self.curl_brown = self.curl_brown.cuda()

        self.curl_grey = CURLLayer()
        self.curl_grey.load_state_dict(model_state_dict_achromatic)
        self.curl_grey = self.curl_grey.cuda()

        self.curl_green = CURLLayer()
        self.curl_green.load_state_dict(model_state_dict_green)
        self.curl_green = self.curl_green.cuda()

        self.curl_orange = CURLLayer()
        self.curl_orange.load_state_dict(model_state_dict_orange)
        self.curl_orange = self.curl_orange.cuda()

        self.curl_pink = CURLLayer()
        self.curl_pink.load_state_dict(model_state_dict_pink)
        self.curl_pink = self.curl_pink.cuda()

        self.curl_purple = CURLLayer()
        self.curl_purple.load_state_dict(model_state_dict_pink)
        self.curl_purple = self.curl_purple.cuda()

        self.curl_red = CURLLayer()
        self.curl_red.load_state_dict(model_state_dict_red)
        self.curl_red = self.curl_red.cuda()

        self.curl_white = CURLLayer()
        self.curl_white.load_state_dict(model_state_dict_achromatic)
        self.curl_white = self.curl_white.cuda()

        self.curl_yellow = CURLLayer()
        self.curl_yellow.load_state_dict(model_state_dict_orange)
        self.curl_yellow = self.curl_yellow.cuda()

    def forward(self, img):
        """Neural network forward function

        :param img: forward the data img through the network
        :returns: residual image
        :rtype: numpy ndarray

        """
        x = self.tednet(img)
        x.contiguous()
        img = x[:, 0:3, :, :].detach().cpu()

        feat = x[:, 3:64, :, :]
        img = torch.clamp(img, 0, 1)
        x_color, x_probs = self.color_naming(img)

        #x_color = x_color.cuda()
        x_probs = x_probs.cuda()

        x_0, gradient_0 = self.curl_black(x)
        x_1, gradient_1 = self.curl_blue(x)
        x_2, gradient_2 = self.curl_brown(x)
        x_3, gradient_3 = self.curl_grey(x)
        x_4, gradient_4 = self.curl_green(x)
        x_5, gradient_5 = self.curl_orange(x)
        x_6, gradient_6 = self.curl_pink(x)
        x_7, gradient_7 = self.curl_purple(x)
        x_8, gradient_8 = self.curl_red(x)
        x_9, gradient_9 = self.curl_white(x)
        x_10, gradient_10 = self.curl_yellow(x)

        x = torch.stack([x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10], dim=0)
        x = torch.sum(x*x_probs.unsqueeze(2), dim=0)

        return x, gradient_0 + gradient_1 + gradient_2 + gradient_3 + gradient_4 + gradient_5 + gradient_6 + gradient_7 + gradient_8 + gradient_9 + gradient_10
        """Color Naming"""
        # from scipy.io import loadmat
        # from torchvision.transforms.functional import to_tensor
        #
        # MATPATH = "/home/david/Downloads/wetransfer_imatges_2023-05-12_1337/Color_naming/ColorNaming/w2c.mat"
        # COLOR_CATEGORIES = [[2, 5, 10], [0, 3, 9], [6, 7], [8], [4], [1]]
        # THRESHOLD = 0.1
        #
        # mat = loadmat(MATPATH)['w2c']
        # img = feat.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # index_im = np.floor(img[..., 0].flatten() / 8).astype(int) + 32 * np.floor(img[..., 1].flatten() / 8).astype(
        #     int) + 32 * 32 * np.floor(img[..., 2].flatten() / 8).astype(int)
        # prob_maps = []
        # for idx, w2cM in enumerate(mat.T):
        #     map = w2cM[index_im].reshape(img.shape[:2])
        #     prob_maps.append(map)
        #
        # mask_images = []
        # prob_images = []
        # for idx, category in enumerate(COLOR_CATEGORIES):
        #     mask = np.expand_dims(
        #         analogic_or([np.greater_equal(prob_maps[i], THRESHOLD).astype(int) for i in category]).astype(int),
        #         axis=2)
        #     prob = np.sum([prob_maps[i] for i in category], axis=0)
        #
        #     mask_images.append(mask)
        #     prob_images.append(prob)
        #
        # curl_images = []
        # for idx, (mask, map) in enumerate(zip(mask_images, prob_images)):
        #     input_tensor = feat * torch.from_numpy(mask).float().permute(2, 0, 1).unsqueeze(0).cuda()
        #     net_output_img_example, _ = self.curllayer(input_tensor)
        #
        #     map = torch.from_numpy(map).unsqueeze(0)
        #     net_output_img_example = net_output_img_example.squeeze(0) * map.cuda()
        #
        #     curl_images.append(net_output_img_example)
        #
        # reconstructed_img = torch.sum(torch.stack(curl_images), dim=0)

        return x, gradient_0 + gradient_1

        #img, gradient_regulariser = self.curllayer(feat)
        #return img, gradient_regulariser

from functools import reduce
def analogic_or(masks):
    return reduce(lambda x, y: x | y, masks)

class CurveNet(nn.Module):
    def __init__(self, in_chan=3, out_chan=64, ker_sizes=3, drop_rate=0.2):
        super(CurveNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=int(out_chan/2), kernel_size=ker_sizes, padding=1)
        self.conv2 = nn.Conv2d(in_channels=int(out_chan/2), out_channels=out_chan, kernel_size=ker_sizes, padding=1)
        self.conv3 = nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=ker_sizes, padding=1)

        self.lrelu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adap_avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(out_chan, 48)

        self.curve = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=int(out_chan/2), kernel_size=ker_sizes, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=int(out_chan/2), out_channels=out_chan, kernel_size=ker_sizes, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=ker_sizes, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(drop_rate),
            nn.Linear(out_chan, 48))

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.maxpool(x)
        x = self.lrelu(self.conv2(x))
        x = self.maxpool(x)
        x = self.lrelu(self.conv3(x))
        x = self.adap_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ColorNaming():
    def __init__(self, matrix_path=os.path.join(str(pathlib.Path(__file__).parent.resolve()), "color_naming_images/w2c.mat"), threshold=0.1, mode='6'):
        self.matrix = to_tensor(loadmat(matrix_path)['w2c'])
        self.threshold = threshold

        if mode == '6':
            self.color_categories = [[2,5,10], [0,3,9], [6,7], [8], [4], [1]]

    def __call__(self, input_tensor, mode='6'):
        """Converts an RGB image to a color naming image.

        Args:
        input_tensor: batch of RGB images (B x 3 x H x W)

        Returns:
            np.array: Color naming image.
        """
        # Reconvert image to [0-255] range
        img = (input_tensor * 255).int()

        index_tensor = torch.floor(
            img[:, 0, ...].view(img.shape[0], -1) / 8).long() + 32 * torch.floor(
            img[:, 1, ...].view(img.shape[0], -1) / 8).long() + 32 * 32 * torch.floor(
            img[:, 2, ...].view(img.shape[0], -1) / 8).long()

        prob_maps = []
        for w2cM in self.matrix.permute(*torch.arange(self.matrix.ndim-1, -1, -1)):
            out = w2cM[index_tensor].view(input_tensor.size(0), input_tensor.size(2), input_tensor.size(3))
            prob_maps.append(out)
        prob_maps = torch.stack(prob_maps, dim=0)

        category_probs = []  # prob maps for each color category. [0, 1]
        category_images = [] # binary masks for each color category. {0, 1}
        for category in self.color_categories:
            cat_tensors = torch.index_select(prob_maps, 0, torch.tensor(category)).sum(dim=0)
            category_probs.append(cat_tensors)

            cat_tensors = input_tensor * torch.where(cat_tensors >= self.threshold, torch.tensor(1), torch.tensor(0)).unsqueeze(1)
            category_images.append(cat_tensors)

        category_images = torch.stack(category_images, dim=0)
        category_probs = torch.stack(category_probs, dim=0)

        return None, prob_maps
        #return category_images, category_probs