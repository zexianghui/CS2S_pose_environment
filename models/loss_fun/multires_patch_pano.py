# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved
import functools
import warnings

import numpy as np
import torch
import torch.nn as nn

from models.sat2density.layers import Conv2dBlock
from imaginaire.utils.distributed import master_only_print
import cv2

class Equirectangular():
    """
    Random sample a panorama image into a perspective view
    take https://github.com/fuenwang/Equirec2Perspec/blob/master/Equirec2Perspec.py as a reference
    """
    def __init__(self, width = 256, height = 256, FovX = 100, theta = [0, 0]):
        """
        width: output image's width
        height: output image's height
        FovX: perspective camera FOV on x-axis (degree)
        theta: theta field where img's theta degree from 
        """
        self.theta = theta
        self.width = width
        self.height = height
        self.type = type

        #create x-axis coordinates and corresponding y-axis coordinates
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y) 
        
        #create homogenerous coordinates
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        
        #translation matrix
        f = 0.5 * width * 1 / np.tan(np.radians(FovX/2))
        # cx = (width - 1) / 2.0
        # cy = (height - 1) / 2.0
        cx = (width) / 2.0
        cy = (height) / 2.0        
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)
        xyz = xyz @ K_inv.T
        self.xyz = xyz  ### self.xyz is the direction of the each ray in the camera space when camera is fixed



    def __call__(self, img1): 
        batch = img1.shape[0]
        PHI, THETA = self.getRandomRotation(batch)
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        #rotation matrix
        xy_grid = []
        for i in range(batch):
            R1, _ = cv2.Rodrigues(y_axis * np.radians(PHI[i]))
            R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(THETA[i]))
            R = R2 @ R1
            #rotate
            xyz = self.xyz @ R.T  ### ### xyz is the direction of the each ray in the camera space when camera is rotate
            norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
            xyz_norm = xyz / norm
            
            #transfer to image coordinates
            xy = self.xyz2xy(xyz_norm)
            device = img1.device
            xy = torch.from_numpy(xy).to(device).unsqueeze(0)
            xy_grid.append(xy)
        xy = torch.cat(xy_grid,dim=0)

        #resample
        return xy

    def xyz2xy(self, xyz_norm):
        #normlize
        x = xyz_norm[..., 0]
        y = xyz_norm[..., 1]
        z = xyz_norm[..., 2]

        lon = np.arctan2(x, z)
        lat = np.arcsin(y)
        ### transfer to the lon and lat

        X = lon / (np.pi)
        Y = lat / (np.pi) * 2
        xy = np.stack([X, Y], axis=-1)
        xy = xy.astype(np.float32)
        
        return xy

    def getRandomRotation(self,batch_size):
        # phi = np.random.rand(batch_size) * 360 -180
        phi = np.random.randint(-180,180,batch_size)
        assert(self.theta[0]<self.theta[1])
        theta = np.random.randint(self.theta[0],self.theta[1],batch_size)
        # theta = np.random.rand(batch_size)*(self.theta[1]-self.theta[0])-self.theta[0]
        return phi, theta

class Discriminator(nn.Module):
    r"""Multi-resolution patch discriminator.

    Args:
        dis_cfg (obj): Discriminator definition part of the yaml config
            file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, dis_cfg):
        super(Discriminator, self).__init__()
        master_only_print('Multi-resolution patch discriminator initialization.')
        # We assume the first datum is the ground truth image.
        num_input_channels = getattr(dis_cfg, 'input_channels', 3)
        # Calculate number of channels in the input label.

        # Build the discriminator.
        kernel_size = getattr(dis_cfg, 'kernel_size', 3)
        num_filters = getattr(dis_cfg, 'num_filters', 128)
        max_num_filters = getattr(dis_cfg, 'max_num_filters', 512)
        num_discriminators = getattr(dis_cfg, 'num_discriminators', 2)
        num_layers = getattr(dis_cfg, 'num_layers', 5)
        activation_norm_type = getattr(dis_cfg, 'activation_norm_type', 'none')
        weight_norm_type = getattr(dis_cfg, 'weight_norm_type', 'spectral')
        master_only_print('\tBase filter number: %d' % num_filters)
        master_only_print('\tNumber of discriminators: %d' % num_discriminators)
        master_only_print('\tNumber of layers in a discriminator: %d' % num_layers)
        master_only_print('\tWeight norm type: %s' % weight_norm_type)
        self.condition = getattr(dis_cfg, 'condition', None)
        # self.condition = dis_cfg.condition
        self.model = MultiResPatchDiscriminator(num_discriminators,
                                                kernel_size,
                                                num_input_channels,
                                                num_filters,
                                                num_layers,
                                                max_num_filters,
                                                activation_norm_type,
                                                weight_norm_type)
        master_only_print('Done with the Multi-resolution patch '
              'discriminator initialization.')

    def forward(self, data, net_G_output, real=True):
        r"""SPADE Generator forward.

        Args:
            data  (N x C1 x H x W tensor) : Ground truth images.
            net_G_output (dict):
                fake_images  (N x C1 x H x W tensor) : Fake images.
            real (bool): If ``True``, also classifies real images. Otherwise it
                only classifies generated images to save computation during the
                generator update.
        Returns:
            (tuple):
              - real_outputs (list): list of output tensors produced by
              - individual patch discriminators for real images.
              - real_features (list): list of lists of features produced by
                individual patch discriminators for real images.
              - fake_outputs (list): list of output tensors produced by
                individual patch discriminators for fake images.
              - fake_features (list): list of lists of features produced by
                individual patch discriminators for fake images.
        """
        output_x = dict()
        if self.condition:
            fake_input_x = torch.cat([net_G_output['pred'],net_G_output['generator_inputs']],dim=1)
        else:
            fake_input_x = net_G_output['pred']
        output_x['fake_outputs'], output_x['fake_features'], _ = \
            self.model.forward(fake_input_x)
        if real:
            if self.condition:
                real_input_x = torch.cat([net_G_output['pred'],net_G_output['generator_inputs']],dim=1)
            else:
                real_input_x = data
            output_x['real_outputs'], output_x['real_features'], _ = \
                self.model.forward(real_input_x)
        return output_x


class MultiResPatchDiscriminator(nn.Module):
    r"""Multi-resolution patch discriminator.

    Args:
        num_discriminators (int): Num. of discriminators (one per scale).
        kernel_size (int): Convolution kernel size.
        num_image_channels (int): Num. of channels in the real/fake image.
        num_filters (int): Num. of base filters in a layer.
        num_layers (int): Num. of layers for the patch discriminator.
        max_num_filters (int): Maximum num. of filters in a layer.
        activation_norm_type (str): batch_norm/instance_norm/none/....
        weight_norm_type (str): none/spectral_norm/weight_norm
    """

    def __init__(self,
                 num_discriminators=3,
                 kernel_size=3,
                 num_image_channels=3,
                 num_filters=64,
                 num_layers=4,
                 max_num_filters=512,
                 activation_norm_type='',
                 weight_norm_type='',
                 **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type' and key != 'patch_wise':
                warnings.warn(
                    "Discriminator argument {} is not used".format(key))

        self.discriminators = nn.ModuleList()
        for i in range(num_discriminators):
            net_discriminator = NLayerPatchDiscriminator(
                kernel_size,
                num_image_channels,
                num_filters,
                num_layers,
                max_num_filters,
                activation_norm_type,
                weight_norm_type)
            self.discriminators.append(net_discriminator)
        master_only_print('Done with the Multi-resolution patch '
              'discriminator initialization.')
        self.e = Equirectangular(theta=[-40., 40.],width = 128, height = 128,FovX = 100)

    def forward(self, input_x):
        r"""Multi-resolution patch discriminator forward.

        Args:
            input_x (tensor) : Input images.
        Returns:
            (tuple):
              - output_list (list): list of output tensors produced by
                individual patch discriminators.
              - features_list (list): list of lists of features produced by
                individual patch discriminators.
              - input_list (list): list of downsampled input images.
        """
        input_list = []
        output_list = []
        features_list = []
        input_N = nn.functional.interpolate(
            input_x, scale_factor=0.5, mode='bilinear',
            align_corners=True, recompute_scale_factor=True)
        equ= self.e(input_x)
        for i, net_discriminator in enumerate(self.discriminators):
            input_list.append(input_N)
            output, features = net_discriminator(input_N)
            output_list.append(output)
            features_list.append(features)
            if i == 0:
                input_N = torch.nn.functional.grid_sample(input_x, equ.float(), align_corners = True)*0.99
            elif i == 1:
                input_N = nn.functional.interpolate(
                    input_N, scale_factor=0.5, mode='bilinear',
                    align_corners=True, recompute_scale_factor=True)

        return output_list, features_list, input_list

class NLayerPatchDiscriminator(nn.Module):
    r"""Patch Discriminator constructor.

    Args:
        kernel_size (int): Convolution kernel size.
        num_input_channels (int): Num. of channels in the real/fake image.
        num_filters (int): Num. of base filters in a layer.
        num_layers (int): Num. of layers for the patch discriminator.
        max_num_filters (int): Maximum num. of filters in a layer.
        activation_norm_type (str): batch_norm/instance_norm/none/....
        weight_norm_type (str): none/spectral_norm/weight_norm
    """

    def __init__(self,
                 kernel_size,
                 num_input_channels,
                 num_filters,
                 num_layers,
                 max_num_filters,
                 activation_norm_type,
                 weight_norm_type):
        super(NLayerPatchDiscriminator, self).__init__()
        self.num_layers = num_layers
        padding = int(np.floor((kernel_size - 1.0) / 2))
        nonlinearity = 'leakyrelu'
        base_conv2d_block = \
            functools.partial(Conv2dBlock,
                              kernel_size=kernel_size,
                              padding=padding,
                              weight_norm_type=weight_norm_type,
                              activation_norm_type=activation_norm_type,
                              nonlinearity=nonlinearity,
                              # inplace_nonlinearity=True,
                              order='CNA')
        layers = [[base_conv2d_block(
            num_input_channels, num_filters, stride=2)]]
        for n in range(num_layers):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            stride = 2 if n < (num_layers - 1) else 1
            layers += [[base_conv2d_block(num_filters_prev, num_filters,
                                          stride=stride)]]
        layers += [[Conv2dBlock(num_filters, 1,
                                3, 1,
                                padding,
                                weight_norm_type=weight_norm_type)]]
        for n in range(len(layers)):
            setattr(self, 'layer' + str(n), nn.Sequential(*layers[n]))
        

    def forward(self, input_x):
        r"""Patch Discriminator forward.

        Args:
            input_x (N x C x H1 x W2 tensor): Concatenation of images and
                semantic representations.
        Returns:
            (tuple):
              - output (N x 1 x H2 x W2 tensor): Discriminator output value.
                Before the sigmoid when using NSGAN.
              - features (list): lists of tensors of the intermediate
                activations.
        """
        res = [input_x]
        for n in range(self.num_layers + 2):
            layer = getattr(self, 'layer' + str(n))
            x = res[-1]
            res.append(layer(x))
        output = res[-1]
        features = res[1:-1]
        return output, features
