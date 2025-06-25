import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

import torch

def sample_within_bounds(signal, x, y, bounds):
    ''' '''
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    sample = torch.zeros((signal.shape[0], x.shape[0], x.shape[1], signal.shape[-1])).to(signal.device)
    
    sample[:, idxs, :] = signal[:, x[idxs], y[idxs], :]

    return sample


def sample_bilinear(signal, rx, ry):
    ''' '''

    signal_dim_x = signal.shape[1]
    signal_dim_y = signal.shape[2]
    

    # obtain four sample coordinates
    ix0 = rx.long()
    iy0 = ry.long()

    ix1 = ix0 + 1
    iy1 = iy0 + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    # na = np.newaxis
    # linear interpolation in x-direction
    fx1 = (ix1 - rx)[..., None] * signal_00 + (rx - ix0)[..., None] * signal_10
    fx2 = (ix1 - rx)[..., None] * signal_01 + (rx - ix0)[..., None] * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry)[..., None] * fx1 + (ry - iy0)[..., None] * fx2

def cvusa_aer2grd(img, token_h, token_w):
    B, H, W, C= img.shape
    S = H
    height = 2 * token_h #image height = 128
    width = token_w #image width = 512

    s = S / 55
    grd_height = -2
    
    i = torch.arange(0, height)
    j = torch.arange(0, width)
    jj, ii = torch.meshgrid(j, i, indexing='xy')
    
    tanii = torch.tan(ii * torch.pi / height)
    y = (S / 2. + 0.) - s * grd_height * tanii * torch.sin(2 * torch.pi * jj / width)
    x = (S / 2. + 0.) + s * grd_height * tanii * torch.cos(2 * torch.pi * jj / width)
    y[:height // 2, ...] = -1
    x[:height // 2, ...] = -1
    signal = sample_bilinear(img, x.to(img.device), y.to(img.device))

    signal = signal[:, int(0.25 * height) : int(0.75 * height), ...]

    return signal

def cvact_aer2grd(img, token_h, token_w):
    B, H, W, C= img.shape
    S = H
    height = token_h #image height = 128
    width = token_w #image width = 512

    s = S / 50
    grd_height = -2
    
    i = torch.arange(0, height)
    j = torch.arange(0, width)
    jj, ii = torch.meshgrid(j, i, indexing='xy')
    
    tanii = torch.tan(ii * torch.pi / height)
    y = (S / 2. + 0.) - s * grd_height * tanii * torch.sin(2 * torch.pi * jj / width)
    x = (S / 2. + 0.) + s * grd_height * tanii * torch.cos(2 * torch.pi * jj / width)
    y[:height // 2, ...] = -1
    x[:height // 2, ...] = -1
    signal = sample_bilinear(img, x, y)

    return signal

def cvusa_grd2aer(img, token_h, token_w, grd_s):
    height = token_h * 2
    width = token_w

    S = grd_s
    s = S / 27.5
    grd_height = -2
    i = torch.arange(0, S)
    j = torch.arange(0, S)
    jj, ii = torch.meshgrid(j, i, indexing='xy')

    radius = torch.sqrt((ii - (S / 2 - 0.5)) ** 2 + (jj - (S / 2 - 0.5)) ** 2)

    Theta = torch.zeros([S, S])
    Theta[:, 0:int(S / 2)] = torch.arctan(
        (ii[:, 0:int(S / 2)] - (S / 2 - 0.5)) / (jj[:, 0:int(S / 2)] - (S / 2 - 0.5))) + 0.5 * torch.pi
    Theta[:, int(S / 2):] = torch.arctan(
        (ii[:, int(S / 2):] - (S / 2 - 0.5)) / (jj[:, int(S / 2):] - (S / 2 - 0.5))) + 1.5 * torch.pi
    Phimin = torch.pi + torch.arctan(radius / s / grd_height)
    # Phimin = height - radius/S*2*height/2   # for a regular polar

    Theta = Theta / 2 / torch.pi * width
    Phimin = Phimin / torch.pi * height
    
    padding = (height//4 - 2) if (height//4 - 2) > 0 else 1
    signal = torch.nn.functional.pad(img, (0, 0, 0, 0, padding, padding), mode='constant', value=0)

    signal = sample_bilinear(signal, Phimin, Theta)

    return signal

def cvact_grd2aer(img, token_h, token_w, grd_s):
    height = token_h
    width = token_w

    S = grd_s
    s = S / 40
    grd_height = -2
    i = torch.arange(0, S)
    j = torch.arange(0, S)
    jj, ii = torch.meshgrid(j, i, indexing='xy')

    radius = torch.sqrt((ii - (S / 2 - 0.5)) ** 2 + (jj - (S / 2 - 0.5)) ** 2)

    Theta = torch.zeros([S, S])
    Theta[:, 0:int(S / 2)] = torch.arctan(
        (ii[:, 0:int(S / 2)] - (S / 2 - 0.5)) / (jj[:, 0:int(S / 2)] - (S / 2 - 0.5))) + 0.5 * torch.pi
    Theta[:, int(S / 2):] = torch.arctan(
        (ii[:, int(S / 2):] - (S / 2 - 0.5)) / (jj[:, int(S / 2):] - (S / 2 - 0.5))) + 1.5 * torch.pi
    Phimin = torch.pi + torch.arctan(radius / s / grd_height)
    # Phimin = height - radius/S*2*height/2   # for a regular polar

    Theta = Theta / 2 / torch.pi * width
    Phimin = Phimin / torch.pi * height
    
    signal = img
    # signal = nn.pad(img, (0, 0, 0, 0, height//5, height//5), mode='constant', value=0)

    signal = sample_bilinear(signal, Phimin, Theta)

    return signal

# CVUSA_sat2grd_uv 
level = 0
H = 128
W = 512
grd_path = "dataset/CVUSA/streetview/panos/0000003.jpg"
image = Image.open(grd_path).resize((W,H))
transform = transforms.Compose([ 
    transforms.ToTensor()  
])
tensor_image = transform(image)
tensor_image = tensor_image.unsqueeze(0).permute(0,2,3,1)

feat_H = 256
meter_per_pixel = torch.ones(1) * (55/256) * 256 / feat_H

out_val = cvusa_grd2aer(tensor_image, 128, 512, 256).permute(0,3,1,2)
# out_val, mask = grid_sample(tensor_image, optical)
transform = transforms.ToPILImage()
image = transform(out_val[0])
image.save("output_image.jpg")

image = Image.open(grd_path)
image.save("grd_img.jpg")

image = Image.open(grd_path.replace('streetview/panos', 'bingmap/19')).resize((256,256))
image.save("sat_img.jpg")
