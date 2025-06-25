
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

def VIGOR_sat2grd_uv(grd_H, grd_W, sat_H, sat_W):
    '''
    grd_H:  128
    grd_W:  512
    sat_H:  256
    sat_W:  256
    meter_per_pixel: 0.1171 

    return:
    reference_points_rebatch: [1, 8, max_len 2]  
    indexes: [8, max_len]  
    '''
    grd_H = grd_H 
    grd_W = grd_W 
    meter_per_pixel = torch.ones(1) * 0.11 * 640 / 512 
    B = meter_per_pixel.shape[0]

    sample_h = [-2]
    # sample_h = torch.tensor(sample_h, device=meter_per_pixel.device).unsqueeze(0).repeat(B, 1)

    ii, jj = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32, device=meter_per_pixel.device),
                            torch.arange(0, grd_W, dtype=torch.float32, device=meter_per_pixel.device))
    ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [1,8,32] v dimension 8
    jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [1,8,32] u dimension 32
    # ii = true[..., 1]  # [1,8,32] v dimension 8
    # jj = true[..., 0]  # [1,8,32] u dimension 32

    # ii = (ii / grd_H * np.pi/180*170) + np.pi/180*10/2 #CVACT
    ii = (ii / grd_H * np.pi)
    radius = torch.zeros((B, 1, grd_H, grd_W), device=meter_per_pixel.device)
    
    for idx_h in range(len(sample_h)):
        h = sample_h[idx_h]
        if h<0:
            radius[:, idx_h, int(grd_H/2):, :] =  torch.tensor(h) * torch.tan(ii)[:, int(grd_H/2):, :]
        if h>0:
            radius[:, idx_h, :int(grd_H/2), :] =  torch.tensor(h) * torch.tan(ii)[:, :int(grd_H/2), :]
    radius = radius / meter_per_pixel[:, None, None]
    # radius = torch.clamp(radius, min=0, max=sat_W/2)

    # theta = (jj / grd_W * 2 * np.pi - np.pi /2).unsqueeze(1).repeat(1, 8, 1, 1)
    theta = (jj / grd_W * 2 * np.pi - np.pi /2).unsqueeze(1).repeat(1, 1, 1, 1)

    sat_v = sat_W / 2 - radius * torch.sin(theta)
    sat_u = sat_H / 2 - radius * torch.cos(theta)

    uv = torch.stack([sat_u, sat_v], dim=-1)
    for idx_h in range(len(sample_h)):
        h = sample_h[idx_h]
        if h<0:
            uv[:, idx_h, :int(grd_H/2), :, :] =  torch.zeros_like(uv[:, idx_h, :int(grd_H/2), :, :])
        if h>0:
            uv[:, idx_h, int(grd_H/2):, :, :] =  torch.zeros_like(uv[:, idx_h, int(grd_H/2):, :, :])

    uv = uv.reshape(1, grd_H*grd_W, 2)
    uv[..., 0] /= sat_H #sat size
    uv[..., 1] /= sat_H
    bev_mask = ((uv[..., 1:2] > 0.0)
                & (uv[..., 1:2] < 1.0)
                & (uv[..., 0:1] < 1.0)
                & (uv[..., 0:1] > 0.0))
    # indexes = []
    # for i, mask_per_img in enumerate(bev_mask):
    #     index_query_per_img = mask_per_img.sum(-1).nonzero().squeeze(-1)
    #     indexes.append(index_query_per_img) 

    # max_len = max([len(each) for each in indexes])
    # reference_points_rebatch = uv.new_zeros([1,  max_len, 8, 2])
    # for i in range(8):
    #     index_query_per_img = indexes[i]
    #     reference_points_rebatch[:, :len(index_query_per_img), i] = uv[i,index_query_per_img]
    return uv, bev_mask

def VIGOR_sat2grd_uv_h(grd_H, grd_W, sat_H, sat_W):
    '''
    grd_H:  128
    grd_W:  512
    sat_H:  256
    sat_W:  256
    meter_per_pixel: 0.1171 

    return:
    reference_points_rebatch: [1, 8, max_len 2] 
    indexes: [8, max_len] 
    '''
    grd_H = grd_H 
    grd_W = grd_W 
    meter_per_pixel = torch.ones(1) * 0.11 * 640 / 512 
    B = meter_per_pixel.shape[0]

    sample_h = [-4, -3, -2 , -1, 1, 3, 5, 7]
    # sample_h = torch.tensor(sample_h, device=meter_per_pixel.device).unsqueeze(0).repeat(B, 1)

    ii, jj = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32, device=meter_per_pixel.device),
                            torch.arange(0, grd_W, dtype=torch.float32, device=meter_per_pixel.device))
    ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [1,8,32] v dimension 8
    jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [1,8,32] u dimension 32
    # ii = true[..., 1]  # [1,8,32] v dimension 8
    # jj = true[..., 0]  # [1,8,32] u dimension 32

    # ii = (ii / grd_H * np.pi/180*170) + np.pi/180*10/2 #CVACT
    ii = (ii / grd_H * np.pi)
    radius = torch.zeros((B, 8, grd_H, grd_W), device=meter_per_pixel.device)
    
    for idx_h in range(len(sample_h)):
        h = sample_h[idx_h]
        if h<0:
            radius[:, idx_h, int(grd_H/2):, :] =  torch.tensor(h) * torch.tan(ii)[:, int(grd_H/2):, :]
        if h>0:
            radius[:, idx_h, :int(grd_H/2), :] =  torch.tensor(h) * torch.tan(ii)[:, :int(grd_H/2), :]
    radius = radius / meter_per_pixel[:, None, None]
    # radius = torch.clamp(radius, min=0, max=sat_W/2)

    # theta = (jj / grd_W * 2 * np.pi - np.pi /2).unsqueeze(1).repeat(1, 8, 1, 1)
    theta = (jj / grd_W * 2 * np.pi - np.pi /2).unsqueeze(1).repeat(1, 8, 1, 1)

    sat_v = sat_W / 2 - radius * torch.sin(theta)
    sat_u = sat_H / 2 - radius * torch.cos(theta)

    uv = torch.stack([sat_u, sat_v], dim=-1)
    for idx_h in range(len(sample_h)):
        h = sample_h[idx_h]
        if h<0:
            uv[:, idx_h, :int(grd_H/2), :, :] =  torch.zeros_like(uv[:, idx_h, :int(grd_H/2), :, :])
        if h>0:
            uv[:, idx_h, int(grd_H/2):, :, :] =  torch.zeros_like(uv[:, idx_h, int(grd_H/2):, :, :])

    uv = uv.reshape(8, grd_H*grd_W, 2)
    uv[..., 0] /= sat_H #sat size
    uv[..., 1] /= sat_H
    bev_mask = ((uv[..., 1:2] > 0.0)
                & (uv[..., 1:2] < 1.0)
                & (uv[..., 0:1] < 1.0)
                & (uv[..., 0:1] > 0.0))
    indexes = []
    for i, mask_per_img in enumerate(bev_mask):
        index_query_per_img = mask_per_img.sum(-1).nonzero().squeeze(-1)
        indexes.append(index_query_per_img) 

    max_len = max([len(each) for each in indexes])
    reference_points_rebatch = uv.new_zeros([1,  max_len, 8, 2])
    for i in range(8):
        index_query_per_img = indexes[i]
        reference_points_rebatch[:, :len(index_query_per_img), i] = uv[i,index_query_per_img]
    return reference_points_rebatch, indexes