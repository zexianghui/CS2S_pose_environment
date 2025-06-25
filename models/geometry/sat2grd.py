
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


def CVUSA_sat2grd_uv(level, grd_H, grd_W, grd_cam_h, sat_H, sat_W, meter_per_pixel):
    '''
    rot.shape = [B]
    shift_u.shape = [B]
    shift_v.shape = [B]
    H: scalar  height of grd feature map, from which projection is conducted
    W: scalar  width of grd feature map, from which projection is conducted
    '''

    B = meter_per_pixel.shape[0]

    S = sat_H / np.power(2, level)

    ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=meter_per_pixel.device),
                            torch.arange(0, S, dtype=torch.float32, device=meter_per_pixel.device))
    ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] v dimension
    jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] u dimension
    radius = torch.sqrt((ii-(S/2-0.5))**2 + (jj-(S/2-0.5 ))**2)
    theta = torch.atan2(ii - (S / 2 - 0.5), jj - (S / 2 - 0.5))
    theta = (-np.pi / 2 + (theta) % (2 * np.pi)) % (2 * np.pi)

    theta = theta / 2 / np.pi * grd_W

    # meter_per_pixel = meter_per_pixel * 256 / S
    phimin = torch.atan2(radius * meter_per_pixel[:, None, None], torch.tensor(grd_cam_h * -1))
    # phimin = phimin / np.pi * grd_H
    #CVUSA
    phimin = phimin / np.pi * grd_H*2 - grd_H/2

    uv = torch.stack([theta, phimin.float()], dim=-1)

    return uv

def CVACT_sat2grd_uv(grd_H, grd_W, grd_cam_h, sat_H, sat_W):
    '''
    rot.shape = [B]
    shift_u.shape = [B]
    shift_v.shape = [B]
    H: scalar  height of grd feature map, from which projection is conducted
    W: scalar  width of grd feature map, from which projection is conducted
    '''
    S = sat_H 
    meter_per_pixel = torch.ones(1) * (50 / 256) * 256 / S
    B = meter_per_pixel.shape[0]

    ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=meter_per_pixel.device),
                            torch.arange(0, S, dtype=torch.float32, device=meter_per_pixel.device))
    ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] v dimension
    jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] u dimension
    radius = torch.sqrt((ii-(S/2-0.5))**2 + (jj-(S/2-0.5 ))**2)
    theta = torch.atan2(ii - (S / 2 - 0.5), jj - (S / 2 - 0.5))
    theta = (-np.pi / 2 + (theta) % (2 * np.pi)) % (2 * np.pi)

    theta = theta / 2 / np.pi * grd_W

    # meter_per_pixel = meter_per_pixel * 256 / S
    phimin = torch.atan2(radius * meter_per_pixel[:, None, None], torch.tensor(grd_cam_h * -1))
    # phimin = phimin / np.pi * grd_H
    #CVUSA
    phimin = phimin / np.pi * grd_H

    uv = torch.stack([theta, phimin.float()], dim=-1)

    return uv

def grd2sat_uv(level, grd_H, grd_W, grd_cam_h, sat_H, sat_W, meter_per_pixel):
    '''
    rot.shape = [B]
    shift_u.shape = [B]
    shift_v.shape = [B]
    H: scalar  height of grd feature map, from which projection is conducted
    W: scalar  width of grd feature map, from which projection is conducted
    '''
    grd_H = grd_H / np.power(2, level)
    grd_W = grd_W / np.power(2, level)
    meter_per_pixel = meter_per_pixel * np.power(2, level)
    B = meter_per_pixel.shape[0]

    ii, jj = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32, device=meter_per_pixel.device),
                            torch.arange(0, grd_W, dtype=torch.float32, device=meter_per_pixel.device))
    ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [1,8,32] v dimension 8
    jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [1,8,32] u dimension 32
    # ii = true[..., 1]  # [1,8,32] v dimension 8
    # jj = true[..., 0]  # [1,8,32] u dimension 32

    ii = (ii / grd_H * np.pi / 2) + np.pi/4
    radius =  torch.tensor(grd_cam_h * -1) * torch.tan(ii)    
    radius = radius / meter_per_pixel[:, None, None]

    theta = jj / grd_W * 2 * np.pi - np.pi /2 

    sat_v = sat_W / 2 - radius * torch.sin(theta) 
    sat_u = sat_H / 2 - radius * torch.cos(theta) 

    uv = torch.stack([sat_u, sat_v], dim=-1)

    return uv

def CVUSA_grd2sat_uv_h(grd_H, grd_W, sat_H, sat_W, meter_per_pixel):
    '''
    grd_H:  128
    grd_W:  512
    sat_H:  256
    sat_W:  256
    meter_per_pixel: 0.21 

    return:
    reference_points_rebatch: [1, 8, max_len 2] 
    indexes: [8, max_len]   
    '''
    grd_H = grd_H 
    grd_W = grd_W 
    meter_per_pixel = meter_per_pixel
    B = meter_per_pixel.shape[0]

    sample_h = [-3, -2 , -1, 1, 2, 3, 4, 5]
    # sample_h = torch.tensor(sample_h, device=meter_per_pixel.device).unsqueeze(0).repeat(B, 1)

    ii, jj = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32, device=meter_per_pixel.device),
                            torch.arange(0, grd_W, dtype=torch.float32, device=meter_per_pixel.device))
    ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [1,8,32] v dimension 8
    jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [1,8,32] u dimension 32
    # ii = true[..., 1]  # [1,8,32] v dimension 8
    # jj = true[..., 0]  # [1,8,32] u dimension 32

    ii = (ii / grd_H * np.pi / 2) + np.pi/4
    radius = torch.zeros((B, 8, grd_H, grd_W), device=meter_per_pixel.device)
    
    for idx_h in range(len(sample_h)):
        h = sample_h[idx_h]
        if h<0:
            radius[:, idx_h, int(grd_H/2):, :] =  torch.tensor(h) * torch.tan(ii)[:, int(grd_H/2):, :]
        if h>0:
            radius[:, idx_h, :int(grd_H/2), :] =  torch.tensor(h) * torch.tan(ii)[:, :int(grd_H/2), :]
    radius = radius / meter_per_pixel[:, None, None]
    # radius = torch.clamp(radius, min=0, max=sat_W/2)

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


def CVACT_grd2sat_uv_h(grd_H, grd_W, sat_H, sat_W):
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
    meter_per_pixel = torch.ones(1) * (50 / 256) * 256 / sat_H 
    B = meter_per_pixel.shape[0]

    sample_h = [-3, -2 , -1, 1, 2, 3, 4, 5]
    # sample_h = torch.tensor(sample_h, device=meter_per_pixel.device).unsqueeze(0).repeat(B, 1)

    ii, jj = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32, device=meter_per_pixel.device),
                            torch.arange(0, grd_W, dtype=torch.float32, device=meter_per_pixel.device))
    ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [1,8,32] v dimension 8
    jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [1,8,32] u dimension 32
    # ii = true[..., 1]  # [1,8,32] v dimension 8
    # jj = true[..., 0]  # [1,8,32] u dimension 32

    ii = (ii / grd_H * np.pi) #CVACT
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

def grid_sample(image, optical):
        # values in optical within range of [0, H], and [0, W]
        N, C, IH, IW = image.shape
        _, H, W, _ = optical.shape

        ix = optical[..., 0].view(N, 1, H, W)
        iy = optical[..., 1].view(N, 1, H, W)

        with torch.no_grad():
            ix_nw = torch.floor(ix)  # north-west  upper-left-x
            iy_nw = torch.floor(iy)  # north-west  upper-left-y
            ix_ne = ix_nw + 1        # north-east  upper-right-x
            iy_ne = iy_nw            # north-east  upper-right-y
            ix_sw = ix_nw            # south-west  lower-left-x
            iy_sw = iy_nw + 1        # south-west  lower-left-y
            ix_se = ix_nw + 1        # south-east  lower-right-x
            iy_se = iy_nw + 1        # south-east  lower-right-y

            torch.clamp(ix_nw, 0, IW -1, out=ix_nw)
            torch.clamp(iy_nw, 0, IH -1, out=iy_nw)

            torch.clamp(ix_ne, 0, IW -1, out=ix_ne)
            torch.clamp(iy_ne, 0, IH -1, out=iy_ne)

            torch.clamp(ix_sw, 0, IW -1, out=ix_sw)
            torch.clamp(iy_sw, 0, IH -1, out=iy_sw)

            torch.clamp(ix_se, 0, IW -1, out=ix_se)
            torch.clamp(iy_se, 0, IH -1, out=iy_se)

        mask_x = (ix >= 0) & (ix <= IW - 1)
        mask_y = (iy >= 0) & (iy <= IH - 1)
        mask = mask_x * mask_y

        assert torch.sum(mask) > 0

        nw = (ix_se - ix) * (iy_se - iy) * mask
        ne = (ix - ix_sw) * (iy_sw - iy) * mask
        sw = (ix_ne - ix) * (iy - iy_ne) * mask
        se = (ix - ix_nw) * (iy - iy_nw) * mask

        image = image.view(N, C, IH * IW)

        nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
        ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
        sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
        se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)

        out_val = (nw_val * nw + ne_val * ne + sw_val * sw + se_val * se)

        # mask_pano = torch.ones_like(out_val)
        # mask_pano[:,:, :int(H/2), :] = 0
        # mask = mask * mask_pano

        # out_val = out_val * mask_pano
        return out_val, mask