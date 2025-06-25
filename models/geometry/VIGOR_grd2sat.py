import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import functools

def VIGOR_grd2sat_uv(level, grd_H, grd_W, grd_cam_h, sat_H, sat_W, meter_per_pixel):
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
    phimin = phimin / np.pi * grd_H
    #CVUSA
    # phimin = phimin / np.pi * grd_H /170 *180 - (grd_H/170 *180 - grd_H) /2

    uv = torch.stack([theta, phimin.float()], dim=-1)

    return uv

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
