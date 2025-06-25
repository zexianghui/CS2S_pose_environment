import torch
import numpy as np
import dataloader.KITTI_utils as utils

def grd_img2cam(ori_camera_k, grd_H, grd_W, ori_grdH, ori_grdW):
    # ori_camera_k = torch.tensor([[[582.9802, 0.0000, 496.2420],
    #                                 [0.0000, 482.7076, 125.0034],
    #                                 [0.0000, 0.0000, 1.0000]]],
    #                             dtype=torch.float32, requires_grad=True)  # [1, 3, 3]

    camera_height = utils.get_camera_height()

    camera_k = ori_camera_k.clone()
    camera_k[:, :1, :] = ori_camera_k[:, :1,
                            :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
    camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
    camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]

    v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                            torch.arange(0, grd_W, dtype=torch.float32))
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0).to(camera_k_inv.device)  # [1, grd_H, grd_W, 3]
    xyz_w = torch.sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :], dim=-1)  # [1, grd_H, grd_W, 3]

    w = camera_height / torch.where(torch.abs(xyz_w[..., 1:2]) > utils.EPS, xyz_w[..., 1:2],
                                    utils.EPS * torch.ones_like(xyz_w[..., 1:2]))  # [BN, grd_H, grd_W, 1]
    xyz_grd = xyz_w * w  # [1, grd_H, grd_W, 3] under the grd camera coordinates

    mask = (xyz_grd[..., -1] > 0).float()  # # [1, grd_H, grd_W]

    return xyz_grd, mask, xyz_w

def grid_sample(image, optical):
    # values in optical within range of [0, H], and [0, W]
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0].view(N, 1, H, W).to(image.device)
    iy = optical[..., 1].view(N, 1, H, W).to(image.device)

    with torch.no_grad():
        ix_nw = torch.floor(ix).to(image.device)  # north-west  upper-left-x
        iy_nw = torch.floor(iy).to(image.device)  # north-west  upper-left-y
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

    return out_val, None

def grd2cam2world2sat(xyz_grds, ori_shift_u, ori_shift_v, ori_heading, level, satmap_sidelength,):
    '''
    realword: X: south, Y:down, Z: east
    camera: u:south, v: down from center (when heading east, need to rotate heading angle)
    Args:
        ori_shift_u: [B, 1]
        ori_shift_v: [B, 1]
        heading: [B, 1]
        XYZ_1: [H,W,4]
        ori_camera_k: [B,3,3]
        grd_H:
        grd_W:
        ori_grdH:
        ori_grdW:

    Returns:
    '''
    B, _ = ori_heading.shape
    rotation_range = 0
    shift_range_lon = 0
    shift_range_lat = 0
    
    heading = ori_heading * rotation_range / 180 * np.pi
    shift_u = ori_shift_u * shift_range_lon
    shift_v = ori_shift_v * shift_range_lat

    cos = torch.cos(heading)
    sin = torch.sin(heading)
    zeros = torch.zeros_like(cos)
    ones = torch.ones_like(cos)
    R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B, 9]
    R = R.view(B, 3, 3)  # shape = [B, N, 3, 3]
    # this R is the inverse of the R in G2SP

    camera_height = utils.get_camera_height()
    # camera offset, shift[0]:east,Z, shift[1]:north,X
    height = camera_height * torch.ones_like(shift_u[:, :1])
    T0 = torch.cat([shift_v, height, -shift_u], dim=-1)  # shape = [B, 3]
    T = torch.sum(-R * T0[:, None, :], dim=-1)  # [B, 3]
    # The above R, T define transformation from camera to world

    xyz_grd = xyz_grds[level][0].detach().to(ori_shift_u.device)#.repeat(B, 1, 1, 1)
    mask = xyz_grds[level][1].detach().to(ori_shift_u.device)#.repeat(B, 1, 1)  # [B, grd_H, grd_W]

    grd_H, grd_W = xyz_grd.shape[1:3]

    xyz = torch.sum(R[:, None, None, :, :] * xyz_grd[:, :, :, None, :], dim=-1) + T[:, None, None, :]
    # [B, grd_H, grd_W, 3]

    R_sat = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.float32, device=ori_shift_u.device, requires_grad=True) \
        .reshape(2, 3)
    zx = torch.sum(R_sat[None, None, None, :, :] * xyz[:, :, :, None, :], dim=-1)
    # [B, grd_H, grd_W, 2]

    meter_per_pixel = utils.get_meter_per_pixel()
    meter_per_pixel *= utils.SatMap_end_sidelength / satmap_sidelength
    sat_uv = zx / meter_per_pixel + satmap_sidelength / 2  # [B, grd_H, grd_W, 2] sat map uv

    return sat_uv, mask

def project_map_to_grd(xyz_grds, sat_f, sat_c, shift_u, shift_v, heading, level):
    '''
    Args:
        sat_f: [B, C, H, W]
        sat_c: [B, 1, H, W]
        shift_u: [B, 2]
        shift_v: [B, 2]
        heading: [B, 1]
        camera_k: [B, 3, 3]

        ori_grdH:
        ori_grdW:

    Returns:

    '''
    B, C, satmap_sidelength, _ = sat_f.size()

    uv, mask = grd2cam2world2sat(xyz_grds, shift_u, shift_v, heading, level, satmap_sidelength)
    # [B, H, W, 2], [B, H, W], [B, H, W, 2], [B, H, W, 2], [B,H, W, 2]

    B, grd_H, grd_W, _ = uv.shape

    sat_f_trans, _ = grid_sample(sat_f, uv)
    sat_f_trans = sat_f_trans * mask[:, None, :, :]

    if sat_c is not None:
        sat_c_trans, _ = grid_sample(sat_c, uv)
        sat_c_trans = sat_c_trans * mask[:, None, :, :]
    else:
        sat_c_trans = None

    return sat_f_trans, sat_c_trans, uv * mask[:, :, :, None], mask



def get_xyz_grds(ori_camera_k, ori_grdH, ori_grdW, num_levels):
    xyz_grds = []
    grd_H, grd_W = 16, 64
    for level in range(num_levels):
        # if level <3:
        #     xyz_grds.append((0, 0, 0))
        #     continue
        # grd_H, grd_W = ori_grdH / (2 **  level), ori_grdW / (2 ** level)

        xyz_grd, mask, xyz_w = grd_img2cam(ori_camera_k, grd_H, grd_W, ori_grdH,
                                                ori_grdW)  # [1, grd_H, grd_W, 3] under the grd camera coordinates
        xyz_grds.append((xyz_grd, mask, xyz_w))

    return xyz_grds