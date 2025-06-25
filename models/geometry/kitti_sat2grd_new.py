import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import dataloader.KITTI_utils as utils
import os
import torchvision.transforms.functional as TF
# from ConvLSTM import VE_LSTM3D, VE_LSTM2D, VE_conv, S_LSTM2D
#from models_ford import loss_func
#from RNNs import NNrefine

EPS = utils.EPS

class gen_KITTI_sat2grd():
    def __init__(self):  # device='cuda:0',
        super(gen_KITTI_sat2grd, self).__init__()
    

    def grd_img2cam(self, grd_H, grd_W, ori_grdH, ori_grdW, ori_camera_k):
        
        # ori_camera_k = torch.tensor([[[582.9802,   0.0000, 496.2420],
        #                               [0.0000, 482.7076, 125.0034],
        #                               [0.0000,   0.0000,   1.0000]]], 
        #                             dtype=torch.float32, requires_grad=True)  # [1, 3, 3]
        
        camera_height = utils.get_camera_height()

        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]

        v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                              torch.arange(0, grd_W, dtype=torch.float32))
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0).to(ori_camera_k.device)  # [1, grd_H, grd_W, 3]
        xyz_w = torch.sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :], dim=-1)  # [1, grd_H, grd_W, 3]

        w = camera_height / torch.where(torch.abs(xyz_w[..., 1:2]) > utils.EPS, xyz_w[..., 1:2],
                                        utils.EPS * torch.ones_like(xyz_w[..., 1:2]))  # [BN, grd_H, grd_W, 1]
        xyz_grd = xyz_w * w  # [1, grd_H, grd_W, 3] under the grd camera coordinates
        # xyz_grd = xyz_grd.reshape(B, N, grd_H, grd_W, 3)

        mask = (xyz_grd[..., -1] > 0).float()  # # [1, grd_H, grd_W]

        return xyz_grd, mask, xyz_w
    
    def grd_img2cam_h(self, grd_H, grd_W, ori_grdH, ori_grdW, ori_camera_k):
        
        # ori_camera_k = torch.tensor([[[582.9802,   0.0000, 496.2420],
        #                               [0.0000, 482.7076, 125.0034],
        #                               [0.0000,   0.0000,   1.0000]]], 
        #                             dtype=torch.float32, requires_grad=True)  # [1, 3, 3]
        
        camera_height = [-3.3, -2.475, -1.65, -0.825, 0.825, 1.65, 2.475, 3.3]
        camera_height = torch.tensor(camera_height, dtype=torch.float32, requires_grad=True, device=ori_camera_k.device)

        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]

        v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                              torch.arange(0, grd_W, dtype=torch.float32))
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0).to(ori_camera_k.device)  # [1, grd_H, grd_W, 3]
        xyz_w = torch.sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :], dim=-1)  # [1, grd_H, grd_W, 3]

        w = camera_height[None,:,None,None,None] / torch.where(torch.abs(xyz_w[..., 1:2]) > utils.EPS, xyz_w[..., 1:2],
                                        utils.EPS * torch.ones_like(xyz_w[..., 1:2]))[:,None,:,:,:]  # [BN, grd_H, grd_W, 1]
        xyz_grd = xyz_w[:,None,:,:,:] * w  # [1, grd_H, grd_W, 3] under the grd camera coordinates
        # xyz_grd = xyz_grd.reshape(B, N, grd_H, grd_W, 3)
        B,_,H,W,_ = xyz_grd.size()
        for i in range(len(camera_height)):
            if camera_height[i] < 0:
                xyz_grd[:, i, int(H/2):] = -1e5
            if camera_height[i] > 0:
                xyz_grd[:, i, :int(H/2)] = -1e5
        mask = (xyz_grd[..., -1] > 0).float()  # # [1, grd_H, grd_W]

        return xyz_grd, mask, xyz_w

    def grd2cam2world2sat(self, xyz_grds, ori_shift_u, ori_shift_v, ori_heading, level,
                          satmap_sidelength):
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
        heading = ori_heading * 10 / 180 * np.pi
        shift_u = ori_shift_u * 20
        shift_v = ori_shift_v * 20

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
        # T0 = torch.unsqueeze(T0, dim=-1)  # shape = [B, N, 3, 1]
        # T = torch.einsum('bnij, bnj -> bni', -R, T0) # [B, N, 3]
        T = torch.sum(-R * T0[:, None, :], dim=-1)   # [B, 3]

        xyz_grd = xyz_grds.detach().to(ori_shift_u.device)
        mask = xyz_grds.detach().to(ori_shift_u.device)  # [B, grd_H, grd_W]
        grd_H, grd_W = xyz_grd.shape[1:3]
        
        xyz = torch.sum(R[:, None, None, :, :] * xyz_grd[:, :, :, None, :], dim=-1) + T[:, None, None, :]
        # [B, grd_H, grd_W, 3]
        # zx0 = torch.stack([xyz[..., 2], xyz[..., 0]], dim=-1)  # [B, N, grd_H, grd_W, 2]
        R_sat = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.float32, device=ori_shift_u.device, requires_grad=True)\
            .reshape(2, 3)
        zx = torch.sum(R_sat[None, None, None, :, :] * xyz[:, :, :, None, :], dim=-1)
        # [B, grd_H, grd_W, 2]
        # assert zx == zx0

        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.SatMap_end_sidelength / satmap_sidelength
        sat_uv = zx/meter_per_pixel + satmap_sidelength / 2  # [B, grd_H, grd_W, 2] sat map uv
        return sat_uv, mask,

    def grd2cam2world2sat_h(self, xyz_grds, ori_shift_u, ori_shift_v, ori_heading, level,
                          satmap_sidelength):
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
        heading = ori_heading * 10 / 180 * np.pi
        shift_u = ori_shift_u * 20
        shift_v = ori_shift_v * 20

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
        # T0 = torch.unsqueeze(T0, dim=-1)  # shape = [B, N, 3, 1]
        # T = torch.einsum('bnij, bnj -> bni', -R, T0) # [B, N, 3]
        T = torch.sum(-R * T0[:, None, :], dim=-1)   # [B, 3]

        xyz_grd = xyz_grds.detach().to(ori_shift_u.device)
        mask = xyz_grds.detach().to(ori_shift_u.device)  # [B, grd_H, grd_W]
        # grd_H, grd_W = xyz_grd.shape[1:3]
        
        xyz = torch.sum(R[:, None, None, None, :, :] * xyz_grd[:, :, :, :, None, :], dim=-1) + T[:,None, None, None, :]
        # [B, grd_H, grd_W, 3]
        # zx0 = torch.stack([xyz[..., 2], xyz[..., 0]], dim=-1)  # [B, N, grd_H, grd_W, 2]
        R_sat = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.float32, device=ori_shift_u.device, requires_grad=True)\
            .reshape(2, 3)
        zx = torch.sum(R_sat[None, None, None, None, :, :] * xyz[:, :, :, :, None, :], dim=-1)
        # [B, grd_H, grd_W, 2]
        # assert zx == zx0

        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.SatMap_end_sidelength / satmap_sidelength
        sat_uv = zx/meter_per_pixel + satmap_sidelength / 2  # [B, grd_H, grd_W, 2] sat map uv
        return sat_uv, mask,

    def grid_sample(self, image, optical):
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
        return out_val, None
    
    def sat2grd_h(self, sat_feat_A, gen_grd_H, gen_grd_W, left_camera_k, shift_u=None, shift_v=None, heading=None):
        ori_grdH = 128
        ori_grdW = 512
        # _,C, grd_H, grd_W = grd_feat.size()
        
        B = left_camera_k.size(0)
        A = sat_feat_A
        xyz_grd, mask, xyz_w = self.grd_img2cam_h(gen_grd_H, gen_grd_W, ori_grdH, ori_grdW, left_camera_k)
        uv, _ = self.grd2cam2world2sat_h(xyz_grd, shift_u, shift_v, heading, 0, A)

        uv = uv.reshape(B, 8, gen_grd_H*gen_grd_W, 2)
        uv[..., 0] /= A #sat size
        uv[..., 1] /= A
        bev_mask = ((uv[..., 1:2] > 0.0)
                    & (uv[..., 1:2] < 1.0)
                    & (uv[..., 0:1] < 1.0)
                    & (uv[..., 0:1] > 0.0))
        indexes = []
        bev_mask = bev_mask.reshape(B*8, gen_grd_H*gen_grd_W, 1)
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img.sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img) 

        max_len = max([len(each) for each in indexes])
        return uv.transpose(1,2)

        
    
    def __call__(self, sat_feat, gen_grd_H, gen_grd_W, left_camera_k, shift_u=None, shift_v=None, heading=None):
        
        ori_grdH = 128
        ori_grdW = 512
        # _,C, grd_H, grd_W = grd_feat.size()

        A = sat_feat.size(-1)
        xyz_grd, mask, xyz_w = self.grd_img2cam(gen_grd_H, gen_grd_W, ori_grdH, ori_grdW, left_camera_k)
        sat_uv, mask = self.grd2cam2world2sat(xyz_grd, shift_u, shift_v, heading, 0, A)
        sat2grd,_ = self.grid_sample(sat_feat, sat_uv)
        return sat2grd

# if __name__ =="__main__": 
#     from dataset.dataloader.KITTI_wo_loc import load_train_data
#     load_train_data
#     gen_KITTI_sat2grd = gen_KITTI_sat2grd()
#     gen_KITTI_sat2grd(tensor_image, )

#     transform = transforms.ToPILImage()
#     image = transform(bev_project)

#     image.save("output_image.jpg")
#     print(1)