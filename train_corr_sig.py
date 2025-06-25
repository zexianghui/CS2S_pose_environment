import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['MASTER_PORT'] = '7319'

import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from models.geometry.sat2grd import CVUSA_sat2grd_uv, grid_sample, CVACT_sat2grd_uv
from models.geometry.VIGOR_grd2sat import VIGOR_grd2sat_uv
from models.geometry.kitti_grd2sat import project_grd_to_map
from models.grd_sat_middle_pth.grd_sat_middel_end import grd_solve, sat_solve, infoNCELoss, crossEntropy

from utils.util import instantiate_from_config
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms
import torchvision.transforms.functional as TF

import random
import dataset.dataloader.KITTI_utils as utils
def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--base", nargs="*", metavar="base_config.yaml",help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    parser.add_argument("--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=True, help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument("--logdir", type=str, const=True, default="result", nargs="?", help="logdir")

    parser.add_argument("--train", type=str2bool, const=True, default=False, nargs="?", help="train")
    parser.add_argument("--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir")
    
    parser.add_argument("--test", type=str, default="", nargs="?")
    return parser


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    seed_everything(opt.seed)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    data = instantiate_from_config(config.data)
    data.setup()
    test_dataset = data._test_dataloader()
    # train_dataset = data._train_dataloader()
    
    model = instantiate_from_config(config.model).cuda()
    state_dict = torch.load(opt.test, map_location='cpu')
    state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict = True)
    model = model.eval()

    grd_solver = grd_solve(pth_path=None).cuda()
    sat_solver = sat_solve(pth_path=None, encoder_model = model.pre_AE_model.encoder, quant_conv = model.pre_AE_model.quant_conv).cuda()
    grd_solver.train()
    sat_solver.train()
    
    for param in grd_solver.parameters():
        param.requires_grad = True
    for param in sat_solver.parameters():
        param.requires_grad = True
    params = [p for p in grd_solver.parameters()] + [p for p in sat_solver.parameters()]
    optimizer = torch.optim.Adam(params, lr=1e-5, betas=(0.9, 0.999))
    global_step = 0
    
    for epoch in range(0, 500):
        for batch in test_dataset:
            if 'CVUSA' in config.data.params.train.target or 'CVACT' in config.data.params.train.target:
                with torch.no_grad():
                    inputs = batch["sat"].cuda()
                    outputs = batch["pano"].cuda()
                    grd_path = batch['paths']
                    gt_shift_x = batch['shift_x']
                    gt_shift_y = batch['shift_y']

                    outputs = outputs*2 - 1
                    inputs = inputs*2 - 1
                    grd_laten = model.pre_AE_model.encode(outputs).sample()*model.scale_factor

                    grd_laten = grd_laten.detach()

                    t = torch.randint(0, 100, (grd_laten.shape[0],), device=grd_laten.device).long()
                    grd_noisy = model.DDPM.q_sample(grd_laten, t)

                    B = grd_noisy.size()[0]
                    A = 32
                    crop_H = int(26)
                    crop_W = int(26)
                    d = []
                    for i in range(B):
                        gt_h = torch.linspace(-(A - crop_H)*4, (A - crop_H)*4, A - crop_H + 1).to(grd_noisy.device)
                        gt_w = torch.linspace(-(A - crop_W)*4, (A - crop_W)*4, A - crop_W + 1).to(grd_noisy.device)
                        gt_h = gt_h - gt_shift_y[i]
                        gt_w = gt_w - gt_shift_x[i]
                        gt_h, gt_w = torch.meshgrid(gt_h, gt_w, indexing='ij')

                        d_batch = torch.sqrt(gt_h**2 + gt_w**2)
                        d.append(d_batch.unsqueeze(0))
                    d = torch.cat(d, dim=0)
                    sigma, mu = 4, 0.0
                    gt_grid = torch.exp(-((d - mu) ** 2) / (2.0 * sigma ** 2)).to(grd_noisy.device)
                    if 'CVUSA' in config.data.params.train.target:
                        meter_per_pixel = torch.ones(1) * 0.21 * 256 / A
                        optical = CVUSA_sat2grd_uv(0, grd_noisy.size(2), grd_noisy.size(3), 2, A, A, meter_per_pixel)
                    if 'CVACT' in config.data.params.train.target:
                        optical = CVACT_sat2grd_uv(grd_noisy.size(2), grd_noisy.size(3), 2, A, A)
                    optical = optical.to(grd_noisy.device).repeat(B, 1, 1, 1)
                    # optical = grd2sat_uv(level, H, W, 2, feat_H, feat_H, meter_per_pixel)
                    BEV_map, _ = grid_sample(grd_noisy, optical)
                    g2s_feat = TF.center_crop(BEV_map, [crop_H, crop_W])
                

                sat_laten = sat_solver.encode(inputs).mode()*model.scale_factor
                sat_noisy = model.DDPM.q_sample(sat_laten, t)
                sat_noisy_patch = []
                gt_label = []
                for b in range(B):
                    sat_noisy_patch_batch = []
                    for h in range(0, sat_noisy.shape[-2] - crop_H + 1, 1):
                        for w in range(0, sat_noisy.shape[-1] - crop_W + 1, 1):
                            orin_sat_feat_t = sat_noisy[b,:,h:h+crop_H,w:w+crop_W]
                            gt_grid_t = gt_grid[b]
                            # orin_sat_feat_t_norm = F.normalize(orin_sat_feat_t.unsqueeze(0).reshape(1,-1))
                            sat_noisy_patch_batch.append([orin_sat_feat_t.unsqueeze(0), gt_grid_t[h,w][None]])
                    random.shuffle(sat_noisy_patch_batch)
                    sat_noisy_patch.append(torch.cat([item[0] for item in sat_noisy_patch_batch], dim=0).unsqueeze(0))
                    gt_label.append(torch.cat([item[1] for item in sat_noisy_patch_batch], dim=0).unsqueeze(0))
                sat_noisy_patch = torch.cat(sat_noisy_patch, dim=0)
                gt_label = torch.cat(gt_label, dim=0)

                optimizer.zero_grad()
                g2s_feat_middle = grd_solver(g2s_feat).reshape(B, -1)
                sat_noisy_patch_middle = sat_solver(sat_noisy_patch.reshape(B*49, 4, crop_H, crop_W)).reshape(B, 49, 4*crop_W*crop_H)

                g2s_feat_middle_norm = F.normalize(g2s_feat_middle, dim=-1)
                sat_noisy_patch_middle_norm = F.normalize(sat_noisy_patch_middle, dim=-1)

                matching_score_stacked = torch.bmm(sat_noisy_patch_middle_norm, g2s_feat_middle_norm.unsqueeze(-1)).squeeze(-1)
                
                loss = infoNCELoss(matching_score_stacked, gt_label)

                loss.backward()
                optimizer.step()

                if global_step % 10 == 0:    # print every 200 mini-batches
                    # print((matching_score_stacked>0).sum(), "/", matching_score_stacked.size(0)*matching_score_stacked.size(1))
                    print(f'epoch: {epoch} global_step: {global_step} loss: {loss.item()}')

                global_step += 1
            
            if 'KITTI' in config.data.params.test.target:
                with torch.no_grad():
                    # gt_inputs = batch["sat_map_gt"].cuda()
                    inputs = batch["sat_map"].cuda()
                    left_camera_k = batch['left_camera_k'].cuda()
                    outputs = batch["grd_left_imgs"].cuda()
                    gt_shift_x = batch['gt_shift_x'].cuda()
                    gt_shift_y = batch['gt_shift_y'].cuda()

                    toPIL = transforms.ToPILImage()

                    outputs = outputs*2 - 1
                    inputs = inputs*2 - 1
                    grd_laten = model.pre_AE_model.encode(outputs).sample()*model.scale_factor
                    # sat_laten = model.pre_AE_model.encode(inputs).sample()*model.scale_factor
                    
                    grd_laten = grd_laten.detach()
                    # sat_con = sat_laten.detach() #len 5

                    t = torch.randint(0, 100, (grd_laten.shape[0],), device=grd_laten.device).long()
                    grd_noisy = model.DDPM.q_sample(grd_laten, t)
                    # sat_noisy = model.DDPM.q_sample(sat_con, t)

                    B = grd_noisy.size()[0]
                    orin_A_2 = inputs.size()[-1] /2
                    A = 32
                    crop_H = int(26)
                    crop_W = int(26)

                    shift_range = 20
                    rote_range = 10
                    meter_per_pixel = utils.get_meter_per_pixel()/inputs.size()[-1] *utils.SatMap_end_sidelength
                    d = []
                    for i in range(B):
                        gt_h = torch.linspace(-(A - crop_H)*4, (A - crop_H)*4, A - crop_H + 1).to(grd_noisy.device)
                        gt_w = torch.linspace(-(A - crop_W)*4, (A - crop_W)*4, A - crop_W + 1).to(grd_noisy.device)
                        gt_h = gt_h - gt_shift_y[i]
                        gt_w = gt_w - gt_shift_x[i]
                        gt_h, gt_w = torch.meshgrid(gt_h, gt_w, indexing='ij')

                        d_batch = torch.sqrt(gt_h**2 + gt_w**2)
                        d.append(d_batch.unsqueeze(0))
                    d = torch.cat(d, dim=0)
                    sigma, mu = 4, 0.0
                    gt_grid = torch.exp(-((d - mu) ** 2) / (2.0 * sigma ** 2)).to(grd_noisy.device)

                    BEV_map,_,_,mask = project_grd_to_map(grd_noisy, None, left_camera_k, 32, outputs.size(-2), outputs.size(-1))
                    g2s_feat = TF.center_crop(BEV_map, [crop_H, crop_W])
                    mask = mask.permute(0,3,1,2)
                    mask = TF.center_crop(mask, [crop_H, crop_W])

                sat_laten = sat_solver.encode(inputs).sample()*model.scale_factor
                sat_noisy = model.DDPM.q_sample(sat_laten, t)
                sat_noisy_patch = []
                gt_label = []
                for b in range(B):
                    sat_noisy_patch_batch = []
                    for h in range(0, sat_noisy.shape[-2] - crop_H + 1, 1):
                        for w in range(0, sat_noisy.shape[-1] - crop_W + 1, 1):
                            orin_sat_feat_t = sat_noisy[b,:,h:h+crop_H,w:w+crop_W]*mask[b]
                            gt_grid_t = gt_grid[b]
                            sat_noisy_patch_batch.append([orin_sat_feat_t.unsqueeze(0), gt_grid_t[h,w][None]])
                    random.shuffle(sat_noisy_patch_batch)
                    sat_noisy_patch.append(torch.cat([item[0] for item in sat_noisy_patch_batch], dim=0).unsqueeze(0))
                    gt_label.append(torch.cat([item[1] for item in sat_noisy_patch_batch], dim=0).unsqueeze(0))
                sat_noisy_patch = torch.cat(sat_noisy_patch, dim=0)
                gt_label = torch.cat(gt_label, dim=0)
                optimizer.zero_grad()
                g2s_feat_middle = grd_solver(g2s_feat).reshape(B, -1)
                sat_noisy_patch_middle = sat_solver(sat_noisy_patch.reshape(B*49, 4, crop_H, crop_W)).reshape(B, 49, 4 * crop_H * crop_W)

                g2s_feat_middle_norm = F.normalize(g2s_feat_middle, dim=-1)
                sat_noisy_patch_middle_norm = F.normalize(sat_noisy_patch_middle, dim=-1)

                matching_score_stacked = torch.bmm(sat_noisy_patch_middle_norm, g2s_feat_middle_norm.unsqueeze(-1)).squeeze(-1)
                loss = infoNCELoss(matching_score_stacked, gt_label)

                loss.backward()
                optimizer.step()

                if global_step % 10 == 0:    # print every 200 mini-batches
                    print(f'epoch: {epoch} global_step: {global_step} loss: {loss.item()}')

                global_step += 1
            if 'VIGOR' in config.data.params.train.target:
                with torch.no_grad():
                    # gt_inputs = batch["sat_map_gt"].cuda()
                    inputs = batch["sat"].cuda()
                    outputs = batch["pano"].cuda()
                    gt_shift_x = batch['shift_x']
                    gt_shift_y = batch['shift_y']
                    meter_per_pixel = batch['meter_per_pixel']
             
                    toPIL = transforms.ToPILImage()

                    outputs = outputs*2 - 1
                    inputs = inputs*2 - 1
                    grd_laten = model.pre_AE_model.encode(outputs).sample()*model.scale_factor
                    
                    grd_laten = grd_laten.detach()

                    t = torch.randint(0, 100, (grd_laten.shape[0],), device=grd_laten.device).long()
                    grd_noisy = model.DDPM.q_sample(grd_laten, t)

                    B = grd_noisy.size()[0]
                    A = 32
                    crop_H = int(26)
                    crop_W = int(26)
                    d = []
                    for i in range(B):
                        gt_h = torch.linspace(-(A - crop_H)*4, (A - crop_H)*4, A - crop_H + 1).to(grd_noisy.device)
                        gt_w = torch.linspace(-(A - crop_W)*4, (A - crop_W)*4, A - crop_W + 1).to(grd_noisy.device)
                        gt_h = gt_h - gt_shift_y[i]
                        gt_w = gt_w - gt_shift_x[i]
                        gt_h, gt_w = torch.meshgrid(gt_h, gt_w, indexing='ij')

                        d_batch = torch.sqrt(gt_h**2 + gt_w**2)
                        d.append(d_batch.unsqueeze(0))
                    d = torch.cat(d, dim=0)
                    sigma, mu = 4, 0.0
                    gt_grid = torch.exp(-((d - mu) ** 2) / (2.0 * sigma ** 2)).to(grd_noisy.device)
                    
                    meter_per_pixel = meter_per_pixel / A *256
                    optical = VIGOR_grd2sat_uv(0, grd_noisy.size(2), grd_noisy.size(3), 2, A, A, meter_per_pixel).to(grd_noisy.device)

                    # optical = grd2sat_uv(level, H, W, 2, feat_H, feat_H, meter_per_pixel)
                    BEV_map, _ = grid_sample(grd_noisy, optical)
                    g2s_feat = TF.center_crop(BEV_map, [crop_H, crop_W])
                
                sat_laten = sat_solver.encode(inputs).sample()*model.scale_factor
                sat_noisy = model.DDPM.q_sample(sat_laten, t)
                sat_noisy_patch = []
                gt_label = []
                for b in range(B):
                    sat_noisy_patch_batch = []
                    for h in range(0, sat_noisy.shape[-2] - crop_H + 1, 1):
                        for w in range(0, sat_noisy.shape[-1] - crop_W + 1, 1):
                            orin_sat_feat_t = sat_noisy[b,:,h:h+crop_H,w:w+crop_W]
                            gt_grid_t = gt_grid[b]
                            # orin_sat_feat_t_norm = F.normalize(orin_sat_feat_t.unsqueeze(0).reshape(1,-1))
                            sat_noisy_patch_batch.append([orin_sat_feat_t.unsqueeze(0), gt_grid_t[h,w][None]])
                    random.shuffle(sat_noisy_patch_batch)
                    sat_noisy_patch.append(torch.cat([item[0] for item in sat_noisy_patch_batch], dim=0).unsqueeze(0))
                    gt_label.append(torch.cat([item[1] for item in sat_noisy_patch_batch], dim=0).unsqueeze(0))
                sat_noisy_patch = torch.cat(sat_noisy_patch, dim=0)
                gt_label = torch.cat(gt_label, dim=0)

                optimizer.zero_grad()
                g2s_feat_middle = grd_solver(g2s_feat).reshape(B, -1)
                sat_noisy_patch_middle = sat_solver(sat_noisy_patch.reshape(B*49, 4, crop_H, crop_W)).reshape(B, 49, 4*crop_W*crop_H)

                g2s_feat_middle_norm = F.normalize(g2s_feat_middle, dim=-1)
                sat_noisy_patch_middle_norm = F.normalize(sat_noisy_patch_middle, dim=-1)

                matching_score_stacked = torch.bmm(sat_noisy_patch_middle_norm, g2s_feat_middle_norm.unsqueeze(-1)).squeeze(-1)
                
                loss = infoNCELoss(matching_score_stacked, gt_label)

                loss.backward()
                optimizer.step()


                if global_step % 10 == 0:   
                    print(f'epoch: {epoch} global_step: {global_step} loss: {loss.item()}')

                global_step += 1
        if epoch % 5 == 0:
            os.makedirs(f"{config.data.params.train.target}", exist_ok=True)
            torch.save(grd_solver.state_dict(), f"{config.data.params.train.target}/grd_solver_epoch{epoch}.pth")
            torch.save(sat_solver.state_dict(), f"{config.data.params.train.target}/sat_solver_epoch{epoch}.pth")
        
    print('Finished Training')