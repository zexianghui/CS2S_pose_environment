import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np

import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
import time

from pytorch_lightning import seed_everything
seed_everything(25)
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from models.geometry.kitti_sat2grd import get_xyz_grds, project_map_to_grd
from models.geometry.kitti_grd2sat import project_grd_to_map

from utils.util import instantiate_from_config
from collections import OrderedDict
from models.CVUSA_geo_ldm_diffusion.ddim_CVUSA import CVUSA_DDIMSampler
from models.KITTI_geo_ldm_diffusion.ddim_KITTI import KITTI_DDIMSampler
from models.VIGOR_geo_ldm_diffusion.ddim_VIGOR import VIGOR_DDIMSampler


import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms


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
    parser.add_argument("--seed", type=int, default=24, help="seed for seed_everything")
    parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=True, help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument("--logdir", type=str, const=True, default="result", nargs="?", help="logdir")

    parser.add_argument("--train", type=str2bool, const=True, default=False, nargs="?", help="train")
    parser.add_argument("--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir")
    
    parser.add_argument("--test", type=str, default="", nargs="?")
    parser.add_argument("--grd_solve_pth", type=str, default=None, nargs="?")
    parser.add_argument("--sat_solve_pth", type=str, default=None, nargs="?")

    parser.add_argument("--function", type=int, default=1, nargs="?", help="1: orin, 2: w/ Pose Correction, 3: w/ Env Control")


    return parser

if __name__ == "__main__":
    #get parser
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    data = instantiate_from_config(config.data)
    data.setup()
    test_dataset = data._test_dataloader()
    model = instantiate_from_config(config.model).cuda()
    state_dict = torch.load(opt.test)['state_dict']
    model.load_state_dict(state_dict, strict = True)
    model = model.eval()

    step = 0
    if 'KITTI' in config.data.params.test.target:
        opt.grd_solve_pth = 'result/localization_corr/KITTI/grd_solver.pth'
        opt.sat_solve_pth = 'result/localization_corr/KITTI/sat_solver.pth'
        sampler = KITTI_DDIMSampler(model.DDPM, model.pre_AE_model, model.scale_factor, grd_solve_pth=opt.grd_solve_pth, sat_solve_pth=opt.sat_solve_pth)
    elif 'CVUSA' in config.data.params.test.target:
        opt.grd_solve_pth = 'result/localization_corr/CVUSA/grd_solver.pth'
        opt.sat_solve_pth = 'result/localization_corr/CVUSA/sat_solver.pth'
        sampler = CVUSA_DDIMSampler(model.DDPM, model.pre_AE_model, model.scale_factor, grd_solve_pth=opt.grd_solve_pth, sat_solve_pth=opt.sat_solve_pth)
    elif 'VIGOR' in config.data.params.test.target:
        opt.grd_solve_pth = 'result/localization_corr/VIGOR/grd_solver.pth'
        opt.sat_solve_pth = 'result/localization_corr/VIGOR/sat_solver.pth'
        sampler = VIGOR_DDIMSampler(model.DDPM, model.pre_AE_model, model.scale_factor, grd_solve_pth=opt.grd_solve_pth, sat_solve_pth=opt.sat_solve_pth)
    with torch.no_grad():
        for batch in test_dataset:
            if 'CVUSA' in config.data.params.test.target:
                inputs = batch["sat"].cuda()
                outputs = batch["pano"].cuda()
                grd_path = batch['paths']

                sat_con = model.condition_model_sat( inputs*2 - 1)[:,1:,:].detach()

                n_samples = outputs.size()[0] #gen num
                shape = [1, 4, 16, 64] #laten size
                x_T = torch.randn(shape, device=inputs.device).repeat(sat_con.size()[0], 1, 1, 1)
                start_code = x_T #init noise

                for test_i in range(opt.function):
                    shape = [4, 16, 64] #laten size
                    ddim_steps = 50
                    scale = 7.5
                    ddim_eta = 1
                    temperature = 1
                    txt = None
                    inputs_feat = None
                    if test_i == 0:
                        txt = None
                        inputs_feat = None
                    if test_i == 1:
                        txt = None
                        inputs_feat = sampler.sat_solver.encode(inputs*2 - 1).mode().detach()
                        inputs_feat = inputs_feat * model.scale_factor
                    if test_i == 2:
                        txt = "in autumn"
                        inputs_feat = sampler.sat_solver.encode(inputs*2 - 1).mode().detach()
                        inputs_feat = inputs_feat * model.scale_factor
                   
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=sat_con,
                                                    txt = txt,
                                                    orin_sat_feat = inputs_feat,
                                                    inter_ref = None,
                                                    batch_size=n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=None,
                                                    eta=ddim_eta,
                                                    x_T=start_code,
                                                    temperature = temperature)
                    
                    samples_ddim = samples_ddim * (1 / model.scale_factor)
                    pre_residual = model.pre_AE_model.decode(samples_ddim.detach())

                    pre_residual = torch.clamp((pre_residual + 1.0) / 2.0, min=0.0, max=1.0)
                
                    toPIL = transforms.ToPILImage()
                    if test_i == 0:
                        for batch_num in range(pre_residual.size(0)):
                            name = grd_path[batch_num].split('/')[-1].split('.')[0]
                            os.makedirs(f"vis_CVUSA/gt_img", exist_ok=True)
                            val_save = toPIL(outputs[batch_num])
                            val_save.save(f"vis_CVUSA/gt_img/{name}.png")

                            os.makedirs(f"vis_CVUSA/gen_img", exist_ok=True)
                            val_save = toPIL(pre_residual[batch_num])
                            val_save.save(f"vis_CVUSA/gen_img/{name}.png")
                    if test_i == 1:
                        for batch_num in range(pre_residual.size(0)):
                            name = grd_path[batch_num].split('/')[-1].split('.')[0]
                            os.makedirs(f"vis_CVUSA/gen_img_correct", exist_ok=True)
                            val_save = toPIL(pre_residual[batch_num])
                            val_save.save(f"vis_CVUSA/gen_img_correct/{name}.png")
                    if test_i > 1:
                        for batch_num in range(pre_residual.size(0)):
                            name = grd_path[batch_num].split('/')[-1].split('.')[0]
                            os.makedirs(f"vis_CVUSA/{txt}", exist_ok=True)
                            val_save = toPIL(pre_residual[batch_num])
                            val_save.save(f"vis_CVUSA/{txt}/{name}.png")
        
            if 'KITTI' in config.data.params.test.target:
                inputs = batch["sat_map"].cuda()
                left_camera_k = batch['left_camera_k'].cuda()
                outputs = batch["grd_left_imgs"].cuda()
                gt_shift_x = batch['gt_shift_x']
                gt_shift_y = batch['gt_shift_y']
                theta = batch['theta']
                file_name = batch['file_name']
                toPIL = transforms.ToPILImage()
    
                outputs = outputs*2 - 1

                cond_label = model.condition_model_sat(inputs*2 -1)
                cond_label = cond_label[:,1:,:]
                sat_con = cond_label.detach()

                n_samples = outputs.size()[0] #gen num
                outputs = (outputs +1)/2
                shape = [1, 4, 16, 64] #laten size
                x_T = torch.randn(shape, device=inputs.device).repeat(sat_con.size()[0], 1, 1, 1)
                start_code = x_T #init noise

                for test_i in range(opt.function):
                    shape = [4, 16, 64] #laten size
                    ddim_steps = 50
                    scale = 7.5
                    ddim_eta = 1
                    temperature = 1
                    if test_i == 0:
                        txt = None
                        inputs_feat = None
                    if test_i == 1:
                        txt = None
                        inputs_feat = sampler.sat_solver.encode(inputs*2 - 1).mode().detach()
                        inputs_feat = inputs_feat * model.scale_factor
                    if test_i == 2:
                        txt = "in automn"
                        inputs_feat = sampler.sat_solver.encode(inputs*2 - 1).mode().detach()
                        inputs_feat = inputs_feat * model.scale_factor
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=sat_con,
                                                    txt = txt,
                                                    orin_sat_feat = inputs_feat,
                                                    batch_size=n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=None,
                                                    eta=ddim_eta,
                                                    x_T=start_code,
                                                    temperature = temperature,left_camera_k = left_camera_k, gt_shift_x = gt_shift_x, gt_shift_y = gt_shift_y, theta = theta
                                                    )
                    
                    samples_ddim = samples_ddim * (1 / model.scale_factor)
                    pre_residual = model.pre_AE_model.decode(samples_ddim.detach())
                    pre_residual = torch.clamp((pre_residual + 1.0) / 2.0, min=0.0, max=1.0)
                   
                    toPIL = transforms.ToPILImage()
                    if test_i == 0:
                        for batch_num in range(pre_residual.size(0)):
                            name = os.path.dirname(file_name[batch_num])
                            os.makedirs(f"vis_KITTI/gt_img/{name}", exist_ok=True)
                            val_save = toPIL(outputs[batch_num])
                            val_save.save(f"vis_KITTI/gt_img/{file_name[batch_num]}.png")

                            name = os.path.dirname(file_name[batch_num])
                            os.makedirs(f"vis_KITTI/gen_img/{name}", exist_ok=True)
                            val_save = toPIL(pre_residual[batch_num])
                            val_save.save(f"vis_KITTI/gen_img/{file_name[batch_num]}.png")
                    if test_i == 1:
                        for batch_num in range(pre_residual.size(0)):
                            name = os.path.dirname(file_name[batch_num])
                            os.makedirs(f"vis_KITTI/gen_img_correct/{name}", exist_ok=True)
                            val_save = toPIL(pre_residual[batch_num])
                            val_save.save(f"vis_KITTI/gen_img_correct/{file_name[batch_num]}.png")
                    if test_i > 1:
                        for batch_num in range(pre_residual.size(0)):
                            name = os.path.dirname(file_name[batch_num])
                            os.makedirs(f"vis_KITTI/{txt}/{name}", exist_ok=True)
                            val_save = toPIL(pre_residual[batch_num])
                            val_save.save(f"vis_KITTI/{txt}/{file_name[batch_num]}.png")
            if 'VIGOR' in config.data.params.test.target:
                inputs = batch["sat"].cuda()
                outputs = batch["pano"].cuda()
                meter_per_pixel = batch['meter_per_pixel']
                file_name = batch['paths']
                toPIL = transforms.ToPILImage()
    
                outputs = outputs*2 - 1

                cond_label = model.condition_model_sat(inputs*2 -1)
                cond_label = cond_label[:,1:,:]
                sat_con = cond_label.detach()

                n_samples = outputs.size()[0] 
                outputs = (outputs +1)/2
                shape = [1, 4, 16, 64] 
                x_T = torch.randn(shape, device=inputs.device).repeat(sat_con.size()[0], 1, 1, 1)
                start_code = x_T 
                for test_i in range(opt.function):
                    shape = [4, 16, 64] #laten size
                    ddim_steps = 50
                    scale = 7.5
                    ddim_eta = 1
                    temperature = 1
                    if test_i == 0:
                        txt = None
                        inputs_feat = None
                    if test_i == 1:
                        txt = None
                        inputs_feat = sampler.sat_solver.encode(inputs*2 - 1).mode().detach()
                        inputs_feat = inputs_feat * model.scale_factor
                    if test_i == 2:
                        txt = "in automn"
                        inputs_feat = sampler.sat_solver.encode(inputs*2 - 1).mode().detach()
                        inputs_feat = inputs_feat * model.scale_factor
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=sat_con,
                                                    txt = txt,
                                                    orin_sat_feat = inputs_feat,
                                                    batch_size=n_samples,
                                                    shape=shape,    
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=None,
                                                    eta=ddim_eta,
                                                    x_T=start_code,
                                                    temperature = temperature,
                                                    meter_per_pixel = meter_per_pixel
                                                    )
                    samples_ddim = samples_ddim * (1 / model.scale_factor)
                    pre_residual = model.pre_AE_model.decode(samples_ddim.detach())
                    

                    pre_residual = torch.clamp((pre_residual + 1.0) / 2.0, min=0.0, max=1.0)
                    toPIL = transforms.ToPILImage()
                    if test_i == 0:
                        for batch_num in range(pre_residual.size(0)):
                            name = os.path.dirname(file_name[batch_num])
                            os.makedirs(f"vis_VIGOR/gt_img/{name}", exist_ok=True)
                            val_save = toPIL(outputs[batch_num])
                            val_save.save(f"vis_VIGOR/gt_img/{file_name[batch_num]}.png")

                            name = os.path.dirname(file_name[batch_num])
                            os.makedirs(f"vis_VIGOR/gen_img/{name}", exist_ok=True)
                            val_save = toPIL(pre_residual[batch_num])
                            val_save.save(f"vis_VIGOR/gen_img/{file_name[batch_num]}.png")
                    if test_i == 1:
                        for batch_num in range(pre_residual.size(0)):
                            name = os.path.dirname(file_name[batch_num])
                            os.makedirs(f"vis_VIGOR/gen_img_correct/{name}", exist_ok=True)
                            val_save = toPIL(pre_residual[batch_num])
                            val_save.save(f"vis_VIGOR/gen_img_correct/{file_name[batch_num]}.png")
                    if test_i > 1:
                        for batch_num in range(pre_residual.size(0)):
                            name = os.path.dirname(file_name[batch_num])
                            os.makedirs(f"vis_VIGOR/{txt}/{name}", exist_ok=True)
                            val_save = toPIL(pre_residual[batch_num])
                            val_save.save(f"vis_VIGOR/{txt}/{file_name[batch_num]}.png")
