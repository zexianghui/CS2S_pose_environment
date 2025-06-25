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

import torchvision.transforms.functional as TF
from PIL import Image

class SetupCallback(Callback):
    """
    save ckpt
    save test result
    """
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config, testckpt_path):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.testckpt_path = testckpt_path

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)
    
    def on_train_epoch_end(self, trainer, pl_module):
        # pl_module.sch_G.step()
        # pl_module.sch_D.step()
        if (trainer.current_epoch) % 10 == 0:
            ckpt_path = os.path.join(self.ckptdir, "epoch_" + str(trainer.current_epoch) + ".ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        # Create logdirs and save configs
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)
        
        print("Project config")
        OmegaConf.save(self.config, os.path.join(self.cfgdir, "project.yaml"))
        OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                        os.path.join(self.cfgdir, "lightning.yaml"))

    
    def on_test_start(self, trainer, pl_module):
        # Create logdirs and save configs
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)

        print("Project config")
        OmegaConf.save(self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))
        OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                        os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))


    def on_test_epoch_end(self, trainer, pl_module):
        # test_epoch = int(self.testckpt_path.split("/")[-1].split("_")[1].split(".")[0])
        test_epoch = int(300)

        RMSE, SSIM, PSNR, SD, P_alex, P_squeeze = pl_module.eval_data()
        os.makedirs(os.path.join(self.logdir, "test"), exist_ok=True)
        with open(os.path.join(self.logdir, "test", "test.txt"), "a") as f:
            f.write("===================\n")
            f.write("epoch: " + str(test_epoch) + "\n")
            f.write("RMSE: {:.4f}, SSIM: {:.4f}, PSNR: {:.4f}, SD: {:.4f}, P_alex: {:.4f}, P_squeeze: {:.4f}\n".format(RMSE.item(), SSIM.item(), PSNR.item(), SD.item(), P_alex.item(), P_squeeze.item()))
            # f.write("res_RMSE: {:.4f}, res_SSIM: {:.4f}, res_PSNR: {:.4f}, res_SD: {:.4f}, res_P_alex: {:.4f}, res_P_squeeze: {:.4f}\n".format(res_RMSE.item(), res_SSIM.item(), res_PSNR.item(), res_SD.item(), res_P_alex.item(), res_P_squeeze.item()))
            # f.write("gt_res_RMSE: {:.4f}, gt_res_SSIM: {:.4f}, gt_res_PSNR: {:.4f}, gt_res_SD: {:.4f}, gt_res_P_alex: {:.4f}, gt_res_P_squeeze: {:.4f}\n".format(gt_res_RMSE.item(), gt_res_SSIM.item(), gt_res_PSNR.item(), gt_res_SD.item(), gt_res_P_alex.item(), gt_res_P_squeeze.item()))


class ImageLogger(Callback):
    """
    save generate images
    """
    def __init__(self, batch_frequency, max_images, clamp=True, logdir='./',
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images

        self.log_steps = [int(self.batch_freq)*n for n in range(50)]
        # self.log_steps = [2312411]
 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

        self.logdir = logdir


    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = self.logdir
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
        
        # for k in images:
        #     images[k] = (images[k] + 1.0) / 2.0
        #     images[k] = torch.clamp(images[k], 0., 1.)

        B = images['inputs'].size()[0]
        full_image = torch.zeros(B, 3, 256, 768)
        full_image[:, :, :256, :256] = images['inputs'].detach().cpu()
        full_image[:, :, :128, 256:768] = images['outputs'].detach().cpu() 
        full_image[:, :, 128:, 256:768] = images['pre_residual'].detach().cpu()

        # full_image = torch.zeros(B, 3, 256, 2560)
        # full_image[:, :, :128, :512] = images['outputs'].detach().cpu()
        # full_image[:, :, 128:, :512] = images['reconstructions'].detach().cpu()
        # full_image[:, :, :128, 512:1024] = images['pre_residual'].detach().cpu()
        # full_image[:, :, 128:, 512:1024] = images['end_reconstructions'].detach().cpu()
        # full_image[:, :, :128, 1024:1536] = (torch.cat((images['outputs'][3:],images['outputs'][:3]))).detach().cpu()
        # full_image[:, :, 128:, 1024:1536] = (torch.cat((images['reconstructions'][3:],images['reconstructions'][:3]))).detach().cpu()
        # full_image[:, :, :128, 1536:2048] = (torch.cat((images['pre_residual'][3:],images['pre_residual'][:3]))).detach().cpu()
        # full_image[:, :, 128:, 1536:2048] = (torch.cat((images['end_reconstructions'][3:],images['end_reconstructions'][:3]))).detach().cpu()
        # full_image[:, :, :128, 2048:] = full_image[:, :, :128, :512] + full_image[:, :, :128, 1536:2048]
        # full_image[:, :, 128:, 2048:] = full_image[:, :, :128, 1024:1536] + full_image[:, :, :128, 512:1024]

        # if "end_reconstructions" in images:
        #     full_image = torch.zeros(B, 3, 256, 1792)
        #     full_image[:, :, :256, :256] = images['inputs'].detach().cpu()
        #     full_image[:, :, :128, 256:768] = images['outputs'].detach().cpu() 
        #     full_image[:, :, 128:, 256:768] = images['reconstructions'].detach().cpu()
        #     # full_image[:, :, :128, 768:1280] = images['style_img'].detach().cpu()
        #     full_image[:, :, 128:, 768:1280] = images['pre_residual'].detach().cpu()
        #     full_image[:, :, :128, 1280:] = images['end_reconstructions'].detach().cpu()
        #     full_image[:, :, 128:, 1280:] = images['sample'].detach().cpu()
        #     # full_image[:, :, 128:, 1280:] = images['sample'].detach().cpu()

        # else:
        #     # if pl_module.pre_train:
        #     if 0:
        #         full_image = torch.zeros(B, 3, 384, 512)
        #         full_image[:, :, :128, :] = (images['inputs'].detach().cpu())
        #         full_image[:, :, 128:256, :] = (images['reconstructions'].detach().cpu())
        #         full_image[:, :, 256:, :] = (images['samples'].detach().cpu())
        #     else:
        #         full_image = torch.zeros(B, 3, 256, 1280)
        #         full_image[:, :, :256, :256] = images['inputs'].detach().cpu()
        #         full_image[:, :, :128, 256:768] = images['outputs'].detach().cpu()
        #         full_image[:, :, 128:, 256:768] = images['reconstructions'].detach().cpu()
        #         full_image[:, :, :128, 768:1280] = images['samples'].detach().cpu()
        #         # full_image[:, :, :128, 768:1280] = images['pre_residual'].detach().cpu()
        #         # full_image[:, :, 128:, 768:1280] = images['end_reconstructions'].detach().cpu()

        os.makedirs(self.logdir, exist_ok=True)
        for i in range(min(B,16)):
            full_image_pil = TF.to_pil_image(full_image[i])
            filename = "full_{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                i,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx)
            path = os.path.join(self.logdir, filename)
            full_image_pil.save(path)


        # for k in images:
        #     N = min(images[k].shape[0], self.max_images)
        #     images[k] = images[k][:N]
        #     if isinstance(images[k], torch.Tensor):
        #         images[k] = images[k].detach().cpu()
        #         if self.clamp:
        #             images[k] = torch.clamp(images[k], -1., 1.)

        # self.log_local(pl_module.logger.save_dir, split, images,
        #                 pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.global_step % self.batch_freq == 0:
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (batch_idx) % 50 == 0:
            self.log_img(pl_module, batch, batch_idx, split="test")