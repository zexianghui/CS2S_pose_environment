import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import numpy as np

from tqdm import tqdm
from models.eval.evaluate import Evaluate_indic
from utils.util import instantiate_from_config
from torchvision import transforms
from collections import OrderedDict

# from models.CVUSA_geo_ldm_diffusion.openaimodel import AttentionBlock
# from ldm.modules.CVUSA_attention import CrossAttention

from models.VIGOR_geo_ldm_diffusion.ddim_VIGOR import VIGOR_DDIMSampler
import random
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def save_img(img, save_path):
    img = transforms.functional.to_pil_image(img, mode='RGB')
    img.save(save_path)

class Boost_Sat2Den_ddpm(pl.LightningModule):
    def __init__(self,
                # Sat2Den_config,
                # Sat2Den_ckpt_path,
                AE_config,
                AE_ckpt_path,
                DDPM_config,
                Condition_config_grd,
                Condition_config_sat,
                scale_factor,
                pre_sat2grd_model_path,
                Condition_config_txt,
                # control_txt,
                pre_ldm_model_path,
                # lossconfig
                 ):
        super().__init__()

        self.pre_AE_model = self.init_AE(AE_config, AE_ckpt_path)
        self.DDPM = instantiate_from_config(DDPM_config)
        # self.condition_model_txt = instantiate_from_config(Condition_config_txt)
        # self.control_txt = instantiate_from_config(control_txt)
        if pre_ldm_model_path is not None:
            pre_ldm_model = torch.load(pre_ldm_model_path)
            self.load_pre_ldm_model(pre_ldm_model['state_dict'])


        # self.Sat2Den = self.init_Sat2Den(Sat2Den_config, Sat2Den_ckpt_path)
        # self.condition_model_grd = instantiate_from_config(Condition_config_grd)
        self.condition_model_sat = instantiate_from_config(Condition_config_sat)
        self.scale_factor = scale_factor
        if pre_sat2grd_model_path is not None:
            pre_sat2grd_model = torch.load(pre_sat2grd_model_path)
            self.load_pre_sat2grd_model(pre_sat2grd_model['state_dict'])

        # self.attention_blocks = self.extract_attention_blocks()

        self.evaluate = Evaluate_indic()
        self.RMSE = []
        self.SSIM = []
        self.PSNR = []
        self.SD = []
        self.P_alex = []
        self.P_squeeze = []

        self.res_RMSE = []
        self.res_SSIM = []
        self.res_PSNR = []
        self.res_SD = []
        self.res_P_alex = []
        self.res_P_squeeze = []

        self.gt_res_RMSE = []
        self.gt_res_SSIM = []
        self.gt_res_PSNR = []
        self.gt_res_SD = []
        self.gt_res_P_alex = []
        self.gt_res_P_squeeze = []

    def load_pre_sat2grd_model(self, pre_sat2grd_model):
        # self.load_pth_rematch(pre_sat2grd_model, self.Sat2Den, 'Sat2Den.', None)
        self.load_pth_rematch(pre_sat2grd_model, self.pre_AE_model, 'pre_AE_model.', None)
        self.load_pth_rematch(pre_sat2grd_model, self.DDPM.denoise_model, 'DDPM.denoise_model.', None)
        self.load_pth_rematch(pre_sat2grd_model, self.condition_model_grd, 'condition_model_grd.', None)
        self.load_pth_rematch(pre_sat2grd_model, self.condition_model_sat, 'condition_model_sat.', None)

    # def extract_attention_blocks(self):
    #     """
    #     Extracts and returns a dictionary of all AttentionBlock and CrossAttention modules in the model.
        
    #     Returns:
    #         `dict` of modules: A dictionary containing all AttentionBlock and CrossAttention modules in the model,
    #         indexed by their names.
    #     """
    #     attention_blocks = {}

    #     def fn_recursive_extract(name, module, attention_blocks):
    #         if isinstance(module, (AttentionBlock, CrossAttention)):
    #             attention_blocks[name] = module.parameters()

    #         for sub_name, child in module.named_children():
    #             fn_recursive_extract(f"{name}.{sub_name}", child, attention_blocks)

    #         return attention_blocks

    #     for name, module in self.named_children():
    #         fn_recursive_extract(name, module, attention_blocks)

    #     return attention_blocks


    def load_pre_ldm_model(self, pre_sat2grd_model):
        AE_state_dict = OrderedDict()
        for k, v in pre_sat2grd_model.items():
            if 'first_stage_model' in k:
                new_k = k.replace('first_stage_model.', '')
                AE_state_dict[new_k] = v
        self.pre_AE_model.load_state_dict(AE_state_dict)

        # cond_state_dict = OrderedDict()
        # for k, v in pre_sat2grd_model.items():
        #     if 'cond_stage_model' in k:
        #         new_k = k.replace('cond_stage_model.', '')
        #         cond_state_dict[new_k] = v
        # self.condition_model_txt.load_state_dict(cond_state_dict)

        DDPM_state_dict = OrderedDict()
        for k, v in pre_sat2grd_model.items():
            # if '.' not in k:
            #     DDPM_state_dict[k] = v
            if 'diffusion_model' in k:
                new_k = k.replace('model.diffusion_model.', '')
                DDPM_state_dict[new_k] = v
        self.DDPM.denoise_model.load_state_dict(DDPM_state_dict, strict=False)  
        
        
    def load_pth_rematch(self, state_dict, model, orin_key, aim_key):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if orin_key in k:
                new_k = k.replace(orin_key, '') 
                new_state_dict[new_k] = v
        if aim_key:
            eval("model." + aim_key).load_state_dict(new_state_dict)
        else:
            model.load_state_dict(new_state_dict)

    def init_AE(self, AE_config, AE_ckpt_path):
        model = instantiate_from_config(AE_config)
        model = model.eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False

        checkpoint = torch.load(AE_ckpt_path)['state_dict']
        model_state_dict = OrderedDict()
        for key, value in checkpoint.items():
            if 'first_stage_model' in key:
                new_k = key.replace('first_stage_model.', '') 
                model_state_dict[new_k] = checkpoint[key]
        # self.load_pth_rematch(checkpoint['state_dict'], model, 'pre_AE_model.', None)
        model.load_state_dict(model_state_dict)
        return model
        

    def init_Sat2Den(self, Sat2Den_config, AE_ckpt_path):
        model = instantiate_from_config(Sat2Den_config)
        model = model.eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        checkpoint = torch.load(AE_ckpt_path)
        self.load_pth_rematch(checkpoint['netG'], model, 'depth_model.', 'depth_model')
        self.load_pth_rematch(checkpoint['netG'], model, 'denoise_model.', 'denoise_model')
        self.load_pth_rematch(checkpoint['netG'], model, 'style_encode.', 'style_encode')
        self.load_pth_rematch(checkpoint['netG'], model, 'style_model.', 'style_model')
        return model

    def logsnr_schedule_cosine(self, t, *, logsnr_min=-20., logsnr_max=20.):
        b = np.arctan(np.exp(-.5 * logsnr_max))
        a = np.arctan(np.exp(-.5 * logsnr_min)) - b
        
        return -2. * torch.log(torch.tan(a * t + b))

    def q_sample(self, gt, logsnr, noise):
        
        # lambdas = logsnr_schedule_cosine(t)
        
        alpha = logsnr.sigmoid().sqrt().to(gt.device)
        sigma = (-logsnr).sigmoid().sqrt().to(gt.device)
        
        alpha = alpha[:,None, None, None]
        sigma = sigma[:,None, None, None]

        return alpha * gt + sigma * noise
    
    @torch.no_grad()
    def p_mean_variance(self, oriimg, noise, logsnr, logsnr_next, w=2.0):
        
        b = oriimg.shape[0]
        w = w[:, None, None, None]
        
        c = - torch.special.expm1(logsnr - logsnr_next)
        
        squared_alpha, squared_alpha_next = logsnr.sigmoid(), logsnr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-logsnr).sigmoid(), (-logsnr_next).sigmoid()
        
        alpha, sigma, alpha_next = map(lambda x: x.sqrt(), (squared_alpha, squared_sigma, squared_alpha_next))
    
        pred_noise = self.denoise_model(oriimg, noise, logsnr.repeat(b).to(oriimg.device))
        
        pred_noise_final = pred_noise
        
        noise = noise
        
        z_start = (noise - sigma * pred_noise_final) / alpha
        z_start.clamp_(-1., 1.)
        
        model_mean = alpha_next * (noise * (1 - c) / alpha + c * z_start)
        
        posterior_variance = squared_sigma_next * c
        
        return model_mean, posterior_variance
    
    @torch.no_grad()
    def p_sample(self, oriimg, noise, logsnr, logsnr_next, w):
        model_mean, model_variance = self.p_mean_variance( oriimg, noise, logsnr=logsnr, logsnr_next=logsnr_next, w = w)
        
        if logsnr_next==0:
            return model_mean
        
        return model_mean + model_variance.sqrt() * torch.randn_like(oriimg).to(oriimg.device)

 
    @torch.no_grad()
    def sample(self, oriimg, w, timesteps=256):
        res_img = torch.randn_like(oriimg).to(oriimg.device)

        logsnrs = self.logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[:-1])
        logsnr_nexts = self.logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[1:])

        res_imgs = []
        for logsnr, logsnr_next in zip(logsnrs, logsnr_nexts): # [1, ..., 0] = size is 257
            res_img = self.p_sample(oriimg, res_img, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
            res_imgs.append(res_img.detach().cpu())
        return res_imgs


    def forward(self, init_pre):
        pre_residual, posterior = self.pre_AE_model(init_pre)
        return pre_residual, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def gen_cond(self, inputs):
        cond_label = self.condition_model_sat(inputs)
        cond_label = cond_label[:,1:,:]
        return cond_label


    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, "sat")
        # style_img = self.get_input(batch, 'sky_histc')
        outputs = self.get_input(batch, "pano")
        label = batch['label']
        
        inputs = inputs*2 - 1
        outputs = outputs*2 - 1


        cond_label = self.condition_model_sat(inputs)
        cond_label = cond_label[:,1:,:]
        pre_residual_laten = self.pre_AE_model.encode(outputs).sample().detach()
        # cond_sat = self.condition_model_sat(inputs).detach()
        # cond_init_grd = self.condition_model_grd(init_result['pred']).detach()

        pre_residual_laten = pre_residual_laten * self.scale_factor
        loss = self.DDPM.t_losses(pre_residual_laten, cond_init_grd=None, cond_sat=None, cond_txt = cond_label)
        self.log("L1_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
        

    def test_step(self, batch, batch_idx):
        log = dict()
        inputs = self.get_input(batch, "sat")
        label = batch['label']
        outputs = self.get_input(batch, "pano")
        grd_path = batch['paths']

        inputs = inputs*2-1
        outputs = outputs*2 -1


        cond_label = self.condition_model_sat(inputs)
        cond_label = cond_label[:,1:,:]
        sat_con = cond_label.detach()

        sampler = VIGOR_DDIMSampler(self.DDPM, self.pre_AE_model, self.scale_factor)

        # cond_label = self.condition_model_txt(txt_prompt).detach()

        n_samples = outputs.size()[0] 
        shape = [1, 4, 16, 64] 
        x_T = torch.randn(shape, device=inputs.device).repeat(sat_con.size()[0], 1, 1, 1)
        start_code = x_T 
        shape = [4, 16, 64]
        ddim_steps = 50
        scale = 7.5
        ddim_eta = 1
        temperature = 1
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                        cond_sat=None, cond_grd=None,
                                        conditioning=sat_con,
                                        batch_size=n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=None,
                                        eta=ddim_eta,
                                        x_T=start_code,
                                        temperature = temperature)
        
        samples_ddim = samples_ddim * (1 / self.scale_factor)
        pre_residual = self.pre_AE_model.decode(samples_ddim)

        pre_residual = torch.clamp((pre_residual + 1.0) / 2.0, min=0.0, max=1.0)
        inputs = torch.clamp((inputs + 1.0) / 2.0, min=0.0, max=1.0)
        outputs = torch.clamp((outputs + 1.0) / 2.0, min=0.0, max=1.0)

        toPIL = transforms.ToPILImage()
        # for batch_num in range(pre_residual.size(0)):
        # #     # name = os.path.basename(grd_path[batch_num]).replace('.jpg', '.png')
        # #     name = grd_path[batch_num].replace('.jpg', '.png').replace('./dataset/VIGOR', './result/2024-09-28T06-46-10_VIGOR_geo_ldm_txt_control/test/ours_epoch210/gt')
        # #     os.makedirs(os.path.dirname(name), exist_ok=True)
        # #     val_save = toPIL(outputs[batch_num])
        # #     val_save.save(name)
        # #     name = grd_path[batch_num].replace('.jpg', '.png').replace('./dataset/VIGOR', './result/2024-09-28T06-46-10_VIGOR_geo_ldm_txt_control/test/ours_epoch210/sat')
        # #     os.makedirs(os.path.dirname(name), exist_ok=True)
        # #     val_save = toPIL(inputs[batch_num])
        # #     val_save.save(name)
        #     name = grd_path[batch_num].replace('.jpg', '.png').replace('./dataset/VIGOR', './result/2024-09-28T06-46-10_VIGOR_geo_ldm_txt_control/test/ours_epoch210/pred')
        #     os.makedirs(os.path.dirname(name), exist_ok=True)
        #     val_save = toPIL(pre_residual[batch_num])
        #     val_save.save(name)

        log_eval = self.evaluate((pre_residual).clamp(0, 1), outputs, split="test")
        self.RMSE.append(log_eval["RMSE"])
        self.SSIM.append(log_eval["SSIM"])
        self.PSNR.append(log_eval["PSNR"])
        self.SD.append(log_eval["SD"])
        self.P_alex.append(log_eval["P_alex"])
        self.P_squeeze.append(log_eval["P_squeeze"])
        self.log("RMSE", log_eval["RMSE"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("SSIM", log_eval["SSIM"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("P_alex", log_eval["P_alex"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return self.log_dict
    
    def eval_data(self):
        RMSE = torch.tensor(self.RMSE).mean()
        SSIM = torch.tensor(self.SSIM).mean()
        PSNR = torch.tensor(self.PSNR).mean()
        SD = torch.tensor(self.SD).mean()
        P_alex = torch.tensor(self.P_alex).mean()
        P_squeeze = torch.tensor(self.P_squeeze).mean()
        self.log("test/RMSE", RMSE)
        self.log("test/SSIM", SSIM)
        self.log("test/PSNR", PSNR)
        self.log("test/SD", SD)
        self.log("test/P_alex", P_alex)
        self.log("test/P_squeeze", P_squeeze)
        self.RMSE = []
        self.SSIM = []
        self.PSNR = []
        self.SD = []
        self.P_alex = []
        self.P_squeeze = []
        return RMSE, SSIM, PSNR, SD, P_alex, P_squeeze

    def configure_optimizers(self):
        lr = self.learning_rate
        opt= torch.optim.AdamW(list(self.DDPM.denoise_model.parameters()) +
                               list(self.condition_model_sat.parameters()),
                                  lr=lr)
        return [opt]

    def get_last_layer(self):
        return self.pre_AE_model.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        inputs = self.get_input(batch, "sat")
        label = batch['label']
        outputs = self.get_input(batch, "pano")


        inputs = inputs*2-1
        outputs = outputs*2 -1


        cond_label = self.condition_model_sat(inputs)
        cond_label = cond_label[:,1:,:]
        sat_con = cond_label.detach()

        sampler = VIGOR_DDIMSampler(self.DDPM, self.pre_AE_model, self.scale_factor)

        # cond_label = self.condition_model_txt(txt_prompt).detach()

        n_samples = outputs.size()[0] 
        shape = [1, 4, 16, 64] 
        x_T = torch.randn(shape, device=inputs.device).repeat(sat_con.size()[0], 1, 1, 1)
        start_code = x_T 
        shape = [4, 16, 64] 
        ddim_steps = 50
        scale = 7.5
        ddim_eta = 1
        temperature = 1
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                        cond_sat=None, cond_grd=None,
                                        conditioning=sat_con,
                                        batch_size=n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=None,
                                        eta=ddim_eta,
                                        x_T=start_code,
                                        temperature = temperature)
        
        samples_ddim = samples_ddim * (1 / self.scale_factor)
        # img_list = self.DDPM.sample(noise, cond_init_grd=cond_init_grd, cond_sat=cond_sat)
        pre_residual = self.pre_AE_model.decode(samples_ddim)
        pre_residual = torch.clamp((pre_residual + 1.0) / 2.0, min=0.0, max=1.0)
        # init_result['pred'] = torch.clamp((init_result['pred'] + 1.0) / 2.0, min=0.0, max=1.0)
        inputs = torch.clamp((inputs + 1.0) / 2.0, min=0.0, max=1.0)
        outputs = torch.clamp((outputs + 1.0) / 2.0, min=0.0, max=1.0)

        # optical = grd2sat_uv(0, 128, 512, 2, 256, 256, meter_per_pixel).to(inputs.device)
        # out_val, mask = grid_sample(inputs, optical)
        # log["reconstructions"] = outputs
        log["inputs"] = inputs
        # log["style_img"] = (pre_residual).clamp(0, 1)
        log["outputs"] = outputs
        log["pre_residual"] = pre_residual.clamp(0, 1)
        # log["end_reconstructions"] = (pre_residual).clamp(0, 1)

        # z = samples_ddim
        # B, C, _, _ = z.size()
        # sample = self.pre_AE_model.decode(torch.randn_like(z.reshape(B, C, 16, 64)))
        # log["sample"] = sample
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

