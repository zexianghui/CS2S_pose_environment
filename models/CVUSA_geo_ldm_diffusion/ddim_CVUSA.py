"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from torchvision import transforms
from torch import nn
import kornia.augmentation as K
from collections import defaultdict
from torch.nn import functional as F
import torchvision.transforms.functional as TF

from PIL import Image
from clip import clip
from torch.nn.functional import mse_loss

# from models.geometry.kitti_grd2sat import project_grd_to_map
from models.geometry.sat2grd import CVUSA_sat2grd_uv, grid_sample

from models.grd_sat_middle_pth.grd_sat_middel_end import grd_solve, sat_solve, infoNCELoss, crossEntropy
import torch.optim as optim

default_noise = [torch.randn((1,4,16,64)) for _ in range(100)]

class AdamOptimizer:
    def __init__(self, lr=0.005, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr 
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.epsilon = epsilon  

        self.m = None
        self.v = None
        self.t = 0  

    def __call__(self, param, loss):
        if self.m==None and self.v==None:
            self.m = torch.zeros_like(param, device=param.device)
            self.v = torch.zeros_like(param, device=param.device)
            self.t = 0
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * loss
        self.v = self.beta2 * self.v + (1 - self.beta2) * (loss ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        param = param - self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        return param


def img_read(img,size=None,datatype='RGB'):
    img = Image.open(img).convert('RGB' if datatype=='RGB' else "L")
    if size:
        if type(size) is int:
            size = (size,size)
        img = img.resize(size = size,resample=Image.BICUBIC if datatype=='RGB' else Image.NEAREST)
    img = transforms.ToTensor()(img)
    # img = img*2 - 1
    return img

class MetricsAccumulator:
    def __init__(self) -> None:
        self.accumulator = defaultdict(lambda: [])

    def update_metric(self, metric_name, metric_value):
        self.accumulator[metric_name].append(metric_value)

    def print_average_metric(self):
        for k, v in self.accumulator.items():
            average_v = np.array(v).mean()
            print(f"{k} - {average_v:.2f}")

        self.__init__()

class ImageAugmentations(nn.Module):
    def __init__(self, output_size, aug_prob, p_min, p_max, patch=False):
        super().__init__()
        self.output_size = output_size
        
        self.aug_prob = aug_prob
        self.patch = patch
        self.augmentations = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=aug_prob, padding_mode="border"),  # type: ignore
            K.RandomPerspective(0.7, p=aug_prob),
        )
        self.random_patch = K.RandomResizedCrop(size=(128,128), scale=(p_min,p_max))
        self.avg_pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))

    def forward(self, input, num_patch=None, is_global=False):
        """Extents the input batch with augmentations

        If the input is consists of images [I1, I2] the extended augmented output
        will be [I1_resized, I2_resized, I1_aug1, I2_aug1, I1_aug2, I2_aug2 ...]

        Args:
            input ([type]): input batch of shape [batch, C, H, W]

        Returns:
            updated batch: of shape [batch * augmentations_number, C, H, W]
        """
        if self.patch:
            if is_global:
                input = input.repeat(num_patch,1,1,1)
            else:
                input_patches = []
                for i in range(num_patch):
                    if self.aug_prob > 0.0:
                        tmp = self.augmentations(self.random_patch(input))
                    else:
                        tmp = self.random_patch(input)
                    input_patches.append(tmp)
                input = torch.cat(input_patches,dim=0)
        
        else:
            input_patches = []
            for i in range(num_patch):
                input_patches.append(self.augmentations(input))
            input = torch.cat(input_patches,dim=0)
        
        resized_images = self.avg_pool(input)
        return resized_images

class CVUSA_DDIMSampler(object):
    def __init__(self, model, pre_AE_model, scale_factor, schedule="linear", grd_solve_pth=None, sat_solve_pth=None, **kwargs):
        super().__init__()
        self.model = model
        self.pre_AE_model = pre_AE_model
        self.scale_factor = scale_factor
        self.ddpm_num_timesteps = model.timesteps
        self.schedule = schedule
        
        self.clip_model = (
            clip.load("ViT-B/16", device=self.model.device, jit=False)[0].eval().requires_grad_(False)
        )
            
        self.clip_size = self.clip_model.visual.input_resolution
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        self.image_augmentations = ImageAugmentations(224, aug_prob = 1, p_min = 0.2, p_max = 0.5, patch=False)
        self.patch_augmentations = ImageAugmentations(224, aug_prob = 1, p_min = 0.2, p_max = 0.5, patch=True)
        
        self.metrics_accumulator = MetricsAccumulator()
        # self.loc_opt = None

        if grd_solve_pth != None:
            self.grd_solver = grd_solve(pth_path=grd_solve_pth).to(self.model.device)
        if sat_solve_pth != None:
            self.sat_solver = sat_solve(pth_path=sat_solve_pth, encoder_model = pre_AE_model.encoder, quant_conv = pre_AE_model.quant_conv).to(self.model.device)

    def d_clip_loss(self, x, y, use_cosine=False):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        if use_cosine:
            distance = 1 - (x @ y.t()).squeeze()
        else:
            distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

        return distance

    def clip_global_loss(self,x_in,text_embed):
        B = x_in.size()[0]
        clip_loss = torch.tensor(0)
        augmented_input = self.image_augmentations(x_in,num_patch=32).add(1).div(2)
        # augmented_input = F.interpolate(x_in, size=(224, 224), mode='bilinear', align_corners=False)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.encode_image(clip_in).float()
        dists = self.d_clip_loss(image_embeds, text_embed)
        for i in range(B):
            clip_loss = clip_loss + dists[i :: B].mean()
        return clip_loss

    def clip_global_patch_loss(self, x_in, text_embed):
        B = x_in.size()[0]
        clip_loss = torch.tensor(0)
        augmented_input = self.patch_augmentations(x_in,num_patch=32).add(1).div(2)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.encode_image(clip_in).float()
        dists = self.d_clip_loss(image_embeds, text_embed)
        for i in range(B):
            clip_loss = clip_loss + dists[i :: B].mean()

        return clip_loss

    def triplet_loss(self, x, orin_sat_feat):
        B = x.size()[0]
        A = orin_sat_feat.size()[-1]
        crop_H = int(26)
        crop_W = int(26)

        gt_h = torch.linspace(-(A - crop_H)*4, (A - crop_H)*4, A - crop_H + 1).to(x.device)
        gt_w = torch.linspace(-(A - crop_W)*4, (A - crop_W)*4, A - crop_W + 1).to(x.device)
        gt_h, gt_w = torch.meshgrid(gt_h, gt_w, indexing='ij')

        d = torch.sqrt(gt_h**2 + gt_w**2)
        sigma, mu = 4, 0.0
        gt_grid = torch.exp(-((d - mu) ** 2) / (2.0 * sigma ** 2))
        gt_grid = gt_grid.reshape(-1).unsqueeze(0).repeat(B, 1)

        meter_per_pixel = torch.ones(1) * 0.21 * 256 / A
        optical = CVUSA_sat2grd_uv(0, x.size(2), x.size(3), 2, A, A, meter_per_pixel)
        optical = optical.to(x.device).repeat(B, 1, 1, 1)
        BEV_map, _ = grid_sample(x, optical)
        g2s_feat = TF.center_crop(BEV_map, [crop_H, crop_W])

        sat_patch = []
        for h in range(0, orin_sat_feat.shape[-2] - crop_H + 1, 1):
            for w in range(0, orin_sat_feat.shape[-1] - crop_W + 1, 1):
                orin_sat_feat_t = orin_sat_feat[:,:,h:h+crop_H,w:w+crop_W]
                sat_patch.append(orin_sat_feat_t.unsqueeze(1))
        sat_patch = torch.cat(sat_patch, dim=1)

        g2s_feat_middle = self.grd_solver(g2s_feat).reshape(B, -1)
        sat_noisy_patch_middle = self.sat_solver(sat_patch.reshape(B*49, 4, crop_H, crop_W)).reshape(B, 49, 4*crop_H*crop_W)
        g2s_feat_middle_norm = F.normalize(g2s_feat_middle, dim=-1)
        sat_noisy_patch_middle_norm = F.normalize(sat_noisy_patch_middle, dim=-1)

        matching_score_stacked = torch.bmm(sat_noisy_patch_middle_norm, g2s_feat_middle_norm.unsqueeze(-1)).squeeze(-1)

        # print(matching_score_stacked.reshape(B, 7, 7))
        loss = infoNCELoss(matching_score_stacked, gt_grid)
        return loss

    def cond_fn(self, x, c, text_embed, orin_sat_feat, inter_ref, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                unconditional_guidance_scale=1., unconditional_conditioning=None):
        para_dict = {
            "l_clip_global": 0,
            "l_clip_global_patch": 15000, #5000 15000
            "l_triplet": 0,
                }
        clip_loss = 0    
        clip_patch_loss = 0
        l_triplet = 0
        with torch.set_grad_enabled(True):
            x = x.detach().requires_grad_()
            x.requires_grad_(True)
            self.pre_AE_model.requires_grad_(True)
            self.model.requires_grad_(True)
            e_t, x_prev, pred_x0 = self.p_sample_ddim(x, c, t, index=index,       
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning)
        
            fac = self.sqrt_one_minus_alphas_cumprod[t[0].item()]
            x_in = pred_x0 * fac + x * (1 - fac)
            x_in_decoder = x_in * (1 / self.scale_factor)
            x_in_decoder = self.pre_AE_model.decode(x_in_decoder)
            loss = torch.tensor(0)
            if para_dict["l_clip_global"] != 0:
                clip_loss = self.clip_global_loss(x_in_decoder, text_embed) * para_dict["l_clip_global"]
                loss = loss + clip_loss
            
            if para_dict["l_clip_global_patch"] != 0: 
                clip_patch_loss = self.clip_global_patch_loss(x_in_decoder, text_embed) * para_dict["l_clip_global_patch"]
                loss = loss + clip_patch_loss
        return e_t, -torch.autograd.grad(loss, x)[0]
        
    def cond_sat_fn(self, x, c, text_embed, orin_sat_feat, inter_ref, t, index, repeat_noise=False, 
                use_original_steps=False, quantize_denoised=False,
                temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                unconditional_guidance_scale=1., unconditional_conditioning=None):
        if (not (index>15 and index<50)) or index%5 != 0:
            with torch.set_grad_enabled(False):
                e_t, x_prev, pred_x0 = self.p_sample_ddim(x, c, t, index=index,       
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning)
                return e_t, x_prev, pred_x0, x
        with torch.set_grad_enabled(True):
            x = x.detach().requires_grad_()

            H_matrix = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=x.device).unsqueeze(0).repeat(x.size()[0],1,1)
            H_matrix.requires_grad = True
            grid = F.affine_grid(H_matrix[:,:2,:], (x.size()[0],4,16,64))
            x = F.grid_sample(x, grid, padding_mode='zeros')

            e_t, x_prev, pred_x0 = self.p_sample_ddim(x, c, t, index=index,       
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning)
        
            fac = self.sqrt_one_minus_alphas_cumprod[t[0].item()]
            x_in = pred_x0 * fac + x * (1 - fac)

            fac = self.sqrt_one_minus_alphas_cumprod[t[0].item()]
            orin_sat_feat_t = self.model.q_sample(orin_sat_feat, t)
            orin_sat_feat_t = orin_sat_feat * fac + orin_sat_feat_t * (1 - fac)

            l_triplet = self.triplet_loss(x_in, orin_sat_feat_t)
            # print(l_triplet)
            # if l_triplet<2:
            #     return x
            grd = torch.autograd.grad(l_triplet, H_matrix)[0]
            grd = grd * 0.5 * 0.8 ** (50 - index)
            # print(grd)
            H_matrix = H_matrix - grd
            grid = F.affine_grid(H_matrix[:,:2,:], (x.size()[0],4,16,64))
            x = F.grid_sample(x, grid, padding_mode='zeros')
            x_prev = F.grid_sample(x_prev, grid, padding_mode='zeros')
            pred_x0 = F.grid_sample(pred_x0, grid, padding_mode='zeros')

            noise = torch.randn_like(x)
            mask = (x == 0)
            x[mask] = noise[mask]
            x_prev[mask] = noise[mask]
            pred_x0[mask] = noise[mask]
            return e_t, x_prev, pred_x0, x

    def cond_sat_test_fn(self, x, c, text_embed, orin_sat_feat, inter_ref, t, index, repeat_noise=False, 
                use_original_steps=False, quantize_denoised=False,
                temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                unconditional_guidance_scale=1., unconditional_conditioning=None):
        x = x.detach().requires_grad_()
        x.requires_grad_(True)
        self.pre_AE_model.requires_grad_(True)
        self.model.requires_grad_(True)

        H_matrix = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=x.device).unsqueeze(0).repeat(x.size()[0],1,1)
        H_matrix.requires_grad = True
        grid = F.affine_grid(H_matrix[:,:2,:], (x.size()[0],4,16,64))
        x = F.grid_sample(x, grid, padding_mode='zeros')

        _, x_prev, pred_x0 = self.p_sample_ddim(x, c, t, index=index,       
                                    quantize_denoised=quantize_denoised, temperature=temperature,
                                    noise_dropout=noise_dropout, score_corrector=score_corrector,
                                    corrector_kwargs=corrector_kwargs,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
    
        fac = self.sqrt_one_minus_alphas_cumprod[t[0].item()]
        x_in = pred_x0 * fac + x * (1 - fac)

        fac = self.sqrt_one_minus_alphas_cumprod[t[0].item()]
        orin_sat_feat_t = self.model.q_sample(orin_sat_feat, t)
        orin_sat_feat_t = orin_sat_feat * fac + orin_sat_feat_t * (1 - fac)

        l_triplet = self.triplet_loss(x_in, orin_sat_feat_t, inter_ref)
        # print("=========================")
        # print(l_triplet)
        return x
    
    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               txt = None,
               orin_sat_feat = None,
               inter_ref = None,   
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if txt is not None:
            txt = self.clip_model.encode_text(
                clip.tokenize(txt).to(conditioning.device)
            ).float()
        # if orin_sat_feat is not None:
        #     self.loc_opt = AdamOptimizer(lr=0.01)
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, txt, orin_sat_feat, inter_ref, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, txt_embed, orin_sat_feat, inter_ref, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.condition_score(img, cond, txt_embed, orin_sat_feat, inter_ref, ts, index=index,       
                                      use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        # control_grd_para = self.model.control_grd(x, t, cond_init_grd = cond_grd, cond_txt = c)
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.denoise_model(x, t, context = c, control_grd = None)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            cond_sat = torch.cat([cond_sat] * 2)
            cond_grd = torch.cat([cond_grd] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.denoise_model(x_in, t_in, cond_sat = cond_sat, cond_grd = cond_grd,cond_txt = c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # if index < 1:
        #     # sigma_t = torch.zeros_like(sigma_t, device=sigma_t.device)
        #     temperature = 0
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        # noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        noise = sigma_t * default_noise[index].repeat(x.shape[0],1,1,1).to(device) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return e_t, x_prev, pred_x0

    def condition_score(self, x, c, txt_embed, orin_sat_feat, inter_ref, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        if index<=1:
            txt_embed = None
            orin_sat_feat = None
        if txt_embed != None or orin_sat_feat != None:
            b, *_, device = *x.shape, x.device

            alphas = self.ddim_alphas
            alphas_prev = self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
            
            if orin_sat_feat is not None:
                e_t, x_prev, pred_x0, x = self.cond_sat_fn(x, c, txt_embed, orin_sat_feat, inter_ref, t, index=index,       
                                        use_original_steps=use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning)
            # self.cond_sat_test_fn(x, c, txt_embed, orin_sat_feat, inter_ref, t, index=index,       
            #                         use_original_steps=use_original_steps,
            #                         quantize_denoised=quantize_denoised, temperature=temperature,
            #                         noise_dropout=noise_dropout, score_corrector=score_corrector,
            #                         corrector_kwargs=corrector_kwargs,
            #                         unconditional_guidance_scale=unconditional_guidance_scale,
            #                         unconditional_conditioning=unconditional_conditioning)
            # control_grd_para = self.model.control_grd(x, t, cond_init_grd = cond_grd, cond_txt = c)
            # if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            #     e_t = self.model.denoise_model(x, t, context = c, control_grd = None)
            # else:
            #     x_in = torch.cat([x] * 2)
            #     t_in = torch.cat([t] * 2)
            #     cond_sat = torch.cat([cond_sat] * 2)
            #     cond_grd = torch.cat([cond_grd] * 2)
            #     c_in = torch.cat([unconditional_conditioning, c])
            #     e_t_uncond, e_t = self.model.denoise_model(x_in, t_in, cond_sat = cond_sat, cond_grd = cond_grd,cond_txt = c_in).chunk(2)
            #     e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            # if score_corrector is not None:
            #     assert self.model.parameterization == "eps"
            #     e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
            
            if txt_embed is not None:
                e_t, cond_grd = self.cond_fn(x, c, txt_embed, orin_sat_feat, inter_ref, t, index=index,       
                                        use_original_steps=use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning)
                e_t = e_t - (1 - a_t).sqrt() * cond_grd
            ################################################
            
            ##################################################
            # current prediction for x_0 
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            # noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            noise = sigma_t * default_noise[index].repeat(x.shape[0],1,1,1).to(device) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0
            #Image.fromarray((np.transpose(((self.pre_AE_model.decode(pred_x0 * (1 / self.scale_factor))[0]+1)/2).detach().cpu().numpy(), (1, 2, 0))* 255).astype(np.uint8)).save('output_image.png')
        else: 
            e_t, x_prev, pred_x0 = self.p_sample_ddim(x, c, t, index=index,       
                                    use_original_steps=use_original_steps,
                                    quantize_denoised=quantize_denoised, temperature=temperature,
                                    noise_dropout=noise_dropout, score_corrector=score_corrector,
                                    corrector_kwargs=corrector_kwargs,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
            return x_prev, pred_x0