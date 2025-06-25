import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from utils.util import instantiate_from_config
from models.loss_fun.losses import FeatureMatchingLoss, GaussianKLLoss, PerceptualLoss,GANLoss
from pytorch_msssim import ssim, SSIM
import pytorch_lightning as pl

class Sat2Den_loss(pl.LightningModule):
    def __init__(self, dis_cfg, opt_cfg):
        super().__init__()
        self.netD = instantiate_from_config(dis_cfg)
        self.criteria = {}
        self.weights = {}
        
        self.Percep_model = \
                    PerceptualLoss(
                        network=opt_cfg.perceptual_loss.mode,
                        layers=opt_cfg.perceptual_loss.layers,
                        weights=opt_cfg.perceptual_loss.weights)
        self.weights['Perceptual'] = opt_cfg.loss_weight.Perceptual
        if hasattr(opt_cfg.loss_weight, 'GaussianKL'):
            if opt_cfg.loss_weight.GaussianKL:
                self.criteria['GaussianKL'] = GaussianKLLoss()
                self.weights['GaussianKL'] = opt_cfg.loss_weight.GaussianKL
        if hasattr(opt_cfg.loss_weight, 'L1'):
            if opt_cfg.loss_weight.L1:
                self.criteria['L1']  = torch.nn.L1Loss(True,True)
                self.weights['L1'] = opt_cfg.loss_weight.L1
        if hasattr(opt_cfg.loss_weight, 'L2'):
            if opt_cfg.loss_weight.L2: 
                self.criteria['L2'] = torch.nn.MSELoss(True,True)
                self.weights['L2'] = opt_cfg.loss_weight.L2
        if hasattr(opt_cfg.loss_weight, 'SSIM'):
            if opt_cfg.loss_weight.SSIM: 
                self.criteria['SSIM'] = SSIM(data_range =1., size_average=True, channel=3)
                self.weights['SSIM']  = opt_cfg.loss_weight.SSIM
        if hasattr(opt_cfg.loss_weight, 'sky_inner'):
            if opt_cfg.loss_weight.sky_inner:
                self.criteria['sky_inner'] = torch.nn.L1Loss(True,True)
                self.weights['sky_inner'] = opt_cfg.loss_weight.sky_inner
        if hasattr(opt_cfg.loss_weight, 'feature_matching'):
            if opt_cfg.loss_weight.feature_matching:
                self.criteria['feature_matching'] = FeatureMatchingLoss()
                self.weights['feature_matching'] = opt_cfg.loss_weight.feature_matching
        self.weights['GAN'] = opt_cfg.loss_weight.GAN
        self.criteria['GAN'] = GANLoss(gan_mode=opt_cfg.gan_mode)
    
    def _get_outputs(self, net_D_output, real=True):
        r"""Return output values. Note that when the gan mode is relativistic.
        It will do the difference before returning.

        Args:
           net_D_output (dict):
               real_outputs (tensor): Real output values.
               fake_outputs (tensor): Fake output values.
           real (bool): Return real or fake.
        """

        def _get_difference(a, b):
            r"""Get difference between two lists of tensors or two tensors.

            Args:
                a: list of tensors or tensor
                b: list of tensors or tensor
            """
            out = list()
            for x, y in zip(a, b):
                if isinstance(x, list):
                    res = _get_difference(x, y)
                else:
                    res = x - y
                out.append(res)
            return out

        if real:
            return net_D_output['real_outputs']
        else:
            return net_D_output['fake_outputs']

    def forward(self, gt_outputs, pre_outputs, sky_mask, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
         # now the GEN part
        self.loss = {}
        
        if optimizer_idx == 0:
            net_D_output = self.netD(gt_outputs, pre_outputs)
            self.loss['Perceptual'] =  self.Percep_model(pre_outputs['pred'],gt_outputs)
            pred_fake = self._get_outputs(net_D_output, real=False)

            self.loss['GAN'] = self.criteria['GAN'](pred_fake, True, dis_update=False)
            if 'GaussianKL' in self.criteria:
                self.loss['GaussianKL'] = self.criteria['GaussianKL'](pre_outputs['mu'], pre_outputs['logvar'])
            if 'L1' in self.criteria:
                self.loss['L1'] = self.criteria['L1'](gt_outputs,pre_outputs['pred'])
            if 'L2' in self.criteria:
                self.loss['L2'] = self.criteria['L2'](gt_outputs,pre_outputs['pred'])
            if 'SSIM' in self.criteria:
                self.loss['SSIM'] = 1-self.criteria['SSIM'](gt_outputs, pre_outputs['pred'])
            if 'GaussianKL' in self.criteria:
                self.loss['GaussianKL'] = self.criteria['GaussianKL'](pre_outputs['mu'],pre_outputs['logvar'])
            if 'sky_inner' in self.criteria:
                self.loss['sky_inner'] = self.criteria['sky_inner'](pre_outputs['opacity'], 1 - sky_mask)
           
            if 'feature_matching' in self.criteria:
                self.loss['feature_matching']  = self.criteria['feature_matching'](net_D_output['fake_features'], net_D_output['real_features'])
            self.loss_G = 0
            for key in self.loss:
                self.loss_G += self.loss[key] * self.weights[key]
            self.loss['total'] = self.loss_G 
            return self.loss['total'], self.loss

        if optimizer_idx == 1:
            pre_outputs['pred'] = pre_outputs['pred'].detach()
            net_D_output = self.netD(gt_outputs, pre_outputs)

            output_fake = self._get_outputs(net_D_output, real=False)
            output_real = self._get_outputs(net_D_output, real=True)
            fake_loss = self.criteria['GAN'](output_fake, False, dis_update=True)
            true_loss = self.criteria['GAN'](output_real, True, dis_update=True)
            self.dis_losses = dict()
            self.dis_losses['GAN/fake'] = fake_loss
            self.dis_losses['GAN/true'] = true_loss
            self.dis_losses['DIS'] = fake_loss + true_loss
            return self.dis_losses['DIS'], self.dis_losses