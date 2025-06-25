import torch
import torch.nn as nn
import lpips
from pytorch_msssim import ssim

def sd_func(real, fake):
    '''
    ref: page 6 in https://arxiv.org/abs/1511.05440
    '''
    dgt1 = torch.abs(torch.diff(real,dim=-2))[:, :, 1:, 1:-1]
    dgt2 = torch.abs(torch.diff(real, dim=-1))[:, :, 1:-1, 1:]
    dpred1 = torch.abs(torch.diff(fake, dim=-2))[:, :, 1:, 1:-1]
    dpred2 = torch.abs(torch.diff(fake, dim=-1))[:, :, 1:-1, 1:]
    return 10*torch.log10(1.**2/torch.mean(torch.abs(dgt1+dgt2-dpred1-dpred2))).cpu().item()

def deprocess(image):
    image = (image + 1)/2
    image = torch.clamp(image, 0., 1.)
    return image

class Evaluate_indic(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn_alex = lpips.LPIPS(net='alex',eval_mode=True).cuda()
        self.loss_fn_sque = lpips.LPIPS(net='squeeze',eval_mode=True).cuda()
        self.mseloss = torch.nn.MSELoss(True,True)
        

    def forward(self, fake_B, real_B, split="test"):
        # fake_B = deprocess(fake_B)
        # real_B = deprocess(real_B)
        RMSE = torch.sqrt(self.mseloss(fake_B*255.,real_B*255.)).item()
        SSIM = ssim(real_B, fake_B,data_range=1.).item()
        PSNR = -10*self.mseloss(fake_B,real_B).log10().item() 
        SD = sd_func(real_B,fake_B)  
        
        
        P_alex = torch.mean(self.loss_fn_alex((real_B*2.)-1, (2.*fake_B)-1)).cpu() 
        P_squeeze = torch.mean(self.loss_fn_sque((real_B*2.)-1, (2.*fake_B)-1)).cpu() 

        eva_log = {"RMSE": RMSE,
                "SSIM": SSIM,
                "PSNR": PSNR,
                "SD": SD,
                "P_alex": P_alex,
                "P_squeeze": P_squeeze
            }
        
        return eva_log