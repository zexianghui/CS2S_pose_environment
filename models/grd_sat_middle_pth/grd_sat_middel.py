import torch
import torch.nn as nn

def crossEntropy(y_pred, y_true):
    y_pred = nn.Softmax(dim=-1)(y_pred)
    y_pred = torch.clamp(y_pred, min=1e-10, max=1.0)
    loss = -torch.sum(y_true * torch.log(y_pred), dim=1)  
    return torch.mean(loss)*1000


def infoNCELoss(scores, labels, temperature=0.1):
    """
    Contrastive loss over matching score. Adapted from https://arxiv.org/pdf/2004.11362.pdf Eq.2
    We extraly weigh the positive samples using the ground truth likelihood on those positions
    
    loss = - 1/sum(weights) * sum(inner_element*weights)
    inner_element = log( exp(score_pos/temperature) / sum(exp(score/temperature)) )
    """
    
    exp_scores = torch.exp(scores / temperature)
    bool_mask = labels>1e-2 # elements with a likelihood > 1e-2 are considered as positive samples in contrastive learning
    
    denominator = torch.sum(exp_scores, dim=1, keepdim=True)
    inner_element = torch.log(torch.masked_select(exp_scores/denominator, bool_mask))
    loss = -torch.sum(inner_element*torch.masked_select(labels, bool_mask)) / torch.sum(torch.masked_select(labels, bool_mask))
    
    return loss

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        x = self.mean
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='batch', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class grd_solve(nn.Module):
    def __init__(self, pth_path=None, encoder_model=None, quant_conv=None):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=1)
        self.conv2 = ResidualBlock(8, 8)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=1)
        self.conv4 = ResidualBlock(16, 16)

        self.head = nn.Sequential(
            nn.Linear(16*25*25, 16*25*25),
            nn.SELU(),
            nn.Linear(16*25*25, 4*25*25),
        )

        if pth_path is not None:
            self.load_state_dict(torch.load(pth_path), strict=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.SELU()(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = nn.SELU()(x)

        x = x.flatten(1)
        x = self.head(x)
        return x
    

class sat_solve(nn.Module):
    def __init__(self, pth_path=None, encoder_model=None, quant_conv=None):
        super().__init__()
        self.encoder = encoder_model
        self.quant_conv = quant_conv

        self.conv1 = nn.Conv2d(4, 8, kernel_size=1)
        self.conv2 = ResidualBlock(8, 8)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=1)
        self.conv4 = ResidualBlock(16, 16)

        self.head = nn.Sequential(
            nn.Linear(16*25*25, 16*25*25),
            nn.SELU(),
            nn.Linear(16*25*25, 4*25*25),
        )

        if pth_path is not None:
            self.load_state_dict(torch.load(pth_path), strict=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.SELU()(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = nn.SELU()(x)

        x = x.flatten(1)
        x = self.head(x)
        return x
    
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

