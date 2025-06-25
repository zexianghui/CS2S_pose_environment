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

class grd_solve(nn.Module):
    def __init__(self, pth_path=None):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=1)

        self.head = nn.Sequential(
            nn.Linear(32*25*25, 16*25*25),
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
        x = nn.SELU()(x)

        x = x.flatten(1)
        x = self.head(x)
        return x

class sat_solve(nn.Module):
    def __init__(self, pth_path=None):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=1)

        self.head = nn.Sequential(
            nn.Linear(32*25*25, 16*25*25),
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
        x = nn.SELU()(x)

        x = x.flatten(1)
        x = self.head(x)
        return x

