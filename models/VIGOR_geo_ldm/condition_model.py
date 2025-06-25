import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from einops import rearrange

class VGGUnet(nn.Module):
    def __init__(self):
        super(VGGUnet, self).__init__()
        # print('estimate_depth: ', estimate_depth)

        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        # self.vgg16.load_state_dict(torch.load('orin_pth/vgg16-397923af.pth'))

        # load CNN from VGG16, the first three block
        self.conv0 = self.vgg16.features[0]
        self.conv2 = self.vgg16.features[2]  # \\64
        self.conv5 = self.vgg16.features[5]  #
        self.conv7 = self.vgg16.features[7]  # \\128
        self.conv10 = self.vgg16.features[10]
        self.conv12 = self.vgg16.features[12]
        self.conv14 = self.vgg16.features[14]  # \\256
        self.conv17 = self.vgg16.features[17]
        self.conv19 = self.vgg16.features[19]
        self.conv21 = self.vgg16.features[21]
        self.conv24 = self.vgg16.features[24]
        self.conv26 = self.vgg16.features[26]
        self.conv28 = self.vgg16.features[28]

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        # block0
        x0 = self.conv0(x)
        x1 = self.relu(x0)
        x2 = self.conv2(x1)
        x3 = self.relu(x2)
        x4, ind4 = self.max_pool(x3)  # [H/2, W/2]

        # block1
        x5 = self.conv5(x4)
        x6 = self.relu(x5)
        x7 = self.conv7(x6)
        x8 = self.relu(x7)
        x9, ind9 = self.max_pool(x8)  # [H/4, W/4]

        # block2
        x10 = self.conv10(x9)
        x11 = self.relu(x10)
        x12 = self.conv12(x11)
        x13 = self.relu(x12)
        x14 = self.conv14(x13)
        x15 = self.relu(x14)
        x16, ind16 = self.max_pool(x15)  # [H/8, W/8]

        # block3
        x17 = self.conv17(x16)
        x18 = self.relu(x17)
        x19 = self.conv19(x18)
        x20 = self.relu(x19)
        x21 = self.conv21(x20)
        x22 = self.relu(x21)
        # x23, ind23 = self.max_pool(x22) # [H/16, W/16]

        # # block4
        # x24 = self.conv24(x23)
        # x25 = self.relu(x24)
        # x26 = self.conv26(x25)
        # x27 = self.relu(x26)
        # x28 = self.conv28(x27)

        # x28 = rearrange(x15, 'b c h w -> b (h w) c')
        return x22

class VIT_224(nn.Module):
    def __init__(self):
        super(VIT_224, self).__init__()
        self.vit = torchvision.models.vit_b_16(pretrained=True)
        self.x_encoder = sat_encoder()
        self.conv_proj = self.vit.conv_proj
        self.encoder = self.vit.encoder
    
    def forward(self, x):
        x = torchvision.transforms.Resize((224, 224))(x)
        x = self.x_encoder(x)
        x = self.vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        x = x
        return x
    
    def process_input(self, x):
        x = torchvision.transforms.Resize((224, 224))(x)
        x = self.x_encoder(x)
        return x
    
    def encode(self, x):
        n = x.shape[0]
        x = self.vit._process_input(x)
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        x = x
        return x
    
class sat_encoder(nn.Module):
    def __init__(self):
        super(sat_encoder, self).__init__()

        self.model1=nn.Sequential(
            nn.Conv2d(3,3,7,1,3),
            nn.Conv2d(3,3,3,1,1),
            nn.Conv2d(3,3,5,1,2),
            nn.Conv2d(3,3,3,1,1)
        )

    def forward(self, x):
        return self.model1(x)

# vit = VIT_224()
# image = torch.randn(1, 3, 256, 256)
# re = vit(image)
