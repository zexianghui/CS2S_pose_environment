import torch,os
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision.transforms as transforms
import re
from easydict import EasyDict as edict
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils.util import instantiate_from_config


def data_list(img_root,mode):
    data_list=[]
    if mode=='train':
        split_file=os.path.join(img_root, 'splits/train-19zl.csv')
        with open(split_file) as f:
            list = f.readlines()
            for i in list:
                aerial_name=re.split(r',', re.split('\n', i)[0])[0]
                panorama_name = re.split(r',', re.split('\n', i)[0])[1]
                data_list.append([aerial_name, panorama_name])
    else:
        split_file=os.path.join(img_root+'splits/val-19zl.csv')
        with open(split_file) as f:
            list = f.readlines()
            for i in list:
                aerial_name=re.split(r',', re.split('\n', i)[0])[0]
                panorama_name = re.split(r',', re.split('\n', i)[0])[1]
                data_list.append([aerial_name, panorama_name])
    print('length of dataset is: ', len(data_list))
    return [os.path.join(img_root, i[1]) for i in data_list]
    
def img_read(img,size=None,datatype='RGB'):
    img = Image.open(img).convert('RGB' if datatype=='RGB' else "L")
    if size:
        if type(size) is int:
            size = (size,size)
        img = img.resize(size = size,resample=Image.BICUBIC if datatype=='RGB' else Image.NEAREST)
    img = transforms.ToTensor()(img)
    # img = img*2 - 1
    return img


class Dataset(Dataset):
    def __init__(self, data_root, sat_size, pano_size, sky_mask, histo_mode, split='train',sub=None,sty_img=None):
        self.pano_list = data_list(img_root=data_root,mode=split)
        self.sky_mask = sky_mask
        self.sat_size = sat_size
        self.pano_size = pano_size
        self.histo_mode = histo_mode

    def __len__(self):
        return len(self.pano_list)

    def __getitem__(self, index):
        pano = self.pano_list[index]
        aer = pano.replace('streetview/panos', 'bingmap/19')
        if self.sky_mask:
            sky = pano.replace('streetview/panos','sky_mask').replace('jpg', 'png')

        input = {}
        name = pano
        aer = img_read(aer,  size = self.sat_size)
        pano = img_read(pano,size = self.pano_size)
        if self.sky_mask:
            sky = img_read(sky,size=self.pano_size,datatype='L')
            input['sky_mask']=sky

        input['sat']=aer
        input['pano']=pano
        input['paths']=name
        input['label'] = ""
        input['shift_x'] = 0
        input['shift_y'] = 0
        
        return input
       