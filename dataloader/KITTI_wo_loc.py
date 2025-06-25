import random

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

import torch
import dataloader.KITTI_utils as utils
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

root_dir = './dataset/KITTI' # '../../data/Kitti' # '../Data' #'..\\Data' #

test_csv_file_name = 'test.csv'
ignore_csv_file_name = 'ignore.csv'
satmap_dir = 'satmap'
grdimage_dir = 'raw_data'
left_color_camera_dir = 'image_02/data'  # 'image_02\\data' #
right_color_camera_dir = 'image_03/data'  # 'image_03\\data' #
oxts_dir = 'oxts/data'  # 'oxts\\data' #
depth_dir = 'depth/data_depth_annotated/train/'

GrdImg_H = 128  # 256 # original: 375 #224, 256
GrdImg_W = 512  # 1024 # original:1242 #1248, 1024
GrdOriImg_H = 375
GrdOriImg_W = 1242
num_thread_workers = 2

train_file = './dataLoader/train_files.txt'
test1_file = './dataLoader/test1_files.txt'
test2_file = './dataLoader/test2_files.txt'


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth


class SatGrdDataset(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = utils.get_meter_per_pixel()
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters 
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters 

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~

        SatMap_end_sidelength = utils.SatMap_end_sidelength## 0.2 m per pixel

        self.satmap_transform = transforms.Compose([
            transforms.Resize(size=[SatMap_end_sidelength, SatMap_end_sidelength]),#sat_d*sat_d
            transforms.ToTensor(),
        ])

        Grd_h = GrdImg_H
        Grd_w = GrdImg_W

        self.grdimage_transform = transforms.Compose([
            # transforms.Resize(size=[Grd_h, Grd_w]),#grd_H*grd_W
            transforms.ToTensor(),
        ])

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir#satmap

        with open(file, 'r') as f:
            file_name = f.readlines()

        self.file_name = [file[:-1] for file in file_name]

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        file_name = self.file_name[idx]
        day_dir = file_name[:10]
        drive_dir = file_name[:38]#2011_09_26/2011_09_26_drive_0002_sync/
        image_no = file_name[38:] 

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    break

        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB').resize((utils.SatMap_process_sidelength, utils.SatMap_process_sidelength))

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        # grd_left_depths = torch.tensor([1])
        image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
                content = f.readline().split(' ')
                # get heading
                heading = float(content[5])
                heading = torch.from_numpy(np.asarray(heading))

                left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                             image_no.lower())
                with Image.open(left_img_name, 'r') as GrdImg:
                    grd_img_left = GrdImg.convert('RGB').resize((GrdImg_W, GrdImg_H))
                    if self.grdimage_transform is not None:
                        grd_img_left = self.grdimage_transform(grd_img_left)

                grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
                
        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, utils.CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, utils.CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=Image.BILINEAR)

        sat_align_cam_cut =TF.center_crop(sat_align_cam, utils.SatMap_end_sidelength)
        if self.satmap_transform is not None:
            sat_map_gt = self.satmap_transform(sat_align_cam_cut)

        gt_shift_x = 0 # --> right as positive, parallel to the heading direction
        gt_shift_y = 0  # --> up as positive, vertical to the heading direction
        theta = 0
        
        input = {}
        input['sat_map_gt'] = sat_map_gt
        input['sat_map'] = sat_map_gt
        input['left_camera_k'] = left_camera_k
        input['grd_left_imgs'] = grd_left_imgs[0]
        input['gt_shift_x'] = torch.tensor(-gt_shift_x, dtype=torch.float32).reshape(1)
        input['gt_shift_y'] = torch.tensor(-gt_shift_y, dtype=torch.float32).reshape(1)
        input['theta'] = torch.tensor(theta, dtype=torch.float32).reshape(1)
        input['file_name'] = file_name

        return input


class SatGrdDatasetTest(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = utils.get_meter_per_pixel()
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        SatMap_end_sidelength = utils.SatMap_end_sidelength## 0.2 m per pixel

        self.satmap_transform = transforms.Compose([
            transforms.Resize(size=[SatMap_end_sidelength, SatMap_end_sidelength]),#sat_d*sat_d
            transforms.ToTensor(),
        ])

        Grd_h = GrdImg_H
        Grd_w = GrdImg_W

        self.grdimage_transform = transforms.Compose([
            transforms.ToTensor(),
        ])



        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        self.file_name = [file[:-1] for file in file_name]



    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        line = self.file_name[idx]
        file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    # if not self.stereo:
                    break

        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB').resize((utils.SatMap_process_sidelength, utils.SatMap_process_sidelength))

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])

        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            heading = float(content[5])
            heading = torch.from_numpy(np.asarray(heading))

            left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            with Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB').resize((GrdImg_W, GrdImg_H))
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)

        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, utils.CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, utils.CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=Image.BILINEAR)
        
        sat_align_cam_cut =TF.center_crop(sat_align_cam, utils.SatMap_end_sidelength)
        if self.satmap_transform is not None:
            sat_map_gt = self.satmap_transform(sat_align_cam_cut)

        gt_shift_x = 0  # --> right as positive, parallel to the heading direction
        gt_shift_y = 0  # --> up as positive, vertical to the heading direction

        theta = 0
 
        input = {}
        input['sat_map'] = sat_map_gt
        input['left_camera_k'] = left_camera_k
        input['grd_left_imgs'] = grd_left_imgs[0]
        input['gt_shift_x'] = torch.tensor(-gt_shift_x, dtype=torch.float32).reshape(1)
        input['gt_shift_y'] = torch.tensor(-gt_shift_y, dtype=torch.float32).reshape(1)
        input['theta'] = torch.tensor(theta, dtype=torch.float32).reshape(1)
        input['file_name'] = file_name
        return input



def load_train_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10, dpp=1):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()## 0.2 m per pixel

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),#sat_d*sat_d
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),#grd_H*grd_W
        transforms.ToTensor(),
    ])

    train_set = SatGrdDataset(root=root_dir, file=train_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)  

    if dpp:
        train_sampler = DistributedSampler(train_set)
        train_dataloader=DataLoader(train_set, batch_size=batch_size, sampler = train_sampler, pin_memory=True,
                                num_workers=num_thread_workers, drop_last=True)
        return train_dataloader, train_sampler
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                num_workers=num_thread_workers, drop_last=True)
    return train_loader


def load_test1_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    # # Plz keep the following two lines!!! These are for fair test comparison.
    # np.random.seed(2022)
    # torch.manual_seed(2022)

    test1_set = SatGrdDatasetTest(root=root_dir, file=test1_file,
                            transform=(satmap_transform, grdimage_transform),
                            shift_range_lat=shift_range_lat,
                            shift_range_lon=shift_range_lon,
                            rotation_range=rotation_range)

    test1_loader = DataLoader(test1_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_thread_workers, drop_last=True)
    return test1_loader


def load_test2_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    # # Plz keep the following two lines!!! These are for fair test comparison.
    # np.random.seed(2022)
    # torch.manual_seed(2022)

    test2_set = SatGrdDatasetTest(root=root_dir, file=test2_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)

    test2_loader = DataLoader(test2_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=True)
    return test2_loader



if __name__ == '__main__':
    
    train_loader = load_train_data(64)

    for epoch in range(10):
        #train_loader.sampler.set_epoch(epoch)
        # sat_map, left_camera_k, grd_left_imgs[0], \
        # torch.tensor(-gt_shift_x, dtype=torch.float32).reshape(1), \
        # torch.tensor(-gt_shift_y, dtype=torch.float32).reshape(1), \
        # torch.tensor(theta, dtype=torch.float32).reshape(1), \
        # file_name
        for step, data in enumerate(train_loader):
            sat_map, left_camera, grd_left_img, trans_x, trans_y, theta = data
            B = sat_map.size(0)
            sat_map = F.interpolate(sat_map, size=(64, 64), mode='bilinear', align_corners=False)
            grd_left_img = F.interpolate(grd_left_img, size=(64, 256), mode='bilinear', align_corners=False)
            from models.geometry.kitti_sat2grd import project_map_to_grd, get_xyz_grds
            xyz_grds = get_xyz_grds(left_camera, ori_grdH = 128, ori_grdW = 512, num_levels = 6)
            sat_feat_proj, _, sat_uv, mask = project_map_to_grd(xyz_grds, sat_map, sat_c = None, shift_u = trans_x, shift_v = trans_y, heading = data, level = 3)
            for i in range(5):
                Image.fromarray((np.transpose(sat_map[i].detach().cpu().numpy(), (1, 2, 0))* 255).astype(np.uint8)).save('test/sat.png')
                Image.fromarray((np.transpose(grd_left_img[i].detach().cpu().numpy(), (1, 2, 0))* 255).astype(np.uint8)).save('test/grd.png')
                Image.fromarray((np.transpose(sat_feat_proj[i].detach().cpu().numpy(), (1, 2, 0))* 255).astype(np.uint8)).save('test/output_image.png')



