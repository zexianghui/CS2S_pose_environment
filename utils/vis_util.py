import skimage.io as io
import os.path
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from matplotlib.cm import get_cmap

#pcl_features_to_RGB([feature_map], 0, "result_visualize/")
def pcl_features_to_RGB(grd_feat_list, loop=0):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def reshape_normalize(x):
        '''
        Args:
            x: [B, C, H, W]

        Returns:

        '''
        B, C, H, W = x.shape
        x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator==0, 1, denominator)
        return x / denominator

    def normalize(x):
        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator == 0, 1, denominator)
        return x / denominator

    # sat_shape = []
    # grd_shape = []
    for level in range(len(grd_feat_list)):
    # for level in [len(sat_feat_list)-1]:
        # flatten = []

        grd_feat = grd_feat_list[level].data.cpu().numpy()  # [B, C, H, W]

        B, C, H, W = grd_feat.shape

        # flatten.append(reshape_normalize(grd_feat))

        # flatten = np.concatenate(flatten[:1], axis=0)

        # if level == 0:
        pca_grd = PCA(n_components=3)
        pca_grd.fit(reshape_normalize(grd_feat))

    # for level in range(len(sat_feat_list)):
        grd_feat = grd_feat_list[level].data.cpu().numpy()  # [B, C, H, W]

        B, C, H, W = grd_feat.shape
        grd_feat_new = ((normalize(pca_grd.transform(reshape_normalize(grd_feat))) + 1) / 2).reshape(B, H, W, 3)
        grd_feat_new = np.transpose(grd_feat_new, (0, 3, 1, 2))
        #gt_s2g_new = ((normalize(pca.transform(reshape_normalize(gt_s2g[:, :, H//2:, :]))) + 1) / 2).reshape(B, H//2, W, 3)

        # for idx in range(B):
        #     if not os.path.exists(os.path.join(save_dir)):
        #         os.makedirs(os.path.join(save_dir))

        #     grd = Image.fromarray((grd_feat_new[idx] * 255).astype(np.uint8))
        #     grd = grd.resize((W,H))
        #     grd.save(save_dir + ('feat_' + str(loop * B + idx) + '.jpg'))

    return grd_feat_new