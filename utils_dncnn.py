import math
import torch
import torch.nn as nn
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(
            Iclean[i, 0],
            Img[i, 0],
            data_range=data_range
        )

    return SSIM / Img.shape[0]

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    h, w = out.shape[:2]
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.fliplr(out)
    elif mode == 2:
        # Rotate +5°
        M = cv2.getRotationMatrix2D((w/2, h/2), 5, 1.0)
        out = cv2.warpAffine(out, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        if out.ndim == 2:
            out = np.expand_dims(out, axis=-1)
    elif mode == 3:
         # Rotate -5°
        M = cv2.getRotationMatrix2D((w/2, h/2), -5, 1.0)
        out = cv2.warpAffine(out, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        if out.ndim == 2:
            out = np.expand_dims(out, axis=-1)
    elif mode == 4:
        # Translate kanan
        M = np.float32([[1, 0, 5],
                        [0, 1, 0]])
        out = cv2.warpAffine(out, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        if out.ndim == 2:
            out = np.expand_dims(out, axis=-1)
    elif mode == 5:
       # Translate kiri
        M = np.float32([[1, 0, -5],
                        [0, 1, 0]])
        out = cv2.warpAffine(out, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        if out.ndim == 2:
            out = np.expand_dims(out, axis=-1)
    elif mode == 6:
        # Translate atas
        M = np.float32([[1, 0, 0],
                        [0, 1, -5]])
        out = cv2.warpAffine(out, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        if out.ndim == 2:
            out = np.expand_dims(out, axis=-1)
    elif mode == 7:
        # Translate bawah
        M = np.float32([[1, 0, 0],
                        [0, 1, 5]])
        out = cv2.warpAffine(out, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        if out.ndim == 2:
            out = np.expand_dims(out, axis=-1)
        
    return np.transpose(out, (2,0,1))
