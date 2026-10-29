import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import random
import cv2

class mydata_rldcnn(Dataset):
    def __init__(self, LR_path, GT_path, in_memory = True, transform = None):
        self.LR_path = LR_path
        self.GT_path = GT_path
        self.in_memory = in_memory
        self.transform = transform

        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        self.LR_img = sorted([
            f for f in os.listdir(LR_path)
            if f.lower().endswith(valid_ext)
        ])

        self.GT_img = sorted([
            f for f in os.listdir(GT_path)
            if f.lower().endswith(valid_ext)
        ])
        
        if in_memory:
            # Tetap simpan dalam format uint8 (0-255) di memori RAM agar menghemat tempat
            self.LR_img = [np.array(Image.open(os.path.join(self.LR_path, lr)).convert("RGB")).astype(np.uint8) for lr in self.LR_img]
            self.GT_img = [np.array(Image.open(os.path.join(self.GT_path, gt)).convert("RGB")).astype(np.uint8) for gt in self.GT_img]
        
    def __len__(self):
        return len(self.LR_img)
        
    def __getitem__(self, i):
        img_item = {}
        
        if self.in_memory:
            # Ambil salinan bertipe float32 tapi biarkan nilainya masih 0-255 untuk augmentasi OpenCV
            GT = self.GT_img[i].astype(np.float32)
            LR = self.LR_img[i].astype(np.float32)
        else:
            GT = np.array(Image.open(os.path.join(self.GT_path, self.GT_img[i])).convert("RGB")).astype(np.float32)
            LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_img[i])).convert("RGB")).astype(np.float32)

        img_item['GT'] = GT
        img_item['LR'] = LR
                
        # 1. Jalankan transform (Crop & Augmentasi) saat nilai piksel masih berskala asli (0-255)
        if self.transform is not None:
            img_item = self.transform(img_item)
            
        # 2. SEKARANG baru lakukan normalisasi ke rentang [0.0, 1.0] agar kalkulasi metrik valid
        img_item['GT'] = img_item['GT'] / 255.0
        img_item['LR'] = img_item['LR'] / 255.0
            
        # 3. Ubah format dari (H, W, C) ke (C, H, W) untuk PyTorch
        img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32)
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)
        
        return img_item

class mydata(Dataset):
    def __init__(self, LR_path, GT_path, in_memory = True, transform = None):
        
        self.LR_path = LR_path
        self.GT_path = GT_path
        self.in_memory = in_memory
        self.transform = transform

        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        self.LR_img = sorted([
            f for f in os.listdir(LR_path)
            if f.lower().endswith(valid_ext)
        ])

        self.GT_img = sorted([
            f for f in os.listdir(GT_path)
            if f.lower().endswith(valid_ext)
        ])
        
        if in_memory:
            self.LR_img = [np.array(Image.open(os.path.join(self.LR_path, lr)).convert("RGB")).astype(np.uint8) for lr in self.LR_img]
            self.GT_img = [np.array(Image.open(os.path.join(self.GT_path, gt)).convert("RGB")).astype(np.uint8) for gt in self.GT_img]
        
    def __len__(self):
        
        return len(self.LR_img)
        
    def __getitem__(self, i):
        
        img_item = {}
        
        if self.in_memory:
            GT = self.GT_img[i].astype(np.float32)
            LR = self.LR_img[i].astype(np.float32)
            
        else:
            GT = np.array(Image.open(os.path.join(self.GT_path, self.GT_img[i])).convert("RGB"))
            LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_img[i])).convert("RGB"))

        img_item['GT'] = (GT / 127.5) - 1.0
        img_item['LR'] = (LR / 127.5) - 1.0
                
        if self.transform is not None:
            img_item = self.transform(img_item)
            
        img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32)
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)
        
        return img_item
    

class mydata_dsa_srgan(Dataset):
    def __init__(self, LR_path, GT_path, in_memory = True, transform = None):
        
        self.LR_path = LR_path
        self.GT_path = GT_path
        self.in_memory = in_memory
        self.transform = transform

        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        self.LR_img = sorted([
            f for f in os.listdir(LR_path)
            if f.lower().endswith(valid_ext)
        ])

        self.GT_img = sorted([
            f for f in os.listdir(GT_path)
            if f.lower().endswith(valid_ext)
        ])
        
        if in_memory:
            self.LR_img = [np.array(Image.open(os.path.join(self.LR_path, lr)).convert("L")).astype(np.uint8) for lr in self.LR_img]
            self.GT_img = [np.array(Image.open(os.path.join(self.GT_path, gt)).convert("L")).astype(np.uint8) for gt in self.GT_img]
        
    def __len__(self):
        
        return len(self.LR_img)
        
    def __getitem__(self, i):
        
        img_item = {}
        
        if self.in_memory:
            GT = self.GT_img[i].astype(np.float32)
            LR = self.LR_img[i].astype(np.float32)
            
        else:
            GT = np.array(Image.open(os.path.join(self.GT_path, self.GT_img[i])).convert("L"))
            LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_img[i])).convert("L"))

        img_item['GT'] = (GT / 127.5) - 1.0
        img_item['LR'] = (LR / 127.5) - 1.0

        img_item['GT'] = img_item['GT'][..., np.newaxis]
        img_item['LR'] = img_item['LR'][..., np.newaxis]
            
        if self.transform is not None:
            img_item = self.transform(img_item)
            
        img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32)
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)
        
        return img_item    

class testOnly_data(Dataset):
    def __init__(self, LR_path, in_memory = True, transform = None):
        
        self.LR_path = LR_path
        self.LR_img = sorted(os.listdir(LR_path))
        self.in_memory = in_memory
        if in_memory:
            self.LR_img = [np.array(Image.open(os.path.join(self.LR_path, lr))) for lr in self.LR_img]
        
    def __len__(self):
        
        return len(self.LR_img)
        
    def __getitem__(self, i):
        
        img_item = {}
        
        if self.in_memory:
            LR = self.LR_img[i]
            
        else:
            LR = np.array(
            Image.open(os.path.join(self.LR_path, self.LR_img[i])).convert("RGB")
        ).astype(np.float32)

        img_item['LR'] = (LR / 127.5) - 1.0                
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)
        
        return img_item

class testOnly_data_dsa_srgan(Dataset):
    def __init__(self, LR_path, in_memory=True, transform=None):

        self.LR_path = LR_path
        self.LR_img = sorted([
            f for f in os.listdir(self.LR_path)
            if os.path.isfile(os.path.join(self.LR_path, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ])
        self.in_memory = in_memory

        if in_memory:
            self.LR_img = [
                np.array(
                    Image.open(os.path.join(self.LR_path, lr)).convert("L")
                ).astype(np.uint8)
                for lr in self.LR_img
            ]

    def __len__(self):
        return len(self.LR_img)

    def __getitem__(self, i):

        if self.in_memory:
            LR = self.LR_img[i].astype(np.float32)
        else:
            LR = np.array(
                Image.open(os.path.join(self.LR_path, self.LR_img[i])).convert("L")
            ).astype(np.float32)

        LR = (LR / 127.5) - 1.0
        LR = LR[..., np.newaxis]
        LR = LR.transpose(2, 0, 1).astype(np.float32)

        return {"LR": LR}


class crop_srgan(object):
    def __init__(self, scale, patch_size):
        
        self.scale = scale
        self.patch_size = patch_size
        
    def __call__(self, sample):
        LR_img, GT_img = sample['LR'], sample['GT']
        ih, iw = LR_img.shape[:2]
        
        ix = random.randrange(0, iw - self.patch_size +1)
        iy = random.randrange(0, ih - self.patch_size +1)
        
        tx = ix * self.scale
        ty = iy * self.scale
        
        LR_patch = LR_img[iy : iy + self.patch_size, ix : ix + self.patch_size]
        GT_patch = GT_img[ty : ty + (self.scale * self.patch_size), tx : tx + (self.scale * self.patch_size)]
        
        return {'LR' : LR_patch, 'GT' : GT_patch}

class crop_rldcnn(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, sample):
        LR_img, GT_img = sample['LR'], sample['GT']

        h, w = LR_img.shape[:2]

        x = random.randrange(0, w - self.patch_size + 1)
        y = random.randrange(0, h - self.patch_size + 1)

        LR_patch = LR_img[y:y+self.patch_size, x:x+self.patch_size]
        GT_patch = GT_img[y:y+self.patch_size, x:x+self.patch_size]

        return {
            'LR': LR_patch,
            'GT': GT_patch
        }
    

class augmentation_srgan(object):
    
    def __call__(self, sample):
        LR_img, GT_img = sample['LR'], sample['GT']
        
        hor_flip = random.randrange(0,2)
        # ver_flip = random.randrange(0,2)
        # rot = random.randrange(0,2)
    
        if hor_flip:
            temp_LR = np.fliplr(LR_img)
            LR_img = temp_LR.copy()
            temp_GT = np.fliplr(GT_img)
            GT_img = temp_GT.copy()
            
            del temp_LR, temp_GT
        
        
        
        return {'LR' : LR_img, 'GT' : GT_img}
        
class augmentation_rldcnn(object):

    def __call__(self, sample):
        LR_img, GT_img = sample['LR'], sample['GT']

        # Horizontal Flip
        if random.random() < 0.5:
            LR_img = np.fliplr(LR_img).copy()
            GT_img = np.fliplr(GT_img).copy()

  
        # Random Rotation 
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            h, w = LR_img.shape[:2]
            M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

            LR_img = cv2.warpAffine(
                LR_img, M_rot, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            GT_img = cv2.warpAffine(
                GT_img, M_rot, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

        # Random Translation 
        if random.random() < 0.5:
            h, w = LR_img.shape[:2]
            tx = random.uniform(-0.05, 0.05) * w
            ty = random.uniform(-0.05, 0.05) * h

            M_trans = np.float32([
                [1, 0, tx],
                [0, 1, ty]
            ])

            LR_img = cv2.warpAffine(
                LR_img, M_trans, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            GT_img = cv2.warpAffine(
                GT_img, M_trans, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

        return {
            'LR': LR_img,
            'GT': GT_img
        }

