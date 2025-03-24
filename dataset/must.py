"""
get data loaders
"""
from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import mmcv
import torch


def get_data_folder():
    """
    return server-dependent path to store the data
    """

    return '/data3/PublicDataset/Public/MUST-BIT'

class MustDataset():
    """
    Dataset for MUST
    """

    def __init__(self, root, transform=None):
        # super().__init__(root=root, transform=transform, )
        self.root = root
        self.transform = transform

        self.img_list = []
        self.jpg_path = os.path.join(self.root, 'HSICV')
        self.rgb_path = os.path.join(self.root, 'FalseColor')
        test_jpg_path = os.path.join(self.jpg_path, 'test')
        train_jpg_path = os.path.join(self.jpg_path, 'train')

        # walk the test path
        for dir_path, dir_names, _ in os.walk(test_jpg_path):
            for vid in dir_names:
                img_list = sorted([os.path.join('test', vid, img) for img in os.listdir(os.path.join(dir_path, vid)) if img.endswith('.jpg')])
                assert len(img_list)%3 == 0
                # sample with interval = 5
                img_list = img_list[::15]
                self.img_list += img_list
        
        for dir_path, dir_names, _ in os.walk(train_jpg_path):
            for vid in dir_names:
                img_list = sorted([os.path.join('train', vid, img) for img in os.listdir(os.path.join(dir_path, vid)) if img.endswith('.jpg')])
                assert len(img_list)%3 == 0
                # sample with interval = 5
                img_list = img_list[::15]
                self.img_list += img_list
             
        pass

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img_real_name = img_path.replace('_img1.jpg', '')

        rgb_img = os.path.join(self.rgb_path, f'{img_real_name}.png')

        jpg_im1 = os.path.join(self.jpg_path, f'{img_real_name}_img1.jpg')
        jpg_im2 = os.path.join(self.jpg_path, f'{img_real_name}_img2.jpg')
        jpg_im3 = os.path.join(self.jpg_path, f'{img_real_name}_img3.jpg')


        rgb_img = mmcv.imread(rgb_img, backend='cv2', channel_order='rgb')       
        
        im1 = mmcv.imread(jpg_im1, backend='cv2', channel_order='rgb')
        im2 = mmcv.imread(jpg_im2, backend='cv2', channel_order='rgb')
        im3 = mmcv.imread(jpg_im3, backend='cv2', channel_order='rgb')
        im = np.concatenate((im1[:,:,2:3], im1[:,:,1:2], im1[:,:,0:1], im2[:,:,2:3], im2[:,:,1:2], im2[:,:,0:1], im3[:,:,1:2], im3[:,:,0:1]), axis=2)

        img_rgb_8ch = np.concatenate((rgb_img, im), axis=2)

        img_rgb_8ch = torch.as_tensor(img_rgb_8ch, dtype=torch.float32).permute(2, 0, 1)
        img_rgb_8ch /= 255.0

        img_rgb_8ch = self.transform(img_rgb_8ch)

        img_rgb = img_rgb_8ch[:3, :, :]
        im_8ch = img_rgb_8ch[3:, :, :]
        return img_rgb, im_8ch, idx

    def __len__(self):
        return len(self.img_list)

def get_dataloader_sample(dataset='must', batch_size=128, num_workers=8):
    """Data Loader for ImageNet"""

    data_folder = get_data_folder()

    # add data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.449, 0.328, 0.326, 0.322, 0.328, 0.324, 0.326, 0.327], std=[0.229, 0.224, 0.225, 0.236, 0.173, 0.171, 0.170, 0.176, 0.157, 0.162, 0.164])
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.RandomResizedCrop((768,992)),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        normalize,
    ])

    train_dataset = MustDataset(data_folder, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    print('num_samples', len(train_dataset))
    return train_loader, len(train_dataset)