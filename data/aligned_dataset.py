import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, make_dataset_test
from PIL import Image
import torch
import numpy as np
import glob

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class AlignedDataset(BaseDataset):
    def initialize(self,opt):
        self.opt = opt
        self.root = opt.dataroot 
        self.files_A = sorted(glob.glob(os.path.join(self.root, "train_label") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(self.root, "train_img") + "/*.*"))
        self.files_BR = sorted(glob.glob(os.path.join(self.root, "train_img") + "/*.*"))
   
    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_B = Image.open(self.files_B[index % len(self.files_B)])
        image_BR = Image.open(self.files_B[index % len(self.files_BR)])
        params = get_params(self.opt, image_A.size)
        transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        A_tensor = transform_A(image_A) * 255.0
        image_B = to_rgb(image_B)
        image_BR = to_rgb(image_BR)
        transform_B = get_transform(self.opt, params)      
        B_tensor = transform_B(image_B)
        BR_tensor = transform_B(image_BR)
        input_dict = { 'label': A_tensor,   'image': B_tensor, 'image_ref': BR_tensor, 'path': self.root}
        return input_dict
   

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def name(self):
        return 'AlignedDataset'