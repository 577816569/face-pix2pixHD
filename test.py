  
import os
import sys
import cv2
import time
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from data.base_dataset import BaseDataset, get_params, get_transform, normalize



opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
model = create_model(opt)  

labelpth='./test/label/1.png'
imgpth='./test/image/1.jpg'
respth='./test/result'

if not os.path.exists(respth):
    os.makedirs(respth)

params = get_params(opt, (128,128))
transform_mask = get_transform(opt, params, method=Image.NEAREST, normalize=False)
transform_image = get_transform(opt, params)
image=Image.open(imgpth)
mask=Image.open(labelpth)
mask = transform_mask(mask)*255.0
image = transform_image(image)
start_t = time.time()
generated = model.inference(torch.FloatTensor([mask.numpy()]), torch.FloatTensor([image.numpy()]))   
end_t = time.time()
save_image((generated.data[0] + 1) / 2,respth+'/1.jpg')
            
    
