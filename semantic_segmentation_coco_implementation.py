#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys
import torch
import torchvision.transforms as tvt
from PIL import Image
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tvt
from torchvision import utils
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import random
import time
import skimage.io as io
from pycocotools.coco import COCO
import numpy as np
import copy
from scipy.ndimage import zoom
import torch.optim as optim
from DLStudio import *


# In[ ]:


train_ann = COCO('/Users/avnishkanungo/Desktop/coco-dataset/train2017/train2017/annotations/instances_train2017.json')


# In[ ]:


test_ann = COCO('/Users/avnishkanungo/Desktop/coco-dataset/train2017/train2017/annotations/instances_val2017.json')


# In[ ]:


root_directory_train = '/Users/avnishkanungo/Desktop/coco-dataset/train2017/train2017/train2017'
root_directory_test = '/Users/avnishkanungo/Desktop/coco-dataset/train2017/train2017/val2017'
classes = ['cake', 'dog', 'motorcycle']


# In[ ]:


def getImgAndMask(coco, path, classes):
    l = coco.getImgIds() #list(coco.imgs.keys())
    masks = []
    img = []
    class_ids = [coco.getCatIds(catNms=[class_name])[0] for class_name in classes]
    for i in l:
        x = coco.getAnnIds(i)
        y = coco.loadAnns(x)
        if y:
            if y[0]['category_id'] in class_ids:
                image_info = coco.loadImgs(i)[0]
                mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
                
                if len(y) == 1:
                        bbox = y[0]['bbox']
                        bbox_size = max(bbox[2], bbox[3])
                    
                        if bbox_size >= 200:
                            image_path = f"{path}/{image_info['file_name']}"
                            image = Image.open(image_path).convert("RGB")
                            image = image.resize((256,256))
                            img.append(image)
                            
                            for ann in y:
                                mask += coco.annToMask(ann)
                                mask = zoom(mask, (256/image_info["height"], 256/image_info["width"]), order=1)
                                masks.append(mask)

    return masks, img


# In[ ]:


class CustomCocoDataset(Dataset):
    def __init__(self, masks, images, transform=None):
        self.masks = masks
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        mask = self.masks[idx]
        image = self.images[idx]

        if self.transform:
            
            mask = torch.tensor(mask, dtype=torch.float32)
            image, transformed_mask = self.transform(image, mask)

        return transformed_mask, image


# In[ ]:


class CustomTransform:
    def __init__(self, image_transforms):
        self.image_transforms = image_transforms

    def __call__(self, image, mask):
        transformed_image = self.image_transforms(image)
        transformed_mask = torch.unsqueeze(mask, 0)  # Add channel dimension
        return transformed_image, transformed_mask


# In[ ]:


image_transform = tvt.Compose([
    tvt.ToTensor(),
    tvt.Resize(size=(256,256))
])


# In[ ]:


transform = CustomTransform(image_transform)


# In[ ]:


test_dataset = CustomCocoDataset(im_masks_test, im_img_test, transform=transform)
train_dataset = CustomCocoDataset(im_masks_train, im_img_train, transform=transform)


# In[ ]:


test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)


# In[ ]:


class SkipBlockDN(nn.Module):
                
                def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                    super(SkipBlockDN, self).__init__()
                    self.downsample = downsample
                    self.skip_connections = skip_connections
                    self.in_ch = in_ch
                    self.out_ch = out_ch
                    self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                    self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                    self.bn1 = nn.BatchNorm2d(out_ch)
                    self.bn2 = nn.BatchNorm2d(out_ch)
                    if downsample:
                        self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
                def forward(self, x):
                    identity = x                                     
                    out = self.convo1(x)                              
                    out = self.bn1(out)                              
                    out = nn.functional.relu(out)
                    if self.in_ch == self.out_ch:
                        out = self.convo2(out)                              
                        out = self.bn2(out)                              
                        out = nn.functional.relu(out)
                    if self.downsample:
                        out = self.downsampler(out)
                        identity = self.downsampler(identity)
                    if self.skip_connections:
                        if self.in_ch == self.out_ch:
                            out = out + identity
                        else:
                            out = out + torch.cat((identity, identity), dim=1) 
                    return out
    
    
class SkipBlockUP(nn.Module):
                
                def __init__(self, in_ch, out_ch, upsample=False, skip_connections=True):
                    super(SkipBlockUP, self).__init__()
                    self.upsample = upsample
                    self.skip_connections = skip_connections
                    self.in_ch = in_ch
                    self.out_ch = out_ch
                    self.convoT1 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
                    self.convoT2 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
                    self.bn1 = nn.BatchNorm2d(out_ch)
                    self.bn2 = nn.BatchNorm2d(out_ch)
                    if upsample:
                        self.upsampler = nn.ConvTranspose2d(in_ch, out_ch, 1, stride=2, dilation=2, output_padding=1, padding=0)
                def forward(self, x):
                    identity = x                                     
                    out = self.convoT1(x)                              
                    out = self.bn1(out)                              
                    out = nn.functional.relu(out)
                    out  =  nn.ReLU(inplace=False)(out)            
                    if self.in_ch == self.out_ch:
                        out = self.convoT2(out)                              
                        out = self.bn2(out)                              
                        out = nn.functional.relu(out)
                    if self.upsample:
                        out = self.upsampler(out)
                        identity = self.upsampler(identity)
                    if self.skip_connections:
                        if self.in_ch == self.out_ch:
                            out = out + identity                              
                        else:
                            out = out + identity[:,self.out_ch:,:,:]
                    return out


# In[ ]:


class mUnet(nn.Module):
               
    def __init__(self, skip_connections=True, depth=16):
                    super(mUnet, self).__init__()
                    self.depth = depth // 2
                    self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
                    ##  For the DN arm of the U:
                    self.bn1DN  = nn.BatchNorm2d(64)
                    self.bn2DN  = nn.BatchNorm2d(128)
                    self.skip64DN_arr = nn.ModuleList()
                    for i in range(self.depth):
                        self.skip64DN_arr.append(SkipBlockDN(64, 64, skip_connections=skip_connections))
                    self.skip64dsDN = SkipBlockDN(64, 64,   downsample=True, skip_connections=skip_connections)
                    self.skip64to128DN = SkipBlockDN(64, 128, skip_connections=skip_connections )
                    self.skip128DN_arr = nn.ModuleList()
                    for i in range(self.depth):
                        self.skip128DN_arr.append(SkipBlockDN(128, 128, skip_connections=skip_connections))
                    self.skip128dsDN = SkipBlockDN(128,128, downsample=True, skip_connections=skip_connections)
                    ##  For the UP arm of the U:
                    self.bn1UP  = nn.BatchNorm2d(128)
                    self.bn2UP  = nn.BatchNorm2d(64)
                    self.skip64UP_arr = nn.ModuleList()
                    for i in range(self.depth):
                        self.skip64UP_arr.append(SkipBlockUP(64, 64, skip_connections=skip_connections))
                    self.skip64usUP = SkipBlockUP(64, 64, upsample=True, skip_connections=skip_connections)
                    self.skip128to64UP = SkipBlockUP(128, 64, skip_connections=skip_connections )
                    self.skip128UP_arr = nn.ModuleList()
                    for i in range(self.depth):
                        self.skip128UP_arr.append(SkipBlockUP(128, 128, skip_connections=skip_connections))
                    self.skip128usUP = SkipBlockUP(128,128, upsample=True, skip_connections=skip_connections)
                    self.conv_out = nn.ConvTranspose2d(64, 1, 3, stride=2,dilation=2,output_padding=1,padding=2) #remember to change the output channel to 3 later
    
    def forward(self, x):
                    ##  Going down to the bottom of the U:
                    x = nn.MaxPool2d(2,2)(nn.functional.relu(self.conv_in(x)))          
                    for i,skip64 in enumerate(self.skip64DN_arr[:self.depth//4]):
                        x = skip64(x)                
            
                    num_channels_to_save1 = x.shape[1] // 2
                    save_for_upside_1 = x[:,:num_channels_to_save1,:,:].clone()
                    x = self.skip64dsDN(x)
                    for i,skip64 in enumerate(self.skip64DN_arr[self.depth//4:]):
                        x = skip64(x)                
                    x = self.bn1DN(x)
                    num_channels_to_save2 = x.shape[1] // 2
                    save_for_upside_2 = x[:,:num_channels_to_save2,:,:].clone()
                    x = self.skip64to128DN(x)
                    for i,skip128 in enumerate(self.skip128DN_arr[:self.depth//4]):
                        x = skip128(x)                
            
                    x = self.bn2DN(x)
                    num_channels_to_save3 = x.shape[1] // 2
                    save_for_upside_3 = x[:,:num_channels_to_save3,:,:].clone()
                    for i,skip128 in enumerate(self.skip128DN_arr[self.depth//4:]):
                        x = skip128(x)                
                    x = self.skip128dsDN(x)
                    ## Coming up from the bottom of U on the other side:
                    x = self.skip128usUP(x)          
                    for i,skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
                        x = skip128(x)                
                    x[:,:num_channels_to_save3,:,:] =  save_for_upside_3
                    x = self.bn1UP(x)
                    for i,skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
                        x = skip128(x)                
                    x = self.skip128to64UP(x)
                    for i,skip64 in enumerate(self.skip64UP_arr[self.depth//4:]):
                        x = skip64(x)                
                    x[:,:num_channels_to_save2,:,:] =  save_for_upside_2
                    x = self.bn2UP(x)
                    x = self.skip64usUP(x)
                    for i,skip64 in enumerate(self.skip64UP_arr[:self.depth//4]):
                        x = skip64(x)                
                    x[:,:num_channels_to_save1,:,:] =  save_for_upside_1
                    x = self.conv_out(x)
                    return x


# In[ ]:


class MSEPlusDiceLoss1(nn.Module):
    def __init__(self, alpha=0.5, epsilon=1e-2):
        super(MSEPlusDiceLoss1, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, output, mask_tensor):
        total_loss = 0
        composite_loss = torch.zeros(1, 4, requires_grad=True)
        dice_loss = torch.zeros(1, 4, requires_grad=True)
        mse_loss = torch.zeros(1, 4, requires_grad=True)

        for idx in range(min(output.shape[0], mask_tensor.shape[0])):
            mask = mask_tensor[idx, 0, :, :]
            output_mask = output[idx, 0, :, :]

            # Dice Loss
            if torch.sum(mask)+torch.sum(output_mask)>0:
                intersection = torch.sum(output_mask * mask)
                union = torch.sum(output_mask) + torch.sum(mask) + self.epsilon
                dice_coefficient = (2. * intersection + self.epsilon) / union
                dice_loss_copy = dice_loss.clone()
                dice_loss_copy[0,idx] = 1 - dice_coefficient
                dice_loss = dice_loss_copy

            if torch.sum(mask)>0 and torch.sum(output_mask)>0:
                # MSE Loss
                mse_loss_copy = mse_loss.clone()
                mse_loss_copy[0, idx] = F.mse_loss(output[idx], mask_tensor[idx])
                mse_loss = mse_loss_copy

            if torch.sum(mse_loss)>0 and torch.sum(dice_loss)>0:
                # Composite Loss
                composite_loss_copy = composite_loss.clone()
                composite_loss_copy[0, idx]= torch.sum(mse_loss) + 30*torch.sum(dice_loss)
                composite_loss = composite_loss_copy
                
        average_loss = torch.sum(composite_loss) / min(output.shape[0], mask_tensor.shape[0])
        
        return average_loss


# In[ ]:


def run_code_for_training_for_semantic_segmentation(net, data_loader):        
                net = copy.deepcopy(net)
                net = net.to(torch.device("cpu"))
                criterion1 = MSEPlusDiceLoss1(4, alpha=0.5, epsilon = 1e-3)
                optimizer = optim.SGD(net.parameters(), 
                             lr=1e-4, momentum=0.9)
                start_time = time.perf_counter()
                running_loss = []
                for epoch in range(10):  
                    print("")
                    running_loss_segmentation = 0.0
                    for i, data in enumerate(data_loader):    
                        mask_tensor, im_tensor = data
                        # print(im_tensor.shape)
                        im_tensor   = im_tensor.to(torch.device("cpu"))
                        mask_tensor = mask_tensor.to(torch.device("cpu"))
                        mask_tensor = mask_tensor.type(torch.FloatTensor)
                        optimizer.zero_grad()
                        output = net(im_tensor) 
                        segmentation_loss= criterion1(output, mask_tensor)
                        segmentation_loss.requires_grad
                        output.requires_grad
                        im_tensor.requires_grad = True
                        mask_tensor.requires_grad = True 
                        segmentation_loss.backward()
                        optimizer.step()
                        running_loss_segmentation += segmentation_loss.item()    
                        if i%100==99:    
                            current_time = time.perf_counter()
                            elapsed_time = current_time - start_time
                            avg_loss_segmentation = running_loss_segmentation / float(100)
                            print("[epoch=%d/%d, iter=%4d  elapsed_time=%3d secs]   Combined loss: %f" % (epoch+1, 10, i+1, elapsed_time, avg_loss_segmentation ))
                            running_loss.append(avg_loss_segmentation)
                            running_loss_segmentation = 0.0
                print("\nFinished Training\n")
                return running_loss
                torch.save(net.state_dict(), '/Users/avnishkanungo/Desktop/coco-dataset/train2017/hw7_model_save/save_model_loss_1')


# In[ ]:


def run_code_for_testing_semantic_segmentation( net, dataloader, path):
                net.load_state_dict(torch.load(path))
                batch_size = 4
                image_size = [256,256]
                with torch.no_grad():
                    for i, data in enumerate(dataloader):
                        mask_tensor,im_tensor = data
                        outputs = net(im_tensor)
                        fig = plt.figure(figsize=(10, 7)) 
  
                        # setting values to rows and column variables 
                        rows = 2
                        columns = 2
                        
                        fig.add_subplot(rows, columns, 1) 
  
                        # showing image 
                        plt.imshow(outputs[i-1].permute(1,2,0)) 
                        plt.axis('off') 
                        plt.title("Input Image") 
                          
                        # Adds a subplot at the 2nd position 
                        fig.add_subplot(rows, columns, 2) 
                          
                        # showing image 
                        plt.imshow(im_tensor[i-1].permute(1,2,0)) 
                        plt.axis('off') 
                        plt.title("Output Image") 
                          
                        # Adds a subplot at the 3rd position 
                        fig.add_subplot(rows, columns, 3) 
                          
                        # showing image 
                        plt.imshow(mask_tensor[i-1].permute(1,2,0)) 
                        plt.axis('off') 
                        plt.title("Image Mask") 


# In[ ]:


model = mUnet(skip_connections=True, depth=16)

combined_loss = run_code_for_training_for_semantic_segmentation(model, train_dataloader, '/Users/avnishkanungo/Desktop/coco-dataset/train2017/hw7_model_save/saved_model_combined')

run_code_for_testing_semantic_segmentation(model, test_dataloader, '/Users/avnishkanungo/Desktop/coco-dataset/train2017/hw7_model_save/saved_model_combined')

