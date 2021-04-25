## Author: Chris Reekie CJR2198

## File contains the custom dataset class for image batch and preparation

from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import pandas as pd
from PIL import Image

## Dataset Declaration for melanoma data
class MelaDS (Dataset):
    def __init__(self, image_folder, dataframe, global_res=224, high_res=None, train=True):
        # Input train dataframe, global and high resolution
        self.image_folder = image_folder # folder containing images
        self.dataframe = dataframe #dataframe
        self.train = train #training flag
        self.global_res = global_res #global 'low res' image resolution
        self.high_res = high_res #high resolution for stochastic conv

        ## If doing stochastic conv, augmentation is done to the high res image only
        ## low res image is then derived from high res image after augmentation (this is important)
        ## augmentations (no shear)
        if self.high_res != None:
            self.high_res_train = transforms.Compose([
                transforms.RandomResizedCrop(size=high_res, scale=(0.6,1.0)),
                transforms.RandomHorizontalFlip(), #random horizontal flip
                transforms.RandomVerticalFlip(), #random vertical flip
                transforms.RandomRotation(180) #random rotation
            ])

        ## normalize for effnet
        self.high_res_prepare = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ##standard normalization for effnet
        ])

        ## prep for low res, using both high and low res
        self.low_res_prepare = transforms.Compose([
            transforms.Resize(size=global_res),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        ## dedicated augmentation stream if only using low resolution (as with the base effnet runs)
        self.train_augmentation_lr = transforms.Compose([
            transforms.RandomResizedCrop(size=global_res, scale=(0.6,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item_idx):
        # get item function to prepare image for batch
        y = None

        # convert image to PIL for pytorch
        image = Image.open(self.image_folder + '/' + self.dataframe.iloc[item_idx]['filename']+'.jpg') #cv2.imread(self.image_folder + '/' + self.dataframe.iloc[item_idx]['image_name']+'.jpg')
        image = image.convert('RGB')
        if self.train == True:
            y = self.dataframe.iloc[item_idx]['target'].astype('float')

        # if doing stochastic run, prepare high res image, then create low res image from high res
        # return both
        if self.high_res != None:
            hr_image = self.high_res_train(image)
            lr_image = self.low_res_prepare(hr_image)
            hr_image = self.high_res_prepare(hr_image)
            return (lr_image, hr_image, y)

        # if doing standard run just prepare the low res image
        if self.high_res == None:
            image = self.train_augmentation_lr (image)
            return (image, y)
