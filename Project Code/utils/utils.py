from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import pandas as pd
from PIL import Image

class MelaDS (Dataset):
    def __init__(self, image_folder, dataframe, global_res=224, high_res=None, train=True):
        self.image_folder = image_folder
        self.dataframe = dataframe
        self.train = train
        self.global_res = 224
        self.high_res = high_res

        if self.high_res != None:
            self.high_res_train = transforms.Compose([
                transforms.RandomResizedCrop(size=high_res, scale=(0.6,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180)
            ])

        self.high_res_prepare = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.low_res_prepare = transforms.Compose([
            transforms.Resize(size=global_res),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
        y = None

        image = Image.open(self.image_folder + '/' + self.dataframe.iloc[item_idx]['filename']+'.jpg') #cv2.imread(self.image_folder + '/' + self.dataframe.iloc[item_idx]['image_name']+'.jpg')
        image = image.convert('RGB')
        if self.train == True:
            y = self.dataframe.iloc[item_idx]['target'].astype('float')

        if self.high_res != None:
            hr_image = self.high_res_train(image)
            lr_image = self.low_res_prepare(hr_image)
            hr_image = self.high_res_prepare(hr_image)
            return (lr_image, hr_image, y)


        if self.high_res == None:
            image = self.train_augmentation_lr (image)
            return (image, y)

        ## implement test

class MelaDS_CPU (Dataset):
    def __init__(self, image_folder, dataframe, global_res=224, high_res=None, train=True, high_res_channels = 32):
        self.image_folder = image_folder
        self.dataframe = dataframe
        self.train = train
        self.global_res = 224
        self.high_res = high_res
        self.high_res_channels = high_res_channels
        self.slice_size = 225

        if self.high_res != None:
            self.high_res_train = transforms.Compose([
                transforms.RandomResizedCrop(size=high_res, scale=(0.6,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180)
            ])

        self.high_res_prepare = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.low_res_prepare = transforms.Compose([
            transforms.Resize(size=global_res),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
        y = None

        image = Image.open(self.image_folder + '/' + self.dataframe.iloc[item_idx]['filename']+'.jpg') #cv2.imread(self.image_folder + '/' + self.dataframe.iloc[item_idx]['image_name']+'.jpg')
        image = image.convert('RGB')
        if self.train == True:
            y = self.dataframe.iloc[item_idx]['target'].astype('float')

        if self.high_res != None:
            RandomWindowStart = torch.randint(low=self.slice_size, high=image.shape[3], size=(self.out_channels,)) #(filters, images)
            hr_output = torch.zeros((self.high_res_channels, 3, self.slice_size, self.slice_size), dtype=torch.float32)
            hr_image = self.high_res_prepare(self.high_res_train(image))

            for i in range(0,self.high_res_channels):
                end = RandomWindowStart[idx_k][idx_i]  # end position of random window
                start = end - self.slice_size  # start position of random window
                hr_output[i,:,:,:] = hr_image[idx_i, :, start:end, start:end].unsqueeze(0)

            lr_image = self.low_res_prepare(self.high_res_train(image))

            return (lr_image, hr_output, y)


        if self.high_res == None:
            image = self.train_augmentation_lr (image)
            return (image, y)

        ## implement test

