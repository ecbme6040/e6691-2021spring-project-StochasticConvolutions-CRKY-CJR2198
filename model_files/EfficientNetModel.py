
## Credits / References:
## Official implementation https://github.com/tensorflow/tpu/blob/01574500090fa9c011cb8418c61d442286720211/models/official/efficientnet/efficientnet_model.py#L101-L125
## Discussion https://forums.fast.ai/t/efficientnet/46978/76
## Drop conect rate: https://github.com/tensorflow/tpu/issues/381 0.2

import torch
from torch import nn
from layers import *
from model_architectures import *
from utils import *
import time

class EffNetb0 (nn.Module):
    def __init__(self, image_size, dropout):
        super(EffNetb0, self).__init__()
        self.InvertedResBlocks = nn.ModuleList([])

        ##start architecture
        self.base_conv = nn.Conv2d(in_channels=3, out_channels=32,
                                   stride=2, kernel_size=3, padding=1)
        self.base_bn = nn.BatchNorm2d(num_features=32)

        ##output size of first conv layer calculation (to calc pad in static padding calc)
        input_size = feature_dim_after_conv(image_size, stride=2)

        ### define architecture blocks ###

        block_list = GetArchitectureBlockList('b0')

        self.InvertedResBlocks = build_module_list (block_list, input_size)

        #Set drop connect rate now we have created all the blocks
        drop_rate = 0.2
        for idx, block in enumerate(self.InvertedResBlocks):
            block.DropConnect.set_drop_connect_rate(rate=((drop_rate)*float(idx)/len(self.InvertedResBlocks)))

        self.TopConv = Convolution2dSamePadding(in_channels=320, out_channels=1280,
                                                kernel_size=1, stride=1, input_size=image_size)
        self.TopBN = nn.BatchNorm2d(num_features=1280)

        self.TopAvgPool = nn.AdaptiveAvgPool2d(1)
        self.TopDrop = nn.Dropout(dropout)
        self.Top = nn.Linear(1280, 1)#num classes

    def forward (self, input_batch):

        x = nn.SiLU()((self.base_bn(self.base_conv(input_batch))))

        for idx, resblock in enumerate(self.InvertedResBlocks):
            x = resblock(x)

        x = self.TopBN(self.TopConv(x))
        x = torch.flatten(self.TopAvgPool(x),1)
        x = self.TopDrop(x)
        x = self.Top(x)

        return x

### EfficientNet b0 architecture with 64 channels ###
class EffNetb064 (nn.Module):
    def __init__(self, image_size, dropout):
        super(EffNetb064, self).__init__()
        self.InvertedResBlocks = nn.ModuleList([])

        ##start architecture
        self.base_conv = nn.Conv2d(in_channels=3, out_channels=64,
                                   stride=2, kernel_size=3, padding=1)
        self.base_bn = nn.BatchNorm2d(num_features=64)

        ##output size of first conv layer calculation (to calc pad in static padding calc)
        input_size = feature_dim_after_conv(image_size, stride=2)

        ### get architecture blocks ###
        block_list = GetArchitectureBlockList('b064')

        self.InvertedResBlocks = build_module_list (block_list, input_size)

        #Set drop connect rate now we have created all the blocks
        drop_rate = 0.2
        for idx, block in enumerate(self.InvertedResBlocks):
            block.DropConnect.set_drop_connect_rate(rate=((drop_rate)*float(idx)/len(self.InvertedResBlocks)))

        self.TopConv = Convolution2dSamePadding(in_channels=320, out_channels=1280,
                                                kernel_size=1, stride=1, input_size=image_size)
        self.TopBN = nn.BatchNorm2d(num_features=1280)

        self.TopAvgPool = nn.AdaptiveAvgPool2d(1)
        self.TopDrop = nn.Dropout(dropout)
        self.Top = nn.Linear(1280, 1)#num classes

    def forward (self, input_batch):

        x = nn.SiLU()((self.base_bn(self.base_conv(input_batch))))

        for idx, resblock in enumerate(self.InvertedResBlocks):
            x = resblock(x)

        x = self.TopBN(self.TopConv(x))
        x = torch.flatten(self.TopAvgPool(x),1)
        x = self.TopDrop(x)
        x = self.Top(x)

        return x

### Efficientnet b0 with stochastic stem ###
class EffNetb0Stochastic (nn.Module):
    def __init__(self, image_size, image_size_hr, dropout):
        super(EffNetb0Stochastic, self).__init__()
        self.InvertedResBlocks = nn.ModuleList([])

        ##start architecture
        ##stem low res
        self.base_conv = nn.Conv2d(in_channels=3, out_channels=32,
                                   stride=2, kernel_size=3, padding=1)
        self.base_bn = nn.BatchNorm2d(num_features=32)

        ##stem high res
        self.base_conv_hr = StochasticConv4(stride=2, out_channels=32, kernel_size=3, output_size=(112, 112))
        self.base_bn_hr = nn.BatchNorm2d(num_features=32)


        ##output size of first conv layer calculation (to calc pad in static padding calc)
        input_size = feature_dim_after_conv(image_size, stride=2)


        block_list = GetArchitectureBlockList('b064')

        self.InvertedResBlocks = build_module_list (block_list, input_size)


        #Set drop connect rate now we have created all the blocks
        drop_rate = 0.2
        for idx, block in enumerate(self.InvertedResBlocks):
            block.DropConnect.set_drop_connect_rate(rate=((drop_rate)*float(idx)/len(self.InvertedResBlocks)))

        self.TopConv = Convolution2dSamePadding(in_channels=320, out_channels=1280,
                                                kernel_size=1, stride=1, input_size=image_size)
        self.TopBN = nn.BatchNorm2d(num_features=1280)

        self.TopAvgPool = nn.AdaptiveAvgPool2d(1)
        self.TopDrop = nn.Dropout(dropout)
        self.Top = nn.Linear(1280, 1)#num classes

    def forward (self, input_batch):
        x = nn.SiLU()((self.base_bn(self.base_conv(input_batch[0]))))

        x_hr = nn.SiLU()((self.base_bn_hr(self.base_conv_hr(input_batch[1]))))

        x = torch.cat((x, x_hr), 1)

        for idx, resblock in enumerate(self.InvertedResBlocks):
            x = resblock(x)

        x = self.TopBN(self.TopConv(x))
        x = torch.flatten(self.TopAvgPool(x),1)
        x = self.TopDrop(x)
        x = self.Top(x)

        return x

### EfficientNet b0 architecture with skip connection (3 block layers) ###
class EffNetb0StochasticSkip (nn.Module):
    def __init__(self, image_size, image_size_hr, dropout):
        super(EffNetb0StochasticSkip, self).__init__()
        self.InvertedResBlocks = nn.ModuleList([])
        self.HRInvertedResBlocks = nn.ModuleList([])

        ##start architecture
        ##stem low res
        self.base_conv = nn.Conv2d(in_channels=3, out_channels=32, stride=2, kernel_size=3, padding=1)
        self.base_bn = nn.BatchNorm2d(num_features=32)

        ##stem high res
        self.base_conv_hr = StochasticConv4(stride=2, out_channels=32, kernel_size=3, output_size=(112, 112))
        self.base_bn_hr = nn.BatchNorm2d(num_features=32)


        ##output size of first conv layer calculation (to calc pad in static padding calc)
        input_size = feature_dim_after_conv(image_size, stride=2)

        ### define architecture blocks ###
        block_list = GetArchitectureBlockList('b032')

        self.InvertedResBlocks = build_module_list(block_list, input_size)
        self.HRInvertedResBlocks = build_module_list(block_list, input_size, stop_idx=3)


        #Set drop connect rate now we have created all the blocks
        drop_rate = 0.2
        for idx, block in enumerate(self.InvertedResBlocks):
            block.DropConnect.set_drop_connect_rate(rate=((drop_rate)*float(idx)/len(self.InvertedResBlocks)))
            if idx < len(self.HRInvertedResBlocks):
                self.HRInvertedResBlocks[idx].DropConnect.set_drop_connect_rate(rate=((drop_rate) * float(idx) / len(self.InvertedResBlocks)))

        self.TopConv = Convolution2dSamePadding(in_channels=320, out_channels=1280, kernel_size=1, stride=1, input_size=image_size)
        self.TopBN = nn.BatchNorm2d(num_features=1280)

        self.TopAvgPool = nn.AdaptiveAvgPool2d(1)
        self.TopDrop = nn.Dropout(dropout)
        self.Top = nn.Linear(1280, 1)#num classes

    def forward (self, input_batch):

        x = nn.SiLU()((self.base_bn(self.base_conv(input_batch[0]))))
        x_hr = nn.SiLU()((self.base_bn_hr(self.base_conv_hr(input_batch[1]))))

        for idx, resblock in enumerate(self.InvertedResBlocks):
            x = resblock(x)
            if idx < len(self.HRInvertedResBlocks):
                x_hr = self.HRInvertedResBlocks[idx](x_hr)
            if idx == len(self.HRInvertedResBlocks)-1:
                x = torch.cat((x, x_hr),1)

        x = self.TopBN(self.TopConv(x))
        x = torch.flatten(self.TopAvgPool(x),1)
        x = self.TopDrop(x)
        x = self.Top(x)

        return x


### EfficientNet b0 full split architecture ###
class EffNetb0FullStochasticSplit (nn.Module):
    def __init__(self, image_size, image_size_hr, dropout):
        super(EffNetb0FullStochasticSplit, self).__init__()
        self.InvertedResBlocks = nn.ModuleList([])
        self.HRInvertedResBlocks = nn.ModuleList([])

        ##start architecture
        ##stem low res
        self.base_conv = nn.Conv2d(in_channels=3, out_channels=32, stride=2, kernel_size=3, padding=1)
        self.base_bn = nn.BatchNorm2d(num_features=32)

        ##stem high res
        self.base_conv_hr = StochasticConv4(stride=2, out_channels=32, kernel_size=3, output_size=(112, 112))
        self.base_bn_hr = nn.BatchNorm2d(num_features=32)


        ##output size of first conv layer calculation (to calc pad in static padding calc)
        input_size = feature_dim_after_conv(image_size, stride=2)

        ### define architecture blocks ###
        block_list = GetArchitectureBlockList('b032')

        ### Full split uses 2 full backbones ###
        self.InvertedResBlocks = build_module_list(block_list, input_size)
        self.HRInvertedResBlocks = build_module_list(block_list, input_size)

        #Set drop connect rate now we have created all the blocks
        drop_rate = 0.2
        for idx, block in enumerate(self.InvertedResBlocks):
            block.DropConnect.set_drop_connect_rate(rate=((drop_rate)*float(idx)/len(self.InvertedResBlocks)))
            if idx < len(self.HRInvertedResBlocks):
                self.HRInvertedResBlocks[idx].DropConnect.set_drop_connect_rate(rate=((drop_rate) * float(idx) / len(self.InvertedResBlocks)))

        ## combine both at the top of the architecture
        self.TopConv_lr = Convolution2dSamePadding(in_channels=320, out_channels=1280, kernel_size=1, stride=1, input_size=image_size)
        self.TopBN_lr = nn.BatchNorm2d(num_features=1280)

        self.TopConv_hr = Convolution2dSamePadding(in_channels=320, out_channels=1280, kernel_size=1, stride=1,
                                                   input_size=image_size)
        self.TopBN_hr = nn.BatchNorm2d(num_features=1280)

        self.TopAvgPool = nn.AdaptiveAvgPool2d(1)
        self.TopAvgPool_hr = nn.AdaptiveAvgPool2d(1)

        self.TopDrop = nn.Dropout(dropout)
        self.Top = nn.Linear(1280 * 2, 1)#num classes

    def forward (self, input_batch):

        x = nn.SiLU()((self.base_bn(self.base_conv(input_batch[0]))))
        x_hr = nn.SiLU()((self.base_bn_hr(self.base_conv_hr(input_batch[1]))))

        for idx, resblock in enumerate(self.InvertedResBlocks):
            x = resblock(x)
            x_hr = self.HRInvertedResBlocks[idx](x_hr)

        x = self.TopBN(self.TopConv(x))
        x_hr = self.TopBN_hr(self.TopConv_hr(x_hr))
        x = torch.flatten(self.TopAvgPool(x), 1)
        x_hr = torch.flatten(self.TopAvgPool_hr(x_hr),1)
        x = torch.cat((x, x_hr), 1)

        x = self.TopDrop(x)
        x = self.Top(x)

        return x


### EfficientNet b0 architecture with only stochastic stem ###
class EffNetb0PureStochastic (nn.Module):
    def __init__(self, image_size, image_size_hr, dropout):
        super(EffNetb0PureStochastic, self).__init__()
        self.InvertedResBlocks = nn.ModuleList([])

        ##stem high res
        self.base_conv_hr = StochasticConv4(stride=2, out_channels=32, kernel_size=3, output_size=(112, 112))
        self.base_bn_hr = nn.BatchNorm2d(num_features=32)


        ##output size of first conv layer calculation (to calc pad in static padding calc)
        input_size = feature_dim_after_conv(image_size, stride=2)

        ### define architecture blocks ###
        block_list = GetArchitectureBlockList('b032')

        self.InvertedResBlocks = build_module_list(block_list, input_size)

        #Set drop connect rate now we have created all the blocks
        drop_rate = 0.2
        for idx, block in enumerate(self.InvertedResBlocks):
            block.DropConnect.set_drop_connect_rate(rate=((drop_rate)*float(idx)/len(self.InvertedResBlocks)))

        self.TopConv = Convolution2dSamePadding(in_channels=320, out_channels=1280,
                                                kernel_size=1, stride=1, input_size=image_size)
        self.TopBN = nn.BatchNorm2d(num_features=1280)

        self.TopAvgPool = nn.AdaptiveAvgPool2d(1)
        self.TopDrop = nn.Dropout(dropout)
        self.Top = nn.Linear(1280, 1)#num classes

    def forward (self, input_batch):
        x= nn.SiLU()((self.base_bn_hr(self.base_conv_hr(input_batch[1]))))

        for idx, resblock in enumerate(self.InvertedResBlocks):
            x = resblock(x)

        x = self.TopBN(self.TopConv(x))
        x = torch.flatten(self.TopAvgPool(x),1)
        x = self.TopDrop(x)
        x = self.Top(x)

        return x

### EfficientNet b4 base model ###
class EffNetb4 (nn.Module):
    def __init__(self, image_size, dropout):
        super(EffNetb4, self).__init__()
        self.InvertedResBlocks = nn.ModuleList([])

        ##start architecture
        self.base_conv = nn.Conv2d(in_channels=3, out_channels=48,
                                   stride=2, kernel_size=3, padding=1)
        self.base_bn = nn.BatchNorm2d(num_features=48)

        ##output size of first conv layer calculation (to calc pad in static padding calc)
        input_size = feature_dim_after_conv(image_size, stride=2)

        ### define architecture blocks ###
        block_list = GetArchitectureBlockList('b4')

        self.InvertedResBlocks = build_module_list(block_list, input_size)

        #Set drop connect rate now we have created all the blocks
        drop_rate = 0.2
        for idx, block in enumerate(self.InvertedResBlocks):
            block.DropConnect.set_drop_connect_rate(rate=((drop_rate)*float(idx)/len(self.InvertedResBlocks)))

        # Define the top
        self.TopConv = Convolution2dSamePadding(in_channels=448, out_channels=1792,
                                                kernel_size=1, stride=1, input_size=input_size)
        self.TopBN = nn.BatchNorm2d(num_features=1792)
        self.TopAvgPool = nn.AdaptiveAvgPool2d(1)
        self.TopDrop = nn.Dropout(dropout)
        self.Top = nn.Linear(1792, 1)#num classes

    def forward (self, input_batch):

        x = nn.SiLU()((self.base_bn(self.base_conv(input_batch))))

        for idx, resblock in enumerate(self.InvertedResBlocks):
            x = resblock(x)

        x = self.TopBN(self.TopConv(x))
        x = torch.flatten(self.TopAvgPool(x),1)
        x = self.TopDrop(x)
        x = self.Top(x)

        return x

### EfficientNet B4 Model with Stochastic Stem ###
class EffNetb4Stochastic (nn.Module):
    def __init__(self, image_size, image_size_hr, dropout):
        super(EffNetb4Stochastic, self).__init__()
        self.InvertedResBlocks = nn.ModuleList([])

        ##start architecture
        ##stem low res
        self.base_conv = nn.Conv2d(in_channels=3, out_channels=48,
                                   stride=2, kernel_size=3, padding=1)
        self.base_bn = nn.BatchNorm2d(num_features=48)

        ##stem high res
        ##stochastic conv layer
        self.base_conv_hr = StochasticConv4(stride=2, out_channels=48, kernel_size=3, output_size=(190, 190))
        self.base_bn_hr = nn.BatchNorm2d(num_features=48)


        ##output size of first conv layer calculation (to calc pad in static padding calc)
        input_size = feature_dim_after_conv(image_size, stride=2)

        ### define architecture blocks ###
        block_list = GetArchitectureBlockList('b4_96')

        self.InvertedResBlocks = build_module_list(block_list, input_size)

        #Set drop connect rate now we have created all the blocks
        drop_rate = 0.2
        for idx, block in enumerate(self.InvertedResBlocks):
            block.DropConnect.set_drop_connect_rate(rate=((drop_rate)*float(idx)/len(self.InvertedResBlocks)))

        # Define the top
        self.TopConv = Convolution2dSamePadding(in_channels=448, out_channels=1792,
                                                kernel_size=1, stride=1, input_size=input_size)
        self.TopBN = nn.BatchNorm2d(num_features=1792)
        self.TopAvgPool = nn.AdaptiveAvgPool2d(1)
        self.TopDrop = nn.Dropout(dropout)
        self.Top = nn.Linear(1792, 1)#num classes

    def forward (self, input_batch):
        x = nn.SiLU()((self.base_bn(self.base_conv(input_batch[0]))))

        x_hr = nn.SiLU()((self.base_bn_hr(self.base_conv_hr(input_batch[1]))))

        x = torch.cat((x, x_hr), 1)

        for idx, resblock in enumerate(self.InvertedResBlocks):
            x = resblock(x)

        x = self.TopBN(self.TopConv(x))
        x = torch.flatten(self.TopAvgPool(x),1)
        x = self.TopDrop(x)
        x = self.Top(x)

        return x


