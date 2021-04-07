import torch
from torch import nn
import math
from torch.nn import functional as F


def feature_dim_after_conv (in_size, stride):
    ## function to calculate feature map dimension after strided convolution (for conv2d layers!)
    ## assume square images and convs
    out_size = int(math.ceil(in_size[0] / stride))
    return (out_size, out_size)

##Class for same padding convolution
class Convolution2dSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 input_size=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # Padding is calculated in constructor at initialization to avoid having to calculate it in the forward pass
        # Using an identical approach to static pad as in tensorflow for 'same' padding
        # I assume that the input image is square MxM image
        self.pad_left = 0
        self.pad_top = 0
        self.pad_right = 0
        self.pad_bottom = 0
        self.out_dim = math.ceil(input_size[0] / self.stride[0]) #output feature map dimension

        if input_size != None:
            in_dim = input_size[0]
            kernel_dim = self.weight.shape[3]
            if (in_dim % stride == 0):
                pad = max(kernel_dim - stride, 0)
            else:
                pad = max(kernel_dim - (in_dim % stride), 0)

            self.pad_left = pad // 2
            self.pad_top = pad // 2
            self.pad_right = pad - self.pad_left
            self.pad_bottom = pad - self.pad_top

    def forward(self, x):

        # pad feature map in forward pass
        x = torch.nn.functional.pad(x, (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom))

        # apply convolution to padded feature map
        return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation,
                                          self.groups)

class StochasticConv1(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 input_size=None, output_size=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.output_size = output_size[0]
        self.input_size = input_size[0]
        self.slice_size = (self.output_size - 1) * stride + kernel_size
        self.stride = stride

    def forward(self, x):
        RandomWindowStart = torch.randint(low=self.slice_size, high=x.shape[3], size=(self.out_channels, x.shape[0]))
        output = torch.cuda.FloatTensor(x.shape[0], self.out_channels, self.output_size, self.output_size)
        num_filters = self.out_channels
        num_images = x.shape[0]

        #loop over filters loop over images
        for idx_k in range(0, num_filters):
            for idx_i in range(0, num_images):
                ## create the splice
                end = RandomWindowStart[idx_k][idx_i]  # end position of random window
                start = end - self.slice_size  # start position of random window
                #img_slice = x[idx_i, :, start:end, start:end].unsqueeze(0)  # Image slice for kernel idx

                # conv for this kernel over specified slice
                output[idx_i, idx_k, :, :] = F.conv2d(input=x[idx_i, :, start:end, start:end].unsqueeze(0), weight=self.weight[idx_k, :, :, :].unsqueeze(0),
                                                      stride=self.stride, dilation=self.dilation,
                                                      padding=0, bias=self.bias).squeeze(1)

        return output

class StochasticConv2(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 input_size=None, output_size=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.output_size = output_size[0]
        self.input_size = input_size[0]
        self.slice_size = (self.output_size - 1) * stride + kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        RandomWindowStart = torch.randint(low=self.slice_size, high=x.shape[3], size=(self.out_channels, x.shape[0])) #(filters, images)
        output = torch.zeros((x.shape[0], self.out_channels, self.output_size, self.output_size), dtype=torch.float32).cuda()

        num_filters = self.out_channels
        num_images = x.shape[0]

        image_slice = torch.zeros((x.shape[0], self.in_channels, self.slice_size, self.slice_size), dtype=torch.float32).cuda()


        #loop over filters loop over images
        for idx_k in range(0, num_filters):
            for idx_i in range(0, num_images):
                ## create the splice
                end = RandomWindowStart[idx_k][idx_i]  # end position of random window
                start = end - self.slice_size  # start position of random window
                image_slice[idx_i,:,:,:] = x[idx_i, :, start:end, start:end].unsqueeze(0)  # Image slice for kernel idx

            output[:, idx_k, :, :] = F.conv2d(input=x[idx_i, :, start:end, start:end].unsqueeze(0),
                                              weight=self.weight[idx_k, :, :, :].unsqueeze(0),
                                              stride=self.stride, dilation=self.dilation,
                                              padding=0, bias=self.bias).squeeze(1)

        return output

class StochasticConv3(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 input_size=None, output_size=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.output_size = output_size[0]
        self.input_size = input_size[0]
        self.slice_size = (self.output_size - 1) * stride + kernel_size
        self.stride = stride

    def forward(self, x):
        RandomWindowStart = torch.randint(low=self.slice_size, high=x.shape[3], size=(self.out_channels,)) #(filters, images)
        output = torch.cuda.FloatTensor(x.shape[0], self.out_channels, self.output_size, self.output_size).fill_(0)
        num_filters = self.out_channels

        #loop over filters loop over images
        for idx_k in range(0, num_filters):
            end = RandomWindowStart[idx_k]  # end position of random window
            start = end - self.slice_size  # start position of random window

            #conv for this kernel over specified slice
            output[:, idx_k, :, :] = F.conv2d(input=x[:, :, start:end, start:end], weight=self.weight[idx_k, :, :, :].unsqueeze(0),
                                                  stride=self.stride, dilation=self.dilation,
                                                  padding=0, bias=self.bias).squeeze(1)



        return output

class StochasticConv4(nn.Module):
    def __init__(self, stride, out_channels, kernel_size, output_size=None):
        super(StochasticConv4, self).__init__()
        ## Module block ##
        self.StochasticConvBlocks = nn.ModuleList([])
        self.out_channels = output_size[0]
        self.slice_size = (output_size[0] - 1) * stride + kernel_size

        for i in range(0, out_channels):
            self.StochasticConvBlocks.append(nn.Conv2d(in_channels=3, out_channels=1, kernel_size=kernel_size, stride=stride, padding=0,
                                                       bias=False))
    def forward(self, x):
        RandomWindowStart = torch.randint(low=self.slice_size, high=x.shape[3], size=(self.out_channels,))
        concat_list = []

        for idx, Conv in enumerate(self.StochasticConvBlocks):
            end = RandomWindowStart[idx]  # end position of random window
            start = end - self.slice_size  # start position of random window
            concat_list.append(Conv(x[:, :, start:end, start:end]))


        return torch.cat(concat_list,axis=1)

class StochasticConv5(nn.Module):
    def __init__(self, stride, out_channels, kernel_size, output_size=None):
        super(StochasticConv5, self).__init__()
        ## High Res Image randomly sliced on CPU prior to being passed to GPU

        ## Module block ##
        self.StochasticConvBlocks = nn.ModuleList([])
        self.out_channels = output_size[0]

        for i in range(0, out_channels):
            self.StochasticConvBlocks.append(nn.Conv2d(in_channels=3, out_channels=1, kernel_size=kernel_size, stride=stride,
                                                       padding=0,
                                                       bias=False))
    def forward(self, x):
        concat_list = []
        # input of spec: [batch, channel window, 3, slice_size, slice_size]
        for idx, Conv in enumerate(self.StochasticConvBlocks):
            concat_list.append(Conv(x[:, idx, :, start:end, start:end]))


        return torch.cat(concat_list,axis=1)

class RandomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, out_size, stride, input_size, kernel_size):
        super(RandomConv2d, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.input_size = input_size
        self.out_size = out_size[0]
        self.kernel_weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        torch.nn.init.xavier_uniform(self.kernel_weights)
        self.slice_size = (self.out_size - 1) * stride + kernel_size

    def forward(self, x):
        RandomWindowStart = torch.randint(low=self.slice_size, high=x.shape[3], size=(self.out_channels, x.shape[0]))
        output = torch.zeros((x.shape[0], self.out_channels, self.out_size, self.out_size), dtype=torch.float32).cuda()
        print(x.shape[3] - self.slice_size)
        print(x.shape[3])

        for idx in range(0, len(RandomWindowStart)):
            end = RandomWindowStart[idx]  # start position
            start = end - self.slice_size  # end position
            print(start)
            img_slice = x[:, :, start:end, start:end]  # Image slice for kernel idx

            # conv for this kernel over specified slice
            output[:, idx, :, :] = F.conv2d(input=img_slice, weight=self.kernel_weights[idx, :, :, :].unsqueeze(0), stride=2, dilation=1, padding=0, bias=None).squeeze(1)
        return output

class DropConnect (nn.Module):
    ## DropConnect class
    ## This layer gives stochastic depth to the model
    ## With drop probability drop each element from the feature map
    ## Provides stochastic depth in repeating inverted bottleneck blocks
    ## The drop connect rate reduces in later blocks and depends on the number of blocks

    def __init__(self, drop_rate = 0):
        super(DropConnect, self).__init__()
        self.drop_rate = drop_rate

    def set_drop_connect_rate (self, rate=0): ## set the drop rate, needs to be done after all blocks have been added
        self.drop_rate = rate

    def forward(self, x):
        if self.training == False:
            return x #in eval mode, don't drop features

        rand = torch.rand(x.shape[0], 1, 1, 1) + (1-self.drop_rate) #random uniform tensor + (1- drop rate)
        rand = torch.floor(rand).to('cuda') #creates a binary mask with drop rate % of mask will be floored to 0
        output = torch.div(x, (1-self.drop_rate))*rand # will randomly drop in the forward pass
        return output

class ExpandLayer(nn.Module):
    ## ExpandLayer class
    ## Expands the incoming feature map into more channels
    ## in channels < out channels
    ## Composed of expansion conv and batch norm layer + activation (swisH)

    def __init__(self, in_channels, out_channels, input_size, kernel_size=1, stride=1, bias=False):
        super(ExpandLayer, self).__init__()

        ##kernel size and stride always == 1 in expansion layer
        self.ExpansionConvolution = Convolution2dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                                             kernel_size=kernel_size, stride=stride, bias=bias, input_size=input_size)
        self.ExpandBN = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.ExpansionConvolution(x)
        x = self.ExpandBN(x)
        x = nn.SiLU()(x) ##swish activation
        return x

class SqueezeExcitationLayer(nn.Module):
    ## Squeeze and excite class
    ## Credit to: https://python.plainenglish.io/implementing-efficientnet-in-pytorch-part-3-mbconv-squeeze-and-excitation-and-more-4ca9fd62d302
    ## Helped me understand the layer


    ## Applies global average pooling,
    ## then apply squeeze and unsqueeze convolution instead of FC because it works well with feature map dimensions
    ## after swish activation, apply excitation x * sigmoid (squeeze) to input and return

    def __init__(self, in_channels, squeeze_channel):
        super(SqueezeExcitationLayer, self).__init__()

        ##squeeze convolution
        self.SqueezedConv = Convolution2dSamePadding(in_channels=in_channels, out_channels=squeeze_channel,
                                                     kernel_size=1, stride=1, bias=False)
        ##unsqueeze convolution
        self.UnSqueezedConv = Convolution2dSamePadding(in_channels=squeeze_channel, out_channels=in_channels,
                                                       kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x_squeeze = F.adaptive_avg_pool2d(x, 1)
        x_squeeze = nn.SiLU()(self.SqueezedConv(x_squeeze))
        x_squeeze = self.UnSqueezedConv(x_squeeze)
        x = x * torch.sigmoid(x_squeeze)
        return x

class DepthWiseConv(nn.Module):
    ## Depthwise convolution
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, input_size, groups):
        super(DepthWiseConv, self).__init__()

        ## depthwise convolution
        ## groups input determines the depthwise conv
        self.DepthWiseConv = Convolution2dSamePadding(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=kernel_size, stride=stride, bias=bias,
                                                      input_size=input_size, groups=groups)

        self.DepthwiseBN = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = nn.SiLU()(self.DepthwiseBN(self.DepthWiseConv(x)))
        return x



class InvertedMobileResidualBlock(nn.Module):
    def __init__(self, input_size, input_channels,
                 output_channels, kernel_size, stride,
                 expansion_ratio, squeeze_ratio=None, dropout=0):
        super(InvertedMobileResidualBlock, self).__init__()

        ##predeclaration of layers:
        self.ExpandNeck = None
        self.SqueezeExcitation = None
        self.stride = stride
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dropout = dropout

        # define the expand and compress Conv structure
        # calculate expanded channels in the inverted block
        expanded_channels = input_channels * expansion_ratio

        ##expansion neck
        if expanded_channels > input_channels:
            self.ExpandNeck = ExpandLayer(in_channels=input_channels, out_channels=expanded_channels,
                                          input_size=input_size)

        ## depthwise convolution
        ## groups input determines the depthwise conv
        self.DepthWiseConv = DepthWiseConv(in_channels=expanded_channels, out_channels=expanded_channels,
                                           kernel_size=kernel_size, stride=stride, bias=False,
                                           input_size=input_size, groups=expanded_channels)

        input_size = feature_dim_after_conv(input_size, stride)

        channels_to_squeeze = int(input_channels * squeeze_ratio)
        self.SqueezeExcitation = SqueezeExcitationLayer(in_channels=expanded_channels,
                                                        squeeze_channel=channels_to_squeeze)

        ## output convolution layer
        ## stride is 1 and kernel size is 1
        self.OutputConv = Convolution2dSamePadding(in_channels=expanded_channels, out_channels=output_channels,
                                                   kernel_size=1, stride=1, bias=False, input_size=input_size)
        self.OutputBN = nn.BatchNorm2d(num_features=output_channels)

        ## Drop Connect
        self.DropConnect = DropConnect(drop_rate=dropout)

    def forward(self, x_in):

        x = x_in

        if self.ExpandNeck != None:
            x = self.ExpandNeck(x)

        x = self.DepthWiseConv(x)

        if self.SqueezeExcitation != None:
            x = self.SqueezeExcitation(x)

        x = nn.SiLU()(self.OutputBN(self.OutputConv(x)))


        #skip connect conditions: stride = 1, input channels = output channels (i.e. repeating block)
        if self.input_channels == self.output_channels and self.stride == 1:
            if self.DropConnect.drop_rate > 0:
                x = self.DropConnect (x)

            x = x + x_in

        return x