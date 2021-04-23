
from layers import *
from torch import nn
## Dictionaries that define the backbones of the b0 and b4 models
## Both models have 7 main phases of inverted residual bottleneck blocks
## Differ in number of block repeats (depth), channels (width) and resolution
def GetArchitectureBlockList (model_name):

    ## b0 base architectures
    if 'b0' in model_name:
        ## b0 base architecture 64 input channels from the stem
        ## used for base 64 channel implementation and stochastic stem b0
        if '64' in model_name:
            InvResBlock1 = {'input_channels':64,
                            'output_channels':16,
                            'kernel_size':3,
                            'stride':1,
                            'expansion_ratio':1,
                            'squeeze_ratio':0.25,
                            'repeats':0}
        else:
            InvResBlock1 = {'input_channels': 32,
                            'output_channels': 16,
                            'kernel_size': 3,
                            'stride': 1,
                            'expansion_ratio': 1,
                            'squeeze_ratio': 0.25,
                            'repeats': 0}

        InvResBlock2 = {'input_channels': 16,
                        'output_channels': 24,
                        'kernel_size': 3,
                        'stride': 2,
                        'expansion_ratio': 6,
                        'squeeze_ratio': 0.25,
                        'repeats': 1}

        InvResBlock3 = {'input_channels': 24,
                        'output_channels': 40,
                        'kernel_size': 5,
                        'stride': 2,
                        'expansion_ratio': 6,
                        'squeeze_ratio': 0.25,
                        'repeats': 1}

        InvResBlock4 = {'input_channels': 40,
                        'output_channels': 80,
                        'kernel_size': 3,
                        'stride': 2,
                        'expansion_ratio': 6,
                        'squeeze_ratio': 0.25,
                        'repeats': 2}

        InvResBlock5 = {'input_channels': 80,
                        'output_channels': 112,
                        'kernel_size': 5,
                        'stride': 1,
                        'expansion_ratio': 6,
                        'squeeze_ratio': 0.25,
                        'repeats': 2}

        InvResBlock6 = {'input_channels': 112,
                        'output_channels': 192,
                        'kernel_size': 5,
                        'stride': 2,
                        'expansion_ratio': 6,
                        'squeeze_ratio': 0.25,
                        'repeats': 3}

        InvResBlock7 = {'input_channels': 192,
                        'output_channels': 320,
                        'kernel_size': 3,
                        'stride': 1,
                        'expansion_ratio': 6,
                        'squeeze_ratio': 0.25,
                        'repeats': 0}
    ## b4 architecture
    if 'b4' in model_name:
        ## stochastic stem I double the input channels to the first block
        if '96' in model_name:
            InvResBlock1 = {'input_channels': 96,
                            'output_channels': 24,
                            'kernel_size': 3,
                            'stride': 1,
                            'expansion_ratio': 1,
                            'squeeze_ratio': 0.25,
                            'repeats': 1}
        else:
            InvResBlock1 = {'input_channels': 48,
                            'output_channels': 24,
                            'kernel_size': 3,
                            'stride': 1,
                            'expansion_ratio': 1,
                            'squeeze_ratio': 0.25,
                            'repeats': 1}

        InvResBlock2 = {'input_channels': 24,
                        'output_channels': 32,
                        'kernel_size': 3,
                        'stride': 2,
                        'expansion_ratio': 6,
                        'squeeze_ratio': 0.25,
                        'repeats': 3}

        InvResBlock3 = {'input_channels': 32,
                        'output_channels': 56,
                        'kernel_size': 5,
                        'stride': 2,
                        'expansion_ratio': 6,
                        'squeeze_ratio': 0.25,
                        'repeats': 3}

        InvResBlock4 = {'input_channels': 56,
                        'output_channels': 112,
                        'kernel_size': 3,
                        'stride': 2,
                        'expansion_ratio': 6,
                        'squeeze_ratio': 0.25,
                        'repeats': 5}

        InvResBlock5 = {'input_channels': 112,
                        'output_channels': 160,
                        'kernel_size': 5,
                        'stride': 1,
                        'expansion_ratio': 6,
                        'squeeze_ratio': 0.25,
                        'repeats': 5}

        InvResBlock6 = {'input_channels': 160,
                        'output_channels': 272,
                        'kernel_size': 5,
                        'stride': 2,
                        'expansion_ratio': 6,
                        'squeeze_ratio': 0.25,
                        'repeats': 7}

        InvResBlock7 = {'input_channels': 272,
                        'output_channels': 448,
                        'kernel_size': 3,
                        'stride': 1,
                        'expansion_ratio': 6,
                        'squeeze_ratio': 0.25,
                        'repeats': 1}

    block_list = []
    block_list.extend(
            [InvResBlock1, InvResBlock2, InvResBlock3, InvResBlock4, InvResBlock5, InvResBlock6, InvResBlock7])

    return block_list

def build_module_list (block_list, input_size, stop_idx=99):
    InvertedResBlocks = nn.ModuleList([])

    for idx, block in enumerate(block_list):
        if idx >= stop_idx:
            break
        InvertedResBlocks.append(
            InvertedMobileResidualBlock(input_size=input_size, input_channels=block['input_channels'],
                                        output_channels=block['output_channels'], kernel_size=block['kernel_size'],
                                        stride=block['stride'], expansion_ratio=block['expansion_ratio'],
                                        squeeze_ratio=block['squeeze_ratio'],
                                        dropout=0))

        input_size = feature_dim_after_conv(input_size, stride=block['stride'])

        if block['repeats'] > 0:
            InvMobileBlock = InvertedMobileResidualBlock(input_size=input_size,
                                                         input_channels=block['output_channels'],
                                                         output_channels=block['output_channels'],
                                                         kernel_size=block['kernel_size'],
                                                         stride=1,
                                                         expansion_ratio=block['expansion_ratio'],
                                                         squeeze_ratio=block['squeeze_ratio'], dropout=0)

            for i in range(0, block['repeats']):
                InvertedResBlocks.append(InvMobileBlock)

    return InvertedResBlocks