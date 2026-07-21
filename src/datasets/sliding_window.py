'''
Time seriesed data is classified using 'sliding windows'
A good analogy is how if you had a tape of information with rows of the features.
The window is a cardboard cutout allowing you to see only a specific number of rows at a time
the window is then iterated a set number of rows forward at a time
This allows you to create the sense of the temporal order in the data
'''

import torch

def sliding_data(data: torch.Tensor, W: int, stride = 1) -> torch.Tensor:
    '''
    Slide data into windowed format
    
    :param data: Input formatted data
    :type data: torch.Tensor

    :param W: Window size (number of rows it can see at a time)
    :type W: int

    :param stride: Stride size (number of rows skipped each between each window default = 1 meaning none are skipped)
    :type stride: int

    :return: sliding window dataset
    :rtype: torch.Tensor
    '''
    windowed_data = data.unfold(0, size=W, step=stride).permute(0, 2, 1).contiguous()

    return windowed_data