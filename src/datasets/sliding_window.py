'''
Time seriesed data is classified using 'sliding windows'
A good analogy is how if you had a tape of information with rows of the features.
The window is a cardboard cutout allowing you to see only a specific number of rows at a time
this is window is then flattened to a single vector, the window is then iterated only one row forward at a time
This allows you to create the sense of the temporal order in the rectified data
'''

import pandas as pd
import numpy as np

def convert_sliding_window(df: pd.DataFrame, W: int) -> np.ndarray:
    return sliding_data(df.to_numpy(), W)

def sliding_data(data: np.ndarray, W: int) -> np.ndarray:
    '''
    Slide data into windowed format
    
    :param data: Input formatted data
    :type data: np.ndarray

    :param W: Window size (number of rows it can see at a time)
    :type W: int

    :return: sliding window dataset
    :rtype: ndarray
    '''
    
    windowed_data = np.lib.stride_tricks.sliding_window_view(data, W, axis=0
                    ).transpose((0, 2, 1) # transposes the windows as originally created with in flipped order
                    ).reshape(-1, data.shape[1] * W) # flattens windows into single array

    return windowed_data