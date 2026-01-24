'''
Time seriesed data is classified using 'sliding windows'
A good analogy is how if you had a tape of information with rows of the features.
The window is a cardboard cutout allowing you to see only a specific number of rows at a time
this is window is then flattened to a single vector, the window is then iterated only one row forward at a time
This allows you to create the sense of the temporal order in the rectified data
'''

import pandas as pd
import numpy as np

def sliding_window_data(data: np.ndarray, W: int) -> np.ndarray:
    '''
    Create sliding window dataset
    
    :param data: Input timeseriesed data
    :type data: np.ndarray

    :param W: Window size (number of rows it can see at a time)
    :type W: int

    :return: sliding window dataset
    :rtype: ndarray
    '''
    time_removed = data[:, 1:]

    windowed_data = np.zeros(time_removed.shape[0], time_removed.shape[1] * W)
    for i in range(time_removed.shape[0] - W):
        windowed_data[i] = time_removed[i : i + W].flatten()

    return windowed_data

