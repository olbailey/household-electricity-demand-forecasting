import pytest

import numpy as np

# from src.datasets.sliding_window import sliding_window_data
from datasets.sliding_window import sliding_window_data

def test_valid_output():
    x = np.array(list(range(50)))
    x = x.reshape((10, 5))

    print(x.shape)
    print(x)

if __name__ == "__main__":
    test_valid_output()