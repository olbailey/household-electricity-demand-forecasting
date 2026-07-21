import torch
from datasets import MainDataset

def make_dataset(window_size, stride, prediction_size):
    dataset = MainDataset.__new__(MainDataset)
    dataset.window_size = window_size
    dataset.window_stride = stride
    dataset.prediction_size = prediction_size
    return dataset

def test_format_labels_basic():
    dataset = make_dataset(window_size=5, stride=1, prediction_size=1)

    data = torch.Tensor([
        [0, -1], [1, -1], [3, -1], [6, -1], [10, -1], [15, -1], 
        [21, -1], [28, -1], [36, -1], [45, -1], [55, -1],
    ])

    labels = dataset._format_labels(data)

    expected = torch.Tensor([5, 6, 7, 8, 9, 10])

    assert torch.allclose(labels, expected)

def test_format_labels_with_larger_prediction():
    dataset = make_dataset(window_size=5, stride=1, prediction_size=2)

    data = torch.Tensor([
        [0, -1], [1, -1], [3, -1], [6, -1], [10, -1], [15, -1], 
        [21, -1], [28, -1], [36, -1], [45, -1], [55, -1],
    ])

    labels = dataset._format_labels(data)

    expected = torch.Tensor([
        [5, 6], [6, 7], 
        [7, 8], [8, 9], 
        [9, 10]
    ])

    assert torch.allclose(labels, expected)

if __name__ == "__main__":
    test_format_labels_basic()