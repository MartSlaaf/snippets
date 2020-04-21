from torch.utils.data import Dataset
import numpy as np

class StackedAlongDataSet(Dataset):
    """
    Stacking datasets along their length.
    If given N datasets, __getitem__(k) will return k-th object of each dataset.
    NB: there is no meaningful synchronization of augmentations between datasets, don't use this for the images and masks.
    """
    def __init__(self, *datasets_list):
        self.datasets_list = datasets_list
        if len(np.unique(np.array([len(d) for d in self.datasets_list]))) > 1:
            raise ValueError('All datasets should be of the same length!')

    def __len__(self):
        return len(self.datasets_list[0])

    def __getitem__(self, id):
        return [d[id] for d in self.datasets_list]
