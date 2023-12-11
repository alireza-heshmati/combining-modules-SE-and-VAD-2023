import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample

class DS_FA_v04(Dataset):
    """This creats DS-FA-v04 dataset and its labels.

    Arguments
    ---------
    data_path : str
        Path of dataset file (./dataset).

    Returns
    -------
    data : float (Tensor)
        Readed audio. 

    label : float (Tensor)
        label of data.

    """
    def __init__(self, data_path):
        self.data_path = data_path
        
        
    def __len__(self):
        """This measures the number of total dataset."""
        return len(self.data_path)

    def __getitem__(self, idx):
        """This method read each data, label according to super class Dataset. """
        file_path = self.data_path[idx]

        data, rate = torchaudio.load(file_path)
    
        if rate != 16000:
            transform = Resample(rate, 16000)
            data = transform(data)
            rate = 16000

        if file_path[-2].split("/") == 'SAD_noise_dataset':
            label = torch.tensor([0])

        else :
            label = torch.tensor([1])

        return data, label
    

