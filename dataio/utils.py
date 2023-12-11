import pickle
import random
import pandas as pd
import os

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """help torch.loader give data and labels. indeed preparing
     readed dataset for network with padding.

    Arguments
    ---------
    batch : tuple, Tensor
        It includes the output of Dataset class

    Returns
    -------
    inputs : float (Tensor)
        Padded noisy audios
    targets : float (Tensor)
        Padded clean audios as targets
    length_ratio : float (Tensor)
        Real length of each audio

    """
    inputs, targets, length_ratio = [], [], []
    for data, label in batch:
        inputs.append(data.squeeze())
        targets.append(label.squeeze())
        length_ratio.append(len(inputs[-1]))

    inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    length_ratio = torch.tensor(length_ratio, dtype=torch.long) / inputs.shape[1]

    return inputs, torch.stack(targets), length_ratio

def creat_dataset_path(invalid_filenames, valid_filenames, DATA_path):
    """Prepare dataset pathes.

    Arguments
    ---------
    invalid_filenames : list, str
        List of invalid data files name.

    valid_filenames : list, str
        List of valid data files name.

    DATA_path : str
        Path of dataset.

    Returns
    -------
    total_path : list, str
        Pathes of the total DS-FA-v04 dataset.

    """

    with open(os.path.join(DATA_path, invalid_filenames) , 'rb') as fp:
        invalid_path = pickle.load(fp)

    random.seed(12)
    random.shuffle(invalid_path)

    valid_path = pd.read_csv(os.path.join(DATA_path, valid_filenames))   
    valid_path = list(valid_path['speech_path'])

    random.seed(12)
    random.shuffle(valid_path)
    valid_path = valid_path[:len(invalid_path)]

    for i in range(len(valid_path)):
        valid_path[i] = os.path.join(DATA_path,valid_path[i])
        invalid_path[i] = os.path.join(DATA_path,invalid_path[i])


    total_path = invalid_path + valid_path 

    random.seed(12)
    random.shuffle(total_path)

    return total_path