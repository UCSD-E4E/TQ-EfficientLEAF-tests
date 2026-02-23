## imports
#basic
import os
import warnings

#processing
from scipy.io.wavfile import read as wav_read
import numpy as np
import pandas as pd

#torch
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

#shared
from . import (_compute_split_boundaries, _get_inter_splits_by_group,
               build_dataloaders)
label_mapping = {}

class BirdSet(Dataset):
    def __init__(self, region, split='train', seed=0, sample_rate=16000,
                 fixed_crop=None, random_crop=None):
        #same seed for train and test(!)
        # assert split == 'test' or split == 'train' or split == 'validation', 'split must be "train", "validation" or "test"'
        
        # inits
        #metadata
        self.dataset = load_dataset('DBD-research-group/BirdSet', region, split=split)
        #parameters
        self.split = split
        self.seed = seed
        self.sample_rate = sample_rate
        self.fixed_crop = fixed_crop
        self.random_crop = random_crop
        
        self.filenames = []
        self.labels = []
        self.label_mapping = label_mapping
        
        # fill filenames and labels
        for item in self.dataset:
            self.filenames.append(item['filepath'])
            label = item['ebird_code']
            if label not in label_mapping:
                label_mapping[label] = len(label_mapping)
            self.labels.append(label_mapping[label])
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        _, audio = wav_read(os.path.join(self.root_data, self.filenames[idx]),
                            mmap=True)
        if self.fixed_crop:
            audio = audio[:self.fixed_crop]
        if self.random_crop:
            if len(audio) > self.random_crop:
                pos = np.random.randint(len(audio) - self.random_crop)
                audio = audio[pos:pos + self.random_crop]
            elif len(audio) < self.random_crop:
                audio = np.concatenate((audio,
                                        np.zeros(self.random_crop - len(audio),
                                                 dtype=audio.dtype)))
        if audio.ndim == 2:
            audio = audio.T  # move channels first
        elif audio.ndim == 1:
            audio = audio[np.newaxis]  # add channels dimension
        if not np.issubdtype(audio.dtype, np.floating):
            audio = np.divide(audio, np.iinfo(audio.dtype).max, dtype=np.float32)
        audio = torch.as_tensor(audio, dtype=torch.float32)
        label = self.labels[idx]

        return audio, label
    
def build_dataset(args):
    # test on full recordings
    test_set = BirdSet(region=args.birdset_region, split='test', seed=0, sample_rate=16000)
    # validate on first 16 seconds
    val_set = BirdSet(region=args.birdset_region, split='validation', seed=0, sample_rate=16000, fixed_crop=16*16000)
    # train on random excerpt (args.input_size)
    train_set = BirdSet(region=args.birdset_region, split='train', seed=0, sample_rate=16000, random_crop=args.input_size)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set,
        label_transform_fn=None, args=args)
    nb_classes = 397
    return train_loader, val_loader, test_loader, nb_classes