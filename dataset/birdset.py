## imports
#basic
import os
import warnings
import re
from pathlib import Path

#processing
from scipy.io.wavfile import read as wav_read
from librosa import load
import numpy as np
import pandas as pd
import soundfile as sf

#torch
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

#shared
from . import (_compute_split_boundaries, _get_inter_splits_by_group,
               build_dataloaders)
label_mapping = {}

def check_wav(path):
    """
    If .wav does not exist for path, write it
    """
    path = Path(path)
    if not path.with_suffix(".wav").exists():
        audio, sr = load(str(path))
        sf.write(str(path.with_suffix(".wav")), audio, sr)
    return str(path.with_suffix(".wav")) 
class BirdSet(Dataset):
    def __init__(self, region, split='train', seed=0, sample_rate=16000,
                 fixed_crop=None, random_crop=None, indices=None):
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
        
        self.filenames = np.array([])
        self.labels = np.array([])
        self.label_mapping = label_mapping
        
        # fill filenames and labels
        for item in self.dataset:
            self.filenames = np.append(self.filenames, item['filepath'])
            label = item['ebird_code']
            if label not in label_mapping:
                label_mapping[label] = len(label_mapping)
            self.labels = np.append(self.labels, label_mapping[label])
            
        if indices is not None:
            self.labels = self.labels[indices]
            self.filenames = self.filenames[indices]
        
        self.labels = torch.tensor(self.labels) # must convert label list to tensor for torch
    
    def split_dataset(self, n_splits=2, props=[0.7, 0.3]):
        if not np.isclose(sum(props), 1.0):
            raise ValueError("Proportions must sum to 1.0")
        # print("test")
        indices = (np.cumsum(props) * len(self.filenames)).astype(int)
        
        # Remove the last index which is just the total length of the array
        split_indices = indices[:-1]
        
        dataset_indices = np.arange(len(self.filenames))
        np.random.seed(self.seed)
        np.random.shuffle(dataset_indices)
        
        set_indices = np.split(dataset_indices, split_indices)
        return [BirdSet(region=None, split=self.split, seed=self.seed, sample_rate=self.sample_rate,
                        fixed_crop=self.fixed_crop, random_crop=self.random_crop, indices=idx) for idx in set_indices]
        
        
        
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = self.filenames[idx]
        path = check_wav(path)
        if not os.path.exists(path):
            raise ValueError(f"File {path} not found")
        try:
            _, audio = wav_read(path, mmap=True)
        except Exception as e:
            print("Error reading file: ", path)
            raise e
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
    region = re.findall(r'BIRDSET_(.+)', args.data_set)[0]
    if args.train_with_test:
        test_set = BirdSet(region=region, split='test', seed=0, sample_rate=16000)
        train_set, test_set, val_set = test_set.split_dataset(n_splits=3, props=[0.5, 0.1, 0.4])
    else:
        # test on full recordings
        test_set = BirdSet(region=region, split='test', seed=0, sample_rate=16000)
        # train on random excerpt (args.input_size)
        train_set = BirdSet(region=region, split='train', seed=0, sample_rate=16000, random_crop=args.input_size)
        print(type(train_set))
        train_set, val_set = train_set.split_dataset(n_splits=2, props=[0.7, 0.3])
        # validate on first 16 seconds
    val_set.fixed_crop = 16*16000
    
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set,
        label_transform_fn=None, args=args)
    nb_classes = 397
    return train_loader, val_loader, test_loader, nb_classes