from torch.utils.data import Dataset, DataLoader
import librosa
import torch
import glob
import os
from utils import wav2spec

def create_dataloader(args, train):
    def train_collate_fn(batch):
        dvec_list = list()
        target_spect_list = list()
        mixed_spect_list = list()
        
        for dvec, target_spect, mixed_spect in batch:
            dvec_list.append(dvec)
            target_spect_list.append(target_spect)
            mixed_spect_list.append(mixed_spect)
        target_spect_list = torch.stack(target_spect_list, dim=0)
        mixed_spect_list = torch.stack(mixed_spect_list, dim=0)
        return dvec_list, target_spect_list, mixed_spect_list
    
    def test_collate_fn(batch):
        return batch
    
    if train:
        return DataLoader(dataset=VFDataset(args, True),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=train_collate_fn,
                    pin_memory=True,
                    drop_last=True,
                    sampler=None)
    else:
        return DataLoader(dataset=VFDataset(args, False),
                    collate_fn=test_collate_fn,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0)

class VFDataset(Dataset):
    def __init__(self, args, train):
        def find_all(file_format):
            return sorted(glob.glob(os.path.join(self.data_path, file_format)))
        self.args = args
        self.train = train
        self.data_path = args.train_path if train else args.test_path
        self.dvec_list = find_all('*_dvector.pt')
        self.target_wav_list = find_all('*_target.wav')
        self.mixed_wav_list = find_all('*_mixed.wav')
        self.target_spect_list = find_all('*_target.pt')
        self.mixed_spect_list = find_all('*_mixed.pt')
        
        assert len(self.dvec_list) == len(self.target_wav_list) == len(self.mixed_wav_list) == \
        len(self.target_spect_list) == len(self.mixed_spect_list), "Number of files do not match"
        
        total_size = len(self.dvec_list)
        print(f"Total number of files: {total_size}")

    def __len__(self):
        return len(self.dvec_list)
    
    def __getitem__(self, idx):
        dvec = torch.load(self.dvec_list[idx])
        
        if self.train:
            target_spect = torch.load(self.target_spect_list[idx])
            mixed_spect = torch.load(self.mixed_spect_list[idx])
            return dvec, target_spect, mixed_spect

        else:
            target_wav, _ = librosa.load(self.target_wav_list[idx], sr=16000)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr=16000)
            target_spect, _ = self.wav2spectphase(self.target_wav_list[idx])
            mixed_spect, mixed_phase = self.wav2spectphase(self.mixed_wav_list[idx])
            target_spect = torch.from_numpy(target_spect)
            mixed_spect = torch.from_numpy(mixed_spect)
            return dvec, target_wav, mixed_wav, target_spect, mixed_spect, mixed_phase
            
    def wav2spectphase(self, path):
        wav, _ = librosa.load(path, sr=16000)
        spect, phase = wav2spec(wav)
        return spect, phase