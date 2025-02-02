import argparse
import random
import glob
import os
import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from resemblyzer import VoiceEncoder, preprocess_wav
from utils import wav2spec

# generate dvector and target audio pairs for each speaker
def generate_pairs(spk_list, sample_size):
    pairs = []
    for spk_id, spk in enumerate(spk_list):
        for pair_id in range(sample_size):
            spk1_dvector, spk1_target = random.sample(spk, 2)
            pairs.append((spk_id, pair_id, spk1_dvector, spk1_target))    
    return pairs

def vad_merge(waveform):
    intervals = librosa.effects.split(waveform, top_db=20)
    piece = list()
    for s, e in intervals:
        piece.append(waveform[s:e])
    return np.concatenate(piece, axis=None)

# 
def mix(spk_id, pair_id, spk1_dvector, spk1_target, spk2, output_dir, train, encoder):  
    # load wav files  
    spk1_dvector_wav, _ = librosa.load(spk1_dvector, sr=args.sample_rate)
    spk1_target_wav, _ = librosa.load(spk1_target, sr=args.sample_rate)
    spk2_wav, _ = librosa.load(spk2, sr=args.sample_rate)
    
    # trim silent part, only both ends
    spk1_dvector_wav, _ = librosa.effects.trim(spk1_dvector_wav, top_db=20)
    spk1_target_wav, _ = librosa.effects.trim(spk1_target_wav, top_db=20)
    spk2_wav, _ = librosa.effects.trim(spk2_wav, top_db=20)
    
    # vad to remove silent part
    spk1_target_wav = vad_merge(spk1_target_wav)
    spk2_wav = vad_merge(spk2_wav)
    
    # if dvector is too short, discard
    if spk1_dvector_wav.shape[0] < args.sample_rate:
        return
    # if spk1_wav and spk2_wav is too short than setting, discard
    if spk1_target_wav.shape[0] < args.sample_rate * args.audio_length or spk2_wav.shape[0] < args.sample_rate * args.audio_length:
        return
    
    # get dvector embedding from voice encoder
    dvector = encoder.embed_utterance(spk1_dvector_wav)

    # mixed audio
    spk1_part, spk2_part = spk1_target_wav[:args.sample_rate * args.audio_length], spk2_wav[:args.sample_rate * args.audio_length]
    mixed = spk1_part + spk2_part
    
    # normalize, make sure the maximum value not exceed 1.0
    norm = np.max(np.abs(mixed)) * 1.1
    spk1_part, spk2_part, mixed = spk1_part / norm, spk2_part / norm, mixed / norm

    # save to file
    if train:
        spk1_target_wav_path = os.path.join(output_dir, 'train', f'{spk_id}_{pair_id}_target.wav')
        spk1_target_spec_path = os.path.join(output_dir, 'train', f'{spk_id}_{pair_id}_target.pt')
        mixed_wav_path = os.path.join(output_dir, 'train', f'{spk_id}_{pair_id}_mixed.wav')
        mixed_spec_path = os.path.join(output_dir, 'train', f'{spk_id}_{pair_id}_mixed.pt')
        spk1_dvector_wav_path = os.path.join(output_dir, 'train', f'{spk_id}_{pair_id}_dvector.pt')
    else:
        spk1_target_wav_path = os.path.join(output_dir, 'test', f'{spk_id}_{pair_id}_target.wav')
        spk1_target_spec_path = os.path.join(output_dir, 'test', f'{spk_id}_{pair_id}_target.pt')
        mixed_wav_path = os.path.join(output_dir, 'test', f'{spk_id}_{pair_id}_mixed.wav')
        mixed_spec_path = os.path.join(output_dir, 'test', f'{spk_id}_{pair_id}_mixed.pt')
        spk1_dvector_wav_path = os.path.join(output_dir, 'test', f'{spk_id}_{pair_id}_dvector.pt')
    
    # save the spectrogram of spk1 and mixed audio
    spk1_part_spec, _ = wav2spec(spk1_part)
    mixed_spec, _ = wav2spec(mixed)
    torch.save(torch.from_numpy(spk1_part_spec), spk1_target_spec_path)
    torch.save(torch.from_numpy(mixed_spec), mixed_spec_path)
    
    # save the wav file and dvector
    spk1_part_tensor = torch.from_numpy(spk1_part).unsqueeze(0)
    mixed_tensor = torch.from_numpy(mixed).unsqueeze(0)
    dvector_tensor = torch.from_numpy(dvector).float()
    torchaudio.save(spk1_target_wav_path, spk1_part_tensor, args.sample_rate)
    torchaudio.save(mixed_wav_path, mixed_tensor, args.sample_rate)  
    torch.save(dvector_tensor, spk1_dvector_wav_path)
    
if __name__ == "__main__":
    # Example:
    # python preprocess.py -d ../LibriSpeech -o ./data
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--libri_dir', type=str, required=True, default=None, help='Path to LibriSpeech dataset')    
    parser.add_argument('-o', '--output_dir', type=str, required=True, default=None, help='Path to output directory')
    parser.add_argument('--sample_size', type=int, default=50, help='Number of pairs per speaker')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--audio_length', type=int, default=3, help='Audio length in seconds')
    args = parser.parse_args()
    
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)
    
    # raise error if path is not provided
    if args.libri_dir is None:
        raise ValueError("Please provide path to LibriSpeech dataset")
    if args.output_dir is None:
        raise ValueError("Please provide path to output directory")
    
    # load voice encoder, input is wav file, output is 256 dim vector
    encoder = VoiceEncoder()
    encoder.eval()
    
    # Example
    # in train-clean-100, 251 train speakers, 40 test speakers
    train_spk_dir = [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-100', '*')) if os.path.isdir(x)]
    test_spk_dir = [x for x in glob.glob(os.path.join(args.libri_dir, 'dev-clean', '*')) if os.path.isdir(x)]

    # full path to wav files (every speaker has at least 2 audios)
    train_spk_list = [glob.glob(os.path.join(spk, '**', '*.wav'), recursive=True) for spk in train_spk_dir]
    train_spk_list = [spk for spk in train_spk_list if len(spk) >= 2]
    test_spk_list = [glob.glob(os.path.join(spk, '**', '*.wav'), recursive=True) for spk in test_spk_dir]
    test_spk_list = [spk for spk in test_spk_list if len(spk) >= 2]
        
    # generate 10 pairs per speaker
    train_pairs = generate_pairs(train_spk_list, args.sample_size) # 251 * 10 = 2510
    test_pairs = generate_pairs(test_spk_list, args.sample_size) # 40 * 10 = 400

    # mix audio and prepare for training folder
    for pair in tqdm(train_pairs, total=len(train_pairs), desc="Train Pairs"):
        spk_id, pair_id, spk1_dvector, spk1_target = pair
        # choose spk2 from other speakers
        spk2_list = random.choice([train_spk_list[x] for x in range(len(train_spk_list)) if x != spk_id])
        spk2 = random.choice(spk2_list)
        mix(spk_id, pair_id, spk1_dvector, spk1_target, spk2, args.output_dir, train=True, encoder=encoder)
    
    # mix audio and prepare for testing folder
    for pair in tqdm(test_pairs, total=len(test_pairs), desc="Test Pairs"):
        spk_id, pair_id, spk1_dvector, spk1_target = pair
        # choose spk2 from other speakers
        spk2_list = random.choice([test_spk_list[x] for x in range(len(test_spk_list)) if x != spk_id])
        spk2 = random.choice(spk2_list)
        mix(spk_id, pair_id, spk1_dvector, spk1_target, spk2, args.output_dir, train=False, encoder=encoder)
