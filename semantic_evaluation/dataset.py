import os
import torch
import torchaudio
import numpy as np
import logging
import random

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset

class LibriSpeechDataset(Dataset):
    def __init__(self, dataset_directory, sample_rate):
        """
        初始化 LibriSpeech 数据集。

        参数:
            directory (str): 数据集的根目录路径，例如 "/remote-home1/share/data/SpeechPretrain/librispeech/LibriSpeech/train-clean-100"
        """
        self.data_directory = dataset_directory
        self.pairs = []
        self.sample_rate = sample_rate  
        self.name2transcription_and_filepath = {} # filename -> transcription 的 map        
        
        
        # 遍历目录中的所有 .trans.txt 文件
        for root, _, files in os.walk(dataset_directory):
            for file in files:
                if file.endswith('.trans.txt'):
                    trans_file_path = os.path.join(root, file)
                    self._parse_transcription_file(trans_file_path)
                    
        logging.info(f"Found {len(self.pairs)} audios in {self.data_directory}")
        
    
    def _parse_transcription_file(self, trans_file_path):
        """
        解析单个 .trans.txt 文件，并将 (flac_path, transcription) 对添加到 self.pairs 中。

        参数:
            trans_file_path (str): .trans.txt 文件的完整路径
        """
        with open(trans_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) != 2:
                    continue  # 跳过格式不正确的行
                file_id, transcription = parts
                flac_file = f"{file_id}.flac"
                flac_path = os.path.join(os.path.dirname(trans_file_path), flac_file)
                if os.path.isfile(flac_path):
                    self.pairs.append((flac_path, transcription))
                    self.name2transcription_and_filepath[os.path.basename(flac_path)] = (transcription, flac_path)
                else:
                    logging.info(f"Warning: Can't find file {flac_path}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据对。

        参数:
            idx (int): 索引

        返回:
            tuple: (audio_tensor, transcription)
                   audio_tensor: 音频数据的 Tensor，形状为 (samples, channels)
                   transcription: 对应的转录文本
        """
        flac_path, transcription = self.pairs[idx]
        
        # 读取音频文件
        audio, raw_sample_rate = torchaudio.load(flac_path) # (1, T)   tensor 
       
        if raw_sample_rate != self.sample_rate:   
            audio = torchaudio.functional.resample(audio, raw_sample_rate, self.sample_rate) # tensor 
       
        audio = np.expand_dims(audio.squeeze(0).numpy(), axis=1)
        
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        return flac_path, audio_tensor, transcription


class AIShell2Dataset(Dataset):
    def __init__(self, dataset_directory, sample_rate, usefull):
        """
        初始化 AIShell2 数据集。

        参数:
            dataset_directory (str): AIShell2 根目录路径
                目录下面需要有 trans.txt，每一行为: filename text, 对应音频 ${filename}.txt
            sample_rate
            usefull:
                如果为 True, 则用于微调，取 aishell2 中前 10w 条样本
                如果为 False, 则用于调试，取前 1000 条样本
        """
        super(AIShell2Dataset, self).__init__()
        self.data_directory = dataset_directory
        self.pairs = []
        self.sample_rate = sample_rate  

        # 构建 wav 文件的 basename 到完整路径的映射
        self.wav_mapping = self._build_wav_mapping()

        # 定位 trans.txt 文件
        trans_file_path = os.path.join(dataset_directory, 'trans.txt')
        if not os.path.isfile(trans_file_path):
            raise FileNotFoundError(f"trans.txt not found in {dataset_directory}")

        self.usefull = usefull
        self.max_audio_cnt = 100000 if self.usefull else 1000
        
        # 解析 trans.txt 文件
        self._parse_transcription_file(trans_file_path)

        logging.info(f"Found {len(self.pairs)} audio-transcription pairs in {self.data_directory}")

    def _build_wav_mapping(self):
        """
        构建一个字典，将所有 .wav 文件的 basename 映射到它们的完整路径。

        返回:
            dict: { 'IC0001W0001.wav': '/path/to/AIShell2/wav/IC0001W0001.wav', ... }
        """
        wav_mapping = {}
        for root, _, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith('.wav'):
                    wav_mapping[file] = os.path.join(root, file)
        logging.info(f"Indexed {len(wav_mapping)} wav files.")
        return wav_mapping

    def _parse_transcription_file(self, trans_file_path):
        """
        解析 trans.txt 文件，并将 (wav_path, transcription) 对添加到 self.pairs 中。

        参数:
            trans_file_path (str): trans.txt 文件的完整路径
        """
        with open(trans_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.strip().split('\t', 1)
                if len(parts) != 2:
                    continue  # 跳过格式不正确的行
                file_id, transcription = parts
                wav_file = f"{file_id}.wav"

                # 使用 wav_mapping 查找 wav 文件的完整路径
                wav_path = self.wav_mapping.get(wav_file, None)

                if wav_path and os.path.isfile(wav_path):
                    self.pairs.append((wav_path, transcription))
                    
                if len(self.pairs) >= self.max_audio_cnt:
                    break
                
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        获取指定索引的数据对。

        参数:
            idx (int): 索引

        返回:
            tuple: (audio_tensor, transcription)
                   audio_tensor: 音频数据的 Tensor，形状为 (samples, channels)
                   transcription: 对应的转录文本
        """
        flac_path, transcription = self.pairs[idx]
        
        # 读取音频文件
        audio, raw_sample_rate = torchaudio.load(flac_path) # (1, T)   tensor 
       
        if raw_sample_rate != self.sample_rate:   
            audio = torchaudio.functional.resample(audio, raw_sample_rate, self.sample_rate) # tensor 
       
        audio = np.expand_dims(audio.squeeze(0).numpy(), axis=1)
        
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        return flac_path, audio_tensor, transcription

def split_aishell2_dataset(dataset, usefull):
    """
    将 AIShell2Dataset 拆分为训练集和验证集。

    参数:
        dataset (AIShell2Dataset): AIShell2Dataset 实例
        use_full (int): 取 10w 条还是 1000 条

    返回:
        tuple: (train_dataset, val_dataset)
    """
    total_samples = len(dataset.pairs)
    indices = list(range(total_samples))
    random.shuffle(indices)

    if usefull == 1:
        sample_size = 100000
        if total_samples < sample_size:
            raise ValueError(f"数据集样本不足 {sample_size} 条，仅有 {total_samples} 条。")
        sampled_indices = indices[:sample_size]
        val_size = 1000
        train_size = 99000
    else:
        sample_size = 1000
        if total_samples < sample_size:
            raise ValueError(f"数据集样本不足 {sample_size} 条，仅有 {total_samples} 条。")
        sampled_indices = indices[:sample_size]
        val_size = 100
        train_size = 900

    val_indices = sampled_indices[:val_size]
    train_indices = sampled_indices[val_size:val_size + train_size]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    logging.info(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")

    return train_subset, val_subset

def collate_fn(batch):
    """
    自定义的 collate_fn，用于将 batch 中的音频序列填充到相同的长度。

    参数:
        batch (list): 包含多个 (audio_tensor, transcription) 对的列表

    返回:
        tuple: (padded_audios, transcriptions, lengths)
               padded_audios: 填充后的音频 Tensor，形状为 (batch_size, channels, max_length)
               transcriptions: 转录文本的列表
               lengths: 原始音频长度的列表
    """
    audionames, audios, transcriptions = zip(*batch)
    
    # 获取每个音频的长度
    lengths = torch.tensor([audio.shape[0] for audio in audios], dtype=torch.long)
    
    # 填充音频序列，使它们具有相同的长度
    padded_audios = pad_sequence(audios, batch_first=True, padding_value=0).transpose(1, 2)  # (batch_size, channels, max_length)
    
    return audionames, padded_audios, transcriptions, lengths

def get_dataset(language, usefull, sample_rate, librispeech_path, aishell2_path):
    if language == "EN":
        if usefull == 1:
            train_dir = os.path.join(librispeech_path, "train-clean-100") 
            dev_dir   = os.path.join(librispeech_path, "dev-clean")
        else:
            train_dir = os.path.join(librispeech_path, "train-clean-100", "322/124146")
            dev_dir   = os.path.join(librispeech_path, "dev-clean", "2078/142845/") 
        train_dataset = LibriSpeechDataset(train_dir, sample_rate=sample_rate)
        dev_dataset   = LibriSpeechDataset(dev_dir, sample_rate=sample_rate)
        return train_dataset, dev_dataset
    elif language == "ZH":
        dataset = AIShell2Dataset(dataset_directory=aishell2_path, sample_rate=sample_rate, usefull=usefull)
        train_dataset, dev_dataset = split_aishell2_dataset(dataset, usefull)
        return train_dataset, dev_dataset
    else:
        assert False, f"expect language in ['EN', 'ZH'], found {language}"