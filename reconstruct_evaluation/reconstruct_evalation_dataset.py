import os
import glob
import torch

from typing import Tuple, List
from torch.utils.data import Dataset

from utils.helpers import load_audio, find_audio_files

class DatasetForReconstructEvaluation(Dataset):
    def __init__(self, directory: str, target_sample_rate: int = 16000):
        """
        初始化 TestDataset。

        Args:
            directory (str): 包含音频文件的目录路径。
            target_sample_rate (int): 目标采样率，默认为 16000 Hz。
        """
        self.directory = directory
        self.target_sample_rate = target_sample_rate
        self.audio_paths = self._get_audio_files(directory)
        
    def _get_audio_files(self, directory: str) -> List[str]:
        """
        获取指定目录下所有音频文件的路径。

        Args:
            directory (str): 目录路径。

        Returns:
            List[str]: 所有音频文件的完整路径列表。
        """
        audio_paths = find_audio_files(directory)
        if not audio_paths:
            raise ValueError(f"No audio files found in directory: {directory}")
        return sorted(audio_paths)

    def __len__(self) -> int:
        """
        返回数据集的大小。

        Returns:
            int: 数据集中 audio 文件的数量。
        """
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        """
        获取指定索引的数据项。

        Args:
            idx (int): 数据项的索引。

        Returns:
            Tuple[str, torch.Tensor]: 
            waveform of shape (1, L)
        """
        audio_path = self.audio_paths[idx]
        waveform = load_audio(audio_path=audio_path, target_sample_rate=self.target_sample_rate).reshape(1, -1)
        
        return audio_path, waveform