import numpy as np
import soundfile as sf
import librosa
import glob
import os
import logging

from pesq import pesq
from tqdm import tqdm

from utils.helpers import find_audio_files

def evaluate_pesq(ref_folder, est_folder, mode, target_sr=16000):
    logging.info(f"Evaluating PESQ-{mode}: ref_folder: {ref_folder}, syn_folder: {est_folder}")
    """
    Calculate PESQ (Perceptual evaluation of speech quality) metric between pairs of reference and estimated audio files
    located in the given directories, optionally resampling all files to a specified sample rate.
    
    Parameters:
        ref_folder (str): The folder path containing the reference audio files (.wav).
        est_folder (str): The folder path containing the estimated/generated audio files (.wav).
    
    Returns:
        dict: A dictionary containing the STOI for each pair of audio files, with file names as keys.
    """
    ref_files = find_audio_files(ref_folder)
    est_files = find_audio_files(est_folder)
    
    logging.info(f"Searching ref audios in {os.path.join(ref_folder, '*.flac')}")
    logging.info(f"Searching syn audios in {os.path.join(est_folder, '*.flac')}")
    logging.info(f"ref_files num = {len(ref_files)}")
    logging.info(f"syn_files num = {len(est_files)}")
    
    if len(ref_files) != len(est_files):
        raise ValueError("The number of reference and estimated files do not match.")
    
    pesq_score = {}
    mean_score = []
    index = 0
    
    progress_bar = tqdm(zip(ref_files, est_files), desc=f"PESQ-{mode} evaluating....")
    for ref_path, est_path in progress_bar:
        index += 1
        assert os.path.basename(ref_path) == os.path.basename(est_path), "File Name must be same !"
        
        ref_audio, ref_rate = sf.read(ref_path)
        est_audio, est_rate = sf.read(est_path)
        
        if est_rate != ref_rate:
            est_audio = librosa.resample(est_audio, orig_sr=est_rate, target_sr=ref_rate)

        min_len = min(ref_audio.shape[0], est_audio.shape[0])
        ref_audio = ref_audio[:min_len]
        est_audio = est_audio[:min_len]
        
        if ref_audio.ndim > 1:
            ref_audio = ref_audio[:, 0]
        if est_audio.ndim > 1:
            est_audio = est_audio[:, 0]

        score = pesq(target_sr, ref_audio, est_audio, mode = mode)
        mean_score.append(score)
        pesq_score[os.path.basename(ref_path)] = score
        
        progress_bar.set_postfix(index = index, ref_path = ref_path, est_path = est_path, pesq_score = score)
    
    
    return pesq_score, np.mean(mean_score)