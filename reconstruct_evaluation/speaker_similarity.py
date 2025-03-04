import argparse
import os
import numpy as np
import logging

from stopes.eval.vocal_style_similarity.vocal_style_sim_tool import get_embedder, compute_cosine_similarity

from utils.helpers import find_audio_files

def evaluate_sim(ref_path, syn_path):
    logging.info(f"Evaluating Speaker Similarity: ref_path = {ref_path}, syn_path = {syn_path}")
    ref_audio_list = find_audio_files(ref_path)
    syn_audio_list = find_audio_files(syn_path)
    logging.info(f"ref_files num = {len(ref_audio_list)}")
    logging.info(f"syn_files num = {len(syn_audio_list)}")

    model_path = "/remote-home1/ytgong/model/wavlm_large_finetune/wavlm_large_finetune.pth"
    embedder = get_embedder(model_name="valle", model_path=model_path)
    src_embs = embedder(ref_audio_list)
    tgt_embs = embedder(syn_audio_list)
    similarities = compute_cosine_similarity(src_embs, tgt_embs)
    return similarities, np.mean(similarities)

