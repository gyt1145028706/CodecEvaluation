import torch
import argparse
import sys
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from jiwer import wer, cer
import torchaudio
from tensorboardX import SummaryWriter
import logging
import random
import numpy as np

from semantic_evaluation.dataset import get_dataset, collate_fn
from semantic_evaluation.ctc_model import CodecForCTC

from utils.helpers import set_logging, count_params_by_module, waiting_for_debug
from utils.spt_utils import load_and_fix_codec_model
from utils.asr_utils import ASR_Utils

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finetune_codec_for_ctc(args): 
    # basic settings
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # tensorboard settings
    writer = SummaryWriter(os.path.join(args.exp_root, args.tag))
    save_to_tensorboard_cnt = args.save_to_tensorboard_cnt
    
    # language
    language = args.language
    asr_utils = ASR_Utils(language=language)
    
    # 读取 Codec 模型并固定，获取 CodecForCTC 模型
    codec_model, target_frame_rate_before_ctc = load_and_fix_codec_model(args)
    codec_for_ctc = CodecForCTC(
        codec_model=codec_model, 
        num_chars=asr_utils.num_chars,
        target_frame_rate=target_frame_rate_before_ctc
    )
    codec_for_ctc = codec_for_ctc.to(device)
    count_params_by_module("codec_model", codec_model)
    count_params_by_module("codec_for_ctc", codec_for_ctc)
    logging.info(f"codec_for_ctc model: \n{codec_for_ctc}")
    
    
    # dataset 和 dataloader
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, dev_dataset = get_dataset(
        language=args.language, 
        usefull=args.usefull, 
        sample_rate=codec_for_ctc.codec_model.sampling_rate, 
        librispeech_path=args.librispeech_path,
        aishell2_path=args.aishell2_path
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    logging.info(f"Num train steps in one epoch: {len(train_loader)}")
    logging.info(f"Num dev steps in one epoch :   {len(dev_loader)}")


    # train settings
    lr=args.lr
    num_epochs=args.num_epochs
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, codec_for_ctc.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # start finetune codec for ctc
    global_steps = 0
    for epoch in range(1, num_epochs + 1): 
        codec_for_ctc.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training")
        for batch in progress_bar:
            global_steps += 1
            # prepare input
            audionames, padded_audios, transcriptions, lengths = batch 
            padded_audios = padded_audios.to(device)  # (batch_size, max_length, channels)
            
            # 转录文本转换为标签
            labels, label_lengths = asr_utils.prepare_labels(transcriptions)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            loss, log_probs, predict_lable_lengths = codec_for_ctc(padded_audios, labels, lengths, label_lengths)
            
            # 反向传播
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(codec_for_ctc.parameters(), max_norm=5.0)
            
            # 更新参数
            optimizer.step()
            
            total_loss += loss.item()
            logging.info(f"epochs: {epoch} global_steps: {global_steps}, train loss: {loss.item()}")    
            
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
            writer.add_scalar("train/loss", loss.item(), global_steps)
        
        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")
        
        # 评估
        codec_for_ctc.eval()
        
        eval_total_loss = 0
        with torch.no_grad():
            all_preds = []
            all_targets = []
            progress_bar = tqdm(dev_loader, desc=f"Epoch {epoch}/{num_epochs} - Evaluating")
            saved_cnt = 0
            for batchidx, batch in enumerate(progress_bar):
                audionames, padded_audios, transcriptions, lengths = batch
                padded_audios = padded_audios.to(device)
                
                labels, label_lengths = asr_utils.prepare_labels(transcriptions)
                labels = labels.to(device)
                label_lengths = label_lengths.to(device)
                loss, log_probs, predict_lable_lengths = codec_for_ctc(padded_audios, labels=labels, input_lengths=lengths, label_lengths=label_lengths)
                eval_total_loss += loss.item()
                # 解码预测
                decoded_batch = asr_utils.decode_predictions(log_probs, predict_lable_lengths) # ???? 这边解码的时候是不是要 去掉 pad 的部分？
                
                all_targets.extend(transcriptions)
                all_preds.extend(decoded_batch)
                for i, (audioname, pred, ground_truth) in enumerate(zip(audionames, decoded_batch, transcriptions)): 
                    if saved_cnt >= save_to_tensorboard_cnt:
                        break
                    saved_cnt += 1
                    audio, audio_sample_rate = torchaudio.load(audioname)
                    writer.add_audio(f"val/audio_{saved_cnt}", audio.transpose(0, 1).numpy(), global_steps, sample_rate=audio_sample_rate)
                    writer.add_text(f"val/pred_{saved_cnt}", pred, global_steps)
                    writer.add_text(f"val/ground_truth_{saved_cnt}", ground_truth, global_steps)
                    
                logging.info(f"global_steps: {global_steps}, eval batch index = {batchidx}, eval loss: {loss.item()}") 
            
            eval_total_loss /= len(dev_loader)
            scheduler.step(eval_total_loss)
            logging.info(f"global_steps: {global_steps}, eval_total_loss: {eval_total_loss}")
            
            if language == "EN":
                calculated_wer = wer(all_targets, all_preds)
                writer.add_scalar("val/wer", calculated_wer, global_steps)
                writer.add_scalar("val/loss", eval_total_loss, global_steps)
                logging.info(f"Epoch {epoch}/{num_epochs}, Validation WER: {calculated_wer:.4f}")
            
            elif language == "ZH":
                calculated_cer = cer(all_targets, all_preds)
                writer.add_scalar("val/cer", calculated_cer, global_steps)
                writer.add_scalar("val/loss", eval_total_loss, global_steps)
                logging.info(f"Epoch {epoch}/{num_epochs}, Validation CER: {calculated_cer:.4f}")

            logging.info(f"Epoch {epoch} finished" + "-" * 100)

def main():
    set_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--exp_root", type=str, default="exp")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--codec_ckpt", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--usefull", type=int, required=True)
    
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--save_to_tensorboard_cnt", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--librispeech_path", type=str, default="/remote-home1/share/data/SpeechPretrain/librispeech/LibriSpeech/")
    parser.add_argument("--aishell2_path", type=str, default="/remote-home1/share/data/SpeechPretrain/AIShell-2/data/")
    
    parser.add_argument("--debug", default=0, type=int, nargs="?", help='whether debug or not')    
    parser.add_argument('--debug_ip', default='localhost', type=str)
    parser.add_argument('--debug_port', default=32431, type=int)
    
    args = parser.parse_args()
    if args.debug == 1:
        waiting_for_debug(args.debug_ip, args.debug_port)
        
    finetune_codec_for_ctc(args)

if __name__ == "__main__":
    main()