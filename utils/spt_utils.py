import logging
import torch
import sys

from speechtokenizer.model import SpeechTokenizer

def load_and_fix_speechtokenizer(config_path, ckpt_path, device=torch.device("cuda")):
    speechtokenizer = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    speechtokenizer = speechtokenizer.to(device)
    speechtokenizer.eval()
    
    for param in speechtokenizer.parameters():
        param.requires_grad = False
    
    logging.info(f"Load and fix speechtokenizer of config: {config_path} from checkpoint: {ckpt_path} success")
    
    return speechtokenizer

def load_and_fix_codec_model(args):
    if args.model_type == "SpeechTokenizer":
        codec_model = load_and_fix_speechtokenizer(args.config, args.codec_ckpt)
        target_frame_rate_before_ctc = 50
    elif args.model_type == '<your model>':
        """
        sys.path.append("<model dir>")
        codec_model = ...
        target_frame_rate_before_ctc = ...
        """
    else:
        assert False, f'model type {args.model_type} not support !'
    
    for param in codec_model.parameters():
        param.requires_grad = False
    
    return codec_model, target_frame_rate_before_ctc