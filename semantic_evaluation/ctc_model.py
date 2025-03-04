import torch
import torch.nn as nn
import logging

class CodecForCTC(nn.Module):
    def __init__(self, codec_model, num_chars, target_frame_rate):  # 0 为 ctc 中的空白字符，1-256 为 ascii
        """
            codec_model: 
                codec 基座模型，
                需要固定所有参数，固定 vq
                需要在 forward 的返回的 dict 中提供 key 为 "zq" 的 (key, value) pair
                需要提供如下成员变量： 
                    sampling_rate
                    downsample_rate
                    code_dim, 为 codec model 测 asr 位置的 embedding 的 dim
            num_chars: 
                词表中的大小，包括 CTC 的 <blank>
            target_frame_rate:
                用于 ctc 的部分的目标帧率，必须大于等于 50，且为 codec_model 帧率的倍数
        """
        super().__init__()  # 初始化父类
        self.num_chars = num_chars
        self.codec_model = codec_model
        
        assert self.codec_model.sampling_rate % self.codec_model.downsample_rate == 0
        self.codec_model_frame_rate = self.codec_model.sampling_rate // self.codec_model.downsample_rate 
    
        assert target_frame_rate % self.codec_model_frame_rate == 0 and target_frame_rate >= 50, f"expect target_frame_rate >= 50 and target_frame_rate % self.codec_model_frame_rate == 0, found target_frame_rate = {target_frame_rate}, self.codec_model_frame_rate = {self.codec_model_frame_rate}"
        self.target_frame_rate = target_frame_rate
    
        logging.info(f"codec model's sampling_rate is {self.codec_model.sampling_rate}")
        logging.info(f"codec model's total downsample_rate is {self.codec_model.downsample_rate}")
        logging.info(f"codec model's frame_rate is {self.codec_model_frame_rate} hz")
        logging.info(f"codecForCTC's target_frame_rate is {self.target_frame_rate} hz")
        
        # 定义 2 层、每层 1024 单元的双向 LSTM
        self.blstm = nn.LSTM(
            input_size=self.codec_model.code_dim, 
            hidden_size=1024,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # 定义线性层，将 BLSTM 的输出映射到字符概率分布
        self.classifier = nn.Linear(1024 * 2, self.num_chars)  # 双向 LSTM 输出维度为 2048

        # 定义 CTC 损失
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True) # 空白为 0

    def forward(self, x, labels, input_lengths, label_lengths):
        """
        前向传播方法。

        参数:
            x : 输入波形，形状为 (batch, channels=1, length)
            labels : 真实的字符标签，形状为 (batch, max_label_length)
            input_lengths : 每个输入波形的实际长度，形状为 (batch,)
            label_lengths : 每个标签序列的实际长度，形状为 (batch,)

        返回:
            返回 (loss, log_probs, predict_lable_lengths)
            loss 用于 backward
            log_probs 和 predict_lable_lengths 用于从概率解码为 char
        """
        
        # 无法通过设置 requires_grad 来控制是否更新 vq。这里通过 torch.is_grad_enabled() 来判断是否更新 vq
        with torch.inference_mode():
            codec_forward_result = self.codec_model(x)  # x.shape = (batch, 1, len)
            zq = codec_forward_result['zq']  # zq 形状: (batch, code_dim, len)
            
        # 如果 codec 帧率小于 self.target_frame_rate, 统一通过 repeat 补成 50hz
        replicate_times = self.target_frame_rate // self.codec_model_frame_rate
        zq = torch.repeat_interleave(zq, replicate_times, dim=-1)
        zq = zq.transpose(1, 2) # => zq.shape = (batch, len, code_dim)
        
        # 通过 BLSTM
        blstm_out, _ = self.blstm(zq)  # 形状: (batch, len, 2048)

        # 通过分类器，得到字符概率分布
        logits = self.classifier(blstm_out)  # 形状: (batch, len, num_chars)

        # 转置以符合 CTC 的输入要求 (len, batch, num_chars)
        logits = logits.transpose(0, 1)  # 形状: (len, batch, num_chars)

        # 计算 log_softmax
        log_probs = nn.functional.log_softmax(logits, dim=2)  # 形状: (len, batch, num_chars)

        # 计算 CTC 损失
        predict_lable_lengths = input_lengths * replicate_times // self.codec_model.downsample_rate
        loss = self.ctc_loss(log_probs, labels, predict_lable_lengths, label_lengths) 
        return loss, log_probs, predict_lable_lengths