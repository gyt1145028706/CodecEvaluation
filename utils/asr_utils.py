import torch
import os
import logging
import sys


def get_chinese_characters_and_ascii_chars():
    """
    获取所有常见的汉字（基本汉字和扩展汉字）
    """
    chinese_chars = []
    for codepoint in range(1, 256):
        chinese_chars.append(chr(codepoint))
    
    # 基本汉字范围: 0x4e00 to 0x9fff
    for codepoint in range(0x4e00, 0x9fff + 1):
        chinese_chars.append(chr(codepoint))
    
    # 扩展汉字 A-Z 范围
    for codepoint in range(0x3400, 0x4DBF + 1):
        chinese_chars.append(chr(codepoint))
    
    # 扩展汉字 B-Z 范围 (例如：0x20000到0x2A6DF)
    for codepoint in range(0x20000, 0x2A6DF + 1):
        chinese_chars.append(chr(codepoint))
    
    return chinese_chars

class ASR_Utils:
    def __init__(self, language):
        assert language in ['EN', 'ZH'], f"expect language in ['EN', 'ZH'], found language = {language}"
        self.language = language
        
        # ctc 的 <blank> token，词表中下标固定为 0 
        self.blank_token = '<blank>' 
    
        # 如果是英文，则只考虑 ASCII 范围内的字符
        if self.language == 'EN':
            # 创建字符到索引的映射，[1, 256) 对应 ASCII 字符
            char_to_index = {chr(i): i for i in range(1, 256)}
            char_to_index[self.blank_token] = 0  # <blank> 0，注意与 CodecForCTC 中一致
            
            # 创建索引到字符的映射
            index_to_char = {i: char for char, i in char_to_index.items()}
        
        # 如果是中文，考虑 ASCII 范围内的字符，和所有中文字符
        elif self.language == 'ZH':
            chinese_chars_and_ascii_char = get_chinese_characters_and_ascii_chars()
            
            # 创建字符到索引的映射，[1, chinese_chars] 对应汉字
            
            char_to_index = {char: i + 1 for i, char in enumerate(chinese_chars_and_ascii_char)}
            char_to_index[self.blank_token] = 0    
        
            # 创建索引到字符的映射
            index_to_char = {i: char for char, i in char_to_index.items()}
        else:
            assert False
        
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char
        self.num_chars = len(char_to_index)
        logging.info(f"Init asr utils for language {self.language} success, num_chars(included <blank>) is {self.num_chars}")
    
    
    def text_to_labels(self, text):
        """
        将转录文本转换为标签序列。
        
        参数:
            text (str): 转录文本，例如 "hello world"
        
        返回:
            list of int: 标签序列
        """
        try:
            labels = [self.char_to_index[char] for char in text]
        except Exception as e:
            logging.info(f"Error transform text to lables, language = {self.language}, text = {text}")
            assert False
            
        return labels

    def decode_predictions(self, log_probs, predict_lable_lengths):
        """
        使用贪婪解码从模型的 log_probs 中生成预测的转录文本。
        
        参数:
            log_probs : 模型输出的对数概率，形状为 (len, bs, num_chars)
            predict_lable_lengths: batch 中每个 sample 的实际长度
        返回:
            list of str: 预测的转录文本列表
        """
        assert log_probs.shape[1] == len(predict_lable_lengths)
        
        # 获取每个时间步的最大概率索引
        _, preds = torch.max(log_probs, dim=2)  # (len, bs)
        preds = preds.transpose(0, 1)  # (bs, len)
        
        decoded_batch = []
        for (pred, pred_len) in zip(preds, predict_lable_lengths):
            pred = pred.tolist()
            decoded = []
            previous = None
            for i, p in enumerate(pred):
                # 如果超过了实际长度，则直接退出
                if i >= pred_len:
                    break
                """
                如果当前是  <blank> 则直接跳过
                如果当前不是 <blank> 且
                    前面一个是 <blank> 或
                    前面一个不是 <blank> 且和当前不同
                    则加到结果中 
                """
                if p != previous and p != self.char_to_index[self.blank_token]:
                    decoded.append(self.index_to_char[p])
                previous = p
            decoded_text = ''.join(decoded)
            decoded_batch.append(decoded_text)
        return decoded_batch

    def prepare_labels(self, transcriptions):
        """
        将一组转录文本转换为标签序列，并计算标签长度。
        
        参数:
            transcriptions (list of str): 转录文本列表
        
        返回:
            tuple: 
                batch_labels (torch.Tensor): 标签序列
                label_lengths (torch.Tensor): 每个样本的标签长度
        """
        batch_labels = []
        label_lengths = []
        for transcription in transcriptions:
            labels = self.text_to_labels(transcription)
            batch_labels.extend(labels)
            label_lengths.append(len(labels))
        
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)
        return batch_labels, label_lengths