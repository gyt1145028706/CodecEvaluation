
## Introduction



## 代码结构
- semantic_evaluation 路径下为评测 codec / asr model 的 semantic 性能的代码
- reconstruct_evaluation 路径下为评测 codec 的重建性能的代码
- speechtokenizer 路径下为 codec / asr model
- utils 为一些通用代码

## Installation

```bash
conda create -n codecevaluation python=3.10 -y
pip install -r requirements.txt
```

## Intruduction

### Reconstruct Evaluation
在 librispeech-test-clean 上评测 Speaker similarity, STOI, PESQ 

### Semantic  Evaluation
用 codec 的 encoder + vq 来微调一个 ASR 任务，并评测英文数据集的 WER，中文数据集的 CER
具体来说，在 codec encoder 和 vq 后面接一个 dim 为 1024 的两层双向 LSTM，再接一个 CTC
微调时，训练数据集使用 librispeech train-clean-100, 评测数据集使用 librispeech-test-clean

### Prepare your codec model
以 speechtokenizer 为例，评测 codec model / asr model 需要满足如下条件
1. 在 codec model 类中提供如下成员变量，其中 sampling_rate 表示采样率，downsample_rate 表示下采样率，code_dim 为隐藏层的 embedding 大小
    - sampling_rate
    - downsample_rate
    - code_dim
对于 codec，一般来讲会用 rvq / fsq 后的隐层来微调一个下游 asr 模型
对于 asr 模型，一般会用 transformer 的最顶层 / 所有层的平均来微调一个下游 asr 模型

2. 在 [spt_utils](./utils/spt_utils.py) 添加你的 codec / asr model，例如
```python
    if args.model_type == "SpeechTokenizer":
        codec_model = load_and_fix_speechtokenizer(args.config, args.codec_ckpt)
        target_frame_rate_before_ctc = 50
```

CTC 的注意事项
- 假设 CTC 的输入长度为 x, 目标序列的字符数量为 y, 则必须满足 x >= 2 * y + 1，详见 [ctc](https://distill.pub/2017/ctc/)
- 因此对于比特率较低的 codec / asr 模型，我们在微调 lstm-ctc 的 asr 下游任务时，会将隐层 replicate 到 50hz 及以上
- 例如你的 codec vq 部分为 25hz, 则需要传入 `target_frame_rate_before_ctc = 50`

### Reconstruct Evaluation

注意修改启动脚本中的 `model_type`, `config`, `codec_ckpt` 等变量
```bash 
sbatch reconstruct_evaluation/submit_reconstruct_evaluation.sh
```

### Semantic Evaluation


注意修改启动脚本中的 `model_type`, `config`, `codec_ckpt` 等变量
```bash 
sbatch semantic_evaluation/submit_semantic_evaluation.sh
```