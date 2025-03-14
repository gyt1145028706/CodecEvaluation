# CodecEvaluation

## Introduction

`CodecEvaluation` is a comprehensive framework designed to evaluate the performance of codec and ASR models in both **reconstruction** and **semantic** tasks.

## Code Structure
- `semantic_evaluation/` - Code for evaluating the semantic performance of codec/ASR models.
- `reconstruct_evaluation/` - Code for evaluating the reconstruction performance of codec models.
- `speechtokenizer/` - Contains codec or ASR models.
- `utils/` - Utility scripts and common functions.

## Installation

```bash
# Clone the repository
git clone git@github.com:gyt1145028706/CodecEvaluation.git
cd CodecEvaluation

# Create a Conda environment and install dependencies
conda create -n codecevaluation python=3.10 -y
pip install -r requirements.txt

# Set the Python path
export PYTHONPATH=./
```

## Evaluation Tasks

### Reconstruction Evaluation
Evaluates codec reconstruction performance on the `librispeech-test-clean` dataset using the following metrics:
- **Speaker Similarity** - Assessed using a [WavLM-based speaker verification model](https://huggingface.co/Dongchao/UniAudio/resolve/main/wavlm_large_finetune.pth) (SPK SIM). Code available at [speaker_similarity.py](reconstruct_evaluation/speaker_similarity.py).
- **STOI** - Short-Time Objective Intelligibility. Code available at [stoi.py](reconstruct_evaluation/stoi.py).
- **PESQ** - Perceptual Evaluation of Speech Quality. Code available at [pesq_local.py](reconstruct_evaluation/pesq_local.py).

### Semantic Evaluation
Fine-tunes an ASR task using:

#### Model
- **Codec/ASR encoder**.
- **Two-layer bidirectional LSTM** with a hidden dimension of **1024**.
- **CTC (Connectionist Temporal Classification) decoder**.

Code available at [finetune_codecforctc.py](semantic_evaluation/finetune_codecforctc.py).

#### Datasets
- **Training dataset**: `librispeech train-clean-100`.
- **Evaluation dataset**: `librispeech-test-clean`.

## Preparing Your Codec Model
To integrate a codec or ASR model for evaluation, ensure the model class provides the following attributes:
- `sampling_rate` - Sample rate of the model.
- `downsample_rate` - Downsampling rate.
- `code_dim` - Embedding size for ASR fine-tuning.
- `forward` method returns a dictionary with:
  - A key **"y"** containing synthesized audio `shape = (B, 1, T)` - *not required for ASR models*.
  - A key **"zq"** containing embeddings for downstream ASR fine-tuning `shape = (B, D, L)`.

For codec models, the hidden representation after RVQ/FSQ is typically used for ASR fine-tuning.

For ASR models, either the top Transformer layer or an average of all layers is used.

Code available at [model.py](speechtokenizer/model.py).

To add a new codec/ASR model, modify [`spt_utils.py`](./utils/spt_utils.py) as follows (example for SpeechTokenizer):

```python
if args.model_type == "SpeechTokenizer":
    codec_model = load_and_fix_speechtokenizer(args.config, args.codec_ckpt)
    target_frame_rate_before_ctc = 50
elif args.model_type == "<your codec / asr model type>":
    codec_model = your_codec_or_asr_model # Ensure all parameters are fixed
    target_frame_rate_before_ctc = your_frame_rate  # Must be a multiple of the model's Hz and >= 50
```

### CTC Considerations
CTC requires that the input length `x` satisfies:
```
x >= 2 * y + 1
```
where `y` is the target sequence length. More details can be found in this [CTC guide](https://distill.pub/2017/ctc/).

**If the input hidden sequence length is too short, the prediction results may not be accurate.**
For low-bitrate codec/ASR models, the hidden representations are upsampled to at least **50 Hz** before fine-tuning the LSTM-CTC ASR model. 
For example, if the codec's VQ operates at **25 Hz**, set:
```python
target_frame_rate_before_ctc = 50
```

## Running Evaluations

### Reconstruction Evaluation
Before running, modify `model_type`, `config`, and `codec_ckpt` in the execution script:
```bash
sbatch reconstruct_evaluation/submit_reconstruct_evaluation.sh
```

### Semantic Evaluation
Before running, modify `model_type`, `config`, and `codec_ckpt` in the execution script:
```bash
sbatch semantic_evaluation/submit_semantic_evaluation.sh
```


## Acknowledgments
This project was developed with contributions from [**SpeechTokenizer**](https://github.com/ZhangXInFD/SpeechTokenizer). We appreciate its support in providing fundamental codec model functionalities.
