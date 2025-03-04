# CodecEvaluation

## Introduction

`CodecEvaluation` is a comprehensive evaluation framework for assessing the performance of codec and ASR models in both reconstruction and semantic tasks.

## Code Structure
- `semantic_evaluation/` - Code for evaluating the semantic performance of codec/ASR models.
- `reconstruct_evaluation/` - Code for evaluating the reconstruction performance of codec models.
- `speechtokenizer/` - Contains codec and ASR models.
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
- **Speaker Similarity**
- **STOI** (Short-Time Objective Intelligibility)
- **PESQ** (Perceptual Evaluation of Speech Quality)

### Semantic Evaluation
Fine-tunes an ASR task using the codec's encoder and vector quantization (VQ) components, then evaluates:
- **WER** (Word Error Rate) on English datasets
- **CER** (Character Error Rate) on Chinese datasets

The ASR fine-tuning setup includes:
- A **two-layer bidirectional LSTM** with a hidden dimension of **1024**.
- A **CTC (Connectionist Temporal Classification) decoder**.
- Training dataset: `librispeech train-clean-100`
- Evaluation dataset: `librispeech-test-clean`

## Preparing Your Codec Model
To integrate a codec or ASR model for evaluation, ensure the model class provides the following attributes:
- `sampling_rate`: Sample rate of the model.
- `downsample_rate`: Downsampling rate.
- `code_dim`: Hidden layer embedding size.

For codec models, the hidden representation after RVQ/FSQ is typically used for fine-tuning the ASR model. 
For ASR models, either the top Transformer layer or an average of all layers is used for fine-tuning.

To add a new codec/ASR model, modify [`spt_utils.py`](./utils/spt_utils.py) as follows:

```python
if args.model_type == "SpeechTokenizer":
    codec_model = load_and_fix_speechtokenizer(args.config, args.codec_ckpt)
    target_frame_rate_before_ctc = 50
```

### CTC Considerations
CTC requires that the input length `x` satisfies:
```
x >= 2 * y + 1
```
where `y` is the target sequence length. More details can be found in this [CTC guide](https://distill.pub/2017/ctc/).

For low-bitrate codec/ASR models, the hidden representations are upsampled to at least **50 Hz** before fine-tuning the LSTM-CTC ASR model. 
For example, if the codec's VQ operates at **25 Hz**, set:
```python
target_frame_rate_before_ctc = 50
```

## Running Evaluations

### Reconstruction Evaluation
Modify the variables `model_type`, `config`, and `codec_ckpt` in the execution script before running:
```bash
sbatch reconstruct_evaluation/submit_reconstruct_evaluation.sh
```

### Semantic Evaluation
Modify the variables `model_type`, `config`, and `codec_ckpt` in the execution script before running:
```bash
sbatch semantic_evaluation/submit_semantic_evaluation.sh
