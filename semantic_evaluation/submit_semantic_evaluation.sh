#!/usr/bin/env bash

#SBATCH -p warmup
#SBATCH --job-name=en_finetune_speechtokenizer_release_for_ctc
#SBATCH --nodelist=fnlp-4090-59110
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20G

#SBATCH --output=/remote-home1/ytgong/CodecEvaluation/semantic_evaluation/slurmlogs/%x_stdout.log  # STDOUT
#SBATCH  --error=/remote-home1/ytgong/CodecEvaluation/semantic_evaluation/slurmlogs/%x_stderr.log   # STDERR

source ~/.bashrc
conda activate codecevaluation
which python

work_dir=/remote-home1/ytgong/CodecEvaluation
cd ${work_dir}
export PYTHONPATH=./

model_type=SpeechTokenizer
exp_root=semantic_evaluation/exp


language=EN # ! 需要修改
tag=${model_type}/${language}/v1.0/spt1_release # ! 需要修改

config=config/spt_base_cfg.json # ! 需要修改
codec_ckpt=/remote-home1/ytgong/model/speechtokenizer/SpeechTokenizer.pt # ! 需要修改
usefull=1 # ! 需要修改



# 不需要修改
cmd="python ${work_dir}/semantic_evaluation/finetune_codecforctc.py \
--model_type ${model_type} \
--tag ${tag} \
--exp_root ${exp_root} \
--config ${config} \
--codec_ckpt ${codec_ckpt} \
--language ${language} \
--usefull ${usefull} "
echo "Executing: $cmd"
eval $cmd