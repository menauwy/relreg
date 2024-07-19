#!/bin/bash
#SBATCH --job-name=segenc
#SBATCH --mail-type="ALL"
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-medium
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1

#module load ALICE/default # you can access the AMD software stack like this, also works for Intel nodes
# module load 2023
# module load CUDA/12.1.1
source ~/.bashrc
# conda init bash doesn't work
source /home/wangym/data1/miniconda3/etc/profile.d/conda.sh
conda activate relreg
echo $CONDA_DEFAULT_ENV
echo $PYTHONPATH

NAME=q2id_8_512_strided_gpu
NUM_RUNS=5
START=1
for RUN in $(seq $START $NUM_RUNS)
do
  python -u ../train.py \
  --train_file /home/wangym/workspace/query-focused-sum/data/train_segenc.jsonl \
  --validation_file /home/wangym/workspace/query-focused-sum/data/valid_segenc.jsonl \
  --do_train \
  --do_eval \
  --learning_rate 0.000005 \
  --gradient_checkpointing \
  --model_name_or_path facebook/bart-large \
  --metric_for_best_model eval_mean_rouge \
  --output_dir /home/wangym/data1/output-segenc/${NAME}_${RUN} \
  --per_device_train_batch_size 8 \
  --max_source_length 512 \
  --generation_max_len 128 \
  --val_max_target_length 128 \
  --overwrite_output_dir \
  --per_device_eval_batch_size 8 \
  --multiencoder_type bart \
  --multiencoder_max_num_chunks 8 \
  --multiencoder_stride \
  --predict_with_generate \
  --evaluation_strategy epoch \
  --num_train_epochs 10 \
  --save_strategy epoch \
  --logging_strategy epoch \
  --load_best_model_at_end \
  --compute_rouge_for_train True \
  --seed $RUN &> ${NAME}_${RUN}.out
done
#  --metric_for_best_model rouge1_plus_rouge2 \

