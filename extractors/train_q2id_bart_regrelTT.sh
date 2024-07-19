#!/bin/bash
#SBATCH --job-name=relregTT_bart
#SBATCH --mail-type="ALL"
#SBATCH --time=6-00:00:00
#SBATCH --partition=cpu-long
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=10

#module load ALICE/default # you can access the AMD software stack like this, also works for Intel nodes
# module load 2023
# module load CUDA/12.1.1
source ~/.bashrc
# conda init bash doesn't work
source /home/wangym/data1/miniconda3/etc/profile.d/conda.sh
conda activate relreg
echo $CONDA_DEFAULT_ENV
echo $PYTHONPATH

RELREG_OUTPUT_DIR='/home/wangym/data1/output-relregTT-utt'
PATH_TO_CHECKPOINT='facebook/bart-large'
OUTPUT='/home/wangym/data1/output/relregTT'

NAME=relregTT-q2id-128-bart-large
NUM_RUNS=5
START=1
# CUDA_VISIBLE_DEVICES=0 
for RUN in $(seq $START $NUM_RUNS)
do
 CUDA_VISIBLE_DEVICES="" python -u ../multiencoder/train.py \
 --train_file $RELREG_OUTPUT_DIR/train.csv \
 --validation_file $RELREG_OUTPUT_DIR/val.csv \
 --do_train \
 --do_eval \
 --learning_rate 0.000005 \
 --model_name_or_path $PATH_TO_CHECKPOINT \
 --metric_for_best_model eval_mean_rouge \
 --output_dir $OUTPUT/${NAME}_${RUN} \
 --per_device_train_batch_size 4 \
 --max_source_length 1024 \
 --generation_max_len 128 \
 --val_max_target_length 128 \
 --overwrite_output_dir \
 --per_device_eval_batch_size 4 \
 --predict_with_generate \
 --evaluation_strategy epoch \
 --num_train_epochs 10 \
 --save_strategy epoch \
 --logging_strategy epoch \
 --load_best_model_at_end \
 --compute_rouge_for_train True \
 --seed $RUN &> ${NAME}_${RUN}.out
done