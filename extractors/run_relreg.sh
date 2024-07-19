#!/bin/bash
#SBATCH --job-name=relreg
#SBATCH --mail-type="ALL"
#SBATCH --time=00:15:00
#SBATCH --partition=gpu-medium
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:4g.40gb:1

#module load ALICE/default # you can access the AMD software stack like this, also works for Intel nodes
# module load 2023
# module load CUDA/12.1.1
source ~/.bashrc
# conda init bash doesn't work
source /home/wangym/data1/miniconda3/etc/profile.d/conda.sh
conda activate relreg
echo $CONDA_DEFAULT_ENV
echo $PYTHONPATH

CHUNKS=0
# OUTPUT_DIR='output-relreg-utt'
OUTPUT_DIR='/home/wangym/data1/output-relreg-utt'


# Add chunk/utterance-level ROUGE and convert data to format required for RelReg training and inference; 0 for utterance-level data.
python add_rouge.py $CHUNKS
python prep_data_relreg.py $CHUNKS

# Train RelReg on utterance-level input
CUDA_VISIBLE_DEVICES=0 python transformers/examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path google/electra-large-discriminator \
  --train_file ../data/train.relreg.csv \
  --validation_file ../data/val.relreg.csv \
  --save_steps 3000 \
  --do_train \
  --do_eval \
  --max_seq_length 384 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --save_total_limit 1 \
  --output_dir ${OUTPUT_DIR} ;

# Run inference inference
for split in 'train' 'val' 'test'
do
    CUDA_VISIBLE_DEVICES=0 python transformers/examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path ${OUTPUT_DIR} \
    --train_file ../data/train.relreg.csv \
    --validation_file ../data/val.relreg.csv \
    --test_file ../data/${split}.relreg.csv \
    --save_steps 3000 \
    --do_predict \
    --max_seq_length 384 \
    --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir ${OUTPUT_DIR}/${split} ; 
# done

# # Collect predictions and process to format for seq2seq models; 0 signifies not using the semgneted input
# need to reset outputdir??????
python postprocess_relreg.py $CHUNKS $OUTPUT_DIR