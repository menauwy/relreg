#!/bin/bash
#SBATCH --job-name=relreg-TT
#SBATCH --mail-type="ALL"
#SBATCH --time=00:10:00
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
# OUTPUT_DIR='output-relregTT-utt'
OUTPUT_DIR='/home/wangym/data1/output-relregTT-utt'

# Run relreg-tt with (0/1) utterance/segments and max encoder length of 256 of 'nli-distilroberta-base-v2'
# python add_rouge.py $CHUNKS
# python train_relreg_tt.py nli-distilroberta-base-v2 $CHUNKS $OUTPUT_DIR 256     
python test_relreg_tt.py $OUTPUT_DIR $CHUNKS $OUTPUT_DIR 256     
