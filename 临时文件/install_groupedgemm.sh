#!/bin/bash
#SBATCH -A wenge
#SBATCH --partition=wenge
#SBATCH --qos=normal
#SBATCH --job-name=install_gemm
#SBATCH --time=00:30:00
#SBATCH --chdir=/data/home/zdhs0100/xtuner/GroupedGEMM
#SBATCH --output=log/install_gemm/%j.log
#SBATCH --error=log/install_gemm/%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

if [ -f "modules.sh" ]; then
  # shellcheck disable=SC1091
  source "modules.sh"
fi

module load cuda/12.8

source /data/home/zdhs0100/anaconda3/etc/profile.d/conda.sh
conda activate xtuner

echo "===== GPU Check ====="
nvidia-smi

echo "===== Python ====="
which python
python -c "import torch; print(torch.__version__)"

echo "===== Install ====="
pip install . --no-build-isolation
pip install flash-attn --no-build-isolation
