#!/bin/bash -l
#SBATCH -J my_gpu_job
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4                     # CPUs per task (threads); not MPI tasks
#SBATCH --mem=64G
#SBATCH -t 07:00:00
#SBATCH -D /oscar/scratch/aparasel/interpretability/patchscopes/code
#SBATCH -o /oscar/scratch/aparasel/interpretability/patchscopes/code/logs/%x-%j.out
#SBATCH -e /oscar/scratch/aparasel/interpretability/patchscopes/code/logs/%x-%j.err

set -euo pipefail
mkdir -p /oscar/scratch/aparasel/interpretability/patchscopes/code/logs


module purge
module load cuda               
module load miniconda3/23.11.0s

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
set +u
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-}"
conda activate patchscopes-llava
set -u


which python
python -V
nvidia-smi || true

conda run -n patchscopes-llava python /oscar/scratch/aparasel/interpretability/patchscopes/code/visual_attractive_question.py

