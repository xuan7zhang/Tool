#!/bin/bash
#SBATCH --job-name=conf_pollution
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/project/aip-xli135/xzhan576/DUCK/logs/toolbias/confidence/sbatch_%j.out
export DUCK=/project/aip-xli135/xzhan576/DUCK
cd "$DUCK"
VLLM_GPU=0 TOOL_GPU=1 PORT=8000 bash ducx_noise/run_confidence_pollution.sh
