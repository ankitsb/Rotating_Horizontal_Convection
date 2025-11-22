#!/bin/bash
#SBATCH --job-name=RHCm1
## SBATCH --nodes 1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=102400
#SBATCH --time 48:00:00
#SBATCH --account=punim1661
## Use an account that has GPGPU access

module purge
module load Julia/1.10.8-linux-x86_64

julia code.jl

##Job monitor command to list the resource usage
my-job-stats -a -n -s

