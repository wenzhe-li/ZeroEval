#!/bin/bash
#SBATCH --job-name=llm-evaluation         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=20G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=01:45:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=yl7690@princeton.edu

conda activate zeroeval

bash zero_eval_local.sh -d mmlu-redux -m ../../data/Qwen2-0.5B -p Qwen2-0.5B-test_slurm_s4 -s 4 -z 4
