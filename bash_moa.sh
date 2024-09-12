# model_path="../../models/Qwen2-Math-7B-Instruct"
# model_pretty_name="Qwen2-Math-7B-Instruct"
model_path="../../models/Qwen2-7B-Instruct"
model_pretty_name="Qwen2-7B-Instruct"
# model_path="../../models/CodeQwen1.5-7B-Chat"
# model_pretty_name="CodeQwen1.5-7B-Chat"

# sleep 7200
# 
# ## MOA
# i===> instruct
# m===> math
# c===> code 

tasks=("gsm" "zebra-grid" "crux")
for task in "${tasks[@]}"
do
    sbatch zero_eval_local_slurm.sh -d ${task} -m ${model_path} -p ${model_pretty_name}-iii-moa -s 1 -z 1 -c result_dirs/${task}/Qwen2-7B-Instruct-seed1.json:result_dirs/${task}/Qwen2-7B-Instruct-seed2.json:result_dirs/${task}/Qwen2-7B-Instruct-seed3.json
    sbatch zero_eval_local_slurm.sh -d ${task} -m ${model_path} -p ${model_pretty_name}-imc-moa -s 1 -z 1 -c result_dirs/${task}/Qwen2-7B-Instruct-seed1.json:result_dirs/${task}/Qwen2-Math-7B-Instruct-seed1.json:result_dirs/${task}/CodeQwen1.5-7B-Chat-seed1.json
done
