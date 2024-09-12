model_path="../../models/Qwen2-7B-Instruct"
model_pretty_name="Qwen2-7B-Instruct"
# model_path="../../models/Qwen2-Math-7B-Instruct"
# model_pretty_name="Qwen2-Math-7B-Instruct"
# model_path="../../models/CodeQwen1.5-7B-Chat"
# model_pretty_name="CodeQwen1.5-7B-Chat"
temperature="0"
# task="mmlu-redux"
# task="gsm"
# task="zebra-grid"
# task="crux"
tasks=("gsm" "zebra-grid" "crux")
for task in "${tasks[@]}"
do
    for seed in {1..1}
    do
        sbatch zero_eval_local_slurm.sh -d ${task} -m $model_path -p ${model_pretty_name}-temp${temperature}-seed${seed} -s 1 -z $seed -t  ${temperature}
        # if [ $i -lt 4 ]; then  # Check if i is not the last value
        #      reference_path_name++=":"  # Append ":" if not the last
        # fi
    done
done

# sleep 7200
# 
# ## MOA
# sbatch zero_eval_local_slurm.sh -d mmlu-redux -m ${model_path} -p ${model_pretty_name} -s 4 -z 1 -c result_dirs/mmlu-redux/Qwen2-0.5B-test_slurm_s1.json:result_dirs/mmlu-redux/Qwen2-0.5B-test_slurm_s2.json:result_dirs/mmlu-redux/Qwen2-0.5B-test_slurm_s3.json:result_dirs/mmlu-redux/Qwen2-0.5B-test_slurm_s4.json
