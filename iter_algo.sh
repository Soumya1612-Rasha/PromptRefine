#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY="" # change to your wandb account
export WANDB_API_KEY="" # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

gpu=4 #Number of GPUs #Method Name (affects the output folder where the model will be saved)
num_ice=16  #Number of Incontext Examples
port=5300  #Does not need to be changed


model_name="meta-llama/Meta-Llama-3.1-8B"
n_tokens=6000   #Context Length 
scr_batch_size=2  #Scoring Batch Size
inf_batch_size=1  #Inference Batch Size
task_name='xqor'
test_lang_name="Manipuri"
high_lang_name=["Bengali"]
all_data=False
language_list=("Manipuri" "Bengali")
lang_list_string=$(IFS=, ; echo "${language_list[*]}")
output="output"
for epoch in {1..10}
do  
    echo "Running Epoch ${epoch}"
    for lang_name in "${language_list[@]}"
    do 
        echo "Training on Language ${lang_name}"
        rm -rf "index_data_ours/${task_name}/${model_name}"

        
        method="epoch_${epoch}_${test_lang_name}"
        export WANDB_TAGS="${method},${task_name},${model_name}"
        run_dir="${output}/ours/${task_name}/${method}/${lang_name}/${model_name}"
        index_data="index_data_ours/${task_name}/${model_name}/index_dataset.json"

        mkdir -p "${run_dir}"
        mkdir -p "index_data_ours/${task_name}/${model_name}"

  
        if [ "$epoch" -eq 1 ]; then
            pretrained_model_path="None"
        else
            prev_epoch=$((epoch - 1))
            pretrained_model_path="${output}/ours/${task_name}/epoch_${prev_epoch}_${test_lang_name}/merged/${model_name}"
        fi
        echo "Pretrained model path is ${pretrained_model_path}"

        

        ./run_epr.sh "${lang_name}" "${high_lang_name}" "${all_data}" "${test_lang_name}" "${pretrained_model_path}" "${run_dir}" "${index_data}" "${model_name}"

    done
    python model_merging.py --folder "${output}/ours/${task_name}/epoch_${epoch}_${test_lang_name}" --model "${model_name}" --lang_list "${lang_list_string}"
   
done


