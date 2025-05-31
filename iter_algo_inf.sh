#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY="" # change to your wandb account
export WANDB_API_KEY="" # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

gpu=4 #Number of GPUs #Method Name (affects the output folder where the model will be saved)
num_ice=16  #Number of Incontext Examples
port=5305  #Does not need to be changed

model_name="meta-llama/Meta-Llama-3.1-8B"

n_tokens=6000   #Context Length 
scr_batch_size=1  #Scoring Batch Size
inf_batch_size=2  #Inference Batch Size
task_name='xqor'

test_lang_name="Manipuri"
lang_name="Manipuri"
high_lang_name=["Bengali"]    
all_data=True

for epoch in {1..10}
do  
        rm -rf index_data_inf/${task_name}/${model_name}
        
        method="epoch_${epoch}_${test_lang_name}"
        export WANDB_TAGS="${method},${task_name},${model_name}"
        run_dir="output/ours/${task_name}/${method}/Manipuri/${model_name}"
        index_data="index_data_inf/${task_name}/${model_name}/index_dataset.json"

        mkdir -p "index_data_inf/${task_name}/${model_name}"
        pretrained_model_path="${run_dir}/bert-fix_ctx-shared-bs64"

        echo "Loaded Model Path is ${pretrained_model_path}"

        ./run_epr_inf.sh "${lang_name}" "${high_lang_name}" "${all_data}" "${test_lang_name}" "${pretrained_model_path}" "${run_dir}" "${index_data}" "${model_name}"
   
done


