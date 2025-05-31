#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY="sghosal" # change to your wandb account
export WANDB_API_KEY="5e861ad2eeaa96b2d3d0fe1686a83b74fa5d99f1" # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

gpu=4 #Number of GPUs
 
num_ice=16  #Number of Incontext Examples
port=5303  #Does not need to be changed


n_tokens=6000   #Context Length 
scr_batch_size=1 #Scoring Batch Size
inf_batch_size=1  #Inference Batch Size
lang_name=$1
high_lang_name=$2
all_data=$3
test_lang_name=$4
pretrained_model_path=$5
run_dir=$6
index_data=$7
model_name=$8


echo "${run_dir}"
task_name='xqor'
retrieve_file=${run_dir}/retrieved.json
python bm25_retriever.py \
    hydra.run.dir=${run_dir}/bm25_retriever \
    output_file=${retrieve_file} \
    num_candidates=50 \
    num_ice=1 \
    task_name=${task_name} \
    lang_name=${lang_name} \
    high_lang_name=${high_lang_name} \
    all_data=${all_data} \
    test_lang_name=${test_lang_name} \
    index_reader.dataset_path=${index_data} \
    dataset_split=train \
    ds_size=44000 \
    query_field=a \
    index_reader.field=a



scored_file=${run_dir}/scored.json
accelerate launch --num_processes ${gpu} --main_process_port ${port}  scorer.py \
      hydra.run.dir=${run_dir}/scorer \
      task_name=${task_name} \
      output_file=${scored_file} \
      batch_size=${scr_batch_size} \
      model_name=${model_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data}


run_name=bert-fix_ctx-shared-bs64
run_dir=${run_dir}/${run_name}
accelerate launch --num_processes ${gpu} --main_process_port ${port}  retriever_trainer.py \
      hydra.run.dir=${run_dir}/trainer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${scored_file} \
      index_reader.dataset_path=${index_data} \
      training_args.output_dir=${run_dir} \
      training_args.run_name=${run_name} \
      model_config.ctx_model_name=null  \
      training_args.learning_rate=0.0001 \
      training_args.per_device_train_batch_size=64 \
      training_args.per_device_eval_batch_size=64 \
      training_args.gradient_accumulation_steps=1 \
      pretrained_model_path=${pretrained_model_path} \
      lang_name=${test_lang_name}


