#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=""  # change to your wandb account
export WANDB_API_KEY=""  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

gpu=4 #Number of GPUs

num_ice=50 #Number of Incontext Examples
port=9927 #Does not need to be changed

model_name="meta-llama/Meta-Llama-3.1-8B"

n_tokens=6000 #Context Length 
scr_batch_size=1 #Scoring Batch Size
inf_batch_size=2 #Inference Batch Size

lang_name="Santali"
high_lang_name=["Bengali"]
all_data=True
test_lang_name="Santali"
task_name=flores

epoch=5

method=epoch_${epoch}_${test_lang_name}
method_epr=epoch_${epoch}_${test_lang_name}

for scale_factor in 0.1
do
  rm -rf index_data_dpp/${task_name}/${model_name}
  export WANDB_TAGS="${method},${task_name},${model_name}"
  run_dir=output/ours/${task_name}/${method}/dpp/${model_name}
  index_data=index_data_dpp/${task_name}/${model_name}/index_dataset.json
  mkdir -p ${run_dir}
  mkdir -p index_data_dpp/${task_name}/${model_name}

  #### We need to put the path of EPR trained model here ########
  epr_model=output/ours/${task_name}/${method_epr}/merged/${model_name}
  echo "Loading Model from ${epr_model}"

  retrieve_file=${run_dir}/retrieved.json
  python dense_retriever_inf.py \
      hydra.run.dir=${run_dir}/dense_retriever \
      output_file=${retrieve_file} \
      task_name=${task_name} \
      dataset_reader.dataset_split=train \
      dataset_reader.ds_size=44000 \
      index_reader.dataset_path=${index_data} \
      pretrained_model_path=${epr_model} \
      model_config.norm_embed=true \
      faiss_index=${run_dir}/index \
      dpp_search=true \
      dpp_topk=100 \
      num_ice=16 \
      num_candidates=50 \
      model_config.scale_factor=${scale_factor} \
      lang_name=${lang_name} \
      high_lang_name=${high_lang_name} \
      all_data=${all_data} \
      test_lang_name=${test_lang_name} \
      mode=cand_random_plus # [Can be 'cand_random', 'cand_random_plus': Make it cand_random_plus for sampling from L_s using MAP]


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


  run_name=base-mg0.02-s${scale_factor}-fix
  run_dir=${run_dir}/${run_name}
  accelerate launch --num_processes ${gpu} --main_process_port ${port}  retriever_trainer.py \
      hydra.run.dir=${run_dir}/trainer \
      task_name=${task_name} \
      pair_wise=true \
      dataset_reader.dataset_path=${scored_file} \
      index_reader.dataset_path=${index_data} \
      training_args.output_dir=${run_dir} \
      training_args.run_name=${run_name} \
      pretrained_model_path=${epr_model} \
      training_args.num_train_epochs=10 \
      training_args.per_device_train_batch_size=64 \
      training_args.per_device_eval_batch_size=64 \
      training_args.gradient_accumulation_steps=1 \
      training_args.learning_rate=0.0001 \
      model_config.dpp_training=true \
      model_config.norm_embed=true \
      model_config.margin=0.02 \
      model_config.scale_factor=${scale_factor}


  retrieve_file=${run_dir}/train_retrieved.json
  python dense_retriever_inf.py \
      hydra.run.dir=${run_dir}/dense_retriever \
      output_file=${retrieve_file} \
      num_ice=${num_ice} \
      task_name=${task_name} \
      index_reader.dataset_path=${index_data} \
      pretrained_model_path=${run_dir} \
      faiss_index=${run_dir}/index \
      model_config.norm_embed=true \
      model_config.scale_factor=${scale_factor} \
      dpp_search=true \
      dpp_topk=50 \
      mode=map \
      lang_name=${lang_name} \
      high_lang_name=${high_lang_name} \
      all_data=${all_data} \
      test_lang_name=${test_lang_name} \


  pred_file=${run_dir}/pred.json
  accelerate launch --num_processes ${gpu} --main_process_port ${port}  inferencer.py \
      hydra.run.dir=${run_dir}/inferencer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data} \
      output_file=${pred_file} \
      model_name=${model_name} \
      batch_size=${inf_batch_size}
done


