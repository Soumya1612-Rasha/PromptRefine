#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

gpu=4
method=dense
port=1287

num_ice=5

model_config=hf-gen_a
model_name=meta-llama/Meta-Llama-3.1-8B

n_tokens=6000
inf_batch_size=2
gpu=4

lang_name="Manipuri"
test_lang_name="Manipuri"
high_lang_name=["Hindi"]
all_data=False





for task_name in xqor
do
  rm -rf index_data_dense
  run_dir=output/${method}/${task_name}/${model_name}
  index_data=index_data_dense/${task_name}/index_dataset.json
  retrieve_file=${run_dir}/retrieved.json
  pred_file=${run_dir}/pred.json
  mkdir -p ${run_dir}
  mkdir -p index_data_dense/${task_name}

  python dense_retriever.py \
    hydra.run.dir=${run_dir}/dense_retriever \
    output_file=${retrieve_file} \
    num_ice=${num_ice} \
    task_name=${task_name} \
    index_reader.dataset_path=${index_data} \
    faiss_index=${run_dir}/index \
    lang_name=${lang_name} \
    high_lang_name=${high_lang_name} \
    all_data=${all_data} \
    test_lang_name=${test_lang_name} \


  accelerate launch --num_processes ${gpu} --main_process_port ${port}  inferencer.py \
      hydra.run.dir=${run_dir}/inferencer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data} \
      output_file=${pred_file} \
      model_name=${model_name} \
      batch_size=${inf_batch_size} \
      model_config=${model_config}
done



