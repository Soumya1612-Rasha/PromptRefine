import glob
import json
import os
import logging
import hydra
import hydra.utils as hu
import torch
import tqdm
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import set_seed
from src.metrics import get_metric
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.utils.statistics import show_statistics
from src.models.api_client import run_api
from src.utils.misc import parallel_run, save_json
from src.models.model import ppl_generate
import pdb
import re
import transformers
from string import punctuation, whitespace, ascii_lowercase
import evaluate
from collections import Counter
import re
import string
logger = logging.getLogger(__name__)


chrf = evaluate.load("chrf")
bleu = evaluate.load("bleu")
generation_kwargs = {
        "min_length": -1,
        "max_new_tokens":50,
        "top_p": 0.9,
        "temperature": 1.0,
        "repetition_penalty": 1.2,
        # "pad_token_id": tokenizer.pad_token_id,
        # "eos_token_id": tokenizer.eos_token_id,
    }

def normalize_text(text):
    """Lower text, remove punctuation, and fix extra whitespaces."""
    # Define Hindi-specific punctuation and spaces
    punctuation = set(string.punctuation + '।')  # Adding Hindi '।' to punctuation
    # Remove punctuation
    text = ''.join(ch for ch in text if ch not in punctuation)
    # Normalize spaces and lowercase the text
    return ' '.join(text.lower().strip().split())

def token_f1_score(prediction, ground_truth):
    """Compute the token-level F1 score between prediction and ground truth."""
    # Tokenize and normalize text
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    
    # Count common tokens
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common = sum(common_tokens.values())
    
    # If there are no common tokens, F1 is 0
    if num_common == 0:
        return 0
    
    # Calculate precision and recall
    precision = num_common / len(prediction_tokens)
    recall = num_common / len(ground_truth_tokens)
    
    # Calculate F1 score
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def is_english(text):
    only_chars = list(filter(lambda x: x not in punctuation + whitespace, text))
    return all(x.lower() in ascii_lowercase for x in only_chars)
    
class Inferencer:
    def __init__(self, cfg, accelerator=None) -> None:
        print("In Inferencer init")
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.gen_field = cfg.dataset_reader.field
        self.accelerator = accelerator
        self.output_file = cfg.output_file
        self.generation_kwargs = OmegaConf.to_object(cfg.model_config.generation_kwargs)
        self.evaluator = get_metric(cfg.task_name)

        self.model, self.dataloader = self.init_model_dataloader(cfg)

    def init_model_dataloader(self, cfg):
        self.dataset_reader.shard(self.accelerator)

        if self.accelerator.is_main_process:
            logger.info(f"Statistics after sharding: ")
            show_statistics(self.dataset_reader.encoded_dataset, "main dataset")
            show_statistics(self.dataset_reader.index_reader.encoded_dataset, "index dataset")

        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.accelerator.device)
        dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)
        if cfg.model_name=="CohereForAI/aya-23-8B" or cfg.model_name=="google/mt5-xl" or cfg.model_name=="google/flan-t5-base":
            print(f"Loading Scorer Model: {cfg.model_name}")
            model =  transformers.AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16).eval()
        else:
            model =  transformers.AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        print("Loaded Scorer Model")
        # model = model.to("cuda:0")
        model = self.accelerator.prepare(model)

        if hasattr(model, "module"):
            model = model.module

        return model, dataloader

    def forward(self, cfg):
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader
        avg_ice_num = 0
        res = []
        
        for i, entry in enumerate(dataloader):
            metadata = entry.pop("metadata")
            if 'choices' in self.dataset_reader.dataset_wrapper.field_getter:
                choices = [self.dataset_reader.dataset_wrapper.get_field(meta, 'choices') for meta in metadata]
                choices_list = list(zip(*choices))
                preds = ppl_generate([meta['prompt'] for meta in metadata],
                                     model=self.model,
                                     tokenizer=self.dataset_reader.tokenizer,
                                     choices_list=choices_list,
                                     device=self.accelerator.device)
                for mdata, pred in zip(metadata, preds):
                    mdata['generated'] = pred
                    avg_ice_num += len(mdata['ice_prompts_list'])
            else:
                with torch.no_grad():
                    
                    # self.dataset_reader.tokenizer.padding_side="right"
                    if cfg.model_name=="google/gemma-2-9b":
                        self.dataset_reader.tokenizer.pad_token = self.dataset_reader.tokenizer.eos_token
                        eos_token_id = self.dataset_reader.tokenizer.eos_token_id
                        pad_token_id = self.dataset_reader.tokenizer.pad_token_id
                    else:
                        self.dataset_reader.tokenizer.pad_token = self.dataset_reader.tokenizer.eos_token
                        eos_token_id = self.dataset_reader.tokenizer.eos_token_id
                        pad_token_id= self.dataset_reader.tokenizer.pad_token_id
                        

                    outputs = self.model.generate(input_ids=entry.input_ids,
                                                  attention_mask=entry.attention_mask,
                                                  eos_token_id=eos_token_id,
                                                  pad_token_id=pad_token_id,
                                                  do_sample=False,  # always use greedy decode here
                                                  **generation_kwargs)
                    prompt_len = int(entry.attention_mask.shape[1])
                    for mdata, output in zip(metadata, outputs.tolist()):

                        ########## Turn this on for FLORES-200#################
                        # pdb.set_trace()
                        # if cfg.model_name=="CohereForAI/aya-23-8B":
                        #     generated = self.dataset_reader.tokenizer.decode(output)
                        # else:
                        # pdb.set_trace()
                        generated = self.dataset_reader.tokenizer.decode(output[prompt_len:])
                        # pdb.set_trace()
                        generated = generated.strip(self.dataset_reader.tokenizer.eos_token).strip()
                        # pdb.set_trace()
                        #######TURN THIS ON FOR FLORES-200##########
                        # sentence_endings = re.compile(r'[.!?]')
                        # split_paragraph = sentence_endings.split(generated)
                        # generated = split_paragraph[0] + '.'
                        #######################################

                        # if is_english(split_paragraph[1]):
                        #     generated = generated + " " + split_paragraph[1] + '.'


                        ###############FOR XQUAD-IN##################
                        generated = generated.split("\n")[0].strip()



                        # print(generated)
                        mdata['generated'] = generated
                    
                        ########## Turn this on for Adobe Dataset#####################
                        # generated = self.dataset_reader.tokenizer.decode(output[prompt_len:])
                        # mdata['generated'] = generated.strip(self.dataset_reader.tokenizer.pad_token).strip()
                        ########################################################
                        
                        avg_ice_num += len(mdata['ice_prompts_list'])

            res.extend(metadata)
    
            if i == 0:
                logger.info(f"Prompt: {metadata[0]['prompt']}")
                logger.info(f"Generated: {metadata[0]['generated']}")
                logger.info(f"Number of ICE: {len(metadata[0]['ice_prompts_list'])}")

        save_json(f"{self.output_file}tmp_{self.accelerator.device}.bin", res)

        logger.info(f"Average number of in-context examples after truncating is {avg_ice_num / len(res)}")

    def write_results(self):
        data = []
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            with open(path) as f:
                data.extend(json.load(f))
        # from src.utils.misc import load_json
        # data = load_json(self.output_file)
        gen = [i['generated'] for i in data]
        true = [i['answer'] for i in data]
        total_f1 = 0
        count = 0
        for pred, tr in zip(gen, true):
            total_f1 +=token_f1_score(pred, tr)
            count += 1
        
        # print(data[0])
        # pdb.set_trace()
        # pdb.set_trace()
        # chrf_results = chrf.compute(predictions=gen, references=true)['score']
        # bleu_results= bleu.compute(predictions=gen, references=true)['bleu']

        # metric = self.evaluator.evaluate(preds, data)
        # logger.info(f"CHRF: {chrf_results}")
        logger.info(f"Token F-1 score: {total_f1/count}")
        # logger.info(f"BLEU: {bleu_results}")

        results_log_path = f"{self.output_file}_results_log.txt"
        print(results_log_path)
        # Append CHRF and BLEU results to the log file
        with open(results_log_path, 'a') as results_file:
            # results_file.write(f"CHRF: {chrf_results}\n")
            results_file.write(f"Token F-1 score: {total_f1/count}\n")
            # results_file.write(f"BLEU: {bleu_results}\n")

        save_json(self.output_file, data)

        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)
        return data


class APInferencer(Inferencer):

    def init_model_dataloader(self, cfg):
        model = hu.instantiate(cfg.model_config.model)
        dataloader = self.dataset_reader
        return model, dataloader

    def forward(self):
        prompts = [entry['metadata']['prompt'] for entry in self.dataloader]
        if 'choices' in self.dataset_reader.dataset_wrapper.field_getter:
            choices = [self.dataset_reader.dataset_wrapper.get_field(entry['metadata'], 'choices')
                       for entry in self.dataloader]
            args_list = list(zip(prompts, choices))
        else:
            args_list = prompts
        logger.info(str(prompts[0]))
        responses = parallel_run(run_api, args_list=args_list,
                                 n_processes=self.model.n_processes,
                                 client=self.model,
                                 **self.generation_kwargs)

        data = []
        for i, (entry, response) in enumerate(zip(self.dataloader, responses)):
            if i == 0:
                logger.info(prompts[i])
                logger.info('\n***\n'.join([str(i) for i in response][:3]))
            entry['metadata']['generated'] = response[0]['text']
            data.append(entry['metadata'])

        save_json(self.output_file, data)

        avg_ice_num = sum([len(i['ice_prompts_list']) for i in data])/len(data)
        logger.info(f"Average number of in-context examples after truncating is {avg_ice_num}")
        preds = [i['generated'] for i in data]
        metric = self.evaluator.evaluate(preds, data)
        logger.info(f"metric: {str(metric)}")


@hydra.main(config_path="configs", config_name="inferencer")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    # pdb.set_trace()
    if cfg.model_config.model_type == 'hf':
        accelerator = Accelerator()
        inferencer = Inferencer(cfg, accelerator)
        inferencer.forward(cfg)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            inferencer.write_results()
    else:
        inferencer = APInferencer(cfg)
        inferencer.forward()


if __name__ == "__main__":
    main()
