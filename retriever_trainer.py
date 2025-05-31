import warnings
import logging
import hydra
import hydra.utils as hu
import numpy as np
from datasets import load_metric
from transformers import Trainer, EvalPrediction, EarlyStoppingCallback, set_seed
from src.utils.collators import IBNDataCollatorWithPadding
from src.models.biencoder import BiEncoder
import pdb
from transformers import TrainerCallback
import json
import pandas as pd
logger = logging.getLogger(__name__)

class LossLoggerCallback(TrainerCallback):
    def __init__(self, output_file):
        super().__init__()
        self.output_file = output_file
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Save losses during training and evaluation
        if state.is_world_process_zero:
            if logs is not None:
                loss_data = {
                    'step': state.global_step,
                    'train_loss': logs.get('loss'),
                    'eval_loss': logs.get('eval_loss')
                }
                self.losses.append(loss_data)
                
                # Save losses to a file
                with open(self.output_file, 'a') as f:
                    json.dump(self.losses, f, indent=4)

# Use this custom callback in the Trainer


class RetrieverTrainer:

    def __init__(self, cfg) -> None:
        self.training_args = hu.instantiate(cfg.training_args)
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.index_reader = hu.instantiate(cfg.index_reader)
        encoded_index = list(self.index_reader)
      
        train_dataset, eval_dataset = self.dataset_reader.split_dataset(test_size=0.1, seed=42)
        logger.info(f"train size: {len(train_dataset)}, eval size: {len(eval_dataset)}")
       
        model_config = hu.instantiate(cfg.model_config)
        if cfg.pretrained_model_path!="None":
            print("Using pre-trained initialization")
            self.model = BiEncoder.from_pretrained(cfg.pretrained_model_path, config=model_config)
        else:
            self.model = BiEncoder(model_config)
        
        data_collator = IBNDataCollatorWithPadding(tokenizer=self.dataset_reader.tokenizer,
                                                   encoded_index=encoded_index,
                                                   **cfg.collector)
        self.metric = load_metric('src/metrics/accuracy.py', trust_remote_code=True)
        print(cfg.lang_name)
        self.lang_name = cfg.lang_name
        output_file = f"training_losses_{self.lang_name}.json"
        loss_logger_callback = LossLoggerCallback(output_file)
        # self.csv_file = f"df_{self.lang_name}.csv"
        # pdb.set_trace()
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.dataset_reader.tokenizer,
            data_collator=data_collator,
            callbacks=[loss_logger_callback] 
        )
        # self.trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))
    def train(self):
        # pdb.set_trace()
        self.trainer.train()
        # loss_df = pd.DataFrame(self.trainer.state.log_history)
        # loss_df.to_csv(self.csv_file, mode='a', header=False, index=False)
        # print(len(loss_df))
        # print(loss_df)
        self.trainer.model.save_pretrained(self.training_args.output_dir, safe_serialization=False)
        self.trainer.tokenizer.save_pretrained(self.training_args.output_dir)


@hydra.main(config_path="configs", config_name="retriever_trainer")
def main(cfg):
    logger.info(cfg)
    set_seed(43)

    trainer = RetrieverTrainer(cfg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg.training_args.do_train:
            trainer.train()
        if cfg.training_args.do_eval:
            logger.info(trainer.trainer.evaluate())


if __name__ == "__main__":
    main()
