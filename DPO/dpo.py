import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    HfArgumentParser, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    set_seed,
    T5Tokenizer,
    T5ForConditionalGeneration
    )

import hydra
from omegaconf import DictConfig, OmegaConf
from trl import DPOTrainer, DPOConfig
import transformers
from dotenv import load_dotenv
load_dotenv()

import os
import wandb
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"
def make_dataset(data_dir):

    data_files = {
        'train': f'{data_dir}/traindata/dpo.json',
        'eval':f'{data_dir}/devdata/dpo.json',
    }
    
    dataset = load_dataset('json', data_files=data_files)
    return dataset['train'], dataset['eval']


@hydra.main(version_base=None, config_path="exp_config/t5")
def main(cfg : DictConfig):
    parser = transformers.HfArgumentParser(DPOConfig)
    trainer_args_dict = OmegaConf.to_container(cfg.trainer)
    training_args = parser.parse_dict(trainer_args_dict)[0]
    
    set_seed(training_args.seed)
    
    model_path = cfg.model.model_path
    tokenizer_path = cfg.data.tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model_ref = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    train_dataset, eval_dataset = make_dataset(cfg.data.data_dir)

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,

    )
    dpo_trainer.train()
    dpo_trainer.save_model(training_args.output_dir)
    # dpo_trainer.save_pretrained(training_args.output_dir)
    
    return


if __name__ == "__main__":
    main()
