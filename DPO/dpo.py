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
from ruamel.yaml import YAML
import argparse
from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
import sys
# sys.path.append('/home/gaolr/workspace/DPO-ST-main')
# DATA_DIR = '/home/gaolr/workspace/DPO-ST-main'

import os
import wandb
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"
def make_dataset(data_dir):
    # data_files = {
    #     'train': os.path.join(data_dir, 'train_dpo_processed.jsonl'),
    #     'eval': os.path.join(data_dir, 'eval_dpo_processed.jsonl'),
    # }
    # data_files = {
    #     'train': '/data/gaolr/workspace/AQPG/t5-cnndm/dataset/qwen2/1000-gentexts-bleu-merge/train.gen_prompts.0-9.dpo_weak.json',
    #     'eval':'/data/gaolr/workspace/AQPG/t5-cnndm/dataset/qwen2/1000-gentexts-bleu-merge/dev_100.gen_prompts.0-209.dpo_weak.json',
    # }
    
    # data_files = {
    #     'train': '/data/gaolr/workspace/AQPG/t5-cnndm-llama/dataset/1000-gentexts-bleu-merge-hf/train.gen_prompts.0-39.dpo_weak.json',
    #     'eval':'/data/gaolr/workspace/AQPG/t5-cnndm-llama/dataset/1000-gentexts-bleu-merge-hf/dev_100.gen_prompts.0-999.dpo_weak.json',
    # }
    # webnlg
    # data_files = {
    #     'train': '/data/gaolr/workspace/AQPG/t5-webnlg-qwen2/dataset/1000-gentexts-llama3-bleu-merge/train.gen_prompts.0-39.dpo_weak.json',
    #     'eval':'/data/gaolr/workspace/AQPG/t5-webnlg-qwen2/dataset/1000-gentexts-llama3-bleu-merge/dev_100.gen_prompts.0-999.dpo_weak.json',
    # }
    # data_files = {
    #     'train': '/data/gaolr/workspace/AQPG/t5-webnlg-qwen2/dataset/4-gentexts-qwen14b/train.gen_prompts.0-9.dpo_weak.json',
    #     'eval':'/data/gaolr/workspace/AQPG/t5-webnlg-qwen2/dataset/4-gentexts-qwen14b/dev_100.gen_prompts.0-598.dpo_weak.json',
    # }

    # data_files = {
    #     'train': '/data/gaolr/workspace/AQPG/t5-webnlg-qwen2/dataset/qwen-32b/4-gentexts/train.gen_prompts.0-10.dpo_weak.json',
    #     'eval':'/data/gaolr/workspace/AQPG/t5-webnlg-qwen2/dataset/qwen-32b/4-gentexts/dev_100.gen_prompts.0-500.dpo_weak.json',
    # }
    
    data_files = {
        'train': '/data/gaolr/workspace/AQPG/t5-webnlg-qwen2/dataset/qwen-max/4-gentexts/train.gen_prompts.0-9.dpo_weak.json',
        'eval':'/data/gaolr/workspace/AQPG/t5-webnlg-qwen2/dataset/qwen-max/4-gentexts/dev_100.gen_prompts.0-99.dpo_weak.json',
    }
    
    # data_files = {
    #     'train': '/data/gaolr/workspace/AQPG/t5-webnlg-glm4/dataset/1000-gentexts-bleu-merge/train.gen_prompts.0-19.clean.dpo_weak.json',
    #     'eval':'/data/gaolr/workspace/AQPG/t5-webnlg-glm4/dataset/1000-gentexts-bleu-merge/dev_100.gen_prompts.0-999.clean.dpo_weak.json',
    # }

    # data_files = {
    #     'train': '/data/gaolr/workspace/AQPG/t5-GSM8K/dataset/1000-gentexts-eval-merge/train.gen_prompts.0-59.dpo_weak.json',
    #     'eval':'/data/gaolr/workspace/AQPG/t5-GSM8K/dataset/1000-gentexts-eval-merge/dev_100.gen_prompts.0-65.dpo_weak.json',
    # }
    #e2e
    # data_files = {
    #     'train': '/data/gaolr/workspace/AQPG/t5-e2e/dataset/E2E-cleaned/qwen2/1000-gentexts-llama3-bleu-merge/train.gen_prompts.0-49.dpo_weak.json',
    #     'eval':'/data/gaolr/workspace/AQPG/t5-e2e/dataset/E2E-cleaned/qwen2/1000-gentexts-llama3-bleu-merge/dev_100.gen_prompts.0-488.dpo_weak.json',
    # }
    # gsm8k
    # data_files = {
    #     'train': '/data/gaolr/workspace/AQPG/t5-GSM8K-qwen/dataset/4-gentexts/train.gen_prompts.0-9.dpo_weak.json',
    #     'eval':'/data/gaolr/workspace/AQPG/t5-GSM8K-qwen/dataset/4-gentexts/dev_100.gen_prompts.0-229.dpo_weak.json',
    # }

    
    dataset = load_dataset('json', data_files=data_files)
    return dataset['train'], dataset['eval']


@hydra.main(version_base=None, config_path="exp_config/t5")
def main(cfg : DictConfig):
    parser = transformers.HfArgumentParser(DPOConfig)
    trainer_args_dict = OmegaConf.to_container(cfg.trainer)
    training_args = parser.parse_dict(trainer_args_dict)[0]
    training_args.output_dir = os.path.join(DATA_DIR, training_args.output_dir)
    
    set_seed(training_args.seed)
    
    model_path = os.path.join(DATA_DIR, cfg.model.model_path)
    # if 'llama' in model_path:
    #     tokenizer = AutoTokenizer.from_pretrained(model_path)
    #     model = AutoModelForCausalLM.from_pretrained(model_path)
    #     model_ref = AutoModelForCausalLM.from_pretrained(model_path)
    # else:
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        # model_ref = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # tokenizer = T5Tokenizer.from_pretrained(model_path)
        # model = T5ForConditionalGeneration.from_pretrained(model_path)
        # model_ref = T5ForConditionalGeneration.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
