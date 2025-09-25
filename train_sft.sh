#!/bin/bash

llm_name="qwen2-7b"  # ("qwen2-7b" "llama3-8b" "glm4-9b")
# Stage 1: SFT
CUDA_VISIBLE_DEVICES=0,1,2,3 python cli_gt.py \
        --do_train \
        --model_name t5 \
        --output_dir out/SFT-cnndm-${llm_name} \
        --train_file dataset/cnndm/${llm_name}/train_sft.json \
        --predict_file dataset/cnndm/${llm_name}/dev_sft.json \
        --icl_file ./dataset/cnndm/${llm_name}/icl.json \
        --model_path t5-base \
        --tokenizer_path t5-base \
        --dataset cnndm \
        --train_batch_size 4 \
        --predict_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --max_input_length 2500 \
        --max_output_length 350 \
        --append_another_bos \
        --learning_rate 2e-5 \
        --num_train_epochs 40 \
        --warmup_steps 0 \
        --eval_period 60 \
        --num_beams 5 \
        --clean_up_spaces


# Stage 2: DPO
cd DPO
ACC_CONFIG='acc_config/ddp8.yaml'
CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port='29510' --config_file $ACC_CONFIG dpo.py --config-name=dpo-cnndm-qwen2-7b
cd -

