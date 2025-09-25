#!/bin/bash
# ("qwen2-7b" "llama3-8b" "glm4-9b")

# Stage 2: DPO
cd DPO
ACC_CONFIG='acc_config/ddp8.yaml'
CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port='29510' --config_file $ACC_CONFIG dpo.py --config-name=dpo-webnlg-bs8-qwen-max
cd -

