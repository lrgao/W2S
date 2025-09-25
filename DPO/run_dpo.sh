ACC_CONFIG='acc_config/ddp8.yaml'

# cnndm
CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port='29500' --config_file $ACC_CONFIG dpo.py --config-name=dpo-cnndm-bs8-qwen

