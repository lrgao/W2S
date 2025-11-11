# Stage 2: DPO
ACC_CONFIG='DPO/acc_config/ddp8.yaml'
CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port='29509' --config_file $ACC_CONFIG DPO/dpo.py --config-name=dpo-e2e
