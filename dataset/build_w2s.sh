
MODEL="Qwen/Qwen2.5-7B-Instruct"
# MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL="zai-org/glm-4-9b-chat"

dataset="webnlg"
# Precise Example Prompt Generation
# 1. 3–5 ICL samples, each generating 50–100 prompts
CUDA_VISIBLE_DEVICES=0 python build_w2s.py \
    --model $MODEL \
    --mode "gen_prompt" \
    --sampling_times 50 \
    --dataset ${dataset} \
    --input_file raw/${dataset}/icl.json \
    --output_dir w2s/${dataset}

# 2. Evaluate the 50–100 generated prompts
CUDA_VISIBLE_DEVICES=0 python build_w2s.py \
    --model $MODEL \
    --mode "gen_text" \
    --sampling_times 50 \
    --dataset ${dataset} \
    --input_file raw/${dataset}/icl.json \
    --output_dir w2s/${dataset}
