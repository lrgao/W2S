
MODEL="Qwen/Qwen2.5-7B-Instruct"
# MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL="zai-org/glm-4-9b-chat"
postfix="Qwen2.5-7B-Instruct"

dataset="e2e"
# Generate precise example prompts
# 1. Use 3–5 ICL samples, each producing 50–100 prompts
# CUDA_VISIBLE_DEVICES=1 python -m dataset.build_w2s \
#     --model $MODEL \
#     --mode "gen_prompt" \
#     --sampling_times 50 \
#     --dataset ${dataset} \
#     --input_file dataset/raw/${dataset}/icl.json \
#     --output_dir dataset/w2s/${postfix}/${dataset}

# 2. Evaluate 50–100 prompts and select the best-quality examples
# CUDA_VISIBLE_DEVICES=1 python -m dataset.build_w2s \
#     --model $MODEL \
#     --mode "gen_text" \
#     --sampling_times 50 \
#     --dataset ${dataset} \
#     --input_file dataset/raw/${dataset}/icl.json \
#     --input_dir dataset/w2s/${postfix}/${dataset}/gen_prompt \
#     --output_dir dataset/w2s/${postfix}/${dataset}

# Prompt data generation and grading
# 1. Generate 5–10 prompts for each training sample. Please modify the fewshot_file path
CUDA_VISIBLE_DEVICES=1 python -m dataset.build_w2s \
    --model $MODEL \
    --mode "gen_prompt" \
    --sampling_times 10 \
    --dataset ${dataset} \
    --input_file dataset/raw/${dataset}/train.json \
    --fewshot_file dataset/w2s/${postfix}/${dataset}/gen_text/icl.json_v0-50.eval.sort.jsonltop.json \
    --output_dir dataset/w2s/${postfix}/${dataset}

# 2. Evaluate 5–10 prompts and select high-quality example samples
CUDA_VISIBLE_DEVICES=1 python -m dataset.build_w2s \
    --model $MODEL \
    --mode "gen_text" \
    --sampling_times 10 \
    --dataset ${dataset} \
    --input_file dataset/raw/${dataset}/train.json \
    --input_dir dataset/w2s/${postfix}/${dataset}/gen_prompt \
    --output_dir dataset/w2s/${postfix}/${dataset}

# Generate weak-prompt data. Please modify the fewshot_file path
CUDA_VISIBLE_DEVICES=1 python -m dataset.build_w2s \
    --model $MODEL \
    --mode "gen_prompt" \
    --sampling_times 1 \
    --gen_train_data true \
    --dataset ${dataset} \
    --input_file dataset/raw/${dataset}/train.json \
    --fewshot_file dataset/w2s/${postfix}/${dataset}/gen_text/train.json_v0-10.eval.sort.jsonltop.json \
    --output_dir dataset/w2s/${postfix}/${dataset}/weak

# Synthesize weak-to-strong training data. Please modify the prompts_file and weak_file paths
python -m dataset.build_w2s_traindata \
    --prompts_file dataset/w2s/${postfix}/${dataset}/gen_text/train.json_v0-10.eval.sort.jsonl \
    --weak_file dataset/w2s/${postfix}/${dataset}/weak/gen_prompt/train.json_v0.jsonl \
    --dataset ${dataset} \
    --output dataset/w2s/${postfix}/${dataset}/traindata \