
# MODEL="Qwen/Qwen2.5-7B-Instruct"
# MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL="zai-org/glm-4-9b-chat"
MODEL="/data/gaolr/models/Qwen2.5-1.5B-Instruct"
postfix="Qwen2.5-1.5B-Instruct"

dataset="e2e"
# Generate precise example prompts
# 1. Use 3–5 ICL samples, each producing 50–100 prompts
CUDA_VISIBLE_DEVICES=1 python -m dataset.build_w2s \
    --model $MODEL \
    --mode "gen_prompt" \
    --sampling_times 50 \
    --dataset ${dataset} \
    --input_file dataset/raw/${dataset}/dev.json \
    --output_dir dataset/w2s/${postfix}/${dataset}/dev

# 2. Evaluate 50–100 prompts and select the best-quality examples
CUDA_VISIBLE_DEVICES=1 python -m dataset.build_w2s \
    --model $MODEL \
    --mode "gen_text" \
    --sampling_times 50 \
    --dataset ${dataset} \
    --input_file dataset/raw/${dataset}/dev.json \
    --input_dir dataset/w2s/${postfix}/${dataset}/dev/gen_prompt \
    --output_dir dataset/w2s/${postfix}/${dataset}/dev

# Generate weak-prompt data. Please modify the fewshot_file path
CUDA_VISIBLE_DEVICES=1 python -m dataset.build_w2s \
    --model $MODEL \
    --mode "gen_prompt" \
    --sampling_times 1 \
    --gen_train_data true \
    --dataset ${dataset} \
    --input_file dataset/raw/${dataset}/dev.json \
    --fewshot_file dataset/w2s/${postfix}/${dataset}/gen_text/train.json_v0-10.eval.sort.jsonltop.json \
    --output_dir dataset/w2s/${postfix}/${dataset}/dev/weak

# Synthesize weak-to-strong training data. Please modify the prompts_file and weak_file paths
python -m dataset.build_w2s_traindata \
    --prompts_file dataset/w2s/${postfix}/${dataset}/dev/gen_text/dev.json_v0-50.eval.sort.jsonl \
    --weak_file dataset/w2s/${postfix}/${dataset}/dev/weak/gen_prompt/dev.json_v0.jsonl \
    --dataset ${dataset} \
    --output dataset/w2s/${postfix}/${dataset}/devdata \