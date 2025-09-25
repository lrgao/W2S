
MODEL="Qwen/Qwen2.5-7B-Instruct"
# MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL="zai-org/glm-4-9b-chat"

dataset="webnlg"
# 精确示例prompt生成
# 1. 3～5个icl样本，每个生成50～100个prompt
CUDA_VISIBLE_DEVICES=0 python build_w2s.py \
    --model $MODEL \
    --mode "gen_prompt" \
    --sampling_times 50 \
    --dataset ${dataset} \
    --input_file raw/${dataset}/icl.json \
    --output_dir w2s/${dataset}

# 2. 评估50～100个prompt
CUDA_VISIBLE_DEVICES=0 python build_w2s.py \
    --model $MODEL \
    --mode "gen_text" \
    --sampling_times 50 \
    --dataset ${dataset} \
    --input_file raw/${dataset}/icl.json \
    --output_dir w2s/${dataset}