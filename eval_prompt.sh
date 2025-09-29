model_name="Qwen/Qwen2-7B-Instruct"
input_path=""
example_path=""
CUDA_VISIBLE_DEVICES=0 python eval/eval_prompt.py \
        --input_path ${input_path} \
        --model_name ${model_name} \
        --example_path ${example_path}
