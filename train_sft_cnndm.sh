llm_name="Qwen2-7B-Instruct"  # ("Qwen2-7B-Instruct" "Meta-Llama-3-8B-Instruct" "THUDM-glm-4-9b-chat")
dataset="cnndm"
# Stage 1: SFT
CUDA_VISIBLE_DEVICES=0,1,2,3 python cli_gt.py \
        --do_train \
        --model_name t5 \
        --output_dir out/SFT-${dataset}-${llm_name} \
        --train_file dataset/w2s/${llm_name}/${dataset}/traindata/sft.json \
        --predict_file dataset/w2s/${llm_name}/${dataset}/devdata/sft.json \
        --icl_file dataset/raw/${dataset}/icl.json \
        --model_path t5-base \
        --tokenizer_path t5-base \
        --dataset ${dataset} \
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

