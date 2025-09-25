# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import torch
import re
import json
import pdb
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from common_func import read_json,write_json,read_txt
from vllm import LLM, SamplingParams

device = "cuda"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sampling_times", type=int, default=10)
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_file = args.input_file
    input_list = read_json(input_file)
    print(f'load {args.dataset} data from {input_file}')
    print('数据大小：{0}'.format(len(input_list)))

    model_name = args.model
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize the vLLM engine
    llm = LLM(model=model_name)
    
    def gen_text_multipleDialog(messages,sampling_params,max_new_tokens=4096):
        texts = []
        for message in messages:
            text = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)
        outputs = llm.generate(texts, sampling_params)

        response = []
        for output in outputs:
            generated_text = output.outputs[0].text
            response.append(generated_text)
        return response
    
    if args.mode == "gen_prompt":
        # Configurae the sampling parameters
        sampling_params = SamplingParams(temperature=0.9, top_p=0.95, top_k=20, max_tokens=8192)
        
    
    if args.mode == "gen_text":
        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, top_k=20, max_tokens=8192)
        
    save_path = args.output_dir+'/'+args.mode
    # 如果文件夹不存在，则新建
    if not os.path.exists(save_path):
        os.makedirs(save_path)  
    
    postfix = args.input_file.split('/')[-1]
    for i in range(args.sampling_times):
        
        save_file = f'{save_path}/{postfix}_v{i}.jsonl'
        f = open(save_file,'w',encoding='utf-8')
        for i in tqdm(range(0,len(input_list),args.batch_size)):
            curr_data = input_list[i:i+args.batch_size]
            
            dialogs = []
            for item in curr_data:
                
                if args.mode == "gen_prompt":
                    if args.dataset == "webnlg":
                        dialogs.append([
                            {"role": "system", "content": "Given triples and a corresponding text, please provide a prompt that can help you generate the text given only the triples."},
                            {"role": "user", "content": "Triples: {0}\n\nText: {1}\n\nPlease provide a prompt.".format(item['input_string'],item['text'][0])},
                        ])
                    elif args.dataset == "cnndm":
                        dialogs.append([
                            {"role": "system", "content": "Given an article and a corresponding summary, write a prompt that helps you generate the above summary given only the article."},
                            {"role": "user", "content": "Article: {0}\n\nSummary: {1}\n\nPlease provide a prompt.".format(item['article'],item['summary'])},
                        ])
                    else: # e2e
                        dialogs.append([
                            {"role": "system", "content": "Given some data about a restaurant and a sentence that presents the different aspects of the data about the restaurant, please provide a prompt that can help you generate the sentence given only the data."},
                            {"role": "user", "content": "Data: {0}\n\nSentence: {1}\n\nPlease provide a prompt.".format(item['input_string'],item['text'][0])},
                        ])
                
                if args.mode == "gen_text":
                    if args.dataset == "webnlg":
                        dialogs.append([
                            {"role": "system", "content": "Given triples and a corresponding text, please provide a prompt that can help you generate the text given only the triples."},
                            {"role": "user", "content": "Triples: {0}\n\nText: {1}\n\nPlease provide a prompt.".format(item['input_string'],item['text'][0])},
                        ])
                    elif args.dataset == "cnndm":
                        dialogs.append([
                            {"role": "system", "content": "Given an article and a corresponding summary, write a prompt that helps you generate the above summary given only the article."},
                            {"role": "user", "content": "Article: {0}\n\nSummary: {1}\n\nPlease provide a prompt.".format(item['article'],item['summary'])},
                        ])
                    else: # e2e
                        dialogs.append([
                            {"role": "system", "content": "Given some data about a restaurant and a sentence that presents the different aspects of the data about the restaurant, please provide a prompt that can help you generate the sentence given only the data."},
                            {"role": "user", "content": "Data: {0}\n\nSentence: {1}\n\nPlease provide a prompt.".format(item['input_string'],item['text'][0])},
                        ])
                
                    
            gen_text = gen_text_multipleDialog(dialogs)
            
            for item,gt in zip(curr_data,gen_text):
                item[args.mode] = gt
                f.write(json.dumps(item,ensure_ascii=False)+'\n')
        f.close()