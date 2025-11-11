import json
import os
import argparse
from utils import read_txt,read_json,write_json,list2dict
from copy import deepcopy


def add_weakprompt(input_file,weak_file,is_clean=True, output=''):
    input_data = read_json(input_file)
    weak_data = read_txt(weak_file)
    save_file = output+'/'+'sft.json'
    if not os.path.exists(output):
        os.makedirs(output)  

    weakprompt_data_dict = list2dict(weak_data)

    output_data = []
    for item in input_data:
        data_id = str(item['id'])
        if data_id in weakprompt_data_dict:
            weakprompt_item = weakprompt_data_dict[data_id]
            item['weak_prompt'] = weakprompt_item['gen_prompt']
            if 'sorted_prompts' in item:
                item['target_prompt'] = deepcopy(item['sorted_prompts'][0])
            if is_clean:
                item.pop('sorted_prompts')
                item.pop('sorted_bleu')
                item.pop('sorted_llm_texts')
                item.pop('bleu')
                item.pop('gen_prompt')
                item.pop('gen_text')
            output_data.append(item)
    print(len(output_data))
    write_json(output_data,save_file)
    return None

def gen_dpodata(input_file,weak_file='',dataset='',output=''):
    input_data = read_json(input_file)
    save_file = output+'/'+'dpo.json'
    if not os.path.exists(output):
        os.makedirs(output)  

    if len(weak_file)>1:
        weak_data = read_txt(weak_file)
        # weak_data = read_json(weak_file)
        weakprompt_data_dict = list2dict(weak_data)

    f = open(save_file,'w',encoding='utf-8')

    for item in input_data:
        bad_prompts = []
        # Do not run DPO when the number of prompts is less than 1
        if len(item['sorted_prompts'])<2:
            continue
        # for prompt,score in zip(item['sorted_prompts'],item['sorted_bleu']):
        #     if score<0.1:
        #         bad_prompts.append(prompt)
        
        if len(bad_prompts)<1:
            bad_prompts.append(item['sorted_prompts'][-1])
        for bp in bad_prompts:

            if dataset == 'cnndm':
                new_item ={
                    'id': item['id'],
                    'prompt': '[Article] {0} [Possible prompt] {1}'.format(item['article'],weakprompt_data_dict[item['id']]['gen_prompt']),
                    'chosen': item['sorted_prompts'][0],
                    'rejected': bp,
                }
            elif dataset == 'webnlg':
                new_item ={
                    'id': item['id'],
                    'prompt': '[Triples] {0} [Possible prompt] {1}'.format(item['input_string'],weakprompt_data_dict[str(item['id'])]['gen_prompt']),
                    'chosen': item['sorted_prompts'][0],
                    'rejected': bp,
                }
            else:
                new_item ={
                    'id': item['id'],
                    'prompt': '[Data] {0} [Possible prompt] {1}'.format(item['input_string'],weakprompt_data_dict[str(item['id'])]['gen_prompt']),
                    'chosen': item['sorted_prompts'][0],
                    'rejected': bp,
                }
        f.write(json.dumps(new_item,ensure_ascii=False)+'\n')
    f.close()
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, default='')
    parser.add_argument("--weak_file", type=str, default=0)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prompts_file = args.prompts_file
    weak_file = args.weak_file
    dataset = args.dataset
    output = args.output

    gen_dpodata(prompts_file, weak_file=weak_file, dataset=dataset,output=output)
    add_weakprompt(prompts_file, weak_file=weak_file, output=output, is_clean=True)