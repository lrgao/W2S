
import json
import pdb
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-base')

def read_txt(input_path,split_str='\t'):
    out_list = []
    with open(input_path,encoding='utf-8') as var:
        for line in tqdm(var):
            # out_list.append(eval(line.strip('\n')))
            out_list.append(json.loads(line.strip('\n')))
    return out_list

def cal_len(input_file,dataset="webnlg"):
    if ('.jsonl' in input_file) or ('.txt' in input_file):
        data_list = read_txt(input_file)
    else:
        data_list = json.load(open(input_file, encoding='utf-8'))
    lens = []

    for data in tqdm(data_list):
        if dataset == "cnndm":
            input_string = '[Article] {0} [Possible prompt] {1}'.format(data['article'],data['weak_prompt'])
        elif dataset == "e2e":
            input_string = '[Data] {0} [Possible prompt] {1}'.format(data['input_string'],data['weak_prompt'])
        else:
            input_string = '[Triples] {0} [Possible prompt] {1}'.format(data['input_string'],data['weak_prompt'])
        
        input_str = tokenizer(input_string)['input_ids']
        lens.append(len(input_str))
        
    print('triple_lens', len(lens))
    for perc in [95, 98, 99, 100]:
        a = np.percentile(lens, perc)
        print('{0}% triple_lens is {1}'.format(perc, a))
    return None

cal_len('/data/gaolr/workspace/AQPG/t5-cnndm/dataset/qwen-7b-global/1-gentexts/train.gen_prompts.0-9.weak_clean.json')