import argparse
import os
import json
import collections
from copy import deepcopy
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

from concurrent.futures import ThreadPoolExecutor, as_completed
from data import evaluate_bleu
from utils import read_txt,read_json,write_json,list2dict

device = "cuda"

def predict(params):
    item = params
    # prompt = prompt.format(query)

    # 请求返回结果
    # model：调用的模型名称，是一个字符串，用最新模型直接设置成gpt-3.5-turbo
    # messages：请求的文本内容，是一个列表，列表里每个元素类型是字典
    # role:system：设置gpt人设。
    # role:assistant：表示gpt。
    # role:user：表示用户。
    retry_count = 100
    retry_interval = 1
    for _ in range(retry_count):
        try:
            if 'text' not in item:
                prompt_scores = evaluate_bleu(data_ref=[[item['summary']]], data_sys=[item['gen_text']])
            else:
                prompt_scores = evaluate_bleu(data_ref=[item['text']], data_sys=[item['gen_text']])
            item['bleu'] = prompt_scores
            return item

        except TimeoutError:
            print("任务执行超时：", prompt)
            print('重新请求....')
            retry_count += 1
            retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
            time.sleep(retry_interval)

        except Exception as e:
            print("任务执行出错：", e)
            print('重新请求....')
            retry_count += 1
            retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
            time.sleep(retry_interval)

    return prompt,'api请求失败'

def mergeSort_dataByBLEU(root_paths,out_path,margin):
    original_data = read_txt(root_paths[0])

    original_dict = list2dict(original_data)

    # First merge prompts generated from multiple runs and remove duplicates
    for data_path in root_paths[1:]:
        # For KeyError: 'sorted_bleu', please make sure there are at least two files available for sorting.
        curr_data = read_txt(data_path)
        
        for item in curr_data:
            data_id = str(item['id'])
            if data_id in original_dict:
                if 'sorted_prompts' not in original_dict[data_id]:
                    original_dict[data_id]['sorted_prompts'] = [deepcopy(original_dict[data_id]['gen_prompt'])]
                if 'sorted_bleu' not in original_dict[data_id]:
                    # original_dict[data_id]['sorted_bleu'] = [deepcopy(original_dict[data_id]['bleu'])]
                    original_dict[data_id]['sorted_bleu'] = [deepcopy(original_dict[data_id]['bleu']['Bleu_4'])]
                if 'sorted_llm_texts' not in original_dict[data_id]:
                    original_dict[data_id]['sorted_llm_texts'] = [deepcopy(original_dict[data_id]['gen_text'])]

                if item['gen_prompt'] not in original_dict[data_id]['sorted_prompts']:
                    original_dict[data_id]['sorted_prompts'].append(deepcopy(item['gen_prompt']))
                    # original_dict[data_id]['sorted_bleu'].append(deepcopy(item['bleu']))
                    original_dict[data_id]['sorted_bleu'].append(deepcopy(item['bleu']['Bleu_4']))
                    original_dict[data_id]['sorted_llm_texts'].append(deepcopy(item['gen_text']))

    # Then sort by BLEU score
    for key in original_dict:
        zip_a_b = zip(original_dict[key]['sorted_bleu'],original_dict[key]['sorted_prompts'],original_dict[key]['sorted_llm_texts'])
        sorted_zip = sorted(zip_a_b, key=lambda x:x[0],reverse=True)
        original_dict[key]['sorted_bleu'],original_dict[key]['sorted_prompts'],original_dict[key]['sorted_llm_texts'] = zip(*sorted_zip)
        max_bleu = original_dict[key]['sorted_bleu'][0]

    # Save in the same order as the original data
    out_list = []
    top_bleu_list = []
    for item in original_data:
        data_id = str(item['id'])
        out_list.append(original_dict[data_id])
        if original_dict[data_id]['sorted_bleu'][0]>margin:
            top_bleu_list.append(item)
    
    # print(len(original_dict[data_id]['sorted_bleu']))
    print("Data num:",len(out_list))
    print(f"Samples with BLEU > {margin}:",len(top_bleu_list))

    write_json(out_list,out_path)
    write_json(top_bleu_list,out_path+f'top.json')

    return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sampling_times", type=int, default=10)
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument("--gen_train_data", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--fewshot_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_file = args.input_file
    input_list = read_json(input_file)[:10]
    print(f'load {args.dataset} data from {input_file}')
    print('Data size: {0}'.format(len(input_list)))
    postfix = args.input_file.split('/')[-1]
    
    if args.input_dir:
        input_dir = args.input_dir
        print(f'load {args.dataset} prompts from {input_dir}')
    
    if args.fewshot_file:
        fewshots = read_json(args.fewshot_file)
        print(f'load few-shot data from {args.fewshot_file}')
        print('数据大小：{0}'.format(len(fewshots)))

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
    if not os.path.exists(save_path):
        os.makedirs(save_path)  
    
    
    for sampling_time in range(args.sampling_times):
        print('sampling_time',sampling_time)
        if args.mode == "gen_text":
            input_list = read_txt(f'{input_dir}/{postfix}_v{sampling_time}.jsonl')

        save_file = f'{save_path}/{postfix}_v{sampling_time}.jsonl'

        f = open(save_file,'w',encoding='utf-8')
        for i in tqdm(range(0,len(input_list),args.batch_size)):
            curr_data = input_list[i:i+args.batch_size]
            
            dialogs = []
            for item in curr_data:
                
                if args.mode == "gen_prompt":
                    if args.fewshot_file:
                        if args.dataset == "webnlg":
                            dialog = [{"role": "system", "content": "Given triples and a corresponding text, please provide a prompt that can help you generate the text given only the triples."}]
                            for fewshot in fewshots:
                                dialog.extend([
                                    {"role": "user", "content": "Triples: {0}".format(fewshot['input_string'])},
                                    {"role": "assistant", "content": "{0}".format(fewshot['sorted_prompts'][0])},
                                ])
                            dialog.append({"role": "user", "content": "Triples: {0}".format(item['input_string'])})
                            dialogs.append(dialog)
                        elif args.dataset == "cnndm":
                            dialog = [{"role": "system", "content": "Given an article and a corresponding summary, write a prompt that helps you generate the above summary given only the article."}]
                            for fewshot in fewshots:
                                dialog.extend([
                                    {"role": "user", "content": "Article: {0}".format(fewshot['article'])},
                                    {"role": "assistant", "content": "{0}".format(fewshot['sorted_prompts'][0])},
                                ])
                            dialog.append({"role": "user", "content": "Article: {0}".format(item['article'])})
                            dialogs.append(dialog)
                        else: # e2e
                            dialog = [{"role": "system", "content": "Given some data about a restaurant and a sentence that presents the different aspects of the data about the restaurant, please provide a prompt that can help you generate the sentence given only the data."}]
                            for fewshot in fewshots:
                                dialog.extend([
                                    {"role": "user", "content": "Data: {0}".format(fewshot['input_string'])},
                                    {"role": "assistant", "content": "{0}".format(fewshot['sorted_prompts'][0])},
                                ])
                            dialog.append({"role": "user", "content": "Data: {0}".format(item['input_string'])})
                            dialogs.append(dialog)
                    
                    else:
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
                            {"role": "system", "content": "Given a triple and a prompt, generate a brief 1-2 sentence that follows the prompt exactly."},
                            {"role": "user", "content": "Triples: {0}\n\n{1}".format(item['input_string'],item['gen_prompt'])},
                        ])
                    elif args.dataset == "cnndm":
                        dialogs.append([
                            {"role": "system", "content": "Given an article and a prompt, generate a brief 3-4 sentence summary that follows the prompt exactly."},
                            {"role": "user", "content": "Article: {0}\n\n{1}".format(item['article'],item['gen_prompt'])},
                        ])
                    else: # e2e
                        dialogs.append([
                            {"role": "system", "content": "Given some data about a restaurant and a prompt, generate a brief 1-2 descriptions that follows the prompt exactly."},
                            {"role": "user", "content": "Data: {0}\n\n{1}".format(item['input_string'],item['gen_prompt'])},
                        ])

            gen_text = gen_text_multipleDialog(dialogs,sampling_params)
            
            for item,gt in zip(curr_data,gen_text):
                item[args.mode] = gt
                f.write(json.dumps(item,ensure_ascii=False)+'\n')
        f.close()

        # eval
        if args.mode == "gen_text":
            # eval prompt
            dataset = read_txt(save_file)
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(predict, (item)) for item in dataset]
                idx2res = collections.defaultdict(int) 
                for job in tqdm(as_completed(futures)):
                    res = job.result(timeout=None)
                    idx2res[res['id']] = res
            
            f = open(f'{save_path}/{postfix}_v{sampling_time}.eval.jsonl','a+',encoding='utf-8')
            print(f'{save_path}/{postfix}_v{sampling_time}.eval.jsonl')
            for item in dataset:
                item['predict'] = idx2res[item['id']]['bleu']
                f.write(json.dumps(item,ensure_ascii=False)+'\n')

            f.close()

    # merge & sort & select
    if args.mode == "gen_text":
        paths = []
        for i in range(args.sampling_times):
            path = f'{save_path}/{postfix}_v{i}.eval.jsonl'
            if os.path.exists(path):
                paths.append(path)
        
        sortpath = f'{save_path}/{postfix}_v0-{args.sampling_times}.eval.sort.jsonl'

        # Set the threshold for selecting high-quality samples
        if args.dataset == "webnlg":
            margin = 0.6
        elif args.dataset == "cnndm":
            margin = 0.15
        else: # e2e
            margin = 0.4
        mergeSort_dataByBLEU(paths,sortpath,margin)

