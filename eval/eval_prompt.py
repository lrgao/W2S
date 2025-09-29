import json
import sys
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

def list2dict(input_list):
    out_dict = {}
    for item in input_list:
        out_dict[str(item['id'])] = item
    return out_dict

def gen_text_singleDialog(model,tokenizer,messages,max_new_tokens=512):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def gen_text(model,tokenizer,input_file,example_path='',max_batch_size=1):
    
    out_text_path = input_file.replace('.json','.eval.json')
    input_list = json.load(open(input_file, encoding='utf-8'))

    f = open(out_text_path,'+a',encoding='utf-8')
    
    examples = json.load(open(example_path, encoding='utf-8'))
    examples = list2dict(examples)
    
    # Preserve the order of few-shot samples
    if 'webnlg' in input_file:
        example_ids = ["5312","1385","4368"]
        triples = [examples[idx]['input_string'] for idx in example_ids]
        texts = [examples[idx]['text'][0] for idx in example_ids]
        sys_prompt = "Given a triple and a prompt, generate a brief 1-2 sentence that follows the prompt exactly."
        usr_template = "Triples: {0}\n\n{1}"
        
    elif 'cnndm' in input_file:
        example_ids = ["1","2","3"]
        triples = [examples[idx]['article'] for idx in example_ids]
        texts = [examples[idx]['summary'] for idx in example_ids]
        sys_prompt = "Given an article and a prompt, generate a brief 3-4 sentence summary that follows the prompt exactly."
        usr_template = "Article: {0}\n\n{1}"
        
    else:
        example_ids = ["5312","1385","4368"]
        triples = [examples[idx]['input_string'] for idx in example_ids]
        texts = [examples[idx]['text'][0] for idx in example_ids]
        sys_prompt = "Given some data about a restaurant and a prompt, generate a brief 1-2 descriptions that follows the prompt exactly."
        usr_template = "Data: {0}\n\n{1}"
    
    prompts = [examples[idx]['T5_gen_prompt'] for idx in example_ids]

    print("Generating text from prompt...")
    for i in tqdm(range(0,len(input_list),max_batch_size)):
        curr_data = input_list[i:i+max_batch_size]
        dialogs = []
        for item in curr_data:
            dialog = [
                {"role": "system", "content": "{0}".format(sys_prompt)},

                {"role": "user", "content": usr_template.format(triples[0],prompts[0])},
                {"role": "assistant", "content": "{0}".format(texts[0])},

                {"role": "user", "content": usr_template.format(triples[1],prompts[1])},
                {"role": "assistant", "content": "{0}".format(texts[1])},

                {"role": "user", "content": usr_template.format(triples[2],prompts[2])},
                {"role": "assistant", "content": "{0}".format(texts[2])},
            ]
            
            if 'cnndm' in input_file:
                dialog.append({"role": "user", "content": usr_template.format(item['article'],item['T5_GenPrompt'])})
            else:
                dialog.append({"role": "user", "content": usr_template.format(item['input_string'],item['T5_GenPrompt'])})
            
            dialogs.append(dialog)

        gen_text = [gen_text_singleDialog(model,tokenizer,dialog) for dialog in dialogs]

        assert len(gen_text) == len(curr_data)
        for item,gt in zip(curr_data,gen_text):
            item['llm_gen_text'] = gt
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
    
    f.close()
    
    print('Data saved to:',out_text_path)
    return None

if __name__ == "__main__":
    input_path,model_name,example_path = sys.argv[1],sys.argv[2],sys.argv[3]
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    gen_text(model,tokenizer,input_path,example_path)
