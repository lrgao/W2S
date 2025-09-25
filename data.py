import json
from tqdm import tqdm
import sys
import copy
import json

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from eval.pycocotools.coco import COCO
from eval.pycocoevalcap.eval import COCOEvalCap


def run_coco_eval(data_ref, data_sys):
    """Run the COCO evaluator, return the resulting evaluation object (contains both
    system- and segment-level scores."""
    # convert references and system outputs to MS-COCO format in-memory
    coco_ref = create_coco_refs(data_ref)
    coco_sys = create_coco_sys(data_sys)

    print('Running MS-COCO evaluator...', file=sys.stderr)
    coco = COCO()
    coco.dataset = coco_ref
    coco.createIndex()

    coco_res = coco.loadRes(resData=coco_sys)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    return coco_eval


def create_coco_refs(data_ref):
    """Create MS-COCO human references JSON."""
    out = {'info': {}, 'licenses': [], 'images': [], 'type': 'captions', 'annotations': []}
    ref_id = 0
    for inst_id, refs in enumerate(data_ref):
        out['images'].append({'id': 'inst-%d' % inst_id})
        for ref in refs:
            out['annotations'].append({'image_id': 'inst-%d' % inst_id,
                                       'id': ref_id,
                                       'caption': ref})
            ref_id += 1
    return out


def create_coco_sys(data_sys):
    """Create MS-COCO system outputs JSON."""
    out = []
    for inst_id, inst in enumerate(data_sys):
        out.append({'image_id': 'inst-%d' % inst_id, 'caption': inst})
    return out


def read_txt(input_path,split_str='\t'):
    out_list = []
    with open(input_path,encoding='utf-8') as var:
        for line in tqdm(var):
            # out_list.append(eval(line.strip('\n')))
            out_list.append(json.loads(line.strip('\n')))
    return out_list

class w2sDataLoader(DataLoader):

    def __init__(self, args, dataset, mode):
        if mode == "train":
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(w2sDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size,
                                               num_workers=args.num_workers)


# Downstream dataset (webnlg, webquestions, pathquestions)
# Most parts are similar to WikidataDataset
class w2sDataset(Dataset):
    def __init__(self, logger, args, data_path, tokenizer, mode):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.dataset = args.dataset
        
        if ('.jsonl' in self.data_path) or ('.txt' in self.data_path):
            self.data = read_txt(self.data_path)
        else:
            self.data = json.load(open(self.data_path, encoding='utf-8'))
        
        print("Total samples = {}".format(len(self.data)))

        if args.debug:
            self.data = self.data[:1000]
        
        assert type(self.data) == list
        assert all(["id" in d for d in self.data]), self.data[0].keys()
        if type(self.data[0]["id"]) == int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])

        self.args = args
        self.data_type = mode
        self.metric = "BLEU"

        self.mask_token = self.tokenizer.additional_special_tokens[0]
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0])

        self.add_bos_id = []

    def __len__(self):
        return len(self.data)

    def encode_input(self, a, add_bos_id):
        length_a_b = self.args.max_input_length - len(add_bos_id) - 1
        if len(a) > length_a_b:
            a = a[:length_a_b]
        input_ids = add_bos_id + a + [self.tokenizer.eos_token_id]
        attn_mask = [1] * len(input_ids) + [0] * (self.args.max_input_length - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids))
        assert len(input_ids) == len(attn_mask) == self.args.max_input_length 
        return input_ids, attn_mask

    def encode_input_output(self, answers, questions, add_bos_id):
        # add bos and eos
        decoder_label_ids = copy.deepcopy(answers)
        if len(decoder_label_ids) > self.args.max_output_length - len(add_bos_id) - 1:
            decoder_label_ids = decoder_label_ids[:(self.args.max_output_length - len(add_bos_id) - 1)]
        decoder_label_ids = add_bos_id + decoder_label_ids + [self.tokenizer.eos_token_id]
        decoder_attn_mask = [1] * len(decoder_label_ids) + [0] * (self.args.max_output_length - len(decoder_label_ids))
        decoder_label_ids += [self.tokenizer.pad_token_id] * (self.args.max_output_length - len(decoder_label_ids))
        assert len(decoder_label_ids) == self.args.max_output_length == len(decoder_attn_mask)

        input_ids, input_attn_mask= self.encode_input(questions, add_bos_id)

        return input_ids, input_attn_mask, decoder_label_ids, decoder_attn_mask
    
    def __getitem__(self, idx):

        entry = self.data[idx]
        
        inputs_ids = []
        
        if self.dataset in ['webnlg']:
            input_text = '[Triples] {0} [Possible prompt] {1}'.format(entry['input_string'],entry['weak_prompt'])
        elif self.dataset in ['cnndm']:
            input_text = '[Article] {0} [Possible prompt] {1}'.format(entry['article'],entry['weak_prompt'])
        else:
            input_text = '[Data] {0} [Possible prompt] {1}'.format(entry['input_string'],entry['weak_prompt'])

        for input_ in input_text.split():
            input_id_ = self.tokenizer.encode(" {}".format(input_), add_special_tokens=False)
            inputs_ids += input_id_
        
        words_label_ids, words_label_tokens = [], ''

        if self.data_type == 'train':
            current_text = entry['target_prompt']
        else:
            if self.dataset in ['cnndm']:
                current_text = entry['summary']
            else:
                current_text = entry['text'][0]

        for word in current_text.split():
            word_label_ids = self.tokenizer.encode(" {}".format(word), add_special_tokens=False)
            word_label_tokens = copy.deepcopy(word)

            words_label_ids += word_label_ids
            words_label_tokens += ' ' + word_label_tokens

        input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask = \
                self.encode_input_output(words_label_ids, inputs_ids, self.add_bos_id)

        assert len(decoder_label_ids) == len(decoder_attn_mask) == self.args.max_output_length

        input_ids_ar = torch.LongTensor(input_ids_ar)
        attn_mask_ar = torch.LongTensor(attn_mask_ar)
        
        decoder_label_ids = torch.LongTensor(decoder_label_ids)
        decoder_attn_mask = torch.LongTensor(decoder_attn_mask)

        return input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask

def evaluate_bleu(data_ref, data_sys):
    coco_eval = run_coco_eval(data_ref, data_sys)
    scores = {metric: score for metric, score in list(coco_eval.eval.items())}
    return scores
