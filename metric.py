import collections
import copy
import logging
import os
import random

import statistics
from typing import Callable, Dict, List, Tuple, Union

import pdb
import evaluate
import nltk
import numpy as np
import scipy.stats
import torch
import tqdm
import transformers
from transformers import T5Tokenizer,T5ForConditionalGeneration

# import vec2text

logger = logging.getLogger(__name__)


DEFAULT_INPUT_STRING = "Twas brillig, and the slithy toves, Did gyre and gimble in the wabe, All mimsy were the borogoves, And the mome raths outgrabe."


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def sem(L: List[float]) -> float:
    # 标准差
    result = scipy.stats.sem(np.array(L))
    if isinstance(result, np.ndarray):
        return result.mean().item()
    return result


def mean(L: Union[List[int], List[float]]) -> float:
    return sum(L) / len(L)


def count_overlapping_ngrams(s1: str, s2: str, n: int) -> int:
    ngrams_1 = nltk.ngrams(s1, n)
    ngrams_2 = nltk.ngrams(s2, n)
    ngram_counts_1 = collections.Counter(ngrams_1)
    ngram_counts_2 = collections.Counter(ngrams_2)
    total = 0
    for ngram, count in ngram_counts_1.items():
        total += min(count, ngram_counts_2[ngram])
    return total


def compute_metric(preds_sample_list,preds_sample_labels_list,tokenizer):

    metric_accuracy = evaluate.load("accuracy")
    metric_bleu = evaluate.load("sacrebleu")
    metric_bertscore = evaluate.load("bertscore")
    metric_rouge = evaluate.load("rouge")

    def _text_comparison_metrics(
        predictions_ids: List[List[int]],
        predictions_str: List[str],
        references_ids: List[List[int]],
        references_str: List[str],
    ) -> Dict[str, float]:
        assert len(predictions_ids) == len(references_ids)
        assert len(predictions_ids) == len(predictions_str)
        assert len(predictions_str) == len(references_str)
        num_preds = len(predictions_ids)
        if not num_preds:
            return {}

        ###########################################################

        # Compute token, precision, recall, and ngram-level metrics.
        precision_sum = 0.0
        recall_sum = 0.0
        num_overlapping_words = []
        num_overlapping_bigrams = []
        num_overlapping_trigrams = []
        num_true_words = []
        num_pred_words = []
        f1s = []
        for i in range(num_preds):
            true_words = nltk.tokenize.word_tokenize(references_str[i])
            pred_words = nltk.tokenize.word_tokenize(predictions_str[i])
            num_true_words.append(len(true_words))
            num_pred_words.append(len(pred_words))

            true_words_set = set(true_words)
            pred_words_set = set(pred_words)
            TP = len(true_words_set & pred_words_set)
            FP = len(true_words_set) - len(true_words_set & pred_words_set)
            FN = len(pred_words_set) - len(true_words_set & pred_words_set)

            precision = (TP) / (TP + FP + 1e-20)
            recall = (TP) / (TP + FN + 1e-20)

            try:
                f1 = (2 * precision * recall) / (precision + recall + 1e-20)
            except ZeroDivisionError:
                f1 = 0.0
            f1s.append(f1)

            precision_sum += precision
            recall_sum += recall

            ############################################################
            num_overlapping_words.append(
                count_overlapping_ngrams(true_words, pred_words, 1)
            )
            num_overlapping_bigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 2)
            )
            num_overlapping_trigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 3)
            )

        set_token_metrics = {
            "token_set_precision": (precision_sum / num_preds),
            "token_set_recall": (recall_sum / num_preds),
            "token_set_f1": mean(f1s),
            "token_set_f1_sem": sem(f1s),
            "n_ngrams_match_1": mean(num_overlapping_words),
            "n_ngrams_match_2": mean(num_overlapping_bigrams),
            "n_ngrams_match_3": mean(num_overlapping_trigrams),
            "num_true_words": mean(num_true_words),
            "num_pred_words": mean(num_pred_words),
        }
        
        ############################################################
        bleu_results = np.array(
            [
                metric_bleu.compute(predictions=[p], references=[r])["score"]
                for p, r in zip(predictions_str, references_str)
            ]
        )
        rouge_result = metric_rouge.compute(
            predictions=predictions_str, references=references_str
        )
        # pdb.set_trace()
        bleu_results = (bleu_results.tolist())  # store bleu results in case we want to use them later for t-tests
        # bertscore_result = metric_bertscore.compute(
        #     predictions=predictions_str, references=references_str, lang="en"
        # )
        
        # pdb.set_trace()
        exact_matches = np.array(predictions_str) == np.array(references_str)
        gen_metrics = {
            # "bleu_score": bleu_results.mean(),
            "bleu_score": mean(bleu_results),
            "bleu_score_sem": sem(bleu_results),
            "rouge_score": rouge_result[
                "rouge1"
            ],  # ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
            # "bert_score": statistics.fmean(bertscore_result["f1"]),
            "exact_match": mean(exact_matches),
            "exact_match_sem": sem(exact_matches),
        }
        # P R F1 BLEU ROUGE bert_score exact_match

        all_metrics = {**set_token_metrics, **gen_metrics}

        return all_metrics

    def eval_generation_metrics(
        preds_sample_list,
        preds_sample_labels_list,
    ) -> Dict[str, float]:
        # Get decoded text. Note that this is different than `preds`, which
        # is used to compute the loss.
        # Log BLEU, log table of text.
        decoded_preds = tokenizer.batch_decode(
            preds_sample_list, skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(
            preds_sample_labels_list, skip_special_tokens=True
        )
        # import pdb
        # pdb.set_trace()
        # preds_sample_list/preds_sample_labels_list:数据量*句子最大长度  id [[256,243,...,34],[256,243,...,34],...]
        # decoded_preds/decoded_labels: 数据量 string_list ['prompt1','prompt2',...]

        bleu_result = _text_comparison_metrics(
            predictions_ids=preds_sample_list,
            predictions_str=decoded_preds,
            references_ids=preds_sample_labels_list,
            references_str=decoded_labels,
        )

        if not len(decoded_preds):
            return {}
        
        # pdb.set_trace()
        # Compute sims of eval data using embedder.
        # preds_sample = torch.tensor(preds_sample_list)[:128]
        # preds_sample_labels = torch.tensor(
        #     preds_sample_labels_list)[:128]

        # # Log num tokens.
        # num_tokens_metrics = {
        #     "pred_num_tokens": (
        #         (preds_sample != tokenizer.pad_token_id)
        #         & (preds_sample != tokenizer.bos_token_id)
        #     )
        #     .sum(1)
        #     .float()
        #     .mean()
        #     .item(),
        #     "true_num_tokens": (
        #         (preds_sample_labels != tokenizer.pad_token_id)
        #         & (preds_sample_labels != tokenizer.bos_token_id)
        #     )
        #     .sum(1)
        #     .float()
        #     .mean()
        #     .item(),
        # }

        # # Fix eos token on generated text.
        # # bos_token_id = embedder_tokenizer.pad_token_id
        # # assert (preds_sample[:, 0] == bos_token_id).all()
        # eos_token_id = tokenizer.eos_token_id
        # if eos_token_id is not None:
        #     eos_tokens = (
        #         torch.ones(
        #             (len(preds_sample), 1),
        #             dtype=torch.long,
        #         )
        #         * eos_token_id
        #     )
        #     preds_sample = torch.cat((preds_sample[:, 1:], eos_tokens), dim=1)
            # assert preds_sample.shape == preds_sample_labels.shape

        # try:
        #     with torch.no_grad():
        #         # inversion_trainer.model.noise_level = 0.0
        #         preds_sample_retokenized = embedder_tokenizer(
        #             decoded_preds,
        #             padding=True,
        #             truncation=False,
        #             return_tensors="pt",
        #         )["input_ids"].to(preds_sample.device)
        #         preds_sample_retokenized = preds_sample_retokenized[
        #             : args.per_device_eval_batch_size, :
        #         ]
        #         pad_token_id = pad_token_id
        #         preds_emb = call_embedding_model(
        #             input_ids=preds_sample_retokenized,
        #             attention_mask=(preds_sample_retokenized != pad_token_id).to(
        #                 args.device
        #             ),
        #         )
        #         preds_sample_labels_retokenized = embedder_tokenizer(
        #             decoded_labels, padding=True, truncation=False, return_tensors="pt"
        #         )["input_ids"].to(preds_sample.device)
        #         preds_sample_labels_retokenized = preds_sample_labels_retokenized[
        #             : args.per_device_eval_batch_size, :
        #         ]
        #         labels_emb = call_embedding_model(
        #             input_ids=preds_sample_labels_retokenized,
        #             attention_mask=(preds_sample_labels_retokenized != pad_token_id).to(
        #                 args.device
        #             ),
        #         )
        #         emb_cos_sims = torch.nn.CosineSimilarity(dim=1)(preds_emb, labels_emb)
        #         emb_topk_equal = (
        #             (preds_emb[:, :32000].argmax(1) == labels_emb[:, :32000].argmax(1))
        #             .float()
        #             .cpu()
        #         )
        #         sim_result = {
        #             "emb_cos_sim": emb_cos_sims.mean().item(),
        #             "emb_cos_sim_sem": sem(emb_cos_sims.cpu().numpy()),
        #             "emb_top1_equal": emb_topk_equal.mean().item(),
        #             "emb_top1_equal_sem": sem(emb_topk_equal),
        #         }

        # except (TypeError, RuntimeError,):
        #     pass
        sim_result = {}
        # sim_result = evaluate_system_prompts(decoded_preds, decoded_labels, all_inputs)
        # sim_result = evaluate_kl_divergence(decoded_preds, decoded_labels, all_inputs)

        # Store stuff for access later.
        # preds_emb = preds_emb.cpu()
        # labels_emb = labels_emb.cpu()
        # preds_sample_list = preds_sample_list
        # preds_sample_labels_list = preds_sample_labels_list

        metrics = {**bleu_result, **sim_result}
        # metrics = {**num_tokens_metrics, **bleu_result, **sim_result}
        return metrics
    
    metrics = eval_generation_metrics(preds_sample_list,preds_sample_labels_list)
    # print(metrics)
    return metrics

# tokenizer = T5Tokenizer.from_pretrained('/data/gaolr/models/t5-base')
# # model = T5ForConditionalGeneration.from_pretrained('/data/gaolr/models/t5-base')
# # preds_sample_list/preds_sample_labels_list:数据量*句子最大长度  id [[256,243,...,34],[256,243,...,34],...]
# # decoded_preds/decoded_labels: 数据量 string_list ['prompt1','prompt2',...]

# preds_sample_list = [[256,243,6],[256,243,4]]
# preds_sample_labels_list = [[256,243,6,34],[256,243,7,34]]
# compute_metric(preds_sample_list,preds_sample_labels_list,tokenizer)
