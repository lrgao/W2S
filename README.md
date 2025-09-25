# W2S: Weak-to-Strong Prompt Correction for Large Language Models

## Introduction

W2S is a Weak-to-Strong Prompt Correction (W2S) framework. The core principle of W2S is to simplify the optimization process by starting with weak prompts and progressively correcting them into stronger prompts. W2S begins by constructing a weak-to-strong prompt dataset using LLM, then employs a two-stage weak-to-strong prompt correction: supervised fine-tuning to learn transfer patterns from weak prompts to strong prompts and direct preference optimization to mitigate potential errors. 


## Quick Start

<!-- **NOTE**: At the very beginning, in order to compute the METEOR scores, please download the required [data](https://github.com/xinyadu/nqg/blob/master/qgevalcap/meteor/data/paraphrase-en.gz) and put it under the following two folders: `eval_webnlg/pycocoevalcap/meteor/data/` and `eval_wqpq/meteor/data/`. -->

### Installation
```shell
pip install -r requirements.txt
```

### Datasets

Our experiments contain three downstream datasets, i.e., [WebNLG](https://gitlab.com/shimorina/webnlg-dataset/), [CNN/Daily Mail](https://aclanthology.org/K16-1028.pdf), and [Clean E2E NLG](https://aclanthology.org/W19-8652.pdf). 

### Weak-to-Strong Dataset Construction

```shell
cd dataset
bash build_w2s.sh
```

### Weak-to-Strong Prompt Correction

W2S prompt corrector is built on the [T5-base model](https://huggingface.co/google-t5/t5-base) and trained through a two-stage process: (1) learning a weak-to-strong prompt corrector via supervised fine-tuning (SFT), and (2) refining the corrector to mitigate errors from weak or suboptimal prompts using direct preference optimization (DPO).

#### Stage 1: Learning a weak-to-strong prompt corrector via SFT

```shell
bash train_sft_webnlg.sh
bash train_sft_cnndm.sh
bash train_sft_e2e.sh
```

#### Stage 2: Refining the corrector to mitigate errors present in weak or bad prompts via DPO.

In Stage 2, the model is incrementally refined based on the Stage-1 model.

```shell
bash train.sh
```

### Inference

We also provide the inference scripts to directly acquire the generation results on the test sets.

```shell
bash gen_prompt.sh
```

In the scripts, `--output_dir` denotes the directory of model checkpoint used for inference. The generated results are also saved in this directory.

During evaluation, `model_output_path` can be set to the generated file when running our inference codes. `source_path` can be set to `test.source` / `src-test.txt` in our pre-processed datasets. `reference_path` can be set to `test.target` / `tgt-test.txt` in our pre-processed datasets. Refer to the original repositories for more details.
