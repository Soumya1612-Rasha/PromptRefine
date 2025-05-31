# PromptRefine: Enhancing Few-Shot Performance on Low-Resource Indic Languages with Example Selection from related Example Banks (NAACL 2025)

[![arXiv](https://img.shields.io/badge/arXiv-2412.05710-B31B1B.svg)](https://arxiv.org/pdf/2412.05710) <!-- GitHub Stars Badge (replace YourUser and YourRepo) --> 
[![GitHub Stars](https://img.shields.io/github/stars/Soumya1612-Rasha/PromptRefine.svg?style=social)](https://github.com/Soumya1612-Rasha/PromptRefine/stargazers)


<h3>⭐ Accepted for NAACL 2025</h3>

This codebase provides a Pytorch implementation for the paper: PromptRefine: Enhancing Few-Shot Performance on Low-Resource Indic Languages with Example Selection from Related Example Banks.

## Abstract

Large Language Models (LLMs) have recently demonstrated impressive few-shot learning capabilities through in-context learning (ICL). However, ICL performance is highly dependent on the choice of few-shot demonstrations, making the selection of the most optimal examples a persistent research challenge. This issue is further amplified in low-resource Indic languages, where the scarcity of ground-truth data complicates the selection process. In this work, we propose PromptRefine, a novel Alternating Minimization approach for example selection that improves ICL performance on low-resource Indic languages. PromptRefine leverages auxiliary example banks from related high-resource Indic languages and employs multi-task learning techniques to align language-specific retrievers, enabling effective cross-language retrieval. Additionally, we incorporate diversity in the selected examples to enhance generalization and reduce bias. Through comprehensive evaluations on four text generation tasks -- Cross-Lingual Question Answering, Multilingual Question Answering, Machine Translation, and Cross-Lingual Summarization using state-of-the-art LLMs such as LLAMA-3.1-8B, LLAMA-2-7B, Qwen-2-7B, and Qwen-2.5-7B, we demonstrate that PromptRefine significantly outperforms existing frameworks for retrieving examples.

## 🚦 Getting Started

**Setup in Minutes!**  
We recommend using Python 3.8+ and PyTorch 2.0+. Clone this repo and install dependencies:

```
bash
git clone https://github.com/Soumya1612-Rasha/PromptRefine.git
cd PromptRefine
pip install -r requirements.txt
```

## PromptRefine

For illustration, we have set the low-resource target language as Manipuri (`lang_name`) and the related high-resource auxiliary language as Bengali (`high_lang_name`). These can be easily configured by changing the `lang_name` and `high_lang_name` variables.

# 1. Iterative Prompt Refinement: 

To fine-tune the retriever embeddings iteratively using "Alternating Minimization" strategy(Section 4.1.2 in main paper), run:

```
bash iter_algo.sh
```

- `task_name`: The specific downstream task (e.g., `xquad`, `cross_sum`, `flores`, `xnli`).
- `model_name`: The scorer LLM.
- `lang_name`: Target low-resource Indic language.
- `high_lang_name`: List of high-resource auxiliary language.

After fine-tuning, `bash iter_algo_inf.sh` script runs inference on the validation set using the merged model from each iteration. To change the metric of inference, please refer to `src/inferencer.py`. Note down the iteration number that achieves the highest validation performance.

# 2. Diversity Induced Fine-tuning: 

Now, use the best merged model (based on validation performance) for training with Determinantal Point Processes (DPP) to encourage diversity in example selection:

```
bash run_dpp_ours.sh
```

Inside run_dpp_ours.sh, set the `epoch` variable to the iteration number that yielded the highest validation accuracy in previous step.


## 🛠️ Customization

PromptRefine can be easily customized to support new tasks, datasets, models, and hyperparameters:

### To add a new task or dataset:

1. Dataset Script:
Define your dataset script under src/hf_datasets if it’s not directly available via Huggingface Datasets.

2. Dataset Wrapper:
Create a file under src/dataset_readers/dataset_wrapper/ and implement functions to return the input (e.g., source sentence), output (e.g., target sentence), and prompt formatting for in-context examples.

3. Metrics:
Define the evaluation metric under src/metrics (e.g., BLEU for translation).

### To use a different model for scoring or generation:

Change the `model_name` variable in both `iter_algo.sh` and `iter_algo_inf.sh` scripts. Ensure `model_name` exactly matches its Hugging Face Hub identifier (e.g., meta-llama/Llama-2-7b-hf).

### To change hyperparameters:

Primary training hyperparameter configuration is in `configs/retriever_trainer.yaml`. Ensure that hyperparameters in the YAML file are not overridden by command-line arguments in the shell scripts.


## References

Our codebase has been adopted from [CEIL](https://github.com/HKUNLP/icl-ceil). We extend our sincere gratitude to the authors.


## For bibtex citation

```
@inproceedings{ghosal-etal-2025-promptrefine,
    title = "{P}rompt{R}efine: Enhancing Few-Shot Performance on Low-Resource {I}ndic Languages with Example Selection from related Example Banks",
    author = "Ghosal, Soumya Suvra  and
      Pal, Soumyabrata  and
      Mukherjee, Koyel  and
      Manocha, Dinesh",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    year = "2025",
}


