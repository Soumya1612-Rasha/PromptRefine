# PromptRefine: Enhancing Few-Shot Performance on Low-Resource Indic Languages with Example Selection from related Example Banks (NAACL 2025)

[![arXiv](https://img.shields.io/badge/arXiv-2412.05710-B31B1B.svg)](https://arxiv.org/pdf/2412.05710)

<h3>⭐ Accepted for NAACL 2025</h3>

## Abstract

Large Language Models (LLMs) have recently demonstrated impressive few-shot learning capabilities through in-context learning (ICL). However, ICL performance is highly dependent on the choice of few-shot demonstrations, making the selection of the most optimal examples a persistent research challenge. This issue is further amplified in low-resource Indic languages, where the scarcity of ground-truth data complicates the selection process. In this work, we propose PromptRefine, a novel Alternating Minimization approach for example selection that improves ICL performance on low-resource Indic languages. PromptRefine leverages auxiliary example banks from related high-resource Indic languages and employs multi-task learning techniques to align language-specific retrievers, enabling effective cross-language retrieval. Additionally, we incorporate diversity in the selected examples to enhance generalization and reduce bias. Through comprehensive evaluations on four text generation tasks -- Cross-Lingual Question Answering, Multilingual Question Answering, Machine Translation, and Cross-Lingual Summarization using state-of-the-art LLMs such as LLAMA-3.1-8B, LLAMA-2-7B, Qwen-2-7B, and Qwen-2.5-7B, we demonstrate that PromptRefine significantly outperforms existing frameworks for retrieving examples.


## Code 





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
