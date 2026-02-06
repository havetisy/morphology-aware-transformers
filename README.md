# Morphology-Aware Transformers for Low-Resource NLP

This repository contains the official codebase accompanying my PhD thesis:

**Injecting Linguistic Structure into Transformer Architectures for Robust Low-Resource Natural Language Processing**

---

## Author

Hayastan Avetisyan  
PhD Candidate, Otto-von-Guericke University Magdeburg (OVGU), Faculty of Computer Science

---

## Overview

This repository hosts experimental pipelines and models for morphology-aware transformer architectures in low-resource, morphologically rich languages, with a primary focus on Armenian and Greek.  

The projects explore data-centric and architecture-level strategies for injecting explicit linguistic structure into pretrained transformer models, combining expert-annotated resources, synthetic datasets, and Universal Dependencies treebanks.

---

## Included Projects

### MorphCraft

Morphology-aware transformers for Armenian and Greek noun and verb bundle prediction under low-resource conditions.

MorphCraft provides a unified experimental framework comparing multiple morphology injection strategies—token-level tags, prompt-based conditioning, multitask feature heads, and a lightweight feature projector—on both Universal Dependencies treebanks and expert-validated synthetic datasets. The work shows that explicit morphological supervision consistently improves exact bundle accuracy and per-feature F1, especially for large feature bundles and long-tail labels, offering a scalable architecture for morphologically rich languages.

---

### VerbCraft

Morphology-aware Armenian verb generation using an mBART-50 backbone augmented with explicit morphological classifiers.

VerbCraft integrates auxiliary prediction heads for tense, aspect, mood, voice, person, and number, enabling linguistically grounded verb generation in low-resource settings. The project includes expert-validated synthetic Armenian verb datasets created with LLMs and human-in-the-loop correction, and demonstrates substantial gains in morphological accuracy on rare and irregular forms, prioritizing grammatical precision over surface fluency.

---

### Framing & BERTology

Data-centric issue vs. game framing classification in political news using BERT with explicit linguistic supervision.  

This project compares baseline BERT models against variants enriched with linguistically motivated phrases and domain-specific framing vocabulary on a human-annotated corpus of news paragraphs. In addition to input-level feature injection, the BERT tokenizer and embeddings are extended with framing expressions, showing that both linguistic phrase integration and vocabulary expansion improve framing detection accuracy.

---

### Automated Occupation Coding (German KldB)

Transformer-based automated occupation coding using hierarchical features.

Applies BERT and GPT-3 to German job titles and descriptions, incorporating the hierarchical structure of the KldB classification system to enable fine-grained multi-level occupation prediction and significantly outperform classical ML baselines.

---

### Issue & Game Framing

Computational detection of issue vs. game framing in political news using BERT and fine-grained linguistic features.

This project includes a human-annotated corpus of news paragraphs labeled for issue–game framing, enriched with multi-level linguistic annotations (syntactic, semantic, pragmatic), and experiments fine-tuning BERT for binary frame classification. The work provides both quantitative evaluation and qualitative analysis of linguistic cues such as competition metaphors, evaluative language, subjectivity markers, and discourse features.

---


## Related Previous Codebases

Earlier implementations associated with parts of this PhD research:

### AI-Generated vs Human-Generated Humor

Comparative modeling of AI-generated and human-generated humor using transformer-based classifiers and linguistic feature analysis.  
Includes human-annotated datasets (funniness and originality), RoBERTa fine-tuning, and source attribution experiments to study stylistic and structural differences between AI and human humor.

https://github.com/DZHW-AI4SS/Laughing-Out-Loud-Exploring-AI-Generated-and-Human-Generated-Humor

---

## Citation

If you use this code or datasets, please cite the corresponding papers:

```bibtex
@inproceedings{avetisyan2025morphcraft,
  title     = {MorphCraft: Morphology-Aware Transformers for Armenian and Greek},
  author    = {Avetisyan, Hayastan and Karasavva, Christina and Broneske, David},
  booktitle = {Proceedings of the 20th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP 2025)},
  pages     = {1--6},
  year      = {2025},
  publisher = {IEEE}
}

@inproceedings{avetisyan2025verbcraft,
  title     = {VerbCraft: Morphologically-Aware Armenian Text Generation Using LLMs in Low-Resource Settings},
  author    = {Avetisyan, Hayastan and Broneske, David},
  booktitle = {Proceedings of the Third Workshop on Resources and Representations for Under-Resourced Languages and Domains (RESOURCEFUL 2025)},
  pages     = {111},
  year      = {2025}
}

@inproceedings{avetisyan2023framing,
  title     = {Framing and BERTology: A Data-Centric Approach to Integration of Linguistic Features into Transformer-Based Pre-trained Language Models},
  author    = {Avetisyan, Hayastan and Safikhani, Parisa and Broneske, David},
  booktitle = {Intelligent Systems Conference},
  pages     = {81--90},
  year      = {2023},
  organization = {Springer}
}

@article{safikhani2023automated,
  title={Automated occupation coding with hierarchical features: A data-centric approach to classification with pre-trained language models},
  author={Safikhani, Parisa and Avetisyan, Hayastan and F{\"o}ste-Eggers, Dennis and Broneske, David},
  journal={Discover Artificial Intelligence},
  volume={3},
  number={1},
  pages={6},
  year={2023},
  publisher={Springer}
}

@inproceedings{avetisyan2021identifying,
  title     = {Identifying and understanding game-framing in online news: BERT and fine-grained linguistic features},
  author    = {Avetisyan, Hayastan and Broneske, David},
  booktitle = {Proceedings of the 4th International Conference on Natural Language and Speech Processing (ICNLSP 2021)},
  pages     = {95--107},
  year      = {2021}
}
