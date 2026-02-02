# Morphology-Aware Transformers for Low-Resource NLP

This repository contains the official codebase accompanying my PhD thesis:

**Injecting Linguistic Structure into Transformer Architectures for Robust Low-Resource Natural Language Processing**

---

## Author

Hayastan Avetisyan  
PhD Candidate, Otto-von-Guericke University Magdeburg (OVGU), Faculty of Computer Science

---

## Overview

This repository hosts experimental pipelines and models for morphology-aware transformer architectures in low-resource, morphologically rich languages, with a focus on Armenian and Greek.


---

## Included Projects

### MorphCraft

Morphology-aware transformers for Armenian and Greek noun and verb bundle prediction under low-resource conditions.  
MorphCraft provides a unified experimental framework comparing multiple morphology injection strategies—token-level tags, prompt-based conditioning, multitask feature heads, and a lightweight feature projector—on both Universal Dependencies treebanks and expert-validated synthetic datasets. The work shows that explicit morphological supervision consistently improves exact bundle accuracy and per-feature F1, especially for large feature bundles and long-tail labels, offering a scalable architecture for morphologically rich languages.

### VerbCraft

Morphology-aware Armenian verb generation using an mBART-50 backbone augmented with explicit morphological classifiers.  
VerbCraft integrates auxiliary prediction heads for tense, aspect, mood, voice, person, and number, enabling linguistically grounded verb generation in low-resource settings. The project includes expert-validated synthetic Armenian verb datasets created with LLMs and human-in-the-loop correction, and demonstrates substantial gains in morphological accuracy on rare and irregular forms, prioritizing grammatical precision over surface fluency.

### Issue & Game Framing

Computational detection of issue vs. game framing in political news using BERT and fine-grained linguistic features.  
This project includes a human-annotated corpus of news paragraphs labeled for issue–game framing, enriched with multi-level linguistic annotations (syntactic, semantic, pragmatic), and experiments fine-tuning BERT for binary frame classification. The work provides both quantitative evaluation and qualitative analysis of linguistic cues, such as competition metaphors, evaluative language, subjectivity markers, and discourse features.


## Related Previous Codebases

Earlier implementations associated with parts of this PhD research:

### AI-Generated vs Human-Generated Humor
Comparative modeling of AI-generated and human-generated humor using transformer-based classifiers and linguistic feature analysis.
Includes human-annotated datasets (funniness and originality), RoBERTa fine-tuning, and source attribution experiments to study stylistic and structural differences between AI and human humor.

https://github.com/DZHW-AI4SS/Laughing-Out-Loud-Exploring-AI-Generated-and-Human-Generated-Humor

### Automated Occupation Coding (German KldB)

Transformer-based automated occupation coding using hierarchical features.
Applies BERT and GPT-3 to German job titles and descriptions, incorporating the hierarchical structure of the KldB classification system to enable fine-grained multi-level occupation prediction and significantly outperform classical ML baselines.

https://github.com/DZHW-AI/German_Occupation_Coding_KldB

## Citation

If you use this code or datasets, please cite the corresponding papers:



```bibtex
@inproceedings{avetisyan2025verbcraft,
  title     = {VerbCraft: Morphologically-Aware Armenian Text Generation Using LLMs in Low-Resource Settings},
  author    = {Avetisyan, Hayastan and Broneske, David},
  booktitle = {Proceedings of the Third Workshop on Resources and Representations for Under-Resourced Languages and Domains (RESOURCEFUL 2025)},
  pages     = {111},
  year      = {2025}
}


@inproceedings{avetisyan2025morphcraft,
  title     = {MorphCraft: Morphology-Aware Transformers for Armenian and Greek},
  author    = {Avetisyan, Hayastan and Karasavva, Christina and Broneske, David},
  booktitle = {Proceedings of the 20th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP 2025)},
  pages     = {1--6},
  year      = {2025},
  publisher = {IEEE}
}
```


