---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: 'SECTION: hospital_course. CONTEXT: due to his weakness and severe deconditioning.
    Patient also received his home Coumadin medication for [ENTITY] paroxysmal atrial
    fibrillation [/ENTITY] with regular INR monitoring (goal 2.0-3.0). Infectious
    Diseases followed along during the entire admission and'
- text: 'SECTION: history_present_illness. CONTEXT: has to sit down for a few minutes
    after taking a shower because she feels [ENTITY] lightheaded [/ENTITY] . This
    improves in the afternoon, and she is able to run a few miles'
- text: 'SECTION: history_present_illness. CONTEXT: get a flu shot this year. . In
    the ED, he received [ENTITY] levofloxacin [/ENTITY] 750mg, and CXR showed streaky
    right basal opacity, which may represent atelectasis or pneumonia. Also,'
- text: 'SECTION: discharge_medications. CONTEXT: mg iron) 1 tablet(s) by mouth three
    times a day Disp #*90 Tablet Refills:*0 9. [ENTITY] LOPERamide [/ENTITY] 4 mg
    PO DAILY RX *loperamide [Anti-Diarrheal (loperamide)] 2 mg 2 pills by mouth daily'
- text: 'SECTION: discharge_medications. CONTEXT: 1. [ENTITY] Fentanyl [/ENTITY] 75
    mcg/hr Patch 72 hr Sig: One (1) Patch 72 hr Transdermal Q72H (every 72'
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: BAAI/bge-small-en-v1.5
---

# SetFit with BAAI/bge-small-en-v1.5

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 3 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label      | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|:-----------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| SYMPTOM    | <ul><li>'SECTION: chief_complaint. CONTEXT: [ENTITY] right hip pain [/ENTITY]'</li><li>'SECTION: chief_complaint. CONTEXT: [ENTITY] Chest pain [/ENTITY]'</li><li>'SECTION: history_present_illness. CONTEXT: ___ F with metastatic renal cell carcinoma with LIJ thrombus on prophylactic lovenox with [ENTITY] recurrent hematuria [/ENTITY]'</li></ul>                                                                                                                                                                                             |
| DIAGNOSIS  | <ul><li>'SECTION: chief_complaint. CONTEXT: [ENTITY] hypertensive emergency [/ENTITY]'</li><li>'SECTION: history_present_illness. CONTEXT: ___ F with [ENTITY] metastatic renal cell carcinoma [/ENTITY] with LIJ thrombus on prophylactic lovenox with recurrent hematuria'</li><li>'SECTION: history_present_illness. CONTEXT: ___ F on ASA 81mg and Plavix 75mg hx [ENTITY] CAD [/ENTITY] s/p drug eluting cardiac stents x3 in ___, DM, HTN, COPD who presents from OSH'</li></ul>                                                                |
| MEDICATION | <ul><li>'SECTION: history_present_illness. CONTEXT: ___ F with metastatic renal cell carcinoma with [ENTITY] LIJ thrombus [/ENTITY] on prophylactic lovenox with recurrent hematuria'</li><li>'SECTION: history_present_illness. CONTEXT: ___ F with metastatic renal cell carcinoma with LIJ thrombus on prophylactic [ENTITY] lovenox [/ENTITY] with recurrent hematuria'</li><li>'SECTION: history_present_illness. CONTEXT: ___ F on [ENTITY] ASA [/ENTITY] 81mg and Plavix 75mg hx CAD s/p drug eluting cardiac stents x3 in ___, DM,'</li></ul> |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("SECTION: discharge_medications. CONTEXT: 1. [ENTITY] Fentanyl [/ENTITY] 75 mcg/hr Patch 72 hr Sig: One (1) Patch 72 hr Transdermal Q72H (every 72")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 7   | 32.3210 | 42  |

| Label      | Training Sample Count |
|:-----------|:----------------------|
| DIAGNOSIS  | 261                   |
| MEDICATION | 158                   |
| SYMPTOM    | 176                   |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 10
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0013 | 1    | 0.2914        | -               |

### Framework Versions
- Python: 3.13.7
- SetFit: 1.1.3
- Sentence Transformers: 5.6.0
- Transformers: 4.57.6
- PyTorch: 2.13.0+cu130
- Datasets: 5.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->