---
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
library_name: generic
language:
- vi
widget:
- source_sentence: Làm thế nào Đại học Bách khoa Hà Nội thu hút sinh viên quốc tế?
  sentences:
  - >-
    Đại học Bách khoa Hà Nội đã phát triển các chương trình đào tạo bằng tiếng
    Anh để làm cho việc học tại đây dễ dàng hơn cho sinh viên quốc tế.
  - >-
    Môi trường học tập đa dạng và sự hỗ trợ đầy đủ cho sinh viên quốc tế tại Đại
    học Bách khoa Hà Nội giúp họ thích nghi nhanh chóng.
  - Hà Nội có khí hậu mát mẻ vào mùa thu.
  - Các món ăn ở Hà Nội rất ngon và đa dạng.
license: apache-2.0
---

# bkai-foundation-models/vietnamese-bi-encoder

This is a [sentence-transformers](https://www.SBERT.net) model: It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search.

We train the model on a merged training dataset that consists of: 
  - MS Macro (translated into Vietnamese)
  - SQuAD v2  (translated into Vietnamese)
  - 80% of the training set from the Legal Text Retrieval Zalo 2021 challenge

We use [phobert-base-v2](https://github.com/VinAIResearch/PhoBERT) as the pre-trained backbone.

Here are the results on the remaining 20% of the training set from the Legal Text Retrieval Zalo 2021 challenge:

|     Pretrained Model          |     Training Datasets                  |     Acc@1    |     Acc@10    |     Acc@100    |     Pre@10    |     MRR@10    |
|-------------------------------|---------------------------------------|:------------:|:-------------:|:--------------:|:-------------:|:-------------:|
|     [Vietnamese-SBERT](https://huggingface.co/keepitreal/vietnamese-sbert)     |     -                                 |     32.34    |      52.97    |      89.84     |      7.05     |      45.30    |
|     PhoBERT-base-v2           |     MSMACRO                           |     47.81    |      77.19    |      92.34     |      7.72     |      58.37    |
|     PhoBERT-base-v2                            |     MSMACRO + SQuADv2.0 + 80% Zalo    |     73.28    |      93.59    |      98.85     |      9.36     |      80.73    |


<!--- Describe your model here -->

## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer

# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
sentences = ["Cô ấy là một người vui_tính .", "Cô ấy cười nói suốt cả ngày ."]

model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
embeddings = model.encode(sentences)
print(embeddings)
```


## Usage (Widget HuggingFace)
The widget use custom pipeline on top of the default pipeline by adding additional word segmenter before PhobertTokenizer. So you do not need to segment words before using the API:

An example could be seen in Hosted inference API.
 

## Usage (HuggingFace Transformers)

Without [sentence-transformers](https://www.SBERT.net), you can use the model like this: First, you pass your input through the transformer model, then you have to apply the right pooling-operation on-top of the contextualized word embeddings.

```python
from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings, we could use pyvi, underthesea, RDRSegment to segment words
sentences = ['Cô ấy là một người vui_tính .', 'Cô ấy cười nói suốt cả ngày .']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)
```

## Training

The model was trained with the parameters:

**DataLoader**:

`torch.utils.data.dataloader.DataLoader` of length 17584 with parameters:

```
{'batch_size': 32, 'sampler': 'torch.utils.data.sampler.RandomSampler', 'batch_sampler': 'torch.utils.data.sampler.BatchSampler'}
```

**Loss**:

`sentence_transformers.losses.MultipleNegativesRankingLoss.MultipleNegativesRankingLoss` with parameters:

```
{'scale': 20.0, 'similarity_fct': 'cos_sim'}
```

Parameters of the fit()-Method:

```
{
    "epochs": 15,
    "evaluation_steps": 0,
    "evaluator": "NoneType",
    "max_grad_norm": 1,
    "optimizer_class": "<class 'torch.optim.adamw.AdamW'>",
    "optimizer_params": {
        "lr": 2e-05
    },
    "scheduler": "WarmupLinear",
    "steps_per_epoch": null,
    "warmup_steps": 1000,
    "weight_decay": 0.01
}
```

## Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: RobertaModel
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False})
)
```

### Please cite our manuscript if this dataset is used for your work
```
  @article{duc2024towards,
    title={Towards Comprehensive Vietnamese Retrieval-Augmented Generation and Large Language Models},
    author={Nguyen Quang Duc, Le Hai Son, Nguyen Duc Nhan, Nguyen Dich Nhat Minh, Le Thanh Huong, Dinh Viet Sang},
    journal={arXiv preprint arXiv:2403.01616},
    year={2024}
  }
```