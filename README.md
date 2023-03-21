
# BERT-NER for long documents
This repository implements an NER BERT model factory for long documents of arbitary length. The model can be any BERT-like models in the Huggingface Hub. 

## Introduction

Many BERT-like models have a limit of 512 input sequence length. This repository offers various End2End solutions for long document NER tasks, including the following features:
- Support most of the BERT-like models from HuggingFace 
- Add the CRF layer by setting `use_crf=true`
- Implementation of sliding window chunking for long documents
- Document-wise training: set `doc_wise=true`
    - The model is trained chunk by chunk from the same document. You can feed context embedding to the subsequent chunk to implement a LSTM-like BERT by setting `use_ctx=true`.
- Chunk-wise training: set `doc_wise=false`
    - Chunks are randomly shuffled, the model is trained by a batch of chunks

### Document-wise Training

The main idea behind document-wise training is that we put documents with the same number of chunk into the same batch. Thus, we can process these documents in parallel and chunk by chunk.

That also makes it possible to apply the usage of context embedding. Inspired by LSTM, the output of the [CLS] token from the previous chunk will be the [CLS] token embedding input to the next chunk. This method makes it possible for BERT to process long document without losing context information.

### Chunk-wise Training

The long documents are splited into chunks of length 512. All the chunks are shuffled and batched, the model is trained on randomly selected chunks. 

## Usage

First, prepare `train.jsonl`, `devel.jsonl`, and `test.jsonl` under `data` folder. Each file should contain a list of documents and each json should the same as the requirement of [datasets](https://huggingface.co/docs/datasets/index) library (Noted that the tags are already ids):

```
{"id": 0, "tokens": ["This", "is", "a" "sample"], "ner_tags": [0, 0, 0, 0]}
```

Then select a config json you would like to use in [cfgs](cfgs) folder, update the `tag2id` value

```
"tag2id": {"O": 0, "B-JT": 1}
```

Pass the config json to the `main.py`

```
python main.py <cfg>
```

## Evaluation results on my dataset

The dataset cannot release to the public, and it is hard to find annotated NER document dataset. So only evaluation results can be published, from them you can understand how different methods affect performnaces.

The experiments are only performed once so the difference might also due to randomness.

### CRF Layer

| Model             | Use CRF | F1 Score |
|-------------------|:-------:|:---------:|
| xlm-roberta-base  |   Yes   |   84.66   |
| xlm-roberta-base  |   No    |   83.76   |

### Document-wise training 

| Model             | Use CRF |  Mode  | Stride | Use Ctx | F1 Score  |
|-------------------|:-------:|:------:|:------:|:-------:|:---------:|
| xlm-roberta-base  |   Yes   |  None  |   /    |    /    |   84.66   |
| xlm-roberta-base  |   Yes   |  Doc   |   0    |    No   |   85.33   |
| xlm-roberta-base  |   Yes   |  Doc   |   32   |    No   |    TBA    |
| xlm-roberta-base  |   Yes   |  Doc   |   0    |   Yes   |   85.00   |
| xlm-roberta-base  |   Yes   |  Doc   |   32   |   Yes   |    TBA    |

### Chunk-wise training

| Model             | Use CRF |   Mode  | Stride | Use Ctx | F1 Score  |
|-------------------|:-------:|:-------:|:------:|:-------:|:---------:|
| xlm-roberta-base  |   Yes   |   None  |   /    |    /    |   84.66   |
| xlm-roberta-base  |   Yes   |   Doc   |   0    |    No   |   85.33   |
| xlm-roberta-base  |   Yes   |  Chunk  |   0    |    /    |   84.94   |
| xlm-roberta-base  |   Yes   |  Chunk  |   32   |    /    |    TBA    |


## Plan
- [ ] Add functional tests
- [ ] Smart chunking implementation
- [ ] Compare with Hi-Transformer and LongTransformer

All contributions and ideas are welcome. Free free to report any issue or suggest a pull request.
