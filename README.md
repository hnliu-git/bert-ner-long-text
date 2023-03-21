
# BERT-NER for long documents
This repository implements an NER BERT model factory for long documents of arbitary length. The model can be any BERT-like models in the Huggingface Hub. 

## Main Idea

It is known that many BERT-like models have the 512 input sequence length limit. To overcome that issue, a way to go is to split long documents into chunks and train them chunk by chunk.

## How to run it

Prepare your data in data folder then,
```
python main.py
```

## Plan
- [ ] Train with main entity and use smaller models and report own evaluation module metrics
- [ ] Add evaluation results
- [ ] Refine README
https://github.com/mim-solutions/bert_for_longer_texts
- [ ] Add functional tests
- [ ] Smart chunking implementation
- [ ] Compare with Hi-Transformer and LongTransformer
- [ ] Publish your results
https://github.com/google-research/bert/issues/27

Welcome to add any MRs.
