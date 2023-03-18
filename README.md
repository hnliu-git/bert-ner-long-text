
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
- [ ] Refine README
- [ ] Add demo/public data
- [ ] Add evaluation results
- [ ] Add functional tests

Welcome to add any MRs.
