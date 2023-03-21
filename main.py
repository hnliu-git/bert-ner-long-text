import sys
import json

from transformers import AutoTokenizer

from utils import seed_everything
from model_factory.trainer import Trainer, ChunkTrainer
from model_factory.dataset import NerDataset, NerChunkDataset, NerChunkIterDataset
from model_factory.model import BertNerModel, BertNerChunkModel
from model_factory.utils import tokenize_and_align_labels, tokenize_and_align_labels_and_chunk

import datasets

class GlobalConfig:

    def __init__(self, **cfgs) -> None:
        self.__dict__ = json.load(open(sys.argv[1]))
        self.num_labels = len(self.tag2id)

if len(sys.argv) < 2:
    print("Usage python main.py <cfg-path>")
    sys.exit(1)

config = GlobalConfig()
seed_everything(config.seed)

exp_name = '%s-nl-epoch%d%s'%(config.model_name, config.epochs, config.exp_suffix)

data_folder = 'data/'

data_files = {
    'train': data_folder + 'train.jsonl',
    'validation': data_folder + 'devel.jsonl',
    'test': data_folder + 'test.jsonl'
}

dataset = datasets.load_dataset('json', data_files=data_files)

print(f'train: {len(dataset["train"])}')
print(f'eval: {len(dataset["validation"])}')
print(f'test: {len(dataset["test"])}')

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

if config.use_chunk and config.doc_wise:
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels_and_chunk(x, tokenizer, config.stride))
    train_loaders = NerChunkDataset(tokenizer, tokenized_dataset, 'train', config).build_dataloaders()
    val_loaders = NerChunkDataset(tokenizer, tokenized_dataset, 'validation', config).build_dataloaders()
    test_loaders = NerChunkDataset(tokenizer, tokenized_dataset, 'test', config).build_dataloaders()

    model = BertNerChunkModel(config)

    trainer = ChunkTrainer(
        config,
        train_loaders,
        val_loaders,
        model,
    )
elif config.use_chunk and not config.doc_wise:
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels_and_chunk(x, tokenizer, config.stride))
    train_loader = NerChunkIterDataset(tokenizer, tokenized_dataset, 'train', config).build_dataloader()
    val_loader = NerChunkIterDataset(tokenizer, tokenized_dataset, 'validation', config).build_dataloader()
    test_loader = NerChunkIterDataset(tokenizer, tokenized_dataset, 'test', config).build_dataloader()

    model = BertNerModel(config)

    trainer = Trainer(
        config,
        train_loader,
        val_loader,
        model,
    ) 
else:
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    train_loader = NerDataset(tokenizer, tokenized_dataset, 'train', config).build_dataloader()
    val_loader = NerDataset(tokenizer, tokenized_dataset, 'validation', config).build_dataloader()
    test_loader = NerDataset(tokenizer, tokenized_dataset, 'test', config).build_dataloader()

    model = BertNerModel(config)

    trainer = Trainer(
        config,
        train_loader,
        val_loader,
        model,
    )

if not config.debug:
    import wandb
    wandb.init(project='bert_ner', name=exp_name)

trainer.train()