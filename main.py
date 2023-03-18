import evaluate

from transformers import AutoTokenizer

from utils import seed_everything

from model_factory.trainer import Trainer, ChunkTrainer
from model_factory.dataset import TKDataset, TKChunkDataset
from model_factory.model import BertTkModel, BertChunkTkModel
from model_factory.utils import tokenize_and_align_labels, tokenize_and_align_labels_and_chunk

import datasets

def compute_metrics(predictions, labels):
    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    return {
         "precision": results["overall_precision"],
         "recall": results["overall_recall"],
         "f1": results["overall_f1"],
         "accuracy": results["overall_accuracy"],
     }

class GlobalConfig:   
    def __init__(self):
        # general setting
        self.seed = 2022
        # model setting
        self.model_name = 'xlm-roberta-base'
        self.num_labels = len(vac_tags_dict)
        # data setting
        self.max_length = 512
        self.batch_size = 16
        # training setting
        self.epochs = 20
        self.steps_show = 100
        self.warmup_steps = 0.003
        self.lr = 2e-5
        self.saved_model_path = 'saved_models'
        # Use CRF layer or not
        self.use_crf = True
        # Use chunking or not
        self.use_chunk = False
        # If use_chunk, decide the overlapping range by stride
        self.stride = 0
        # If use_chunk, decide use the context embedding or not
        self.use_ctx = False 
        self.debug = True

config = GlobalConfig()
seed_everything(config.seed)

data_folder = 'data/'
exp_name = '%s-nl-epoch%d-crf'%(config.model_name, config.epochs)

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

seqeval = evaluate.load('seqeval')

if config.use_chunk:
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels_and_chunk(x, tokenizer, config.stride))
    train_loaders = TKChunkDataset(tokenizer, tokenized_dataset, 'train', config).build_dataloaders()
    val_loaders = TKChunkDataset(tokenizer, tokenized_dataset, 'validation', config).build_dataloaders()
    test_loader = TKChunkDataset(tokenizer, tokenized_dataset, 'test', config).build_dataloaders()

    model = BertChunkTkModel(config)

    trainer = ChunkTrainer(
        config,
        train_loaders,
        val_loaders,
        model,
        compute_metrics
    )
else:
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    train_loader = TKDataset(tokenizer, tokenized_dataset, 'train', config).build_dataloader()
    val_loader = TKDataset(tokenizer, tokenized_dataset, 'validation', config).build_dataloader()
    test_loader = TKDataset(tokenizer, tokenized_dataset, 'test', config).build_dataloader()

    model = BertTkModel(config)

    trainer = Trainer(
        config,
        train_loader,
        val_loader,
        model,
        compute_metrics
    )

if not config.debug:
    import wandb
    wandb.init(project='bert_vac_ner', name=exp_name)

trainer.train()

if not config.debug:
    wandb.close()
