import wandb
import torch
import evaluate

from transformers import get_linear_schedule_with_warmup
from torch import optim
from tqdm import tqdm

class TrainerBase:

    def __init__(
        self,
        config,
        train_loader,
        val_loader,
        model,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.seqeval = evaluate.load('seqeval')

    def get_crf_optimizer(self):
        model_params = list(self.model.model.named_parameters())
        no_decay = ['bias']
        optimizer_parameters = [
            {
                'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)],
                'weight_decay': 1e-3
            },
            {
                'params': [p for n, p in model_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.
            }
        ]
        optimizer_model = optim.AdamW(
            optimizer_parameters,
            lr=self.config.lr,
        )
        crf_params = list(self.model.crf.named_parameters())
        crf_optimizer_parameters = [
            {
                'params': [p for n, p in crf_params],
                'weight_decay': 1e-3
            }
        ] 
        optimizer_crf = optim.AdamW(
            crf_optimizer_parameters,
            lr=self.config.lr*100,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer_model,
            num_warmup_steps=int(self.config.warmup_steps * self.config.train_steps),
            num_training_steps=self.config.train_steps
        )
        scheduler_crf = get_linear_schedule_with_warmup(
            optimizer_crf,
            num_warmup_steps=int(self.config.warmup_steps * self.config.train_steps),
            num_training_steps=self.config.train_steps
        )
        
        return optimizer_model, optimizer_crf, scheduler, scheduler_crf 

    def get_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias']
        optimizer_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 1e-3
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.
            }
        ]
        optimizer = optim.AdamW(
            optimizer_parameters,
            lr=self.config.lr,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.config.warmup_steps * self.config.train_steps),
            num_training_steps=self.config.train_steps
        )

        return optimizer, scheduler

    def compute_metrics(self, predictions, labels):
        # Remove ignored index (special tokens)
        id2tag = list(self.config.tag2id.keys())

        true_predictions = [
            [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        performances = {}

        for key in results.keys():
            if type(results[key]) == dict:
                for metric in ['precision', 'recall', 'f1']:
                    performances['eval/%s/%s'%(key, metric)] = results[key][metric]
            else:
                performances['eval/%s'%(key)] = results[key]
        
        return performances


class Trainer(TrainerBase):

    def __init__(self, config, train_loader, val_loader, model):
        super().__init__(config, train_loader, val_loader, model)

    
    def train(self):
        self.config.train_steps = len(self.train_loader)*self.config.epochs
        self.model.train()

        if self.config.use_crf:
            optimizer, optimizer_crf, scheduler, scheduler_crf = self.get_crf_optimizer()
        else:
            optimizer, scheduler = self.get_optimizer()

        pbar = tqdm(total=self.config.train_steps)
        pbar.set_description('Training steps:')

        for _ in range(self.config.epochs):
            for batch in self.train_loader:
                
                loss, _ = self.model(batch)
                if not self.config.debug:
                    wandb.log({'train/loss': loss}, commit=False)

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                if self.config.use_crf: 
                    optimizer_crf.step()
                    scheduler_crf.step()
                    optimizer_crf.zero_grad()

                if pbar.n != 0 and pbar.n % self.config.steps_show == 0:
                    loss, performance = self.evaluation(self.val_loader)
                    if not self.config.debug:
                        wandb.log({'eval/loss': loss})
                        wandb.log({'eval/' + k: v for k, v in performance.items()})
                    
                pbar.update(n=1) 
                
    # if dev_f1 > best_f1:
    #     best_f1 = dev_f1
    #     torch.save(model, f'{config.saved_model_path}/test.pth')
    #     print('save best model   f1:%.6f'%best_f1) 

    def evaluation(self, val_loader):

        pbar = tqdm(total=len(val_loader))
        pbar.set_description('Validation step:')

        self.model.eval()

        y_trues =[]
        y_preds = []
        losses = []

        for batch in val_loader:
            with torch.no_grad():
                loss, y_pred = self.model(batch)
            if torch.cuda.is_available():
                labels = batch['labels'].cpu().numpy().tolist()
            else:
                labels = batch['labels'].numpy().tolist() 

            y_preds.extend(y_pred)
            y_trues.extend(labels)
            losses.append(loss)
            pbar.update(n=1)
            
        performance = self.compute_metrics(y_preds, y_trues)
        loss = sum(losses) / len(losses)

        self.model.train()
        return loss, performance
    
class ChunkTrainer(TrainerBase):

    def __init__(self, config, train_loader, val_loader, model):
        super().__init__(config, train_loader, val_loader, model)

    def train(self):
        self.config.train_steps = sum([len(loader)*num for num, loader in self.train_loader])*self.config.epochs
        self.model.train()

        if self.config.use_crf:
            optimizer, optimizer_crf, scheduler, scheduler_crf = self.get_crf_optimizer()
        else:
            optimizer, scheduler = self.get_optimizer()

        pbar = tqdm(total=self.config.train_steps)
        pbar.set_description('Training steps:')

        for _ in range(self.config.epochs):

            for num_chunk, loader in self.train_loader:
                for batches in loader:
                    for i in range(num_chunk):
                        if i == 0:
                            loss, _, ctx = self.model(batches[i])
                        else:
                            loss, _, ctx = self.model(batches[i], ctx)

                        if not self.config.debug:                   
                            wandb.log({'train/loss': loss}, commit=False)

                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        if self.config.use_crf:
                            optimizer_crf.step()
                            scheduler_crf.step()
                            optimizer_crf.zero_grad()

                        if pbar.n != 0 and pbar.n % self.config.steps_show == 0:
                            loss, performance = self.evaluation(self.val_loader)
                            if not self.config.debug:
                                wandb.log({'eval/loss': loss})
                                wandb.log({'eval/' + k: v for k, v in performance.items()})
                        pbar.update(n=1) 

    # if dev_f1 > best_f1:
    #     best_f1 = dev_f1
    #     torch.save(model, f'{config.saved_model_path}/test.pth')
    #     print('save best model   f1:%.6f'%best_f1) 

    def evaluation(self, val_loader):

        pbar = tqdm(total=sum([len(loader)*num for num, loader in val_loader]))
        pbar.set_description('Validation step:')

        self.model.eval()

        y_trues =[]
        y_preds = []
        losses = []

        for num_chunk, loader in val_loader:
            for batches in loader:
                for i in range(num_chunk):
                    if i == 0:
                        with torch.no_grad():
                            loss, y_pred, ctx = self.model(batches[i])
                    else:
                        with torch.no_grad():
                            loss, y_pred, ctx = self.model(batches[i], ctx)
                    
                    if torch.cuda.is_available():
                        labels = batches[i]['labels'].cpu().numpy().tolist()
                    else:
                        labels = batches[i]['labels'].numpy().tolist()

                    y_preds.extend(y_pred)
                    y_trues.extend(labels)
                    losses.append(loss)
                    pbar.update(n=1)
            
        performance = self.compute_metrics(y_preds, y_trues)
        loss = sum(losses) / len(losses)

        self.model.train()
        return loss, performance