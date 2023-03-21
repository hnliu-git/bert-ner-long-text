
import torch

from torch import nn 
from torchcrf import CRF
from transformers import AutoModel


class BertNerBaseModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(
            config.model_name
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, config.num_labels)

        self.classifier.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

        if config.use_crf:
            self.crf = CRF(config.num_labels, batch_first=True) 

        if torch.cuda.is_available():
            self.cuda()

class BertNerModel(BertNerBaseModel):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_dict):
        """
        input_dict with keys {'input_ids', 'attention_mask', 'labels'}
        """
        if torch.cuda.is_available():
            input_dict = {k: v.cuda() for k, v in input_dict.items()}

        outputs = self.model(
            attention_mask=input_dict['attention_mask'],
            input_ids=input_dict['input_ids']
        )

        last_encoder_layer = outputs[0]
        logits = self.classifier(self.dropout(last_encoder_layer))

        if self.config.use_crf:
            # [CLS] is dropped
            emissions = logits[:, 1:, :]
            masks = input_dict['crf_mask'][:, 1:]
            # [SEP] is masked
            labels = (input_dict['labels'] * input_dict['crf_mask'])[:, 1:]
            loss = -self.crf(emissions, labels, masks)
            tags = [[0] + tag for tag in self.crf.decode(emissions)]
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), input_dict['labels'].view(-1))
            tags = torch.max(logits, 2)[1]
            if torch.cuda.is_available():
                tags = tags.cpu()
            tags = tags.numpy().tolist()

        return loss, tags


class BertNerChunkModel(BertNerBaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.embedding = self.model.get_input_embeddings()
    
    def forward(self, input_dict, ctx_embed=None):
        """
        input_dict with keys {'input_ids', 'attention_mask', 'labels'}
        each value is list of length 'chunk_num'
        """
        if torch.cuda.is_available():
            input_dict = {k: v.cuda() for k, v in input_dict.items()}

        if not self.config.use_ctx or ctx_embed is None:
            inputs_embeds = self.embedding(input_dict['input_ids'])
        else:
            # Detach so the graph is not calculated twice
            ctx_embed = ctx_embed.detach()
            inputs_embeds = torch.cat([
                ctx_embed,
                self.embedding(input_dict['input_ids'][:, 1:])
            ], dim=1)

        outputs = self.model(
            attention_mask=input_dict['attention_mask'],
            inputs_embeds=inputs_embeds
        )

        last_encoder_layer = outputs[0]
        logits = self.classifier(self.dropout(last_encoder_layer))

        if self.config.use_crf:
            # [CLS] is dropped
            emissions = logits[:, 1:, :]
            masks = input_dict['crf_mask'][:, 1:]
            # [SEP] is masked
            labels = (input_dict['labels'] * input_dict['crf_mask'])[:, 1:]
            loss = -self.crf(emissions, labels, masks)
            tags = [[0] + tag for tag in self.crf.decode(emissions)]
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), input_dict['labels'].view(-1))
            tags = torch.max(logits, 2)[1]
            if torch.cuda.is_available():
                tags = tags.cpu()
            tags = tags.numpy().tolist()
        
        # ctx is the [CLS] output
        ctx = outputs[0][:, 0:1, :]

        return loss, tags, ctx