from transformers import GPT2LMHeadModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class MYGPT2LMHeadModel(GPT2LMHeadModel):

    def __init__(self, config):

        super().__init__(config)
        self.init_weights()

    def forward2(self, input_ids=None, token_type_ids=None, labels=None, content_id=None):

        outputs = self.forward(input_ids=input_ids, labels=labels)

        if labels is not None:
            if content_id is None or token_type_ids is None:
                raise Exception("当labels不为None时， content_id和token_type_ids不可以为None。")
            mask = (token_type_ids == content_id).long()
            labels = labels * mask
            loss, logits = outputs[:2]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            num = shift_labels.ne(0).long().sum().item()
            loss = loss / num
            outputs.loss = loss
        return outputs