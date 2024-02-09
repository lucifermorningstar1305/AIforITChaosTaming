from typing import Any, List, Tuple, Dict, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class BertBasedLM(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()

        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.linear2 = nn.Linear(in_features=768, out_features=n_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:

        bert_out = self.bert_model(input_ids, attention_mask).pooler_output
        out = self.linear2(bert_out)

        return out
