import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def mean_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed(encoder_model: PreTrainedModel, inputs) -> torch.Tensor:
    model_output = encoder_model(**inputs)
    embeddings = mean_pool(model_output.last_hidden_state, inputs["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)
