
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from typing import Tuple, List, Union, Optional
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use the remapped device index
class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, prefix_size: int = 512, device=device):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = CustomGPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def get_dummy_token(self, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.interim_hidden_state = None
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        text_important_indices=None,
        text_mean_features_lowconfidence=None,
        text_inlp_axes=None,
        mode=False
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = transformer_outputs[0]

        # TODO: do we need this? !!
        # if self.model_parallel:
        #     torch.cuda.set_device(self.transformer.first_device)
        #     hidden_states = hidden_states.to(self.lm_head.weight.device)

        if mode=='sfid':
            hidden_dim_size = hidden_states.shape[1]
            text_mean_features_lowconfidence = text_mean_features_lowconfidence.unsqueeze(0).unsqueeze(1)
            text_mean_features_lowconfidence = text_mean_features_lowconfidence.expand(1, hidden_dim_size, 768)
            hidden_states[:, :, text_important_indices] = text_mean_features_lowconfidence[:, :, text_important_indices]
        elif mode=='spd':
            # SPD replaces the projection of hidden states onto a discriminative subspace (axes).
            # hidden_states: (B, S, D)
            # text_inlp_axes (U): (k, D), rows are orthonormal directions
            # text_mean_features_lowconfidence (mean_vec): (D,)
            if text_inlp_axes is None:
                raise ValueError("mode='spd' requires `text_inlp_axes` (INLP axes tensor).")
            if text_mean_features_lowconfidence is None:
                raise ValueError("mode='spd' requires `text_mean_features_lowconfidence` (low-confidence mean vector).")

            U = text_inlp_axes
            if not torch.is_tensor(U):
                U = torch.tensor(U)
            if U.dim() == 1:
                U = U.unsqueeze(0)

            mean_vec = text_mean_features_lowconfidence
            if not torch.is_tensor(mean_vec):
                mean_vec = torch.tensor(mean_vec)
            if mean_vec.dim() == 2 and mean_vec.shape[0] == 1:
                mean_vec = mean_vec.squeeze(0)

            U = U.to(device=hidden_states.device, dtype=hidden_states.dtype)
            mean_vec = mean_vec.to(device=hidden_states.device, dtype=hidden_states.dtype)

            # s_low: (k,)
            s_low = U @ mean_vec
            # S_proj: (B, S, k)
            S_proj = hidden_states @ U.transpose(0, 1)
            # delta: (B, S, k)
            delta = s_low.view(1, 1, -1) - S_proj
            # hidden_states_fixed: (B, S, D)
            hidden_states = hidden_states + delta @ U
        else:
            self.interim_hidden_state = hidden_states

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )