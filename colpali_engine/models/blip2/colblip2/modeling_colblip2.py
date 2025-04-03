from transformers import Blip2Model, AutoTokenizer, Blip2Config
from torch import nn
from peft import get_peft_model
import torch

class ColBlip2(Blip2Model):
    def __init__(self, config):
        super().__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.config.text_config.hidden_size, self.dim)

    def forward(self, *args, **kwargs):
        dtype = torch.float16
        proj = None
        if "pixel_values" in kwargs:
            pixel_values = kwargs["pixel_values"]
            vision_outputs = self.vision_model(
                    pixel_values=pixel_values,
                    return_dict=True,
                )
            image_embeds = vision_outputs.last_hidden_state  # Shape: (batch_size, num_patches, vision_hidden_size)
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=dtype, device=image_embeds.device)
            
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            query_output = query_outputs.last_hidden_state  # Shape: (batch_size, num_query_tokens, qformer_hidden_size)

            # Step 3: Project Q-Former outputs to the text embedding dimension
            passage_embeddings = self.language_projection(query_output.to(dtype))  # Shape: (batch_size, num_query_tokens, text_hidden_size)
            passage_embeddings = passage_embeddings.mean(dim=1)
            hidden_states = passage_embeddings
            attention_mask = None
        else:
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                raise ValueError("input_ids is not in kwargs.")
            # print(input_ids.shape)
            text_outputs = self.language_model.get_input_embeddings()(input_ids)
            # print(text_outputs.shape)
            attention_mask = kwargs.get("attention_mask")
            attention_mask = attention_mask.unsqueeze(-1).to(dtype)
            masked_text_outputs = text_outputs * attention_mask
            sum_mask = attention_mask.sum(dim=1, keepdim=True)
            text_outputs = masked_text_outputs.sum(dim=1) / sum_mask.squeeze(1)
            hidden_states = text_outputs
        

        proj = self.custom_text_proj(hidden_states)
         # L2 正则化
        proj = proj / proj.norm(dim=-1, keepdim=True) 
        return proj