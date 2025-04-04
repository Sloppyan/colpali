from transformers import LlavaForConditionalGeneration, AutoTokenizer
from torch import nn
from peft import get_peft_model
import torch

class ColLlavaI(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config=config)
        self.dim = 128
        # Add a custom projection layer for the text embeddings
        self.custom_text_proj = nn.Linear(self.config.text_config.hidden_size, self.dim)
    
    def forward(self, *args, **kwargs):
        dtype = torch.float16
        
        if "pixel_values" in kwargs:
            # Process image input
            pixel_values = kwargs["pixel_values"]
            
            # Process through vision encoder
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )
            image_embeds = vision_outputs.last_hidden_state
            
            # Process image embeddings through the projector
            image_embeds = self.multi_modal_projector(image_embeds)
            
            # Get mean of image embeddings as the representation
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=dtype, device=image_embeds.device)
            
            # Use mean pooling for image representation
            attention_mask_expanded = image_attention_mask.unsqueeze(-1).expand_as(image_embeds)
            sum_embeddings = torch.sum(image_embeds * attention_mask_expanded, 1)
            sum_mask = attention_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            hidden_states = sum_embeddings / sum_mask
            
        else:
            # Process text input
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                raise ValueError("input_ids is not in kwargs.")
            
            # Get text embeddings
            text_outputs = self.language_model.model.embed_tokens(input_ids)
            
            # Apply attention mask if available
            attention_mask = kwargs.get("attention_mask")
            attention_mask = attention_mask.unsqueeze(-1).to(dtype)
            masked_text_outputs = text_outputs * attention_mask
            
            # Aggregate embeddings using attention mask
            sum_mask = attention_mask.sum(dim=1, keepdim=True)
            text_outputs = masked_text_outputs.sum(dim=1) / sum_mask.squeeze(1)
            hidden_states = text_outputs

        # Project embeddings to common space
        proj = self.custom_text_proj(hidden_states)
        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)
        
        return proj
