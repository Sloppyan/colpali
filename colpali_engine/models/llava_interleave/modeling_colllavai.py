from transformers import LlavaForConditionalGeneration
from torch import nn
import torch

class ColLlavaI(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config=config)
        self.dim = 128
        # Add a custom projection layer for the text embeddings
        self.custom_text_proj = nn.Linear(self.config.text_config.hidden_size, self.dim)
        # Add an image projection layer to match dimensions
        self.image_proj = nn.Linear(self.config.text_config.hidden_size, self.dim)
    
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, return_dict=True, output_hidden_states=None, **kwargs):
        text_features = None
        image_features = None
        
        # Process images if provided
        if pixel_values is not None:
            # Process through vision encoder
            vision_outputs = self.vision_tower(
                pixel_values=pixel_values,
                return_dict=True,
            )
            image_embeds = vision_outputs.last_hidden_state
            
            # Process image embeddings through the projector
            image_proj_embeds = self.multi_modal_projector(image_embeds)
            
            # Mean pooling for image representation
            pooled_image_embeds = image_proj_embeds.mean(dim=1)
            
            # Project to common space and normalize
            image_features = self.image_proj(pooled_image_embeds)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Process text if provided
        if input_ids is not None:
            # Process through the language model to get meaningful text representations
            text_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Get the last hidden state from the language model
            text_embeds = text_outputs.hidden_states[-1]
            
            # Handle attention mask for proper pooling
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(-1).to(dtype=text_embeds.dtype)
                masked_embeddings = text_embeds * attention_mask
                sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
                pooled_text_embeds = torch.sum(masked_embeddings, dim=1) / sum_mask
            else:
                pooled_text_embeds = text_embeds.mean(dim=1)
            
            # Project to common space and normalize
            text_features = self.custom_text_proj(pooled_text_embeds)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Handle different cases
        if input_ids is not None and pixel_values is not None:
            # For training with both modalities - BiEncoderLoss expects two tensors
            return text_features, image_features
        elif input_ids is not None:
            # Text only
            return text_features
        elif pixel_values is not None:
            # Image only
            return image_features
        else:
            raise ValueError("Either pixel_values or input_ids must be provided")
