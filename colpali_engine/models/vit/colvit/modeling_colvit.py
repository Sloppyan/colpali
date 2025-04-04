from transformers import ViTModel, BertModel, ViTConfig
from torch import nn
import torch

class ColViT(ViTModel):
    def __init__(self, config):
        super().__init__(config=config)
        
        # 加入 BERT 模块（不是父类一部分）
        self.text_model = BertModel.from_pretrained("google-bert/bert-base-uncased")

        # 自定义投影层
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.text_model.config.hidden_size, self.dim)
        self.custom_vision_proj = nn.Linear(config.hidden_size, self.dim)

        # 冻结 BERT
        for p in self.text_model.parameters():
            p.requires_grad = False

    def forward(self, *args, **kwargs):
        dtype = torch.float32
        proj = None

        if "pixel_values" in kwargs:
            outputs = super().forward(pixel_values=kwargs["pixel_values"], return_dict=True)
            image_embeds = outputs.last_hidden_state[:, 0, :]  # [CLS]
            proj = self.custom_vision_proj(image_embeds.to(dtype))

        elif "input_ids" in kwargs:
            text_out = self.text_model(
                input_ids=kwargs["input_ids"],
                attention_mask=kwargs.get("attention_mask", None),
                return_dict=True
            )
            text_embeds = text_out.last_hidden_state[:, 0, :]
            proj = self.custom_text_proj(text_embeds.to(dtype))

        if proj is None:
            raise ValueError("需要传入 pixel_values 或 input_ids")
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj
    
if __name__ == "__main__":
    pass