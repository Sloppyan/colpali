import torch
from torch import nn
from transformers import ViTModel, BertModel, PreTrainedModel

class ColViT(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 ViT 和 BERT
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.dim = 128  # 目标嵌入维度
        self.custom_text_proj = nn.Linear(self.bert.config.hidden_size, self.dim)
        self.custom_vision_proj = nn.Linear(self.vit.config.hidden_size, self.dim)

    def forward(self, *args, **kwargs):
        dtype = torch.float32  # 适用于 ViT 和 BERT
        proj = None

        if "pixel_values" in kwargs:
            # 处理图像输入
            vision_outputs = self.vit(pixel_values=kwargs["pixel_values"], return_dict=True)
            image_embeds = vision_outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 位置的特征
            proj = self.custom_vision_proj(image_embeds.to(dtype))
        
        elif "input_ids" in kwargs:
            # 处理文本输入
            text_outputs = self.bert(input_ids=kwargs["input_ids"], attention_mask=kwargs.get("attention_mask", None), return_dict=True)
            text_embeds = text_outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 位置的特征
            proj = self.custom_text_proj(text_embeds.to(dtype))

        if proj is None:
            raise ValueError("输入必须包含 'pixel_values' 或 'input_ids' 之一")

        # L2 归一化
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj
    
