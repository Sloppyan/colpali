from transformers import ViTImageProcessor, AutoProcessor, Blip2Processor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image
import torch
from transformers import BatchFeature


from transformers import ViTImageProcessor
from transformers.image_processing_utils import ImageProcessingMixin

class DelegatedImageProcessor(ImageProcessingMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.internal_processor = ViTImageProcessor.from_pretrained("D:\\colpali\\colpali_engine\\models\\vit-base-patch16-224-in21k\\", **kwargs)

    def __call__(self, images, **kwargs):
        # 直接委托给 ViTImageProcessor 处理
        return self.internal_processor(images, **kwargs)

    def to_json_string(self):
        """自定义 JSON 序列化（可选）"""
        return self.internal_processor.to_json_string()

    def from_json_string(self, json_str):
        """自定义 JSON 反序列化（可选）"""
        return self.internal_processor.from_json_string(json_str)


class ColViTProcessor(BaseVisualRetrieverProcessor):
    attributes = ["image_processor"]  # 强制它使用 image_processor 而不是 feature_extractor
    image_processor_class = 'ViTImageProcessor'
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)  

    def process_queries(self, 
                        queries: List[str], 
                        max_length=64) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            text=queries,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
    
    def process_images(self, 
                       images: List[Image.Image]) -> BatchFeature:
        image_features = self.image_processor(
            images, 
            return_tensors="pt",
        )
        return image_features
    
    def get_n_patches(self, 
                      patch_size: int) -> Tuple[int, int]:
        width, height = self.image_processor.size["width"], self.image_processor.size["height"]
        n_patches_x = width // patch_size
        n_patches_y = height // patch_size
        return n_patches_x, n_patches_y
    
    def score(self, 
              qs: List[torch.Tensor], 
              ps: List[torch.Tensor], 
              device: Optional[Union[str, torch.device]] = None, 
              **kwargs) -> torch.Tensor:
        return self.score_multi_vector(qs, ps, device=device, **kwargs)
    
    def score_multi_vector(self, 
                           qs: List[torch.Tensor], 
                           ps: List[torch.Tensor], 
                           device: Optional[Union[str, torch.device]] = None, 
                           **kwargs) -> torch.Tensor:
        # 这里假设通过余弦相似度来计算匹配分数
        # 您可以根据需要自定义 score 函数
        
        # L2 归一化
        qs = [q / q.norm(dim=-1, keepdim=True) for q in qs]
        ps = [p / p.norm(dim=-1, keepdim=True) for p in ps]
        
        # 计算余弦相似度
        cosine_scores = torch.matmul(qs, ps.T)  # 假设 qs 和 ps 已经是归一化的
        return cosine_scores


class ColViTProcessorshuang(BaseVisualRetrieverProcessor, ViTImageProcessor):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)  

    def process_queries(self, 
                        queries: List[str], 
                        max_length=64) -> Dict[str, torch.Tensor]:
        """
        处理文本查询（例如问题或文本）并返回 BERT tokenized 格式

        参数:
        - queries: 查询文本列表
        - max_length: 文本最大长度（默认为 64）

        返回:
        - tokenized 查询的字典，包含 input_ids、attention_mask 等
        """
        return self.tokenizer(
            text=queries,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
    
    def process_images(self, 
                       images: List[Image.Image]) -> BatchFeature:
        """
        处理图像输入并返回适合 ViT 的图像特征

        参数:
        - images: 图像列表，PIL 图像格式
        
        返回:
        - BatchFeature: 包含图像特征的字典
        """
        image_features = self.image_processor(
            images, 
            return_tensors="pt",
        )
        return image_features
    
    def get_n_patches(self, 
                      patch_size: int) -> Tuple[int, int]:
        """
        计算图像被分成的 Patch 数量（ViT 中的 Patch 数量）
        
        参数:
        - patch_size: 图像的 Patch 大小
        
        返回:
        - n_patches_x: 水平方向上的 Patch 数量
        - n_patches_y: 垂直方向上的 Patch 数量
        """
        width, height = self.image_processor.size["width"], self.image_processor.size["height"]
        n_patches_x = width // patch_size
        n_patches_y = height // patch_size
        return n_patches_x, n_patches_y
    
    def score(self, 
              qs: List[torch.Tensor], 
              ps: List[torch.Tensor], 
              device: Optional[Union[str, torch.device]] = None, 
              **kwargs) -> torch.Tensor:
        """
        计算文本查询和图像通过 BERT + ViT 模型的匹配分数
        
        参数:
        - qs: 文本查询的 Embedding 列表
        - ps: 相关图像的 Embedding 列表
        - device: 使用的设备（可选）
        
        返回:
        - score: 文本和图像匹配的分数
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)
    
    def score_multi_vector(self, 
                           qs: List[torch.Tensor], 
                           ps: List[torch.Tensor], 
                           device: Optional[Union[str, torch.device]] = None, 
                           **kwargs) -> torch.Tensor:
        """
        基于多向量计算匹配分数
        
        参数:
        - qs: 查询的多个向量
        - ps: 相关图像的多个向量
        - device: 使用的设备（可选）
        
        返回:
        - score: 匹配分数
        """
        # 这里假设通过余弦相似度来计算匹配分数
        # 您可以根据需要自定义 score 函数
        
        # L2 归一化
        qs = [q / q.norm(dim=-1, keepdim=True) for q in qs]
        ps = [p / p.norm(dim=-1, keepdim=True) for p in ps]
        
        # 计算余弦相似度
        cosine_scores = torch.matmul(qs, ps.T)  # 假设 qs 和 ps 已经是归一化的
        return cosine_scores
