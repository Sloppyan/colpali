from transformers import LlavaProcessor
from PIL import Image
from typing import List, Dict, Any, Optional, Union, Tuple
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
import torch
from transformers import BatchFeature

class ColLlavaIProcessor(BaseVisualRetrieverProcessor, LlavaProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def process_queries(
            self, 
            queries: List[str], 
            max_length=64,
        ):
        return self.tokenizer(
            text=queries,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
    
    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature:
        # Convert images to RGB mode to handle grayscale images
        images = [image.convert("RGB") for image in images]
        
        image_features = self.image_processor(
            images, 
            return_tensors="pt",
        )
        return image_features
    
    def get_n_patches(
        self,
        patch_size: int,
    ) -> Tuple[int, int]:
        n_patches_x = self.image_processor.size["width"] // patch_size
        n_patches_y = self.image_processor.size["height"] // patch_size
        return n_patches_x, n_patches_y
    
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the score between query and passage embeddings.
        
        Args:
            qs: List of query embeddings
            ps: List of passage embeddings
            device: Device to run computation on
            
        Returns:
            Similarity scores
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)
    
    # def score_multi_vector(
    #     self,
    #     qs: torch.Tensor,
    #     ps: torch.Tensor,
    #     device: Optional[Union[str, torch.device]] = None,
    #     **kwargs,
    # ) -> torch.Tensor:
    #     """
    #     Calculate similarity scores between query embeddings and passage embeddings.
        
    #     Args:
    #         qs: Query embeddings tensor or list of tensors
    #         ps: Passage embeddings tensor or list of tensors
    #         device: Device to run computation on
            
    #     Returns:
    #         Similarity scores
    #     """
    #     # Convert lists to tensors if needed
    #     if isinstance(qs, list):
    #         qs = torch.stack(qs)
    #     if isinstance(ps, list):
    #         ps = torch.stack(ps)
            
    #     # Move to device if specified
    #     if device is not None:
    #         qs = qs.to(device)
    #         ps = ps.to(device)
            
    #     # Compute dot product between all queries and passages
    #     return torch.matmul(qs, ps.t())
