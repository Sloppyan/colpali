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
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate similarity score between query and passage embeddings.
        For models like LLaVA that use bi-encoder approach, we use dot product.
        
        Args:
            q: Query embeddings [batch_size, dim]
            p: Passage embeddings [batch_size, dim]
        
        Returns:
            Similarity scores [batch_size]
        """
        # Simple dot product for bi-encoder scores
        return torch.sum(q * p, dim=-1)
    
    def score_multi_vector(
        self,
        qs: torch.Tensor,
        ps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate similarity scores between query embeddings and passage embeddings.
        
        Args:
            qs: Query embeddings [num_queries, dim]
            ps: Passage embeddings [num_passages, dim]
        
        Returns:
            Similarity scores [num_queries, num_passages]
        """
        # Compute dot product between all queries and passages
        return torch.matmul(qs, ps.t())
