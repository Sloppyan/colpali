from transformers import Blip2Processor
from PIL import Image
from typing import List, Dict, Any
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
import torch
from typing import Optional, Union, Tuple
from transformers import BatchFeature

class ColBlip2Processor(BaseVisualRetrieverProcessor, Blip2Processor):
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
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)