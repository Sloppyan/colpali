# Environment Setup from pyproject.toml
pip install -e .[train]

# Training
USE_LOCAL_DATASET=0 python scripts/train/train_colbert.py scripts/configs/llava_interleave/train_colllavai_model.yaml