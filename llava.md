# Some notes on LLaVA

## Commands

- Install dependencies from pyproject.toml

  ```bash
  pip install -e .[train]
  ```

- Train the model

  On Linux:

  ```bash
  USE_LOCAL_DATASET=0 python scripts/train/train_colbert.py scripts/configs/llava_interleave/train_colllavai_model.yaml
  ```

  On Windows:

  ```powershell
  $env:USE_LOCAL_DATASET=0
  ```

  ```powershell
  python scripts/train/train_colbert.py scripts/configs/llava_interleave/train_colllavai_model.yaml
  ```

## LLaVA Model Structure

```python
ColModelTrainingConfig(
    model=PeftModelForFeatureExtraction(
        base_model=LoraModel(
            model=ColLlavaI(
                vision_tower=SiglipVisionModel(
                    vision_model=SiglipVisionTransformer(
                        embeddings=SiglipVisionEmbeddings(
                            patch_embedding=Conv2d(3, 1152, kernel_size=(14, 14), stride=(14, 14), padding='valid'),
                            position_embedding=Embedding(729, 1152)
                        ),
                        encoder=SiglipEncoder(
                            layers=ModuleList(
                                [SiglipEncoderLayer(
                                    self_attn=SiglipSdpaAttention(
                                        k_proj=Linear(1152, 1152),
                                        v_proj=Linear(1152, 1152),
                                        q_proj=Linear(1152, 1152),
                                        out_proj=Linear(1152, 1152)
                                    ),
                                    layer_norm1=LayerNorm((1152,), eps=1e-06),
                                    mlp=SiglipMLP(
                                        activation_fn=PytorchGELUTanh(),
                                        fc1=Linear(1152, 4304),
                                        fc2=Linear(4304, 1152)
                                    ),
                                    layer_norm2=LayerNorm((1152,), eps=1e-06)
                                ) for _ in range(26)]
                            )
                        ),
                        post_layernorm=LayerNorm((1152,), eps=1e-06)
                    )
                ),
                multi_modal_projector=LlavaMultiModalProjector(
                    linear_1=lora.Linear(
                        base_layer=Linear(1152, 1024),
                        lora_dropout=ModuleDict({'default': Dropout(p=0.1)}),
                        lora_A=ModuleDict({'default': Linear(1152, 32, bias=False)}),
                        lora_B=ModuleDict({'default': Linear(32, 1024, bias=False)}),
                        lora_embedding_A=ParameterDict(),
                        lora_embedding_B=ParameterDict(),
                        lora_magnitude_vector=ModuleDict()
                    ),
                    act=GELUActivation(),
                    linear_2=lora.Linear(
                        base_layer=Linear(1024, 1024),
                        lora_dropout=ModuleDict({'default': Dropout(p=0.1)}),
                        lora_A=ModuleDict({'default': Linear(1024, 32, bias=False)}),
                        lora_B=ModuleDict({'default': Linear(32, 1024, bias=False)}),
                        lora_embedding_A=ParameterDict(),
                        lora_embedding_B=ParameterDict(),
                        lora_magnitude_vector=ModuleDict()
                    )
                ),
                language_model=Qwen2ForCausalLM(
                    model=Qwen2Model(
                        embed_tokens=Embedding(152000, 1024),
                        layers=ModuleList(
                            [Qwen2DecoderLayer(
                                self_attn=Qwen2Attention(
                                    q_proj=Linear(1024, 1024),
                                    k_proj=Linear(1024, 1024),
                                    v_proj=Linear(1024, 1024),
                                    o_proj=Linear(1024, 1024, bias=False)
                                ),
                                mlp=Qwen2MLP(
                                    gate_proj=Linear(1024, 2816, bias=False),
                                    up_proj=Linear(1024, 2816, bias=False),
                                    down_proj=Linear(2816, 1024, bias=False),
                                    act_fn=SiLU()
                                ),
                                input_layernorm=Qwen2RMSNorm((1024,), eps=1e-06),
                                post_attention_layernorm=Qwen2RMSNorm((1024,), eps=1e-06)
                            ) for _ in range(24)]
                        ),
                        norm=Qwen2RMSNorm((1024,), eps=1e-06),
                        rotary_emb=Qwen2RotaryEmbedding()
                    ),
                    lm_head=Linear(1024, 152000, bias=False)
                ),
                custom_text_proj=lora.Linear(
                    base_layer=Linear(1024, 128),
                    lora_dropout=ModuleDict({'default': Dropout(p=0.1)}),
                    lora_A=ModuleDict({'default': Linear(1024, 32, bias=False)}),
                    lora_B=ModuleDict({'default': Linear(32, 128, bias=False)}),
                    lora_embedding_A=ParameterDict(),
                    lora_embedding_B=ParameterDict(),
                    lora_magnitude_vector=ModuleDict()
                ),
                image_proj=lora.Linear(
                    base_layer=Linear(1024, 128),
                    lora_dropout=ModuleDict({'default': Dropout(p=0.1)}),
                    lora_A=ModuleDict({'default': Linear(1024, 32, bias=False)}),
                    lora_B=ModuleDict({'default': Linear(32, 128, bias=False)}),
                    lora_embedding_A=ParameterDict(),
                    lora_embedding_B=ParameterDict(),
                    lora_magnitude_vector=ModuleDict()
                )
            )
        )
    )
)
```
