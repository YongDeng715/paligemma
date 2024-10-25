
import torch 
import torch.nn as nn 

class SiglipVisionConfig:
    
    def __init__(
        self, 
        hidden_size=768, 
        intermidiate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,     # 'RGB' 
        image_size=224,     # 224, 448, 896
        patch_size=16,      # size of each patch:16x16
        layer_norm_eps=1e-6, 
        attention_dropout=0.0,
        num_image_tokens: int=None,
        **kwargs
    ):
        super().__init__() 
        
        self.hidden_size = hidden_size, 
        self.intermidiate_size = intermidiate_size, 
        self.num_hidden_layers = num_hidden_layers,
        self.num_attention_heads = num_attention_heads,
        self.num_channels = num_channels, 
        self.patch_size = patch_size, 
        self.image_size = image_size, 
         