# VisionTransform Model
from typing import Optional, Tuple 
import torch 
import torch.nn as nn 


class SiglipVisionConfig:
    def __init__(
        self, 
        hidden_size=768, 
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6, 
        attention_dropout=0.0,
        num_image_tokens: int=None,
        **kwargs
    ):
        super().__init__() 
        
        self.hidden_size = hidden_size 
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens 
        self.layer_norm_eps = layer_norm_eps


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        embed_dim = config.hidden_size 
        
        self.embeddings = SiglipVisionEmbeddings(config) 
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [batch_size, num_channels, height, width] -> [batch_size, num_patches, embed_dim] 
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__() 
        self.config = config 
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self, pixel_values) -> Tuple:
        # [batch_size, num_channels, height, width] -> [batch_size, num_patches, embed_dim] 
        return self.vision_model(pixel_values) 
    



class SiglipVisionEmbeddings(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size 
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        # [batch_size, num_channels, height, width] -> [batch_size, embed_dim, num_patches_H, num_patches_W]
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size,
            padding="valid", # This indicates no padding is added
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2 
        self.num_positions = self.num_patches 
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)), 
            persistent=False,
        )
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [batch_size, num_channels, height, width]
        
        """
        Convolve the `patch_size` kernel over the image, with no overlapping since stride == kernel size. 
        The output shape will be [batch_size, embed_dim, num_patches_H, num_patches_W], where 
        num_patches_H = height // patch_size and num_patches_W = width // patch_size.
        """
        patch_embeds = self.patch_embedding(pixel_values)
        
        # [batch_size, embed_dim, num_patches_H, num_patches_W] -> [batch_size, num_patches, embed_dim]
        # where num_patches = num_patches_H * num_patches_W
        embeddings = patch_embeds.flatten(2)
        embeddings = embeddings.transpose(1, 2) 
        # Add position embeddings to each patch. Each positional encoding is a vector of size [embed_dim].
        embeddings += self.position_embedding(self.position_ids)
        # [batch_size, num_patches, embed_dim]
        return embeddings
        

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor: 
        hidden_states = inputs_embeds 
        
        for encoder_layer in self.layers: 
            hidden_states = encoder_layer(hidden_states)
        return hidden_states 

   
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size 
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states  
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states) 
        hidden_states += residual
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual

        return hidden_states
    

class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads 
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** (-0.5)
        self.dropout = config.attention_dropout 
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size() 
        
        # q, k, v: [batch_size, seq_len, embed_dim] ->-> [batch_size, num_heads, seq_len, head_dim]
        # where seq_len = num_patches = num_patches_H * num_patches_W
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # calculate attention using the formula Q * K^T / sqrt(d_k)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale 
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}"
                f"but is {attn_weights.size()}S"
            )
        
        # apply softmax row-wise, attn_weights: [batch_size, num_heads, num_patches, num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # multiply attention weights weights with value states. 
        attn_outputs = torch.matmul(attn_weights, value_states)
        
        if attn_outputs.size() != (batch_size, self.num_heads, seq_len, self.head_dim): 
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}"
                f"but is {attn_outputs.size()}"
            )
        
        # attn_outputs: -> [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_patches, embed_dim]
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()
        attn_outputs = attn_outputs.view(batch_size, seq_len, self.embed_dim)
        
        attn_outputs = self.out_proj(attn_outputs)

        return attn_outputs, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__() 
        self.config = config 
        
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: -> [batch_size, num_patchs, embed_dim] ->->->
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
