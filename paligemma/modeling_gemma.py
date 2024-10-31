# Language Model, decode the answer from the input image tensor and text tokens
from typing import Optional, Tuple, List 
import torch
import torch.nn as nn 
import math

from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


# configuration of the decoder Gemma (2B Language Model)
class GemmaConfig(): 

    def __init__(
        self, 
        vocab_size,     
        hidden_size,     
        intermediate_size, 
        num_hidden_layers, 
        num_attention_heads,
        num_key_value_heads, 
        head_dim=256, 
        max_position_embeddings=8192, 
        rms_norm_eps=1e-6, 
        rope_theta=10000.0, 
        attention_bias=False, 
        attention_dropout=0.0, 
        pad_token_id=None, 
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


# All of the combination of vision encoder(ViT), text encoder(LM) and decoder(Gamma LM)
class PaliGemmaConfig():

    def __init__(
        self, 
        vision_config=None, 
        text_config=None, 
        ignore_index=-100, 
        image_token_index=256000, 
        vocab_size=257152,
        projection_dim=2048, 
        hidden_size=2048,
        pad_token_id=None, 
        **kwargs, 
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size            #      
        self.projection_dim = projection_dim    #
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        
        self.vocab_size = self.text_config.vocab_size 
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = self.projection_dim 


class KVCache():
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = [] 
        self.value_cache: List[torch.Tensor] = []
        
    def num_items(self) -> int: 
        if len(self.key_cache) == 0: 
            return 0
        else: 
            # Shape of key_cache: [batch_size, num_heads_KV, seq_len, head_dim]
            return self.key_cache[0].shape[-2]
    
    def update(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        if len(self.key_cache) <= layer_idx: 
            # If we never add anything to the KV-cache of this layer, we create it.
            self.key_cache.append(key_states) 
            self.value_cache.append(value_states)
        else:
            # ... otherwise, we concatenate the new keys with the existing ones.
            # each tensor's shape: [batch_size, num_heads_KV, seq_len, head_dim] 
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        # ... then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]



     
class PaliGemmaForConditionalGeneration(nn.Module): 
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config 
        
        self.vision_model = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config) # Linear Layer
        self.vocab_size = config.vocab_size
        
        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model 
        
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1 
    
    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self, 
        image_features: torch.Tensor, 
        inputs_embeds: torch.Tensor,    # embeddings of image palceholder we don't use.
        input_ids: torch.Tensor,  
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache]=None,
    ):
        _, _, embed_dim = image_features.shape 
        batch_size, seq_length = input_ids.shape 
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # Image feature shape: [batch_size, num_patches(seq_length), hidden_size]
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)
        # Combine the embeddings of image tokens, text tokens and mask out all padding tokens 
        final_embedding = torch.zeros(batch_size, seq_length, embed_dim, dytpe=inputs_embeds.dtype, device=inputs_embeds.device)
        
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = (input_ids == self.config.image_token_index)
        pad_mask = (input_ids == self.pad_token_id)  
        
        # We need to expand the masks to embedding dimension otherwise we cannot use them in torch.
        """
        text_mask_expanded: [batch_size, seq_length, embed_dim] = shape of final_embedding
        image_mask_expanded: [batch_size, num_patches, embed_dim]
        pad_mask_expanded: [batch_size, seq_length, embed_dim] = shape of final_embedding
        """ 
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim) 
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim) 
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim) 

        # Add text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding) 
        # Insert image embeddings. We cannot use `torch.where` because the sequence length of 
        # scaled_image_features is no equal to the sequence length of final_embedding.
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens 
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding) 
        
        #### CREATE ATTENTION MASK ####
        dtype, device = attention_mask.dtype, attention_mask.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
        
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase.
            # This only works if we have no padding.
            casual_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must have 1 single token.
            assert q_len == 1 
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't mask any token only when we have no padding.
            # This is becasue each query should attend to all previous tokens.
            casual_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )
        
        # Add the head dimension 
        # casual_mask: [batch_size, q_len, kv_len] -> [batch_size, num_heads_Q, q_len, kv_len] 
        casual_mask = casual_mask.unsqueeze(1)
        
        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of query is just the last position 
            position_ids = attention_mask.cumsum(-1)[:, -1] 
            if position_ids.dim() == 1: 
                position_ids = position_ids.unsqueeze(0)
        else: 
            # Create a position_ids tensor based on the size of the attention mask 
            # For masked tokens, use the number 1 as position 
            position_ids = (attention_mask.cumsum(-1)).masked_fill_(\
                (attention_mask == 0), 1).to(device)
        
        return final_embedding, casual_mask, position_ids 

    def forward(
        self, 
        input_ids: torch.LongTensor=None, 
        pixel_values: torch.FloatTensor=None, 
        attention_mask: Optional[torch.Tensor]=None, 
        kv_cache: Optional[KVCache]=None, 
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded."
        
        # 1. Extra the input embeddings
        # shape: (batch_size, seq_len, hidden_size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids) 
        
        # 2. Merge text and images
        # [batch_size, num_channels, height, width] -> [batch_size, num_patches, embed_dim]
        selected_image_feature = self.vision_model(pixel_values.to(inputs_embeds.dtype))
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, hidden_size(projector_dim)]
        image_features = self.multi_modal_projector(selected_image_feature) 
        
        # Merge the embeddings of text tokens and image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features,
            inputs_embeds,
            input_ids, 
            attention_mask,
            kv_cache
        )
        
        outputs = self.language_model(
            attention_mask=attention_mask, 
            position_ids=position_ids,
            inputs_embeds=inputs_embeds, 
            kv_cache=kv_cache, 
        )
        return outputs
    

# 
class PaliGemmaMultiModalProjector(nn.Module): 
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size, 
            config.vision_config.projection_dim, 
            bias = True           
        )
    
    def forward(self, image_features):
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, projection_dim] 
        hidden_states = self.linear(image_features)
        return hidden_states


    
#### Gemma Language model #####
class GemmaForCausalLM(nn.Module):
    
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size 
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight 
    
    def forward(
        self, 
        attention_mask: Optional[torch.Tensor]=None, 
        position_ids: Optional[torch.LongTensor]=None, 
        inputs_embeds: Optional[torch.FloatTensor]=None, 
        kv_cache: Optional[KVCache]=None
    ) -> Tuple: 
        """
        inputs_embeds: [batch_size, seq_len, hidden_size]
        outputs: [batch_size, seq_len, hidden_size] 
        """
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        
        # Finally, after all decoder layers and final RMSNorm layer, apply linear to get logits.
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        return_data = {"logits": logits, }
        
        if kv_cache is not None: 
            return_data["kv_cache"] = kv_cache
        
        return return_data
    

class GemmaModel(nn.Module): 
    
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size 
        
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            self.padding_idx
        )
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) 
        
    def get_input_embeddings(self): 
        return self.embed_tokens 
    
    def forward(
        self, 
        attention_mask: Optional[torch.Tensor]=None, 
        position_ids: Optional[torch.LongTensor]=None, 
        inputs_embeds: Optional[torch.FloatTensor]=None, 
        kv_cache: Optional[KVCache]=None
    ) -> torch.FloatTensor: 
        # [batch_size, seq_len, hidden_size]
        hidden_states = inputs_embeds 
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer 
        
        for decoder_layer in self.layers: 
            hidden_states = decoder_layer(
                hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                kv_cache=kv_cache
            )
            
        hidden_states = self.norm(hidden_states) 
        return hidden_states
    

# Root mean normalization applyed in PaliGemma 
class GemmaRMSNorm(nn.Module): 
    def __init__(self, dim: int, eps: float=1e-6): 
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps 
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())  
        output = output * (1 + self.weight.float()) 
        return output.type_as(x)




#### Gemma Decoder Layer ####

class GemmaDecoderLayer(nn.Module): 
    def __init__(self, config: GemmaConfig, layer_idx: int): 
        super().__init__()
        self.config = config 
        self.padding_idx = config.pad_token_id 
        
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None, 
        position_ids: Optional[torch.LongTensor]=None, 
        kv_cache: Optional[KVCache]=None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,  
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            kv_cache=kv_cache
        )
        hidden_states += residual 
        
        residual = hidden_states 
        hidden_states = self.post_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) 
        hidden_states += residual 

        return hidden_states


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig): 
        super().__init__()
        self.config = config 
        
        self.hidden_size = config.hidden_size 
        self.intermediate_size = config.intermediate_size 
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x):
        """# Equivalent to: 
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, intermediate_size) 
        y = torch.gelu(self.gate_proj(x), approximate="tanh")
        z = y * self.up_proj(x)
        # (batch_size, seq_len, intermediate_size) -> (batch_size, seq_len, hidden_size)
        return self.down_proj(z)
        """
        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )
        

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int]=None): 
        super().__init__()
        self.config = config 
        self.layer_idx = layer_idx
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size 
        self.num_heads = config.num_attention_heads # which is the head number of query
        self.head_dim = config.head_dim  
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads 
        self.max_position_embeddings = config.max_position_embeddings 
        self.rope_theta = config.rope_theta
        self.is_causal = True 
        
        assert self.hidden_size % self.num_heads == 0 
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias) 
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias) 
        
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim, 
            max_position_embeddings=self.max_position_embeddings, 
            base=self.rope_theta
        )
       
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        kv_cache: Optional[KVCache]=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()  # [batch_size, seq_len, hidden_size] 
        
        # [bsz, q_len, hidden_size] ->-> [bsz, num_heads_Q=256, q_len, head_dim]
        query_states = self.q_proj(hidden_states)
        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # [bsz, q_len, hidden_size] ->-> [bsz, num_heads_KV=1, q_len, head_dim]]
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # cos, sin: [batch_size, seq_len, head_dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [batch_size, num_heads_Q, seq_len, head_dim], [batch_size, num_heads_KV, seq_len, head_dim] 
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin) 
        
        if kv_cache is not None: 
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)
        
        # Repeat the key and values to match the number of the query. 
        key_states = key_states.repeat_kv(key_states, self.num_key_value_groups)
        value_states = value_states.repeat_kv(value_states, self.num_key_value_groups)
        # For the calculation: Q * K^T / sqrt(head_dim), shape: [bsz, num_heads_Q, seq_len_Q, seq_len_KV]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        assert attention_mask is not None 
        attn_weights = attn_weights + attention_mask 
        # [bsz, num_heads_Q, seq_len_Q, seq_len_KV]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
        attn_weights = attn_weights.dropout(attn_weights, p=self.dropout, training=self.training) 
        attn_output = torch.matmul(attn_weights, value_states) 
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim): 
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # Make sure the sequence length is the second dimension. # [bsz, num_heads_Q, seq_len_Q, seq_len_KV]
        attn_output = attn_output.transpose(1, 2).contiguous() 
        attn_output = attn_output.view(bsz, q_len, -1) 
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, seq_len, hidden_size] 
        attn_output = self.o_proj(attn_output) 
        return attn_output, attn_weights
    

def rotate_half(x):
    # Bulid the [-x2, x1, -x4, x3, -x6, x5, ...] tensor for sin part of the position encoding.
    x1 = x[..., : x.shape[-1] // 2] # Take the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Take the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1) 

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1): 
    # Add the head dimension. [bsz, seq_len, head_dim] -> [bsz, 1, seq_len, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Apply the formula (34) in the Rotary Position Embedding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin) 
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: 
    batch, num_key_value, slen, head_dim = hidden_states.shape 
    if n_rep == 1:
        return hidden_states 
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value * n_rep, slen, head_dim)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None): 
        super().__init__()
        
        self.dim = dim # it is set to the head_dim 
        self.max_position_embeddings = max_position_embeddings 
        self.base = base 
        
       # calculate the theta according to the formula: theta_i = base^(2i/dim), where i=0, 1, 2, ..., dim//2
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)) 
        self.register_buffer("inv_freq", tensor=self.inv_freq, persistent=False)
        
    @torch.no_grad() 
    def forward(self, x, position_ids, seq_len=None): 
        # x: [batch_size, num_attention_heads, seq_len, head_dim] 
        self.inv_freq.to(x.device) 
        # copy the inv_freq tensor for batch in the sequence 
        # inv_freq_expended: [batch_size, head_dim//2, 1] 
        inv_freq_expended = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, -1)
        # position_ids_expened: [batch_size, 1, seq_len] 
        position_ids_expended = position_ids[:, None, :].float()
        
        device_type = x.device.type 
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu" 
        with torch.autocast(device_type=device_type, enabled=False): 
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [bsz, head_dim//2, seq_len] @ [bsz, 1, seq_len] -> [bsz, head_dim//2, seq_len] -> [bsz, seq_len, head_dim//2]
            freqs = (inv_freq_expended.float() @ position_ids_expended.float()).transpose(1, 2)
            # emb: [batch_size, seq_len, head_dim] 
            emb = torch.cat((freqs, freqs), dim=-1) 
            # cos, sin: [batch_size, seq_len, head_dim]
            cos, sin = emb.cos(), emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

