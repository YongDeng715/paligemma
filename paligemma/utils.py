from transformers import AutoTokenizer 
from safetensors import safe_open 
from typing import Tuple 
import json
import glob
import os 

from modeling_gemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration


def load_hf_model(
    model_path: str, device: str
) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # load the tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    
    # find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors")) 
    
    # ... and load them one by on in the tensors dictionary 
    tensors = {} 
    for safetensors_file in safetensors_files: 
        with safe_open(safetensors_file, framework="pt", device="cpu") as f: 
            for key in f.keys(): 
                tensors[key] = f.get_tensor(key)
            
    # load the model's config 
    with open(os.path.join(model_path, "config.json"), "r") as f: 
        model_config_file = json.load(f) 
        config = PaliGemmaConfig(**model_config_file) 
    
    # create the model using the configuration 
    model = PaliGemmaForConditionalGeneration(config).to(device) 
    # load the state dict of the model
    model.load_state_dict(tensors) 
    
    model.tie_wieghts()
    return (model, tokenizer)
