import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import json
import random
import torch.nn.functional as F
from huggingface_hub import login
from collections import defaultdict
import time
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def top_k_top_p_filter_3d(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
   
    assert logits.dim() == 3, f"Expected 3D tensor, got {logits.dim()}D"
    batch_size, seq_len, vocab_size = logits.shape
    if top_k > 0 :                
        top_k_values = torch.topk(logits, top_k, dim=-1)[0]                
        filter_values = top_k_values[:, :, -1].unsqueeze(-1)                
        logits = torch.where(logits < filter_values,
                             torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype),
                             logits)
    
    if top_p > 0.0:        
        logits_reshaped = logits.view(-1, vocab_size)
        sorted_logits, sorted_indices = torch.sort(logits_reshaped, descending=True, dim=-1)    
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter_mask = cumulative_probs > top_p
        filter_mask[..., 1:] = filter_mask[..., :-1].clone()
        filter_mask[..., 0] = False 
        indices_to_remove = torch.zeros_like(logits_reshaped, dtype=torch.bool)                
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=filter_mask)
        logits_reshaped[indices_to_remove] = float('-inf')
        logits = logits_reshaped.view(batch_size, seq_len, vocab_size)

    return logits
    
def norm_logits_3d(logits: torch.Tensor, temperature: float, top_k: float, top_p: float) -> torch.Tensor:
    assert logits.dim() == 3, f"Expected 3D tensor, got {logits.dim()}D"
    batch_size, seq_len, vocab_size = logits.shape
    if temperature == 0:                
        indices = logits.argmax(dim=-1)        
        probs = torch.zeros_like(logits)                        
        probs.scatter_(dim=-1, index=indices.unsqueeze(-1), value=1.0)
        return probs.float() 
    if temperature > 0:
         logits = logits / temperature
    logits = top_k_top_p_filter_3d(logits, top_k=int(top_k), top_p=top_p) 
    probs = F.softmax(logits, dim=-1)

    return probs
    
def find_last_valid_token(x):
    nonzero_indices = torch.nonzero(x > 0)
    unique_rows = torch.unique(nonzero_indices[:, 0])
    max_cols = torch.zeros_like(unique_rows)
    for i, row in enumerate(unique_rows):
        row_mask = nonzero_indices[:, 0] == row
        max_cols[i] = nonzero_indices[row_mask, 1].max()
    return unique_rows, max_cols