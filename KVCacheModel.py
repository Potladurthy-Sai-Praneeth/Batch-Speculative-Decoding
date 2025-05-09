import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import json
import random
import torch.nn.functional as F
from collections import defaultdict
import time
from torch.nn.utils.rnn import pad_sequence
from utils import *
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


class KVCacheModel:
    def __init__(self,model=None,tokenizer= None,temperature=None , top_k = None , top_p = None ,device="cuda" if torch.cuda.is_available() else "cpu"):
        assert model is not None, "Model cannot be None in KVCacheModel"
        assert tokenizer is not None, "Tokenizer cannot be None in KVCacheModel"
        assert temperature is not None, "Temperature cannot be None in KVCacheModel"
        assert top_k is not None, "Top_k cannot be None in KVCacheModel"
        assert top_p is not None, "Top_p cannot be None in KVCacheModel"
        self._model = model
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._device = device
        self._prob_history = None
        self._past_key_values = None
        self._attention_mask = None
        self.tokenizer = tokenizer
        self.prefil_flag = True
        self.gamma = 1
    
    
    @torch.no_grad()
    def draft_generate(self,input_ids,attention_mask,gamma=5):
        self.gamma = gamma
       
        generated_tokens = []
        if self.prefil_flag:
            self._attention_mask = attention_mask
            outputs = self._model(input_ids=input_ids,attention_mask=self._attention_mask,use_cache=True)
            self._past_key_values = outputs.past_key_values
            raw_logits = outputs.logits.to(torch.float32)
            self._prob_history = norm_logits_3d(raw_logits[:,-1:,:],self._temperature,self._top_k,self._top_p)
            self._attention_mask = torch.cat([self._attention_mask, torch.ones((input_ids.shape[0],1), device=self._device)],dim=1)
            self.prefil_flag = False
            generated_tokens.append( torch.multinomial(self._prob_history[:,-1,:],num_samples=1))
            gamma -=1

        for i in range(gamma):
            if generated_tokens == []:
                r,c = find_last_valid_token(self._attention_mask)
                inputs = input_ids[r,c].view(-1,1)
                # print(f'Input_ids  is {inputs.shape}, {inputs}')
            else:
                inputs = generated_tokens[-1]
            
            self._attention_mask = torch.cat([self._attention_mask, torch.ones((inputs.shape[0],1), device=self._device)],dim=1)
            outputs = self._model(input_ids=inputs,attention_mask=self._attention_mask ,past_key_values=self._past_key_values,use_cache=True)
            self._past_key_values = outputs.past_key_values
            raw_logits = outputs.logits.to(torch.float32)
            self._prob_history = torch.cat([self._prob_history, norm_logits_3d(raw_logits,self._temperature,self._top_k,self._top_p)],dim=1)
            generated_tokens.append( torch.multinomial(self._prob_history[:,-1,:],num_samples=1))
            
        
        generated_tokens = torch.cat(generated_tokens,dim=1)
        
        assert generated_tokens.shape[1] == self._prob_history[:,-self.gamma:,:].shape[1], f"Generated tokens shape {generated_tokens.shape} and prob history shape {self._prob_history[:,-self.gamma:,:].shape} do not match"

        return generated_tokens, self._prob_history[:,-self.gamma:,:], self._attention_mask

            
    @torch.no_grad()
    def target_generate(self,input_ids,attention_mask,gamma=5):
        if self.prefil_flag:
            self._attention_mask = torch.cat([attention_mask, torch.ones((input_ids.shape[0],1), device=self._device)],dim=1)
            outputs = self._model(input_ids=input_ids,attention_mask=self._attention_mask,use_cache=True)
            self._past_key_values = outputs.past_key_values
            raw_logits = outputs.logits.to(torch.float32)
            self._prob_history = norm_logits_3d(raw_logits[:,-1:,:],self._temperature,self._top_k,self._top_p)
            # self._attention_mask = attention_mask
            self.prefil_flag = False
            return None
        
        else:
            input_ids = input_ids[:,-gamma-1:]
            self._attention_mask = torch.cat([self._attention_mask, torch.ones_like(input_ids, device=self._device)],dim=1)
            outputs = self._model(input_ids=input_ids,attention_mask=self._attention_mask ,past_key_values=self._past_key_values,use_cache=True)
            self._past_key_values = outputs.past_key_values
            raw_logits = outputs.logits.to(torch.float32)
            self._prob_history = torch.cat([self._prob_history, norm_logits_3d(raw_logits,self._temperature,self._top_k,self._top_p)],dim=1)

            return self._prob_history[:,-gamma-1:-1,:], self._prob_history[:,-1:,:]
    

    def rollback(self,batch_idx_dict):
        """
        Args:
            batch_idx_dict (dict): Dictionary mapping batch indices to index of the last accepted positions.
                Example: {0: 5, 1: 3} means for batch item 0, end position is at 5 and for batch item 1, last accepted token is at 3.
        In this function , we need to perform three operations:
        1. Truncate the probability history, attention_mask and past_key_values to the maximum of batch_idx_dict for each batch item.
        2. Zero out the positions in the probability history and past_key_values that are beyond the end position for each batch item.
        3. Adjust the attention_mask beyond the end position for each batch item to let the model know that these positions are not valid.
        """
        max_index = max(batch_idx_dict.values())

        # Step 1
        self._attention_mask = self._attention_mask[:,:max_index]
        self._past_key_values.crop(max_index)
    
        # Step 2
        for batch_idx, end_idx in batch_idx_dict.items():
            self._attention_mask[batch_idx, end_idx:] = 0

        # print(f'After rollback prob history shape is {self._prob_history.shape}')
        # print(f'After rollback attention_mask shape is {self._attention_mask.shape}')
        for kv in self._past_key_values:
            k,v = kv
            # print(f'After rollback, K:{k.shape} and V:{v.shape}')
            break
        # print(f'Attention mask is {self._attention_mask}')