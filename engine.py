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
from torch.nn.utils.rnn import pad_sequence
from KVCacheModel import KVCacheModel
from Speculative_decoding import SpeculativeDecoding
from utils import *



class Engine:
    def __init__(self,target_id = None, draft_id= None,  max_length=512, temperature=0.6,gamma=5,top_k=0,top_p=0.9):
        self.target_model_id = target_id
        self.draft_model_id = draft_id
        
        assert self.target_model_id is not None, "Target model ID is not provided."
        assert self.draft_model_id is not None, "Draft model ID is not provided."

        self.max_length = max_length
        self.temperature = temperature            
        self.gamma = gamma
        self.top_k = top_k
        self.top_p = top_p

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

        self.speculative_decoding = SpeculativeDecoding(
            target=self.target_model,
            draft=self.draft_model,
            tokenizer=self.tokenizer, 
            gamma=self.gamma,
            max_length=self.max_length,
            temperature=self.temperature, 
            top_k=self.top_k, top_p=self.top_p, 
            target_device=self.device, 
            draft_device=self.device)


    def load_model(self):
        # Load target model
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)

        # Load draft model
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.draft_model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)
       
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.target_model_id, trust_remote_code=True)
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.target_model.eval()
        self.draft_model.eval()

    def tokenize_text(self,prompts):
        return self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
    
    def generate(self,prompts):
        assert len(prompts) > 0, "prompts should not be empty"

        if type(prompts) == str:
            prompts = [prompts]

        assert type(prompts) == list, "prompts should be a list of strings for batch processing"

        self.batch_size = len(prompts)
        self.tokenized_text = self.tokenize_text(prompts)

        generated_text = self.speculative_decoding.generate(self.tokenized_text)

        return self.tokenizer , generated_text