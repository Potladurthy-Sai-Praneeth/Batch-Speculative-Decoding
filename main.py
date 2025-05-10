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
import argparse
from engine import Engine

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Speculative Decoding Demo')
    parser.add_argument('--target_model', type=str, required=True, help='Target model ID (e.g., "gpt2-xl")')
    parser.add_argument('--draft_model', type=str, required=True, help='Draft model ID (e.g., "gpt2-medium")')
    parser.add_argument('--prompts', type=str, nargs='+', required=True, help='List of prompts to process')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum output length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--gamma', type=int, default=5, help='Number of tokens to predict in each draft step')
    parser.add_argument('--auto_regressive', action='store_true', default=False, 
                        help='If set, use standard auto-regressive decoding instead of speculative decoding')
    
    args = parser.parse_args()
        
    # Initialize the speculative decoding engine
    print(f"Initializing engine with target model: {args.target_model}, draft model: {args.draft_model}")
    speculate = Engine(
        target_id=args.target_model, 
        draft_id=args.draft_model, 
        max_length=args.max_length, 
        temperature=args.temperature,
        gamma=args.gamma
    )
    
    # Process prompts
    if args.auto_regressive:
        print("Using auto-regressive decoding...")
        start_time = time.time()
        
        # Tokenize all prompts
        inputs = speculate.tokenize_text(args.prompts)
        
        # Process each prompt in batch mode for auto-regressive generation
        outputs = speculate.target_model.generate(
            inputs.input_ids.to(speculate.device),
            attention_mask=inputs.attention_mask.to(speculate.device),
            max_length=inputs.input_ids.shape[1] + args.max_length,
            do_sample=True,
            temperature=args.temperature
        )
        
        end_time = time.time()
        
        # Print results
        for i, prompt in enumerate(args.prompts):
            print(f"Prompt: {prompt}")
            print(f"Generated Text: {speculate.tokenizer.decode(outputs[i], skip_special_tokens=True)}\n")
            print('---' * 30)
        
        print(f"Auto-regressive generation completed in {end_time - start_time:.2f} seconds")
        
    else:
        print("Using speculative decoding...")
        start_time = time.time()
        
        # Generate using speculative decoding
        tokenizer,outputs = speculate.generate(args.prompts)
        
        end_time = time.time()
        
        # Print results
        for i, prompt in enumerate(args.prompts):
            print(f"Prompt: {prompt}")
            print(f"Generated Text: {tokenizer.decode(outputs[i], skip_special_tokens=True)}\n")
            print('---' * 30)
        
        print(f"Draft model forward times: {speculate.speculative_decoding.draft_forward_times}")
        print(f"Target model forward times: {speculate.speculative_decoding.target_forward_times}")
        print(f"Speculative decoding completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()