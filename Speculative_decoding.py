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
from utils import *
from torch.nn.utils.rnn import pad_sequence
from KVCacheModel import KVCacheModel
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

class SpeculativeDecoding:
    def __init__(self,target= None, draft = None, tokenizer = None, gamma=5,max_length = 512, temperature = 1.0, top_k = 0, top_p = 0.9, target_device = "cuda" if torch.cuda.is_available() else "cpu", draft_device = "cuda" if torch.cuda.is_available() else "cpu"):
        
        assert target is not None, "Target model cannot be None in SpeculativeDecoding"
        assert draft is not None, "Draft model cannot be None in SpeculativeDecoding"
        assert tokenizer is not None, "Tokenizer cannot be None in SpeculativeDecoding"

        self.target = target
        self.draft = draft
        self.tokenizer = tokenizer

        self.gamma = gamma
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.target_device = target_device
        self.draft_device = draft_device

        self.draft_model = KVCacheModel(model=self.draft,tokenizer=self.tokenizer,temperature=self.temperature,top_k=self.top_k,top_p=self.top_p,device=self.draft_device)
        self.target_model = KVCacheModel(model=self.target,tokenizer=self.tokenizer,temperature=self.temperature,top_k=self.top_k,top_p=self.top_p,device=self.target_device)

        self.seed = 687
        self.draft_forward_times = 0
        self.target_forward_times = 0
    
    def generate(self, tokenized_text):
        assert tokenized_text is not None, "Tokenized text cannot be None in generate method"
        assert hasattr(tokenized_text, 'input_ids'), "Tokenized text must have input_ids attribute"
        assert hasattr(tokenized_text, 'attention_mask'), "Tokenized text must have attention_mask attribute"

        prefix_len = tokenized_text.input_ids.shape[1]
        prefix = tokenized_text.input_ids
        max_token_len = self.max_length + prefix.shape[1]
        attention_mask = tokenized_text.attention_mask
        # Prefill target model
        self.target_model.target_generate(prefix, attention_mask, gamma=self.gamma)

        while prefix_len < max_token_len:
            # print(f'Starting generation with prefix shape {prefix.shape}')
            # print('--'*30)
            # print(f"Draft model generation step with prefix:{prefix.shape} but prefix_len is {prefix_len}...")
            draft_tokens, draft_probs, attention_mask = self.draft_model.draft_generate(prefix,attention_mask ,gamma=self.gamma)
            self.draft_forward_times += self.gamma

            # print(f'Generated draft tokens shape is {draft_tokens.shape}')
            # print(f'Generated draft probabilities shape is {draft_probs.shape}')
            # print(f'Draft attention mask shape is {attention_mask.shape}')
            # print(f'Draft prob history shape is {self.draft_model._prob_history.shape}')
            # print('---'*30)

            # print(f"Target model validation step with prefix:{prefix.shape[1] +self.gamma }...")
            target_probs, last_token_probs = self.target_model.target_generate(torch.cat([prefix,draft_tokens],dim=1), attention_mask, gamma=self.gamma)
            self.target_forward_times += 1
            # print(f'Generated target probabilities shape is {target_probs.shape}')
            # print('---'*30)

            assert draft_probs.shape == target_probs.shape, f"Draft:{draft_probs.shape} and target:{target_probs.shape} probabilities must have the same shape"

            draft_gathered = draft_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
            target_gathered = target_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
            # print(f'Gathered draft probabilities shape is {draft_gathered.shape}, {draft_gathered}')
            # print(f'Gathered target probabilities shape is {target_gathered.shape}, {target_gathered}')

            epsilon = 1e-6
            ratio = torch.minimum(
                torch.ones_like(draft_gathered), 
                target_gathered / (draft_gathered + epsilon)
            )
            
            torch.manual_seed(self.seed+prefix_len)
            random_value = torch.rand_like(ratio,device=self.draft_device)
            reject_mask = (ratio < random_value)
            # print(f'Reject mask is {reject_mask.shape}, {reject_mask}')

            min_probs = torch.min(target_probs, draft_probs)
            rejection_probs = 1.0 - torch.sum(min_probs, dim=-1, keepdim=True)
            adjusted_distribution = (target_probs - min_probs) / (rejection_probs + epsilon)

            # Calculate first rejection index for each sequence in batch
            first_reject_idx = torch.argmax(reject_mask.int(), dim=1)
            no_rejects = ~reject_mask.any(dim=1) 

            # When no rejections, set to draft_tokens length
            first_reject_idx[no_rejects] = draft_tokens.shape[1]
            needs_sampling = first_reject_idx < draft_tokens.shape[1]

            # Sample replacement tokens where needed
            sampled_replacement_tokens = torch.full((draft_tokens.shape[0],), self.tokenizer.pad_token_id, dtype=torch.long, device=self.draft_device)
            batch_indices_to_sample = torch.where(needs_sampling)[0]

            if batch_indices_to_sample.numel() > 0:
                seq_indices_to_sample = first_reject_idx[needs_sampling]
                distributions_to_sample_from = adjusted_distribution[batch_indices_to_sample, seq_indices_to_sample, :]
                samples = torch.multinomial(distributions_to_sample_from, num_samples=1).squeeze(-1)
                sampled_replacement_tokens[batch_indices_to_sample] = samples
                
            batch_size, seq_len = draft_tokens.shape
            mask = torch.arange(seq_len, device=self.draft_device).unsqueeze(0) < first_reject_idx.unsqueeze(1)
            accepted_tokens = draft_tokens.masked_fill(~mask, self.tokenizer.pad_token_id)

            # Create final sequences
            final_sequences_list = []
            for b in range(batch_size):
                if needs_sampling[b]:
                    # Take accepted tokens and append sampled token
                    accepted_prefix = accepted_tokens[b, :first_reject_idx[b]]
                    sampled_token = sampled_replacement_tokens[b].unsqueeze(0)
                    final_sequence = torch.cat((accepted_prefix, sampled_token), dim=0)
                else:
                    # All draft tokens were accepted, so take all of them
                    final_sequence = draft_tokens[b, :first_reject_idx[b]]
                    print(f'last token probs :{last_token_probs.shape} and final sequence is {final_sequence.shape}')
                    # final_sequence = torch.cat([final_sequence,torch.multinomial(last_token_probs, num_samples=1)],dim=1)
                final_sequences_list.append(final_sequence)

            index_accepted_len_dict = {i: seq.shape[0] for i, seq in enumerate(final_sequences_list)} 
            new_total_lengths = {i: prefix.shape[1] + seq.shape[0] for i, seq in enumerate(final_sequences_list)} 

            # print(f'Index max length dict is {index_accepted_len_dict}')
            
            # Pad sequences to same length
            final_tokens_padded = pad_sequence(
                final_sequences_list,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )

            # print(f'Final tokens padded shape is {final_tokens_padded.shape}, {final_tokens_padded}')
            # Create new prefix by concatenating with original prefix
            prefix = torch.cat([prefix, final_tokens_padded], dim=1)

            prefix_len = min(new_total_lengths.values())

            # print(f'---'*20)
            # print(f'Draft roll back')
            self.draft_model.rollback(new_total_lengths)
            # print(f'--'*20)
            # print(f'Target roll back')
            self.target_model.rollback(new_total_lengths)

            attention_mask = self.draft_model._attention_mask
            
            # print('***'*30)

        return prefix