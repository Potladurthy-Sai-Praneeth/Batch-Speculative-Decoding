# Static Batch Speculative Decoding

This repository implements a static batch speculative decoding algorithm for accelerating language model inference. Speculative decoding uses a smaller draft model to predict multiple tokens at once, which are then verified by a larger target model, reducing the number of sequential operations needed during generation.

## Overview

Speculative decoding works by:
1. Using a smaller, faster "draft" model to predict multiple tokens (γ tokens)
2. Verifying these speculative tokens with a larger "target" model
3. Accepting or rejecting tokens based on probability distributions
4. Generating replacement tokens for rejected predictions

## Key Components

### `KVCacheModel`

The `KVCacheModel` class handles key-value cache management for both draft and target models:

- Maintains past key-values for efficient autoregressive generation
- Implements prefill and generation phases
- Handles proper KV cache rollback when tokens are rejected

### `SpeculativeDecoding`

The `SpeculativeDecoding` class implements the core algorithm:

- Manages both draft and target models
- Implements the token acceptance/rejection logic
- Handles probability distribution adjustments for replacement token sampling
- Tracks statistics like forward pass counts

### `Engine`

The `Engine` class provides a high-level interface:

- Loads and initializes models and tokenizers
- Manages batch processing
- Handles tokenization and output formatting

## Key Implementation Details

### KV Cache Management

One of the most critical aspects of this implementation is proper KV cache management:

1. **Prefill Phase**: 
   - Initial cache is filled with the prompt tokens
   - Both draft and target models generate initial hidden states

2. **Incremental Generation**:
   - Draft model generates γ tokens at once, updating its KV cache
   - Target model verifies these tokens, updating its KV cache

3. **KV Cache Truncation**:
   - When tokens are rejected, both models' KV caches must be properly truncated
   - The `rollback` method handles this by:
     - Truncating the KV cache to the maximum valid position across the batch
     - Zeroing out attention masks for invalid positions

```python
def rollback(self, batch_idx_dict, index_accepted_len_dict):
    max_index = max(batch_idx_dict.values())
    self._attention_mask = self._attention_mask[:, :max_index]
    self._past_key_values.crop(max_index)
    
    for batch_idx, end_idx in batch_idx_dict.items():
        self._attention_mask[batch_idx, end_idx:] = 0


### Usage
- Speculative Decoding
```
python main.py --target_model "HuggingFaceTB/SmolLM-1.7B-Instruct" --draft_model "HuggingFaceTB/SmolLM-360M-Instruct" --prompts "What is random forest?" "Explain black holes" "Who is the father of python programming?" "What is the capital of France?" --gamma 5 --max_tokens 100 --temperature 0.75
```
- Auto-regressive Decoding
```python
python main.py --target_model "HuggingFaceTB/SmolLM-1.7B-Instruct" --draft_model "HuggingFaceTB/SmolLM-360M-Instruct" --prompts "What is random forest?" "Explain black holes" "Who is the father of python programming?" "What is the capital of France?" --gamma 5 --max_tokens 100 --temperature 0.75 --auto_regressive
```