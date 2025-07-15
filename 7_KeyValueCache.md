# Build DeepSeek from Scratch - Key-Value Cache (KV Cache)

**By: Ayushmaan Singh**

## Overview
This lecture covers one of the most critical optimizations in transformer inference: the Key-Value Cache. Understanding KV Cache is essential before diving into DeepSeek's Multi-Head Latent Attention innovation.

## Learning Objectives
- Understand what happens during LLM inference
- Learn why Key-Value Cache is necessary
- Master the mathematical implementation of KV Cache
- Understand the advantages and disadvantages (the "dark side") of KV Cache
- Prepare for advanced concepts like Multi-Head Latent Attention

## Prerequisites
- Self-attention mechanism
- Causal attention and masking
- Multi-head attention implementation

---

## 1. Introduction: Why Context Size Affects Pricing

### Real-World Example: OpenAI API Pricing
- **GPT-4 (8K context)**: $30 per million tokens
- **GPT-4 (32K context)**: $60 per million tokens
- **Key insight**: Price doubles with 4x context increase

### The Connection
The Key-Value Cache is a major reason why larger context sizes cost more. As we'll see, the memory requirements scale directly with context length.

---

## 2. Fundamental Concepts

### 2.1 Pre-training vs Inference
**Critical Understanding**: KV Cache **ONLY** applies during inference, not pre-training.

#### Pre-training Phase
- Model parameters are learned from training data
- All 175B+ parameters are optimized through backpropagation
- Parameters become fixed after training

#### Inference Phase
- Pre-trained model is used to predict next tokens
- Parameters are fixed (no learning)
- One token generated at a time
- **This is where KV Cache matters**

### 2.2 The Inference Loop
```
Input: "The next day is"
↓
LLM Architecture
↓ 
Predict: "bright"
↓
New Input: "The next day is bright"
↓
LLM Architecture  
↓
Predict: "and"
↓
Continue until max tokens or end...
```

---

## 3. The Core Problem: Repeated Computations

### 3.1 The Inefficiency
During standard inference:
1. **Step 1**: Process "The next day is" → predict "bright"
2. **Step 2**: Process "The next day is bright" → predict "and"
3. **Step 3**: Process "The next day is bright and" → predict next token

**Problem**: We recompute attention for "The next day is" in every step!

### 3.2 Mathematical Proof of Repetition

Consider the attention mechanism:
```
Input: [4×8] → Q,K,V: [4×4] → Attention: [4×4] → Context: [4×4]
```

When we add one token:
```
Input: [5×8] → Q,K,V: [5×4] → Attention: [5×5] → Context: [5×4]
```

**The repetition**: The first [4×4] block in all matrices is identical to the previous computation!

### 3.3 The Key Insight
To predict the next token, we **only need the context vector for the last token**.

**Why?** 
- Final logits matrix has shape [num_tokens × vocab_size]
- To predict next token, we only use the last row (last token's logits)
- Last row depends only on the last token's context vector

---

## 4. The Key-Value Cache Solution

### 4.1 What to Cache
**We only need to cache Keys and Values, NOT Queries!**

**Reasoning**:
- To get context for new token, we need its query vs all keys
- Previous keys don't change → cache them
- Previous values don't change → cache them  
- Query is only needed for the new token → compute fresh

### 4.2 The Caching Process

#### Step 1: Initial Computation
```python
# Input: "The next day is" [4 tokens]
Q = X @ W_q  # [4×4]
K = X @ W_k  # [4×4] ← CACHE THIS
V = X @ W_v  # [4×4] ← CACHE THIS

# Compute attention and context vectors
attention_scores = Q @ K.T
attention_weights = softmax(scaled_masked_scores)
context = attention_weights @ V
```

#### Step 2: New Token Arrives
```python
# New token: "bright" 
new_token_embedding = embed("bright")

# Only compute Q,K,V for new token
q_new = new_token_embedding @ W_q  # [1×4]
k_new = new_token_embedding @ W_k  # [1×4]  
v_new = new_token_embedding @ W_v  # [1×4]

# Use cached values + new values
K_full = torch.cat([K_cached, k_new], dim=0)  # [5×4]
V_full = torch.cat([V_cached, v_new], dim=0)  # [5×4]

# Compute attention only for new token
attention_scores = q_new @ K_full.T  # [1×5]
attention_weights = softmax(masked_scores)
context_new = attention_weights @ V_full  # [1×4]
```

### 4.3 What We Save
**Only 3 new computations per token**:
1. `new_embedding @ W_q`
2. `new_embedding @ W_k` 
3. `new_embedding @ W_v`

Everything else is cached!

---

## 5. Advantages of Key-Value Cache

### 5.1 Computational Complexity
- **Without caching**: O(n²) - quadratic complexity
- **With caching**: O(n) - linear complexity

### 5.2 Speed Improvements
**Real example from transcript**:
- With KV Cache: 2 seconds for 100 tokens
- Without KV Cache: 7 seconds for 100 tokens
- **Speedup**: 3.5x faster!

### 5.3 Why the Speedup?
- No recomputation of previous keys/values
- Linear scaling with sequence length
- Efficient memory reuse

---

## 6. The Dark Side: Memory Requirements

### 6.1 KV Cache Size Formula
```
KV_Cache_Size = L × B × N × H × S × 2 × 2
```

Where:
- **L**: Number of transformer layers
- **B**: Batch size  
- **N**: Number of attention heads
- **H**: Head dimension
- **S**: Context length (sequence length)
- **2**: For both Keys and Values
- **2**: Bytes per float16

### 6.2 Real-World Examples

#### 30B Model Example
- L=48, N=96, H=128, S=1024, B=128
- **KV Cache Size**: 180 GB!

#### DeepSeek R1 Example  
- L=61, N=128, H=128, S=100K, B=1
- **KV Cache Size**: 400 GB!

### 6.3 Why Memory Matters
- Every piece of data in memory costs money
- Higher memory usage → higher inference costs
- Memory bandwidth becomes bottleneck
- Limits batch sizes and context lengths

---

## 7. The Context Length Problem

### 7.1 Linear Scaling Issue
KV Cache size scales **linearly** with context length:
- 8K context → X memory
- 32K context → 4X memory  
- 100K context → 12.5X memory

### 7.2 Pricing Impact
This is why OpenAI charges more for larger contexts:
- More context → more memory → higher costs
- DeepSeek's innovation addresses this problem

---

## 8. Implementation Details

### 8.1 What Gets Cached
```python
# Cache these matrices
K_cache = []  # Keys for all previous tokens
V_cache = []  # Values for all previous tokens

# Don't cache these  
Q_new = []    # Only current token's query needed
```

### 8.2 Cache Management
```python
def kv_cache_attention(new_token, K_cache, V_cache):
    # Compute only for new token
    q_new = new_token @ W_q
    k_new = new_token @ W_k  
    v_new = new_token @ W_v
    
    # Update cache
    K_cache.append(k_new)
    V_cache.append(v_new)
    
    # Compute attention
    attention_scores = q_new @ torch.cat(K_cache).T
    attention_weights = softmax(masked_scores)
    context = attention_weights @ torch.cat(V_cache)
    
    return context, K_cache, V_cache
```

---

## 9. Connection to Advanced Architectures

### 9.1 Why KV Cache Led to Innovation
The memory problem motivated several solutions:

1. **Multi-Query Attention** (MQA)
   - Share keys/values across heads
   - Reduces memory by factor of num_heads

2. **Grouped Query Attention** (GQA)  
   - Groups of heads share keys/values
   - Balance between MHA and MQA

3. **Multi-Head Latent Attention** (DeepSeek's innovation)
   - Compresses key/value representations
   - Maintains quality while reducing memory

### 9.2 The Evolution Path
```
Standard Attention → KV Cache → Memory Problem → Advanced Solutions
                                     ↓
                            MQA → GQA → Latent Attention
```

---

## 10. Key Takeaways

### 10.1 Core Concepts
1. **KV Cache only applies during inference**
2. **Only cache Keys and Values, not Queries**
3. **Only need context vector for last token**
4. **Trades computation for memory**

### 10.2 Trade-offs
| Aspect | Without KV Cache | With KV Cache |
|--------|------------------|---------------|
| Computation | O(n²) | O(n) |
| Memory | Low | High |
| Speed | Slow | Fast |
| Cost Model | Compute-bound | Memory-bound |

### 10.3 Why This Matters
- Foundation for understanding modern LLM optimizations
- Essential for DeepSeek's Multi-Head Latent Attention
- Key to understanding inference pricing models
- Critical for scaling to long contexts

---

## 11. Next Steps

### 11.1 Upcoming Topics
1. **Multi-Query Attention**: Reducing memory through sharing
2. **Grouped Query Attention**: Balanced approach
3. **Multi-Head Latent Attention**: DeepSeek's breakthrough

### 11.2 Practical Applications
- Implement KV Cache in your own transformers
- Understand memory requirements for deployment
- Optimize inference for production systems

---

## Summary

Key-Value Cache is a fundamental optimization that:
- **Speeds up inference** by avoiding repeated computations
- **Trades memory for computation** 
- **Scales linearly** instead of quadratically
- **Creates new challenges** with memory usage
- **Motivated advanced architectures** like DeepSeek's innovations

Understanding KV Cache is essential for anyone working with modern LLMs, as it underpins most production inference systems and drives the development of memory-efficient attention mechanisms.

The "dark side" of massive memory requirements is what led to the innovations we'll explore next, culminating in DeepSeek's Multi-Head Latent Attention.

---
**Author: Ayushmaan Singh**
 