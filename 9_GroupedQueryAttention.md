# Build DeepSeek from Scratch - Grouped Query Attention (GQA)

**By: Ayushmaan Singh**

## Overview
Grouped Query Attention (GQA) represents the middle ground between Multi-Head Attention and Multi-Query Attention. It solves the performance degradation problem of MQA while still providing significant memory savings compared to standard MHA.

## Learning Objectives
- Understand why GQA was needed as a compromise solution
- Master the grouping concept and how it balances memory vs performance
- Learn the mathematical implications of group-based key/value sharing
- Analyze real-world implementations (LLaMA 3) through visualizations
- Understand the trade-offs and positioning in the attention evolution

## Prerequisites
- Multi-Head Attention implementation and diversity concept
- Multi-Query Attention and its limitations
- Key-Value Cache memory problem understanding

---

# Grouped Query Attention (GQA) - Detailed Notes

## Overview
Grouped Query Attention (GQA) is an optimization technique that sits between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). It was introduced to address the performance degradation issues of MQA while still maintaining memory efficiency benefits.

## Motivation

### The KV Cache Memory Problem
- During inference, we cache the keys (K) and values (V) matrices to avoid repeated computations
- This provides linear computational complexity instead of quadratic
- However, KV cache size grows with:
  - Number of transformer blocks
  - Batch size
  - Number of attention heads
  - Attention head size
  - Context length

### Memory Requirements Examples
- **GPT-2**: 36 MB KV cache
- **GPT-3**: 4.5 GB KV cache  
- **DeepSeek R1/V3**: ~400 GB KV cache (61 blocks, 128 heads, head_dim=128, context=100k)

### Previous Solutions
1. **Multi-Head Attention (MHA)**: Best performance, highest memory usage
2. **Multi-Query Attention (MQA)**: Lowest memory usage, significant performance degradation

## Core Concept of Grouped Query Attention

### The Middle Ground Approach
GQA creates **groups** of attention heads where:
- Heads within the same group share identical K and V matrices
- Different groups have different K and V matrices
- This provides a balance between diversity and memory efficiency

### Mathematical Formulation

For a model with `n` attention heads and `g` groups:
- Each group contains `n/g` heads
- Within group `i`: All heads share the same `K_i` and `V_i`
- Across groups: `K_1 ≠ K_2 ≠ ... ≠ K_g` and `V_1 ≠ V_2 ≠ ... ≠ V_g`

## Detailed Mechanism

### Group Structure Example
Consider 32 attention heads with 8 groups (4 heads per group):

```
Group 1: Heads 1-4    → Same K₁, V₁
Group 2: Heads 5-8    → Same K₂, V₂  
Group 3: Heads 9-12   → Same K₃, V₃
...
Group 8: Heads 29-32  → Same K₈, V₈
```

### Key-Value Sharing Pattern
- **Within Group**: K₁ = K₂ = K₃ = K₄ and V₁ = V₂ = V₃ = V₄
- **Across Groups**: K₁ ≠ K₅ ≠ K₉ and V₁ ≠ V₅ ≠ V₉

### Query Matrices
- Q matrices remain unique for each head (Q₁ ≠ Q₂ ≠ Q₃ ≠ Q₄)
- This preserves some diversity in attention patterns

## Performance Analysis

### Diversity Comparison
1. **Multi-Head Attention**: Maximum diversity (all heads different)
2. **Grouped Query Attention**: Moderate diversity (group-level differences)
3. **Multi-Query Attention**: Minimum diversity (all heads identical)

### Memory Efficiency
The KV cache size reduction factor:
- **MHA**: No reduction (factor = 1)
- **GQA**: Reduction by `n/g` (e.g., 128/8 = 16x reduction)
- **MQA**: Maximum reduction by `n` (e.g., 128x reduction)

## KV Cache Size Formula

### Original Formula (MHA)
```
KV_cache_size = 2 × batch_size × seq_len × n_heads × head_dim × n_layers
```

### GQA Formula
```
KV_cache_size = 2 × batch_size × seq_len × n_groups × head_dim × n_layers
```

Where `n_groups = n_heads / heads_per_group`

## Real-World Implementation: LLaMA 3

### LLaMA 3 Architecture
- **Model**: LLaMA 3 8B and 70B
- **Groups**: 8 groups
- **Heads per group**: 4 heads
- **Total heads**: 32 heads
- **Announced**: April 18, 2024 by Meta

### Experimental Verification
The transcript describes visualization experiments showing:
- Within each group: Identical K and V matrices across heads
- Across groups: Different K and V matrices
- Heat map visualizations confirm the group structure

## Advantages of GQA

### Performance Benefits
1. **Better than MQA**: More diversity in attention patterns
2. **Reasonable memory usage**: Moderate reduction in KV cache size
3. **Balanced trade-off**: Optimizes both performance and memory

### Memory Benefits
1. **Reduced storage**: Only need to cache one K,V pair per group
2. **Scalable**: Can adjust group size based on requirements
3. **Practical**: Enables deployment of larger models

## Disadvantages of GQA

### Performance Limitations
1. **Still not MHA-level**: Some diversity loss compared to full MHA
2. **Group constraints**: Heads within groups are artificially constrained
3. **Compromise solution**: Neither optimal performance nor minimal memory

### Implementation Complexity
1. **Group management**: Need to handle group structures
2. **Hyperparameter tuning**: Optimal group size depends on model/task
3. **Hardware considerations**: May not utilize all available parallelism

## Comparison Table

| Method | KV Cache Size | Performance | Diversity | Memory Factor |
|--------|---------------|-------------|-----------|---------------|
| MHA    | Largest       | Best        | Maximum   | 1× (baseline) |
| GQA    | Medium        | Good        | Moderate  | n/g× reduction |
| MQA    | Smallest      | Worst       | Minimum   | n× reduction  |

## Mathematical Example

### DeepSeek Model with GQA
- **Original heads**: 128
- **Groups**: 8
- **Heads per group**: 16
- **Memory reduction**: 128/8 = 16× smaller than MHA
- **Performance**: Better than MQA, worse than MHA

## Code Implementation Insights

### Visualization Approach
1. **Heat map generation**: Shows K and V matrices across groups
2. **Group verification**: Confirms identical values within groups
3. **Cross-group differences**: Validates different values across groups

### Key Observations
- Same-group matrices are visually identical
- Different-group matrices show clear variations
- Pattern holds for both keys and values matrices

## Transition to Multi-Head Latent Attention

### The Golden Question
"Can we create something that has:
- **Low KV cache size** (like MQA)
- **High performance** (like MHA)"

### DeepSeek's Innovation
- GQA was a stepping stone toward Multi-Head Latent Attention
- DeepSeek achieved the "best of both worlds" through latent attention mechanism
- This represents the next evolution in attention optimization

## Key Takeaways

1. **GQA is a compromise**: Balances memory and performance trade-offs
2. **Group structure matters**: Optimal group size depends on model requirements
3. **Real-world adoption**: Major models like LLaMA 3 use GQA
4. **Stepping stone**: GQA paved the way for more advanced techniques like Multi-Head Latent Attention
5. **Practical solution**: Enables deployment of larger models with reasonable memory requirements

## Next Steps

The next lecture will cover **Multi-Head Latent Attention**, which represents DeepSeek's ultimate solution to the KV cache memory problem while maintaining high performance.

---
**Author: Ayushmaan Singh**
