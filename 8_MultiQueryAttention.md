# Build DeepSeek from Scratch - Multi-Query Attention (MQA)

**By: Ayushmaan Singh**

## Overview
Multi-Query Attention is the first major innovation designed to solve the Key-Value Cache memory problem. It addresses the "dark side" of KV Cache by dramatically reducing memory requirements, though it comes with its own trade-offs.

## Learning Objectives
- Understand the KV Cache memory problem and why it needs solving
- Master the core concept of Multi-Query Attention
- Learn how MQA reduces memory consumption by factor of N (number of heads)
- Understand the performance trade-offs of MQA
- Compare MQA vs standard Multi-Head Attention through practical examples

## Prerequisites
- Key-Value Cache mechanism and its memory problem
- Multi-Head Attention implementation
- Understanding of attention head diversity and perspectives

---

## 1. Recap: The KV Cache Memory Problem

### 1.1 Why KV Cache Exists
During LLM inference:
1. **Input sequence** goes through transformer architecture
2. **Only the context vector of the last token** is needed for next token prediction
3. **Previous computations are repeated** in each iteration
4. **KV Cache stores** previous keys and values to avoid recomputation

### 1.2 The Memory Problem
**KV Cache Size Formula:**
```
KV_Cache_Size = L × B × N × H × S × 2 × 2
```
Where:
- **L**: Number of transformer layers
- **B**: Batch size
- **N**: Number of attention heads ← KEY FACTOR
- **H**: Head dimension
- **S**: Context length (sequence length)
- **2**: For both Keys and Values
- **2**: Bytes per float16

### 1.3 Real-World Examples
- **GPT-2**: 36 MB KV Cache
- **GPT-3**: 4.5 GB KV Cache  
- **DeepSeek R1/V3**: 400 GB KV Cache!
  - L=61, N=128, H=128, S=100K, B=1

**The Problem**: 400 GB memory consumption makes inference extremely expensive.

---

## 2. Multi-Query Attention: The Core Idea

### 2.1 The Simplest Solution
**Key Question**: What's the simplest way to solve the KV Cache memory problem?

**MQA Answer**: What if all attention heads share the same key and value matrices?

### 2.2 Visual Understanding

#### Standard Multi-Head Attention
```
Input: [5 tokens × 8 dimensions]
      ↓
WK split into 4 heads: [WK1, WK2, WK3, WK4] - All DIFFERENT
WV split into 4 heads: [WV1, WV2, WV3, WV4] - All DIFFERENT
      ↓
Keys:   [K1, K2, K3, K4] - All DIFFERENT colors
Values: [V1, V2, V3, V4] - All DIFFERENT colors
```

#### Multi-Query Attention
```
Input: [5 tokens × 8 dimensions]
      ↓
WK for all heads: [WK1, WK1, WK1, WK1] - All SAME
WV for all heads: [WV1, WV1, WV1, WV1] - All SAME
      ↓
Keys:   [K1, K1, K1, K1] - All SAME color
Values: [V1, V1, V1, V1] - All SAME color
```

### 2.3 What Stays Different
**Important**: Queries remain different across heads!
```
Queries: [Q1, Q2, Q3, Q4] - Still DIFFERENT colors
```

**Why?** Each head can still have different attention patterns through different queries, but they all use the same keys and values.

---

## 3. Memory Reduction Mathematics

### 3.1 The Factor N Reduction
**Standard MHA**: Store K,V for all N heads
**MQA**: Store K,V for only 1 head (shared across all)

**Memory Reduction**: Factor of N (number of attention heads)

### 3.2 Practical Examples

#### DeepSeek Model
- **Original KV Cache**: 400 GB
- **Number of heads**: 128
- **With MQA**: 400 GB ÷ 128 = **3 GB**
- **Reduction**: 133x smaller memory footprint!

#### GPT-3 Model  
- **Original KV Cache**: 4.5 GB
- **Number of heads**: 96
- **With MQA**: 4.5 GB ÷ 96 = **47 MB**
- **Reduction**: 96x smaller memory footprint!

---

## 4. Implementation Details

### 4.1 Weight Matrix Sharing
```python
# Standard Multi-Head Attention
class StandardMHA:
    def __init__(self, d_model, num_heads):
        self.W_q = nn.Linear(d_model, d_model)  # Different per head
        self.W_k = nn.Linear(d_model, d_model)  # Different per head  
        self.W_v = nn.Linear(d_model, d_model)  # Different per head

# Multi-Query Attention
class MultiQueryAttention:
    def __init__(self, d_model, num_heads):
        self.W_q = nn.Linear(d_model, d_model)     # Different per head
        self.W_k = nn.Linear(d_model, head_dim)    # SINGLE shared matrix
        self.W_v = nn.Linear(d_model, head_dim)    # SINGLE shared matrix
```

### 4.2 Cache Storage
```python
# Standard MHA: Store for all heads
K_cache = torch.zeros(num_heads, seq_len, head_dim)
V_cache = torch.zeros(num_heads, seq_len, head_dim)

# MQA: Store for only one head
K_cache = torch.zeros(1, seq_len, head_dim)  # Shared across heads
V_cache = torch.zeros(1, seq_len, head_dim)  # Shared across heads
```

### 4.3 Attention Computation
```python
# MQA Forward Pass
def forward(self, x):
    Q = self.W_q(x).view(batch_size, seq_len, num_heads, head_dim)
    K = self.W_k(x).view(batch_size, seq_len, 1, head_dim)        # Single head
    V = self.W_v(x).view(batch_size, seq_len, 1, head_dim)        # Single head
    
    # Expand K,V to all heads (same values)
    K = K.expand(batch_size, seq_len, num_heads, head_dim)
    V = V.expand(batch_size, seq_len, num_heads, head_dim)
    
    # Standard attention computation
    attention_scores = Q @ K.transpose(-2, -1)
    attention_weights = softmax(attention_scores)
    output = attention_weights @ V
    
    return output
```

---

## 5. Advantages of Multi-Query Attention

### 5.1 Dramatic Memory Reduction
- **Factor of N reduction** in KV Cache size
- **Enables longer context lengths** with same memory
- **Reduces inference costs** significantly

### 5.2 Computational Benefits
- **Faster inference**: Less memory access overhead
- **Better GPU utilization**: More compute resources available
- **Linear scaling**: Memory no longer bottleneck for many applications

### 5.3 Practical Performance
**Experimental Results** (from transcript):
- **Multi-Head Attention**: 1.62 seconds inference time
- **Multi-Query Attention**: 0.64 seconds inference time  
- **Speedup**: ~40% improvement in inference speed

---

## 6. The Dark Side: Performance Degradation

### 6.1 The Fundamental Problem
**MHA Purpose**: Capture different perspectives through different heads
**MQA Problem**: Limits diversity by sharing K,V across heads

### 6.2 Why Diversity Matters
Consider the ambiguous sentence:
> "The artist painted the portrait of a woman with a brush"

**Two interpretations**:
1. "The artist painted the portrait of a woman **using** a brush"
2. "The artist painted the portrait of a woman **with a brush in her hand**"

**MHA Solution**: Different heads capture different perspectives
**MQA Limitation**: Shared K,V reduces perspective diversity

### 6.3 Mathematical Explanation
In standard MHA:
- **K1 ≠ K2 ≠ K3 ≠ K4**: Each head sees different "key" patterns
- **V1 ≠ V2 ≠ V3 ≠ V4**: Each head contributes different "value" information

In MQA:
- **K1 = K2 = K3 = K4**: All heads see same "key" patterns
- **V1 = V2 = V3 = V4**: All heads contribute same "value" information
- **Q1 ≠ Q2 ≠ Q3 ≠ Q4**: Still different, but limited by shared K,V

**Result**: Reduced capacity to capture diverse linguistic phenomena

---

## 7. Experimental Analysis

### 7.1 Models Compared
- **GPT-2 Medium** (355M parameters): Standard Multi-Head Attention
- **Falcon 1B**: Multi-Query Attention implementation

### 7.2 Key Matrix Visualization
**GPT-2 (MHA) Results**:
```
Head 0: [50 × 64] - Unique heat map pattern
Head 1: [50 × 64] - Different heat map pattern  
Head 2: [50 × 64] - Different heat map pattern
Head 3: [50 × 64] - Different heat map pattern
```

**Falcon (MQA) Results**:
```
Head 0: [50 × 64] - Shared heat map pattern
Head 1: [50 × 64] - SAME heat map pattern
Head 2: [50 × 64] - SAME heat map pattern  
Head 3: [50 × 64] - SAME heat map pattern
```

### 7.3 Attention Score Analysis
**Input**: "The quick brown fox jumps over the lazy dog" (9 tokens)

**GPT-2 Attention Scores** (16 heads):
- Each head shows **distinct attention patterns**
- High diversity across heads
- Different heads focus on different token relationships

**Falcon Attention Scores** (32 heads):  
- Many heads show **similar attention patterns**
- Reduced diversity due to shared K,V
- Less linguistic perspective capture

---

## 8. Trade-off Analysis

### 8.1 Memory vs Performance Trade-off

| Aspect | Standard MHA | Multi-Query Attention |
|--------|--------------|----------------------|
| **Memory Usage** | High (N × size) | Low (1 × size) |
| **KV Cache Size** | L×B×N×H×S×4 | L×B×1×H×S×4 |
| **Perspective Diversity** | High | Limited |
| **Model Performance** | Best | Degraded |
| **Inference Speed** | Slower | Faster |
| **Cost** | High | Low |

### 8.2 When to Use MQA
**Good for**:
- Memory-constrained environments
- Cost-sensitive applications  
- Speed-critical inference
- Simple tasks not requiring complex reasoning

**Not ideal for**:
- Complex language understanding
- Tasks requiring multiple perspectives
- High-performance requirements
- Research/academic applications

---

## 9. Why DeepSeek Didn't Use MQA

### 9.1 Performance Requirements
- **DeepSeek models**: Focus on high performance
- **Complex reasoning**: Requires diverse perspectives
- **Competitive landscape**: Cannot afford performance degradation

### 9.2 The Innovation Path
```
KV Cache Problem → MQA (memory fix, performance loss)
                ↓
            Need better solution
                ↓
        Grouped Query Attention → Multi-Head Latent Attention
```

### 9.3 DeepSeek's Approach
Instead of MQA, DeepSeek developed:
1. **Grouped Query Attention**: Balance between MHA and MQA
2. **Multi-Head Latent Attention**: Novel compression approach
3. **Maintains performance** while reducing memory

---

## 10. Key Insights and Takeaways

### 10.1 Core Concepts
1. **MQA = Shared K,V across heads, different Q per head**
2. **Memory reduction = Factor of N (number of heads)**
3. **Trade-off = Memory savings vs perspective diversity**
4. **First solution to KV Cache problem, but not the best**

### 10.2 Mathematical Summary
```
Standard: Memory ∝ N × other_factors
MQA:      Memory ∝ 1 × other_factors
Reduction: Factor of N

Standard: Perspectives = N (diverse)
MQA:      Perspectives < N (limited)
```

### 10.3 Historical Importance
- **First major solution** to KV Cache memory problem
- **Proved memory reduction possible** without architectural overhaul
- **Highlighted trade-offs** between memory and performance
- **Inspired better solutions** like Grouped Query Attention

---

## 11. Next Steps in the Series

### 11.1 Upcoming Topics
1. **Grouped Query Attention (GQA)**
   - Balance between MHA and MQA
   - Grouped sharing strategy
   - Better performance-memory trade-off

2. **Multi-Head Latent Attention**
   - DeepSeek's breakthrough innovation
   - Compression without significant performance loss
   - Key to DeepSeek's efficiency

### 11.2 Learning Path
```
KV Cache → MQA → GQA → Multi-Head Latent Attention
   ↓        ↓     ↓            ↓
Foundation → Memory → Balance → Innovation
Problem      Solution  Solution   Solution
```

---

## Summary

Multi-Query Attention represents the **first major attempt** to solve the KV Cache memory problem by:

**✅ Advantages:**
- **Dramatic memory reduction** (factor of N)
- **Faster inference** (~40% speedup)
- **Lower costs** for deployment
- **Simpler implementation**

**❌ Disadvantages:**
- **Reduced perspective diversity**
- **Performance degradation**
- **Limited linguistic understanding**
- **Defeats MHA purpose**

**Key Insight**: While MQA successfully addresses the memory problem, it introduces a new problem - performance degradation due to reduced head diversity. This trade-off motivated the development of more sophisticated solutions like Grouped Query Attention and ultimately DeepSeek's Multi-Head Latent Attention.

MQA serves as a crucial stepping stone in understanding how to balance memory efficiency with model performance in modern transformer architectures.

---
**Author: Ayushmaan Singh**
