# Build DeepSeek from Scratch - Causal Attention Mechanism

**By: Ayushmaan Singh**

## Overview
This lecture covers causal attention (also known as masked attention), a crucial modification of self-attention that prevents models from accessing future tokens during training and inference. This is essential for autoregressive language models.

## Recap: Self-Attention Foundation

### What We've Learned So Far
**Self-Attention Process**:
1. **Step 1**: Input embeddings → Query, Key, Value vectors (via trainable matrices WQ, WK, WV)
2. **Step 2**: Attention scores = Query @ Key.transpose()
3. **Step 3**: Scale by √d_k and apply softmax → Attention weights
4. **Step 4**: Context vectors = Attention weights @ Value vectors

### Context Vector Enrichment
- **Input embeddings**: Token semantics only
- **Context vectors**: Semantics + relationships with all other tokens
- **Result**: Richer representations enabling language understanding

## Next Token Prediction: The Foundation

### Harry Potter Training Example
**Text**: "Mr and Mrs Dudley lived off..."

**Training Data Construction**:
```
Context size = 4, Stride = 1

Input sequences:
- Sequence 1: [Mr, and, Mrs, Dudley]
- Sequence 2: [and, Mrs, Dudley, lived]
- Sequence 3: [Mrs, Dudley, lived, off]

Target sequences (shifted right by 1):
- Target 1: [and, Mrs, Dudley, lived]
- Target 2: [Mrs, Dudley, lived, off]
- Target 3: [Dudley, lived, off, ...]
```

### Multiple Prediction Tasks Per Sequence
**Within one input-output pair**: [Mr, and, Mrs, Dudley] → [and, Mrs, Dudley, lived]

**Four prediction tasks**:
1. Input: "Mr" → Predict: "and"
2. Input: "Mr and" → Predict: "Mrs"  
3. Input: "Mr and Mrs" → Predict: "Dudley"
4. Input: "Mr and Mrs Dudley" → Predict: "lived"

### The Critical Constraint
**Key Insight**: For any output token, the input consists ONLY of tokens that come before it.

**"Cannot cheat, cannot look into the future"**

## Why Is Causal Attention Necessary?

### The Fundamental Problem: Train-Test Mismatch

**During Training**: We have the entire sentence available
- Input: "The cat sat on the"
- Target: "cat sat on the mat"
- Model sees ALL tokens at once

**During Inference**: We generate one token at a time
- Step 1: Input "The" → Generate "cat"
- Step 2: Input "The cat" → Generate "sat"  
- Step 3: Input "The cat sat" → Generate "on"
- Step 4: Input "The cat sat on" → Generate "the"
- Step 5: Input "The cat sat on the" → Generate "mat"

### The Cheating Problem

**Without Causal Attention** (using standard self-attention):
```python
# Training time - model sees future tokens
Input:  ["The", "cat", "sat", "on", "the"]
Target: ["cat", "sat", "on", "the", "mat"]

# When predicting "cat", model can see:
# - "The" (past) ✓ 
# - "cat" (current) ✓
# - "sat" (future) ❌ CHEATING!
# - "on" (future) ❌ CHEATING!
# - "the" (future) ❌ CHEATING!
```

**The Result**: Model learns to cheat by using future information that won't be available during actual text generation.

### Real-World Analogy: Exam Cheating

**Imagine a student taking a test**:
- **Training with cheating**: Student sees all answers while answering each question
- **Test day**: Student must answer questions one by one without seeing future answers
- **Result**: Student fails because they never learned to answer questions independently

**Same problem with language models**:
- **Training without causal attention**: Model sees future words while predicting current word
- **Inference**: Model must generate words one by one without seeing future
- **Result**: Model performs poorly because it relied on information that's not available

### Concrete Example: Why Models Break Without Causal Attention

**Sentence**: "The weather is beautiful today"

**Standard Self-Attention Training**:
```python
# Predicting "weather" - model sees ALL tokens
Attention weights for "weather":
- "The": 0.1 (past - legitimate)
- "weather": 0.2 (current - legitimate)  
- "is": 0.3 (future - CHEATING!)
- "beautiful": 0.2 (future - CHEATING!)
- "today": 0.2 (future - CHEATING!)

# Model learns: "weather" comes after "The" AND before "is beautiful today"
```

**Inference Reality**:
```python
# Predicting "weather" - model only has past context
Available tokens: ["The"]
Missing context: ["is", "beautiful", "today"] # NOT AVAILABLE!

# Model expects future context but doesn't have it
# Result: Poor predictions, incoherent text
```

### The Training-Inference Consistency Problem

**Key Issue**: Models must behave identically during training and inference

**Without Causal Attention**:
- Training: Model uses bidirectional context (past + future)
- Inference: Model only has unidirectional context (past only)
- **Mismatch**: Model trained on different data distribution than it sees at test time

**With Causal Attention**:
- Training: Model uses only past context (matches inference)
- Inference: Model uses only past context
- **Consistency**: Same information available in both phases

### Mathematical Proof of Necessity

**Training Loss vs Inference Performance**:

```python
# Without causal attention
Training: P(word_t | word_1, word_2, ..., word_t-1, word_t+1, ..., word_n)
Inference: P(word_t | word_1, word_2, ..., word_t-1)  # Missing future context

# With causal attention  
Training: P(word_t | word_1, word_2, ..., word_t-1)
Inference: P(word_t | word_1, word_2, ..., word_t-1)  # Perfect match!
```

### What Happens Without Causal Attention?

1. **Exposure Bias**: Model never learns to handle its own prediction errors
2. **Distribution Shift**: Training and inference data distributions differ
3. **Cascading Errors**: Early mistakes compound in autoregressive generation
4. **Poor Generalization**: Model can't generate coherent long sequences

### Empirical Evidence

**Experiments show**:
- Models without causal attention: High training accuracy, poor generation quality
- Models with causal attention: Consistent performance, coherent text generation
- The gap increases with sequence length

## The Problem with Standard Self-Attention

### Information Leakage Issue
**Standard Self-Attention**: Each token attends to ALL other tokens (past and future)

**Example Sentence**: "Your journey starts with one step"
- Token "journey" (position 2) attends to:
  - "your" (position 1) ✓ Past - Available
  - "journey" (position 2) ✓ Current - Available  
  - "starts" (position 3) ❌ Future - Should NOT be available
  - "with" (position 4) ❌ Future - Should NOT be available
  - "one" (position 5) ❌ Future - Should NOT be available
  - "step" (position 6) ❌ Future - Should NOT be available

### Attention Matrix Analysis
**6×6 Attention Scores Matrix**:
```
       your  journey  starts  with  one  step
your    [✓      ❌      ❌     ❌    ❌    ❌ ]
journey [✓      ✓      ❌     ❌    ❌    ❌ ]
starts  [✓      ✓      ✓     ❌    ❌    ❌ ]
with    [✓      ✓      ✓     ✓    ❌    ❌ ]
one     [✓      ✓      ✓     ✓    ✓    ❌ ]
step    [✓      ✓      ✓     ✓    ✓    ✓  ]
```

**Problem**: All ❌ positions represent future information that should be masked

## Causal Attention: The Solution

### Definition
**Causal Attention** (also called Masked Attention, Autoregressive Attention, Unidirectional Attention):
- Restricts model to consider only previous and current tokens
- Masks out future tokens in attention computation
- Ensures "no cheating" during training and inference

### Core Principle
**For each token**: Only attend to tokens at current position or before
- Token 1: Can only attend to token 1
- Token 2: Can only attend to tokens 1-2
- Token 3: Can only attend to tokens 1-3
- Token n: Can only attend to tokens 1-n

### Mathematical Implementation
**Goal**: Set all elements above the diagonal in attention matrix to zero

## Two Implementation Strategies

### Strategy 1: Post-Softmax Masking (Less Efficient)

**Process**:
1. Compute attention scores: Q @ K.T
2. Apply scaling and softmax → attention weights
3. Set upper triangular elements to zero
4. Re-normalize rows to sum to 1

**Problem**: Requires two normalization steps

**Example**:
```python
# After softmax
attention_weights = [[0.1, 0.3, 0.4, 0.2],
                    [0.2, 0.5, 0.2, 0.1],
                    [0.1, 0.3, 0.4, 0.2],
                    [0.25, 0.25, 0.25, 0.25]]

# Apply upper triangular mask (set to zero)
masked_weights = [[0.1, 0.0, 0.0, 0.0],
                  [0.2, 0.5, 0.0, 0.0],
                  [0.1, 0.3, 0.4, 0.0],
                  [0.25, 0.25, 0.25, 0.25]]

# Re-normalize rows to sum to 1
normalized = [[1.0, 0.0, 0.0, 0.0],
              [0.29, 0.71, 0.0, 0.0],
              [0.125, 0.375, 0.5, 0.0],
              [0.25, 0.25, 0.25, 0.25]]
```

### Strategy 2: Pre-Softmax Masking (More Efficient) ⭐

**Process**:
1. Compute attention scores: Q @ K.T
2. Set upper triangular elements to **negative infinity**
3. Apply scaling and softmax (single normalization)

**Why Negative Infinity?**
```python
e^(-∞) = 0
```
When softmax encounters -∞, it naturally becomes 0 after exponentiation

**Example**:
```python
# Attention scores
scores = [[6.0, 2.0, 1.0, 3.0],
          [0.1, 1.8, 4.0, 2.0],
          [2.0, 1.1, 1.2, 0.5],
          [1.5, 2.2, 1.8, 2.0]]

# Apply negative infinity mask
masked_scores = [[6.0, -∞, -∞, -∞],
                 [0.1, 1.8, -∞, -∞],
                 [2.0, 1.1, 1.2, -∞],
                 [1.5, 2.2, 1.8, 2.0]]

# Softmax automatically handles -∞ → 0
# Single normalization step, rows sum to 1
```

### Implementation Code
```python
def create_causal_mask(seq_len):
    # Create upper triangular matrix of 1s
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask

def apply_causal_mask(attention_scores):
    seq_len = attention_scores.size(-1)
    mask = create_causal_mask(seq_len)
    
    # Replace upper triangular elements with -∞
    masked_scores = attention_scores.masked_fill(mask == 1, float('-inf'))
    
    # Apply softmax (single normalization)
    attention_weights = F.softmax(masked_scores, dim=-1)
    return attention_weights
```

## Dropout in Attention

### The Lazy Neuron Problem
**Issue**: Some neurons/attention weights don't learn effectively
- Certain weights dominate during training
- Other weights become "lazy" and contribute minimally
- Poor generalization to new data

### Dropout Solution
**Mechanism**: Randomly set attention weights to zero during training
- **Dropout rate**: Percentage of weights to randomly zero (e.g., 50%)
- **Random selection**: Different weights dropped each iteration
- **Scaling**: Remaining weights scaled up to maintain balance

### Benefits
1. **Forces learning**: Lazy weights must contribute when dominant weights are dropped
2. **Prevents overfitting**: Reduces memorization of training patterns
3. **Improves generalization**: Model becomes more robust

### Mathematical Implementation
```python
# Before dropout
attention_weights = [[0.6, 0.4, 0.0],
                    [0.3, 0.5, 0.2],
                    [0.2, 0.3, 0.5]]

# After 50% dropout (random)
dropped_weights = [[1.2, 0.0, 0.0],  # Remaining weights scaled by 1/(1-0.5) = 2
                   [0.0, 1.0, 0.0],
                   [0.0, 0.6, 1.0]]
```

### Group Project Analogy
- **Normal case**: 2 people do all work, others are lazy
- **Dropout**: Hardworking people "fall sick" randomly
- **Result**: Lazy people forced to contribute and learn

## Complete Causal Attention Implementation

### Class Structure
```python
class CausalAttention:
    def __init__(self, d_in, d_out, context_length, dropout_rate=0.0):
        # Trainable weight matrices
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)  
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Register causal mask as buffer
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        # Step 1: Generate Q, K, V
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)
        
        # Step 2: Attention scores  
        attention_scores = Q @ K.transpose(-2, -1)
        
        # Step 3: Apply causal mask
        seq_len = attention_scores.size(-1)
        masked_scores = attention_scores.masked_fill(
            self.mask[:seq_len, :seq_len] == 1, float('-inf')
        )
        
        # Step 4: Scale and softmax
        d_k = K.size(-1)
        attention_weights = F.softmax(masked_scores / math.sqrt(d_k), dim=-1)
        
        # Step 5: Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Step 6: Compute context vectors
        context_vectors = attention_weights @ V
        
        return context_vectors
```

### Key Implementation Details

#### Tensor Dimensions
```python
# Input batch shape: [batch_size, seq_len, d_in]
# Example: [2, 6, 3] = 2 batches, 6 tokens, 3-dim embeddings

# Throughout processing:
Q, K, V: [batch_size, seq_len, d_out]  # [2, 6, 2]
attention_scores: [batch_size, seq_len, seq_len]  # [2, 6, 6]
attention_weights: [batch_size, seq_len, seq_len]  # [2, 6, 6]
context_vectors: [batch_size, seq_len, d_out]  # [2, 6, 2]
```

#### Register Buffer Usage
```python
self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
```
**Benefits**:
- Automatically moves mask to same device as model
- Avoids device mismatch errors during training
- Best practice for fixed tensors

## Comparison: Self-Attention vs Causal Attention

| Aspect | Self-Attention | Causal Attention |
|--------|----------------|------------------|
| **Token visibility** | All tokens | Only past + current |
| **Attention matrix** | Full matrix | Lower triangular |
| **Future information** | Accessible | Masked out |
| **Training complexity** | Simpler | Requires masking |
| **Use case** | Bidirectional tasks (BERT) | Autoregressive tasks (GPT) |
| **Computational overhead** | Lower | Slightly higher |

## Practical Example

### Input Sentence: "Your journey starts with one step"

**Self-Attention Matrix** (all 1s represent attention):
```
       your  journey  starts  with  one  step
your   [ 1      1      1      1    1    1  ]
journey[ 1      1      1      1    1    1  ]
starts [ 1      1      1      1    1    1  ]
with   [ 1      1      1      1    1    1  ]
one    [ 1      1      1      1    1    1  ]
step   [ 1      1      1      1    1    1  ]
```

**Causal Attention Matrix** (0s represent masked positions):
```
       your  journey  starts  with  one  step
your   [ 1      0      0      0    0    0  ]
journey[ 1      1      0      0    0    0  ]
starts [ 1      1      1      0    0    0  ]
with   [ 1      1      1      1    0    0  ]
one    [ 1      1      1      1    1    0  ]
step   [ 1      1      1      1    1    1  ]
```

## Key Takeaways

### Core Principles
1. **No future information**: Tokens can only attend to past and current positions
2. **Upper triangular masking**: Set elements above diagonal to zero/negative infinity
3. **Efficient implementation**: Use pre-softmax masking with negative infinity
4. **Dropout enhancement**: Randomly mask weights to improve generalization

### Implementation Strategy
1. **Start with attention scores** (not weights)
2. **Apply negative infinity mask** to upper triangular elements
3. **Single softmax operation** (efficient)
4. **Add dropout** for regularization
5. **Compute context vectors** as usual

### Why This Matters
- **Foundation for GPT**: All autoregressive language models use causal attention
- **Training-inference consistency**: Same masking pattern in both phases
- **Next token prediction**: Ensures model can't cheat during training
- **Prepares for multi-head attention**: Next step in our journey to MLA
- **Prevents exposure bias**: Model learns to handle sequential generation realistically
- **Eliminates train-test mismatch**: Consistent information access in both phases

### The Bottom Line
**Causal attention isn't just a nice-to-have feature - it's absolutely essential for any model that needs to generate text sequentially.** Without it, models would be like students who can only solve problems when they can see the answer key, but fail completely when they have to work independently.

This is why every major autoregressive language model (GPT, BERT-decoder, T5-decoder, etc.) uses causal attention. It's not an optimization - it's a fundamental requirement for coherent text generation.

## Next Steps
- **Multi-head attention**: Multiple attention mechanisms in parallel
- **Key-value caching**: Optimization for efficient inference
- **Multi-head latent attention**: DeepSeek's innovation

This masking mechanism is fundamental to how modern language models like GPT work, ensuring they learn to predict tokens based only on available context, not by cheating with future information.

---
**Author: Ayushmaan Singh**