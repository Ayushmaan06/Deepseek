# Build DeepSeek from Scratch - Self-Attention with Trainable Weights

**By: Ayushmaan Singh**

## Overview
This lecture provides a detailed, step-by-step explanation of how self-attention works mathematically, focusing on the transformation from input embeddings to context vectors through trainable weight matrices.

## Recap: Why Self-Attention?

### The Core Problem
- **Input Embeddings**: Contain token meaning + position, but NO neighbor information
- **Context Vectors**: Enriched representations including relationships with other tokens
- **Goal**: Transform input embeddings → context vectors to capture inter-token relationships

### The Dot Product Limitation
**Example**: "The dog chased the ball but it couldn't catch it"
- Simple dot product cannot distinguish that second "it" refers to "ball" not "dog"
- Need more sophisticated mechanism to capture contextual relationships

## The Trainable Weight Matrix Solution

### Philosophy Behind the Approach
**Human Limitation**: Cannot derive exact formulas for attention (unlike physics)
**Deep Learning Solution**: 
1. Introduce trainable parameters (matrices)
2. Initialize randomly
3. Let backpropagation discover optimal relationships
4. Same principle used in CNNs, other neural architectures

### Three Key Matrices
1. **WQ (Query Weight Matrix)**: Transforms input to query space
2. **WK (Key Weight Matrix)**: Transforms input to key space  
3. **WV (Value Weight Matrix)**: Transforms input to value space

**Purpose**: Project input embeddings into different dimensional spaces to increase expressivity

## Step-by-Step Self-Attention Process

### Example Setup
**Sentence**: "The next day is bright"
- **5 tokens**: the, next, day, is, bright
- **Input embedding dimension**: 8 (chosen for simplicity)
- **Output dimension**: 4 (can be different from input)

### Matrix Dimensions
```
Input Embedding Matrix: [5 × 8]
- 5 rows (tokens)
- 8 columns (embedding dimension)

Weight Matrices: [8 × 4]
WQ, WK, WV: All same dimensions
- 8 rows (must match input embedding dimension)
- 4 columns (output dimension - flexible)
```

### Step 1: Generate Query, Key, Value Vectors

**Matrix Multiplications**:
```python
Query vectors = Input_embeddings @ WQ    # [5×8] @ [8×4] = [5×4]
Key vectors = Input_embeddings @ WK      # [5×8] @ [8×4] = [5×4]  
Value vectors = Input_embeddings @ WV    # [5×8] @ [8×4] = [5×4]
```

**Interpretation**:
- Each row corresponds to one token
- Now operating in QKV space (dimension 4) instead of input space (dimension 8)
- This projection enables capturing complex relationships

### Step 2: Calculate Attention Scores

**Matrix Multiplication**:
```python
Attention_scores = Query @ Key.transpose()    # [5×4] @ [4×5] = [5×5]
```

**Understanding the 5×5 Matrix**:
- **Row i**: Attention scores between token i (query) and all tokens (keys)
- **Element (i,j)**: How much token i attends to token j

**Example - Row 2 (token "next")**:
- α₂₁: attention between "next" and "the"
- α₂₂: attention between "next" and "next"  
- α₂₃: attention between "next" and "day"
- α₂₄: attention between "next" and "is"
- α₂₅: attention between "next" and "bright"

### Step 3: Scaling and Normalization

#### Problem with Raw Scores
**Issue**: Attention scores don't sum to 1, can't make percentage statements
**Need**: Convert to probabilities (0-1 range, sum to 1)

#### Scaling Before Softmax
**Critical Step**: Divide by √(key_dimension) before applying softmax

**Why √d_k Scaling?**

**Variance Problem**:
- Dot product of random vectors has variance = dimension
- Higher dimensions → higher variance → unstable values
- Large values before softmax → "peaky" distributions (overconfident predictions)

**Mathematical Proof**:
```python
# Without scaling
variance_before = dimension  # If dim=100, variance=100

# With scaling  
variance_after = dimension / √dimension = √dimension / 1 = 1
```

**Dice Analogy**:
- Rolling 1 die: predictable outcomes (variance ≈ 2.9)
- Rolling 100 dice and summing: unpredictable (variance ≈ 290)
- Dot product without scaling = rolling more dice
- Scaling normalizes variance to 1

**Implementation**:
```python
scaled_scores = attention_scores / math.sqrt(key_dimension)  # Divide by √4 = 2
attention_weights = softmax(scaled_scores)
```

#### Softmax Operation
**Formula for each element**:
```
softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ
```

**Properties**:
- All values between 0 and 1
- All values in each row sum to 1
- Emphasizes larger values, suppresses smaller ones

**Result**: Attention weights - normalized, interpretable probabilities

### Step 4: Compute Context Vectors

**Final Multiplication**:
```python
Context_vectors = Attention_weights @ Value_vectors    # [5×5] @ [5×4] = [5×4]
```

#### Intuitive Understanding
**For token "next" context vector**:
1. Take attention weights: [0.1, 0.5, 0.2, 0.1, 0.1]
2. Scale each value vector by corresponding weight:
   - "the" value vector × 0.1
   - "next" value vector × 0.5  
   - "day" value vector × 0.2
   - "is" value vector × 0.1
   - "bright" value vector × 0.1
3. Sum all scaled vectors = context vector for "next"

**Visual Representation**:
- Blue vectors: original input embeddings
- Green vectors: scaled by attention weights
- Final sum: enriched context vector

## Key Differences: Attention Scores vs. Attention Weights

| Aspect | Attention Scores | Attention Weights |
|--------|------------------|-------------------|
| Normalization | No | Yes |
| Range | Unbounded | [0, 1] |
| Row Sum | Arbitrary | 1.0 |
| Interpretation | Raw similarity | Probability distribution |
| Stability | Can be unstable | Stable after scaling |

## Complete Mathematical Summary

### Full Self-Attention Formula
```python
def self_attention(X):
    # Step 1: Project to QKV spaces
    Q = X @ WQ
    K = X @ WK  
    V = X @ WV
    
    # Step 2: Compute attention scores
    scores = Q @ K.T
    
    # Step 3: Scale and normalize
    scaled_scores = scores / sqrt(d_k)
    attention_weights = softmax(scaled_scores)
    
    # Step 4: Weighted combination
    context = attention_weights @ V
    
    return context
```

### Dimensions Throughout Process
```
Input embeddings:    [seq_len × d_in]     [5 × 8]
Query vectors:       [seq_len × d_out]    [5 × 4]
Key vectors:         [seq_len × d_out]    [5 × 4]
Value vectors:       [seq_len × d_out]    [5 × 4]
Attention scores:    [seq_len × seq_len]  [5 × 5]
Attention weights:   [seq_len × seq_len]  [5 × 5]
Context vectors:     [seq_len × d_out]    [5 × 4]
```

## Implementation Code Structure

```python
class SelfAttention:
    def __init__(self, d_in, d_out):
        self.WQ = torch.randn(d_in, d_out)  # Query weight matrix
        self.WK = torch.randn(d_in, d_out)  # Key weight matrix  
        self.WV = torch.randn(d_in, d_out)  # Value weight matrix
        
    def forward(self, X):
        # Step 1: Generate Q, K, V
        Q = X @ self.WQ
        K = X @ self.WK
        V = X @ self.WV
        
        # Step 2: Attention scores
        scores = Q @ K.transpose(-2, -1)
        
        # Step 3: Scale and softmax
        d_k = K.shape[-1]
        scaled_scores = scores / math.sqrt(d_k)
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        
        # Step 4: Context vectors
        context = attention_weights @ V
        
        return context
```

## Why This Works

### Expressivity Gain
- **Input space**: Limited to semantic similarity
- **QKV spaces**: Learnable transformations capture complex relationships
- **Training**: Backpropagation optimizes matrices for specific tasks

### Context Enrichment
**Before**: Each token knows only about itself
**After**: Each token incorporates information from all other tokens
- Weighted by learned importance (attention weights)
- Enables understanding of relationships, dependencies, context

## Connection to Transformer Architecture

### Position in Transformer Block
1. Layer Normalization
2. **Multi-Head Self-Attention** ← This lecture's focus
3. Dropout + Residual connection
4. Layer Normalization  
5. Feed-Forward Network
6. Dropout + Residual connection

### The "Magic" Component
- Most crucial part of transformer architecture
- Enables language understanding capabilities
- Powers modern LLMs (GPT, BERT, etc.)

## Next Steps Preview

### Upcoming Topics
1. **Causal Attention**: Masking future tokens for autoregressive models
2. **Multi-Head Attention**: Multiple attention mechanisms in parallel
3. **Key-Value Caching**: Optimization for inference
4. **Multi-Head Latent Attention**: DeepSeek's innovation

### Building Toward DeepSeek
This foundational understanding is essential for:
- Understanding multi-head attention variants
- Appreciating efficiency optimizations
- Grasping DeepSeek's Multi-Head Latent Attention innovation

## Key Takeaways

1. **Self-attention transforms isolated token representations into context-aware representations**
2. **Trainable weight matrices (Q, K, V) enable learning complex relationships**
3. **Scaling by √d_k is crucial for training stability**
4. **Context vectors are weighted combinations of value vectors**
5. **This mechanism powers the transformer revolution in NLP**

The mathematical elegance lies in how simple matrix operations, when combined with learnable parameters, can capture the complex relationships that make language understanding possible.

---
**Author: Ayushmaan Singh**