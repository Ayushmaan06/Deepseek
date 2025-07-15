# Transformers: Complete Architecture Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Architecture](#core-architecture)
3. [Attention Mechanism](#attention-mechanism)
4. [Transformer Components](#transformer-components)
5. [Training Process](#training-process)
6. [Variants and Applications](#variants-and-applications)
7. [Mathematical Foundations](#mathematical-foundations)
8. [Implementation Details](#implementation-details)

## Introduction

### What is a Transformer?
The Transformer is a neural network architecture introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017). It revolutionized natural language processing by replacing recurrent and convolutional layers with self-attention mechanisms.

### Key Innovations
- **Self-Attention**: Processes all positions simultaneously
- **Parallelization**: No sequential dependencies like RNNs
- **Long-Range Dependencies**: Captures relationships across entire sequences
- **Scalability**: Efficient training on large datasets

### Why Transformers Matter
- Foundation for modern LLMs (GPT, BERT, T5)
- State-of-the-art performance across NLP tasks
- Transferable to computer vision, protein folding, etc.
- Enables few-shot and zero-shot learning

## Core Architecture

### High-Level Overview
```
Input Tokens → Input Embeddings → Encoder Stack → Decoder Stack → Output Probabilities
```

### Original Transformer Structure
```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────┤
│  INPUT PROCESSING                                           │
│  ┌─────────────┐    ┌──────────────────┐                   │
│  │ Tokenization│ -> │ Input Embeddings │                   │
│  └─────────────┘    └──────────────────┘                   │
│                              │                             │
├─────────────────────────────────────────────────────────────┤
│  ENCODER STACK (6 layers)                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ ┌─────────────────┐  ┌─────────────────┐               ││
│  │ │ Multi-Head      │  │ Feed Forward    │               ││
│  │ │ Self-Attention  │  │ Network         │               ││
│  │ └─────────────────┘  └─────────────────┘               ││
│  │        │                       │                       ││
│  │   Residual +             Residual +                    ││
│  │   Layer Norm             Layer Norm                    ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  DECODER STACK (6 layers)                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ ┌─────────────────┐  ┌─────────────────┐ ┌─────────────┐││
│  │ │ Masked          │  │ Cross-Attention │ │ Feed Forward│││
│  │ │ Self-Attention  │  │                 │ │ Network     │││
│  │ └─────────────────┘  └─────────────────┘ └─────────────┘││
│  │        │                       │               │       ││
│  │   Residual +             Residual +       Residual +   ││
│  │   Layer Norm             Layer Norm       Layer Norm   ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  OUTPUT PROCESSING                                          │
│  ┌─────────────┐    ┌──────────────────┐                   │
│  │ Linear      │ -> │ Softmax          │                   │
│  │ Projection  │    │ (Probabilities)  │                   │
│  └─────────────┘    └──────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### Input Processing

#### 1. Tokenization
**Purpose**: Convert text into discrete tokens
**Methods**:
- **Word-level**: "Hello world" → ["Hello", "world"]
- **Subword-level** (BPE): "Hello" → ["He", "llo"]
- **Character-level**: "Hi" → ["H", "i"]

**Vocabulary**: Fixed set of tokens (typically 30K-50K)

#### 2. Token Embeddings
**Matrix Shape**: [vocab_size × d_model]
- **vocab_size**: Number of unique tokens (e.g., 50,000)
- **d_model**: Embedding dimension (e.g., 512)

**Process**:
```python
token_ids = [15, 234, 1028]  # From tokenization
embeddings = embedding_matrix[token_ids]  # Lookup
# Result: [3, 512] tensor
```

#### 3. Positional Encoding
**Purpose**: Inject sequence order information
**Formula** (Sinusoidal):
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Addition**:
```python
input_embeddings = token_embeddings + positional_encodings
```

## Attention Mechanism

### Self-Attention Mathematics

#### Core Formula
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

#### Step-by-Step Process

**1. Linear Transformations**
```python
Q = X @ W_Q  # Query matrix
K = X @ W_K  # Key matrix  
V = X @ W_V  # Value matrix
```

**2. Attention Scores**
```python
scores = Q @ K.transpose(-2, -1)  # [seq_len, seq_len]
scores = scores / math.sqrt(d_k)  # Scaling
```

**3. Attention Weights**
```python
attention_weights = softmax(scores, dim=-1)
```

**4. Weighted Values**
```python
output = attention_weights @ V
```

#### Intuitive Understanding
- **Query**: "What am I looking for?"
- **Key**: "What do I have to offer?"
- **Value**: "What information do I actually provide?"

**Example**: "The cat sat on the mat"
- When processing "sat", query asks about actions
- Keys from all words respond with their relevance
- Values provide the actual information to aggregate

### Multi-Head Attention

#### Concept
Instead of single attention, run h parallel attention heads:
```python
head_i = Attention(Q_i, K_i, V_i)
MultiHead = Concat(head_1, ..., head_h) @ W_O
```

#### Benefits
- **Different perspectives**: Each head learns different relationships
- **Richer representations**: Capture multiple types of dependencies
- **Parallelization**: Heads computed independently

#### Typical Configuration
- **d_model**: 512
- **num_heads**: 8
- **d_k = d_v**: d_model / num_heads = 64

### Masked Self-Attention (Decoder)
**Purpose**: Prevent looking at future tokens during training

**Mask Application**:
```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
scores = scores.masked_fill(mask == 1, -1e9)
attention_weights = softmax(scores, dim=-1)
```

**Effect**: Attention weights for future positions become ~0

## Transformer Components

### 1. Layer Normalization
**Formula**: 
```
LayerNorm(x) = γ × (x - μ) / σ + β
```
Where:
- μ: mean across features
- σ: standard deviation across features
- γ, β: learnable parameters

**Purpose**:
- Stabilize training
- Reduce internal covariate shift
- Enable deeper networks

### 2. Feed-Forward Networks
**Architecture**:
```python
FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
```

**Dimensions**:
- Input: d_model (512)
- Hidden: d_ff (2048, typically 4×d_model)
- Output: d_model (512)

**Purpose**:
- Add non-linearity
- Process attended information
- Increase model capacity

### 3. Residual Connections
**Formula**:
```python
output = LayerNorm(x + Sublayer(x))
```

**Benefits**:
- Gradient flow in deep networks
- Prevent vanishing gradients
- Identity mapping preservation

### 4. Dropout
**Application**:
- After attention weights
- In feed-forward networks
- On embeddings

**Purpose**:
- Regularization
- Prevent overfitting
- Improve generalization

## Training Process

### 1. Teacher Forcing
**Training**: Use ground truth tokens as decoder input
**Inference**: Use model's own predictions

**Example**:
```
Target: "Hello world"
Training input: [START] "Hello"
Training target: "Hello" "world"
```

### 2. Loss Function
**Cross-Entropy Loss**:
```python
loss = -∑ y_true * log(y_pred)
```

**Sequence Level**:
```python
total_loss = ∑(i=1 to seq_len) CrossEntropy(output_i, target_i)
```

### 3. Optimization
**Adam Optimizer** with learning rate scheduling:
```python
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
```

**Warmup**: Gradually increase learning rate for stability

## Variants and Applications

### Encoder-Only Models
**Examples**: BERT, RoBERTa, DeBERTa
**Use Cases**: 
- Text classification
- Named entity recognition
- Question answering
- Sentiment analysis

**Architecture**: Stack of encoder layers only

### Decoder-Only Models  
**Examples**: GPT series, PaLM, LLaMA
**Use Cases**:
- Text generation
- Language modeling
- Code generation
- Conversational AI

**Architecture**: Stack of decoder layers with causal masking

### Encoder-Decoder Models
**Examples**: T5, BART, Transformer (original)
**Use Cases**:
- Machine translation
- Summarization
- Text-to-text tasks

**Architecture**: Encoder processes input, decoder generates output

### Vision Transformers (ViT)
**Adaptation**: 
- Images → patches → embeddings
- Same transformer architecture
- Classification token [CLS]

### Specialized Variants
- **Longformer**: Extended attention for long sequences
- **Reformer**: Memory-efficient attention
- **Linformer**: Linear attention complexity
- **Performer**: FAVOR+ attention approximation

## Mathematical Foundations

### Attention Complexity
**Time Complexity**: O(n² × d)
- n: sequence length
- d: model dimension

**Space Complexity**: O(n²) for attention matrix

### Scaling Laws
**Model Performance** scales with:
- Number of parameters
- Dataset size  
- Compute budget

**Empirical relationship**:
```
Loss ∝ (Parameters)^(-α) × (Data)^(-β) × (Compute)^(-γ)
```

### Theoretical Properties
**Universal Approximation**: Transformers can approximate any sequence-to-sequence function
**Expressivity**: Attention can implement various algorithms
**Generalization**: Strong inductive biases for language

## Implementation Details

### Memory Optimization
**Gradient Checkpointing**: Trade compute for memory
**Mixed Precision**: FP16/BF16 training
**Model Parallelism**: Split across devices
**Sequence Parallelism**: Parallelize attention computation

### Computational Tricks
**Flash Attention**: Memory-efficient attention
**Key-Value Caching**: Reuse computations in generation
**Quantization**: Reduce precision for inference

### Training Stability
**Gradient Clipping**: Prevent exploding gradients
**Layer Normalization**: Stabilize activations
**Warmup Schedule**: Careful learning rate management

### Hyperparameter Guidelines
**Model Size**:
- Small: d_model=512, heads=8, layers=6
- Base: d_model=768, heads=12, layers=12  
- Large: d_model=1024, heads=16, layers=24

**Training**:
- Batch size: 32-512 sequences
- Learning rate: 1e-4 to 5e-4
- Dropout: 0.1
- Weight decay: 0.01

## Performance Characteristics

### Advantages
✅ **Parallelization**: All positions processed simultaneously
✅ **Long-range dependencies**: Direct connections between distant tokens
✅ **Interpretability**: Attention weights show model focus
✅ **Transfer learning**: Pre-trained models adapt well
✅ **Scalability**: Performance improves with size

### Limitations
❌ **Quadratic complexity**: O(n²) attention computation
❌ **Memory requirements**: Large attention matrices
❌ **Limited context**: Fixed maximum sequence length
❌ **Data hungry**: Requires large datasets for training
❌ **Computational cost**: Expensive training and inference

## Modern Developments

### Efficiency Improvements
- **Sparse Attention**: Reduce O(n²) complexity
- **Low-rank Approximations**: Compress attention matrices
- **Local Attention**: Focus on nearby tokens
- **Hierarchical Attention**: Multi-scale processing

### Architectural Innovations
- **Switch Transformer**: Sparse expert models
- **PaLM**: Pathways language model scaling
- **GLaM**: Generalist language model with MoE
- **Chinchilla**: Optimal compute-data trade-offs

### Future Directions
- **Hardware co-design**: Custom chips for transformers
- **Algorithmic improvements**: Better attention mechanisms
- **Multimodal transformers**: Text, vision, audio integration
- **Efficient architectures**: Reduce computational requirements

This comprehensive guide covers the fundamental concepts, mathematical foundations, and practical aspects of transformer architectures, providing a complete reference for understanding these revolutionary models.

**Author: Ayushmaan Singh**