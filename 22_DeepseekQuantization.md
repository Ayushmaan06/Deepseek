# DeepSeek Quantization: FP8 Training Methods

**By: Ayushmaan Singh**

## Overview

DeepSeek V3 implements sophisticated quantization methods in their FP8 training framework to reduce memory requirements and improve computational efficiency while maintaining model accuracy. This lecture covers the first two major quantization techniques implemented by DeepSeek.

## Five Major Quantization Aspects in DeepSeek V3

1. **Mixed Precision Framework** ✅ (Covered)
2. **Fine-Grained Quantization** ✅ (Covered) 
3. **Increasing Accumulation Precision** ✅ (Covered)
4. **Mantissa Over Exponents** ✅ (Covered)
5. **Online Quantization** ✅ (Covered)

## Fundamental Quantization Concepts

### Core Principle
- **Parameter Space**: Every parameter in an LLM takes up memory space
- **Precision Trade-off**: Higher precision (32-bit) = more memory, Lower precision (8-bit) = less memory
- **Goal**: Reduce parameter precision from 32-bit to 8-bit or 16-bit while maintaining acceptable accuracy

### Data Type Formats

| Format | Bits | Range | Characteristics |
|--------|------|-------|----------------|
| **FP32** | 32 | Full range | High precision, high memory |
| **FP16** | 16 | Reduced range | Lower precision than FP32 |
| **BF16** | 16 | Same as FP32 | Brain Float 16 - same range as FP32 but 16 bits |
| **INT8** | 8 | -127 to 127 | Very small range, minimal memory |
| **FP8** | 8 | Very limited | Extremely low precision, fastest computation |

### Basic Quantization Process
**Example: FP32 → INT8**
```
Original numbers: [2.5, 3.7, 10.8, 1.2]
Step 1: Divide by maximum (10.8)
Step 2: Multiply by 127 (INT8 max)
Result: Quantized values ready for INT8 storage
```

## 1. Mixed Precision Framework

### Neural Network Operation
**Basic Formula**: `y = w × x`
- **x**: Input activations
- **w**: Weight parameters  
- **y**: Output activations

### Forward Propagation (F-Prop)

#### Input Processing
- **Storage**: Inputs stored as **BF16**
- **Computation**: Converted to **FP8** on-the-fly
- **Memory Benefit**: FP8 uses less memory than BF16

#### Weight Processing
- **Master Storage**: Weights stored as **FP32** or **BF16** (high precision)
- **Computation**: Converted to **FP8** on-the-fly for multiplication
- **Stability**: Master weights remain high precision

#### Output Processing
- **Initial Computation**: `FP8 × FP8` computed as **FP32**
- **Storage**: Result stored as **BF16**
- **Rationale**: FP32 computation ensures numerical stability, BF16 storage saves memory

### Backward Propagation

#### Input Gradients (dL/dx)
**Formula**: `dL/dx = dL/dz × W^T`
- **dL/dz**: Stored as BF16, converted to FP8 for computation
- **Weights**: Converted to FP8 on-the-fly
- **Computation**: FP32 initially, stored as BF16

#### Weight Gradients (dL/dw)
**Formula**: `dL/dw = x^T × dL/dz`
- **x**: Input stored as FP8
- **dL/dz**: Converted from BF16 to FP8
- **Critical**: Weight gradients stored as **FP32** (NOT converted to lower precision)
- **Reason**: Weight updates require highest precision for training stability

#### Weight Updates
**Process**:
1. **Master weights**: Stored as FP32
2. **Gradients**: Computed as FP32
3. **Update**: `W_new = W_old - learning_rate × dL/dw`
4. **Storage**: Updated weights stored as FP32
5. **Conversion**: Converted to BF16 or FP8 as needed for next iteration

### Hardware Utilization

#### Tensor Cores
- **Purpose**: Handle FP8 × FP8 multiplications
- **Benefit**: Extremely fast low-precision operations
- **Usage**: All GEMM (General Matrix Multiplication) operations

#### CUDA Cores
- **Purpose**: Upscaling operations (FP8 → FP32)
- **Function**: Element-wise scaling and dequantization
- **Stability**: Ensures numerical stability in final outputs

### Precision Retention Strategy

#### High Precision Components (FP32/BF16)
DeepSeek maintains high precision for sensitive operations:
- **Embedding modules** (token and positional embeddings)
- **Output head** (vocabulary projection)
- **MoE gating modules**
- **Normalization operators**
- **Attention operators**

#### Low Precision Components (FP8)
- **Weight multiplications** in standard layers
- **Activation computations** during forward/backward pass
- **Intermediate calculations** that don't affect stability

### Advantages of Mixed Precision
1. **Memory Reduction**: FP8 operations use significantly less memory
2. **Computational Speed**: Tensor cores excel at FP8 operations
3. **Numerical Stability**: Critical operations remain high precision
4. **Training Stability**: Master weights and gradients in FP32

## 2. Fine-Grained Quantization

### The Outlier Problem

#### Traditional Quantization Issue
**Example without outlier**:
```
Original: [2, 3, 4]
Max value: 4
Quantized: [2/4×127, 3/4×127, 4/4×127] = [63.5, 95.25, 127]
Dequantized: [2.0, 3.0, 4.0] ✅ Perfect recovery
```

**Example with outlier**:
```
Original: [2, 3, 4, 500]
Max value: 500
Quantized: [2/500×127, 3/500×127, 4/500×127, 500/500×127]
         = [0.508, 0.762, 1.016, 127]
```

#### Precision Loss Problem
- **FP8 Limitation**: Cannot represent 0.508, 0.762, 1.016 precisely
- **Rounded Values**: Become [0.5, 0.75, 1.0, 127]
- **Dequantization Error**: Recovery gives [1.97, 2.95, 3.94, 500]
- **Impact**: Small values lose significant precision due to single outlier

### Fine-Grained Solution

#### Chunking Strategy
**Vector Quantization**:
- **Original**: 256-dimensional vector
- **Chunks**: Divide into 2 chunks of 128 elements each
- **Scaling**: Each chunk gets its own scaling factor

**Example**:
```
Chunk 1: [values with max = 20] → Scaled by 20
Chunk 2: [values with max = 0.1] → Scaled by 0.1
```

#### Matrix Quantization
**Weight Matrix**: 256×256 matrix
- **Division**: 4 chunks of 128×128 each
- **Scaling**: Each chunk (W11, W12, W21, W22) has separate scaling factor
- **Isolation**: Outlier in W11 doesn't affect W12, W21, W22

### Implementation Details

#### Activation Inputs
- **Representation**: Green color coding in DeepSeek diagrams
- **Chunk Size**: Typically 128 elements (NC=128)
- **Scaling Factors**: Blue color coding shows different scaling factors
- **Benefit**: Localized precision preservation

#### Weight Matrices
- **Representation**: Different color coding for chunks
- **Chunk Size**: 128×128 sub-matrices
- **Scaling Factors**: Purple color coding shows separate scaling factors
- **Advantage**: Outlier containment within chunks

### Hardware Processing

#### Tensor Core Operations
- **Input**: Green rectangles (activations)
- **Weights**: Yellow rectangles (weight matrices)
- **Operation**: FP8 × FP8 = FP8 result
- **Location**: NVIDIA Tensor Cores
- **Speed**: Optimized for low-precision operations

#### CUDA Core Operations
- **Function**: Upscaling and dequantization
- **Input**: FP8 results from tensor cores
- **Process**: Element-wise scaling with chunk-specific factors
- **Output**: FP32 values for stability
- **Location**: CUDA Cores

### Advantages of Fine-Grained Quantization

1. **Outlier Isolation**: Outliers affect only their local chunk
2. **Precision Preservation**: Small values maintain accuracy
3. **Scalability**: Works with any tensor size
4. **Hardware Efficiency**: Optimized for modern GPU architectures
5. **Minimal Accuracy Loss**: Better quantization-dequantization recovery

## Technical Implementation

### Schematic Understanding
The DeepSeek diagrams show four distinct components:
1. **Input Scaling**: Different scaling factors for activation chunks
2. **Weight Scaling**: Different scaling factors for weight matrix chunks  
3. **Tensor Core Computation**: FP8 operations for speed
4. **CUDA Core Upscaling**: Precision recovery for stability

### Memory and Computation Benefits
- **Memory Reduction**: Up to 4x reduction (FP32 → FP8)
- **Speed Improvement**: Significant speedup on tensor cores
- **Stability Maintenance**: Critical operations remain high precision
- **Accuracy Preservation**: Fine-grained approach minimizes quantization error

## Key Insights

### Why Mixed Precision Works
- **Strategic Precision**: Use high precision only where needed
- **Hardware Optimization**: Leverage tensor cores for speed
- **Memory Efficiency**: Reduce storage requirements significantly
- **Training Stability**: Maintain precision for gradient updates

### Why Fine-Grained Quantization Works
- **Localized Scaling**: Prevents global outlier impact
- **Precision Retention**: Small values maintain their relative importance
- **Hardware Friendly**: Aligns with modern GPU architectures
- **Scalable Solution**: Works across different model sizes

## 3. Increasing Accumulation Precision

### The Problem: Limited Accumulation Precision

When performing GEMM (General Matrix Multiplication) operations like `Y = W × X + B`, if both operands are in FP8 format, we quickly lose accuracy due to intermediate results being too small, leading to **underflow issues**.

#### Core Issue
- **Tensor Core Limitation**: NVIDIA tensor cores accumulate GEMM results internally with limited precision (around 14 bits)
- **Precision Loss**: This is far below FP32 accumulation precision, causing numerical errors
- **Impact Scale**: For large matrices with inner dimensions K=4096, low accumulation precision can cause errors as large as 2%

### Solution: Periodic Promotion to CUDA Cores

DeepSeek implements a two-step process to increase accumulation precision:

#### Step 1: Low Precision MMA Accumulation
- **Operation**: Matrix Multiply Accumulate (MMA) operations performed using FP8 precision on tensor cores
- **Technology**: Uses WGMMA (Warp Group Level Matrix Multiply Accumulate)
- **Storage**: Intermediate results accumulate internally with limited precision (~14 bits)
- **Representation**: Light purple dots in DeepSeek schematics

#### Step 2: Promotion to Higher Precision
- **Frequency**: After every 128 elements (NC=128)
- **Process**: Partial low-precision accumulations are promoted to high-precision registers in CUDA cores
- **Result**: Partial results are accumulated in full FP32 precision
- **Representation**: Dark purple dots in DeepSeek schematics

### Implementation Details

#### Warp Group Level Matrix Multiply Accumulate (WGMMA)
- **Definition**: Performs matrix multiply accumulation using groups of warps
- **Warp**: Collection of threads in NVIDIA GPU terminology
- **Efficiency**: Optimized for efficient MMA operations within NVIDIA GPUs

#### Promotion Process
```
Tensor Core (FP8) → Periodic Transfer → CUDA Core (FP32)
    ↓                      ↓                    ↓
Low Precision         After 128 elements    High Precision
Accumulation         (NC interval)         Storage
```

#### Scaling Factors
- **Purpose**: Used during dequantization process
- **Representation**: Blue dots in DeepSeek schematics
- **Function**: Convert quantized elements back to original precision

### Benefits of Increasing Accumulation Precision
1. **Numerical Stability**: Prevents significant errors from low-precision accumulation
2. **Scalability**: Works effectively with large matrix dimensions
3. **Hardware Efficiency**: Balances tensor core speed with CUDA core precision
4. **Accuracy Preservation**: Maintains model performance during FP8 training

## 4. Mantissa Over Exponents

### Understanding FP8 Formats

Every floating-point representation consists of three components:
- **Sign**: Positive or negative
- **Exponent**: Controls dynamic range
- **Mantissa**: Controls precision

#### Two Primary FP8 Formats

| Format | Exponent Bits | Mantissa Bits | Characteristics |
|--------|---------------|---------------|----------------|
| **E4M3** | 4 | 3 | Smaller range, higher precision |
| **E5M2** | 5 | 2 | Larger range, lower precision |

### Traditional Approach vs. DeepSeek's Innovation

#### Traditional Method
- **Forward Pass**: E4M3 (higher precision, lower dynamic range)
- **Backward Pass**: E5M2 (lower precision, higher dynamic range)
- **Rationale**: Different passes have different numerical requirements

#### DeepSeek's Approach
- **Uniform Format**: E4M3 for both forward and backward passes
- **Justification**: Fine-grained quantization eliminates the need for extra exponent bits

### Why E4M3 Works with Fine-Grained Quantization

#### Without Fine-Grained Quantization
```
Original: [2, 3, 4, 500]
Max value: 500
Quantized: [2/500×127, 3/500×127, 4/500×127, 500/500×127]
         = [0.508, 0.762, 1.016, 127]
```
- **Problem**: Small values lose precision significantly
- **Issue**: Only 3 mantissa bits cannot represent small decimals accurately

#### With Fine-Grained Quantization
```
Group 1: [2, 3, 4] → Max = 4 → Better precision retention
Group 2: [500] → Max = 500 → Outlier contained
```
- **Advantage**: Each group has separate scaling factors
- **Result**: Numbers don't become extremely small
- **Outcome**: 3 mantissa bits are sufficient for accurate representation

### Technical Implementation

#### Effective Precision Expansion
- **Shared Exponents**: Each element effectively shares the group's exponent bits
- **Higher Accuracy**: Achieved within each group through fine-grained quantization
- **Precision Expansion**: Effective precision increases without extra exponent bits

#### Benefits of Mantissa Over Exponents
1. **Simplified Format**: Single E4M3 format for all operations
2. **Precision Retention**: Fine-grained quantization preserves small value accuracy
3. **Hardware Efficiency**: Uniform format simplifies hardware implementation
4. **Memory Optimization**: No need for different format handling

## 5. Online Quantization

### The Problem: Delayed Quantization

#### Traditional Delayed Quantization
- **Scale Factor Source**: Derived from past iterations (historical information)
- **Process**: Uses previous maximum value to scale current tensor
- **Risk**: Current tensor with different range can cause overflow/underflow

#### Example of Delayed Quantization Issues
```
Previous iteration max: 240
Current tensor values: [100, 200, 480, 150]
Problem: Value 480 exceeds FP8 range when using past scaling factor
Result: Overflow error and accuracy loss
```

### DeepSeek's Solution: Online Quantization

#### Real-Time Scale Factor Calculation
- **Current Data**: Scale factor computed from current tensor's data
- **On-the-Fly**: Maximum absolute value calculated in real-time
- **Prevention**: Eliminates overflow/underflow issues

#### Online Quantization Process
```
Current tensor: [100, 200, 480, 150]
Real-time max: 480
Scale factor: 480/127 = 3.78
Quantized: [26, 53, 127, 40]
Result: All values within FP8 range, no overflow
```

### Comparison: Online vs. Delayed Quantization

| Aspect | Delayed Quantization | Online Quantization |
|--------|---------------------|-------------------|
| **Scale Factor** | From past iterations | From current tensor |
| **Accuracy** | Risk of overflow/underflow | Optimal range utilization |
| **Computation** | Lower overhead | Slightly higher overhead |
| **Stability** | Less stable | More stable |

### Benefits of Online Quantization
1. **Accuracy**: Eliminates overflow/underflow errors
2. **Adaptability**: Responds to current tensor characteristics
3. **Stability**: Consistent quantization across iterations
4. **Optimization**: Maximizes utilization of available FP8 range

## Complete DeepSeek Quantization Framework

### Integration of All Five Techniques

DeepSeek combines all five quantization techniques to achieve optimal performance:

1. **Mixed Precision Framework**: Strategic use of different precisions
2. **Fine-Grained Quantization**: Localized scaling to handle outliers
3. **Increasing Accumulation Precision**: Periodic promotion to maintain accuracy
4. **Mantissa Over Exponents**: Uniform E4M3 format enabled by fine-grained quantization
5. **Online Quantization**: Real-time scale factor calculation

### Synergistic Effects

#### Fine-Grained + Mantissa Over Exponents
- Fine-grained quantization enables uniform E4M3 format
- Eliminates need for different exponent bit configurations
- Simplifies hardware implementation

#### Mixed Precision + Increasing Accumulation Precision
- Strategic precision use with periodic promotion
- Balances speed (tensor cores) with accuracy (CUDA cores)
- Maintains numerical stability throughout training

#### Online Quantization + All Techniques
- Real-time adaptation complements all other methods
- Ensures optimal range utilization across all precision levels
- Provides stability foundation for other techniques

### Technical Implementation Summary

```
Input (BF16) → Fine-Grained Chunks → FP8 Quantization (Online)
     ↓                                        ↓
Tensor Core (E4M3) → Periodic Promotion → CUDA Core (FP32)
     ↓                                        ↓
Low Precision MMA → Accumulation → High Precision Storage
```

### Performance Gains

1. **Memory Reduction**: Up to 4x reduction (FP32 → FP8)
2. **Speed Improvement**: Significant speedup on tensor cores
3. **Accuracy Preservation**: Minimal accuracy loss through sophisticated framework
4. **Training Stability**: Maintained through strategic precision management
5. **Inference Efficiency**: Reduced computational costs in deployment

## DeepSeek's Quantization Philosophy

### Building on Existing Knowledge
- **Incremental Innovation**: Enhanced existing quantization techniques
- **Efficiency Gains**: Added improvements to proven methods
- **Practical Implementation**: Focus on real-world deployment benefits

### Integration Excellence
- **Synergistic Design**: Techniques complement each other
- **Hardware Optimization**: Designed for modern GPU architectures
- **Scalability**: Works across different model sizes and applications

### Technical Achievement
- **Comprehensive Framework**: Addresses multiple quantization challenges
- **Performance Balance**: Optimal trade-off between speed, memory, and accuracy
- **Public Contribution**: Released detailed technical information for community benefit

## Future Topics

This completes the comprehensive coverage of DeepSeek's FP8 training quantization framework. The five techniques work together to enable:
- Significant memory reduction
- Computational efficiency improvement
- Maintained model accuracy
- Scalable deployment solutions

---

This quantization framework enables DeepSeek to achieve significant memory and computational efficiency while maintaining model performance, making it possible to train and deploy large language models more effectively.

---
**Author: Ayushmaan Singh**
