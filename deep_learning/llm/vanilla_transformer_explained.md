# The Vanilla Transformer Explained: Key Concepts and Source Code

## Table of Contents

1. [Introduction](#introduction)
2. [Overview of the Transformer Model](#overview-of-the-transformer-model)
    - [Architecture](#architecture)
    - [Key Components](#key-components)
3. [Class Introductions with Source Code](#class-introductions-with-source-code)
    - [LayerNorm](#layernorm)
    - [PositionwiseFeedForward](#positionwisefeedforward)
    - [ScaleDotProductAttention](#scaledotproductattention)
    - [MultiHeadAttention](#multiheadattention)
    - [TokenEmbedding](#tokenembedding)
    - [PositionalEncoding](#positionalencoding)
    - [TransformerEmbedding](#transformerembedding)
    - [EncoderLayer](#encoderlayer)
    - [Encoder](#encoder)
    - [DecoderLayer](#decoderlayer)
    - [Decoder](#decoder)
    - [Transformer](#transformer)
4. [Calling Sequence Explanation](#calling-sequence-explanation)
    - [Forward Pass Overview](#forward-pass-overview)
    - [Detailed Calling Sequence](#detailed-calling-sequence)
5. [Model Parameters Explanation](#model-parameters-explanation)
    - [Explanation of Model Parameters](#explanation-of-model-parameters)
    - [Example Configuration](#example-configuration)
6. [Conclusion](#conclusion)
7. [References](#references)

## Introduction
- Brief introduction to the importance of Transformer models in NLP.

## Overview of the Transformer Model

### Architecture
- Description of the overall architecture of the Transformer model.

### Key Components
- Explanation of key components like self-attention, feed-forward networks, and positional encoding.

## Class Introductions with Source Code

### LayerNorm
```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        """
        Initializes the LayerNorm module.

        Args:
            d_model (int): Dimensionality of the input.
            eps (float): Small epsilon value to avoid division by zero.
        """
        super(LayerNorm, self).__init__()

        # 'gamma' and 'beta' are learnable parameters that scale and shift the normalized output, respectively. 
        # Initially, gamma is set to ones and beta is set to zeros.
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

        # a small epsilon value to avoid division by zero during normalization.
        self.eps = eps

    def forward(self, x):
        """
        Forward pass of the LayerNorm module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Normalized tensor of shape [batch_size, seq_len, d_model].
        """
        # Computes the mean of the input tensor x along the last dimension (d_model). The result has shape [batch_size, seq_len, 1].
        mean = x.mean(-1, keepdim=True)
        
        # Computes the variance of the input tensor x along the last dimension. The result also has shape [batch_size, seq_len, 1].
        var = x.var(-1, unbiased=False, keepdim=True)
        
        # Subtract the mean from the input tensor and divide by the standard deviation (square root of variance plus eps). 
        # This standardizes the input to have zero mean and unit variance.
        # The result out has shape [batch_size, seq_len, d_model].
        out = (x - mean) / torch.sqrt(var + self.eps)

        # Multiplies the normalized tensor element-wise to scale the normalized values. It has shape [d_model], but it is broadcasted to match the shape of out.
        # Adds to the scaled tensor element-wise to shift the normalized values. It also has shape [d_model], and it is broadcasted to match the shape of out.
        out = self.gamma * out + self.beta
        
        # The final output tensor has shape [batch_size, seq_len, d_model].
        return out
```

#### LayerNorm Class Explanation
The LayerNorm (Layer Normalization) class normalizes the input tensor along the last dimension, which is typically the feature dimension in a sequence. This helps in stabilizing the learning process and improves the convergence speed.

In terms of matrix processing:
- The input tensor `x` of shape `[batch_size, seq_len, d_model]` is normalized along the `d_model` dimension.
- The normalization process ensures that for each position in the sequence (for each `[batch_size, seq_len]`), the features (of length `d_model`) have a mean of 0 and a variance of 1.
- The learnable parameters `gamma` and `beta` then scale and shift these normalized values to allow the model to learn the optimal scaling and shifting for each feature dimension.

This normalization helps stabilize the training process by reducing the internal covariate shift, making the model training more efficient and effective.


### PositionwiseFeedForward
```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        """
        Initializes the PositionwiseFeedForward module.

        Args:
            d_model (int): Dimensionality of the input.
            hidden (int): Number of hidden units in the feed-forward network.
            drop_prob (float): Dropout probability.
        """
        super(PositionwiseFeedForward, self).__init__()

        # 'linear1' A linear transformation that maps the input from d_model dimensions to hidden dimensions.
        self.linear1 = nn.Linear(d_model, hidden)
        
        # 'linear2' A linear transformation that maps the hidden layer back to d_model dimensions.
        self.linear2 = nn.Linear(hidden, d_model)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        Forward pass of the PositionwiseFeedForward module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        x = self.linear1(x)  # [batch_size, seq_len, hidden]
        x = self.relu(x)  # [batch_size, seq_len, hidden]
        x = self.dropout(x)  # [batch_size, seq_len, hidden]
        x = self.linear2(x)  # [batch_size, seq_len, d_model]
        return x
```

#### PositionwiseFeedForward Class Explanation
The PositionwiseFeedForward class applies a feed-forward neural network to each position of the input sequence independently. This is an essential component of the Transformer model, providing non-linearity and mixing the features after the self-attention mechanism.

- In terms of matrix processing:
- Input: The input tensor x has shape [batch_size, seq_len, d_model].
- Linear1: The input is linearly transformed to shape [batch_size, seq_len, hidden].
- ReLU: The ReLU activation is applied, maintaining the shape [batch_size, seq_len, hidden].
- Dropout: Dropout is applied, maintaining the shape [batch_size, seq_len, hidden].
- Linear2: The output is linearly transformed back to shape [batch_size, seq_len, d_model].
- Output: The output tensor has the same shape as the input, [batch_size, seq_len, d_model].

This feed-forward network is applied independently to each position of the sequence, enabling the model to learn complex transformations and representations for each token in the sequence.

### ScaleDotProductAttention
```python
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        """
        Initializes the ScaleDotProductAttention module.
        """
        super(ScaleDotProductAttention, self).__init__()

        # This initializes the softmax layer, which will be used to convert the attention scores into probabilities.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        """
        Forward pass of the ScaleDotProductAttention module.

        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, n_head, seq_len, d_tensor].
            k (torch.Tensor): Key tensor of shape [batch_size, n_head, seq_len, d_tensor].
            v (torch.Tensor): Value tensor of shape [batch_size, n_head, seq_len, d_tensor].
            mask (torch.Tensor, optional): Mask tensor of shape [batch_size, 1, seq_len, seq_len].
            e (float): Small epsilon value to avoid division by zero.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_head, seq_len, d_tensor].
            torch.Tensor: Attention scores tensor of shape [batch_size, n_head, seq_len, seq_len].
        """
        batch_size, head, length, d_tensor = k.size()
        k_t = k.transpose(2, 3)

        # The @ operator in Python is used for matrix multiplication. 
        # In this case, it's multiplying the query q and the transposed key k_t. 
        # This operation gives a measure of the alignment between the query and the key.
        
        # math.sqrt(d_tensor) is taking the square root of d_tensor, where d_tensor is likely the dimension of the key vectors. 
        # This is a scaling factor that is used in the Transformer model to prevent the dot product between the query and key 
        # from growing too large, which can lead to gradients that are too small and a model that is difficult to train.
        score = (q @ k_t) / math.sqrt(d_tensor)  # [batch_size, n_head, seq_len, seq_len]
        
        if mask is not None:
            # The masked_fill function is a PyTorch method that replaces certain elements of the score tensor based on the mask. 
            # Specifically, it replaces the elements where mask == 0 (i.e., where the mask is False) with the value -10000.
            # The value -10000 is used because the attention scores are typically passed through a softmax function, 
            # which converts them into probabilities. The softmax of -10000 is very close to 0, so this effectively masks out the 
            # corresponding elements in the attention scores, causing them to have very little impact on the final result.
            score = score.masked_fill(mask == 0, -10000)
        
        # The softmax function is used to convert the raw attention scores into probabilities. 
        # It does this by exponentiating each score and then dividing by the sum of all the exponentiated scores. 
        # This ensures that the attention scores are all between 0 and 1 and sum to 1, so they can be interpreted as probabilities.
        score = self.softmax(score)  # [batch_size, n_head, seq_len, seq_len]

        # The v variable is a tensor representing the value vectors, which are part of the input to the attention mechanism. 
        # The line v = score @ v performs a matrix multiplication between the score tensor and the value tensor. 
        # This operation computes a weighted sum of the value vectors, where the weights are given by the attention probabilities. 
        # This weighted sum is the output of the attention layer.
        v = score @ v  # [batch_size, n_head, seq_len, d_tensor]
        return v, score
```

#### ScaleDotProductAttention Class Explanation
The ScaleDotProductAttention class performs the scaled dot-product attention mechanism, a fundamental part of the Transformer model. This mechanism calculates the attention weights and applies them to the values to produce the output.

The self-attention mechanism can be mathematically described by the following formula:
**Scaled Dot-Product Attention**:
\[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]
Where:
- \( Q \) (Query) is a matrix of query vectors.
- \( K \) (Key) is a matrix of key vectors.
- \( V \) (Value) is a matrix of value vectors.
- \( d_k \) is the dimensionality of the key vectors (typically equal to the dimensionality of the queries and values).
- \( \text{softmax} \) is the softmax function applied along the last axis of the matrix.

In terms of matrix processing:
- Input: The input tensors q, k, and v have the shape [batch_size, n_head, seq_len, d_tensor].
- Transpose: The key tensor k is transposed to shape [batch_size, n_head, d_tensor, seq_len].
- Matrix Multiplication: Calculate the attention scores using the scaled dot-product, resulting in a score tensor of shape [batch_size, n_head, seq_len, - seq_len].
- Masking: Optionally mask certain positions in the score tensor.
- Softmax: Apply the softmax function to convert the scores to probabilities, maintaining the shape [batch_size, n_head, seq_len, seq_len].
- Output Calculation: Multiply the score tensor by the value tensor v to get the final output tensor of shape [batch_size, n_head, seq_len, d_tensor].
- Output: Return the output tensor and the attention scores tensor.


### MultiHeadAttention
```python
# Include source code of MultiHeadAttention here
```
- Explanation of the MultiHeadAttention class.

### TokenEmbedding
```python
# Include source code of TokenEmbedding here
```
- Explanation of the TokenEmbedding class.

### PositionalEncoding
```python
# Include source code of PositionalEncoding here
```
- Explanation of the PositionalEncoding class.

### TransformerEmbedding
```python
# Include source code of TransformerEmbedding here
```
- Explanation of the TransformerEmbedding class.

### EncoderLayer
```python
# Include source code of EncoderLayer here
```
- Explanation of the EncoderLayer class.

### Encoder
```python
# Include source code of Encoder here
```
- Explanation of the Encoder class.

### DecoderLayer
```python
# Include source code of DecoderLayer here
```
- Explanation of the DecoderLayer class.

### Decoder
```python
# Include source code of Decoder here
```
- Explanation of the Decoder class.

### Transformer
```python
# Include source code of Transformer here
```
- Explanation of the Transformer class.

## Calling Sequence Explanation

### Forward Pass Overview
- Overview of how the forward pass works in the Transformer model.

### Detailed Calling Sequence
- Detailed explanation of the calling sequence of each method during the forward pass.

## Model Parameters Explanation

### Explanation of Model Parameters
- Detailed explanation of each parameter in the Transformer model, such as `d_model`, `n_head`, `ffn_hidden`, etc.

### Example Configuration
- Provide an example configuration of the model parameters and explain why these values might be chosen.

## Conclusion
- Summary of the key points discussed in the article.

## References
- List of references and further reading materials.





