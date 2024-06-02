# The Vanilla Transformer Explained: Key Concepts and Source Code

## Introduction
- Brief introduction to the importance of Transformer models in NLP.

## Overview of the Transformer Model

### Architecture
- Description of the overall architecture of the Transformer model.

### Key Components
- Explanation of key components like self-attention, feed-forward networks, and positional encoding.

### Model Parameters
Here is a detailed explanation of the model parameters provided in Transformer:

| Parameter      | Description                                                                                         | Example Value | Example Explanation                                                                              |
|----------------|-----------------------------------------------------------------------------------------------------|---------------|--------------------------------------------------------------------------------------------------|
| `batch_size`   | The number of samples processed together in one forward and backward pass through the network.      | 128           | If you have 1,280 samples and `batch_size` is 128, you will complete 10 iterations per epoch.   |
| `max_len`      | The maximum sequence length that the model will handle.                                             | 4096          | Sentences or sequences longer than 4096 tokens will be truncated or padded to this length.        |
| `d_model`      | The dimension of the input embeddings and the hidden states in the model.                           | 512           | Each word/token is represented by a 512-dimensional vector (so called token embeddings).          |
| `n_layers`     | The number of layers (or blocks) in the encoder and decoder of the transformer model.               | 6             | The encoder and decoder each have 6 stacked layers.                                              |
| `n_heads`      | The number of attention heads in the multi-head attention mechanism.                                | 8             | The attention mechanism splits into 8 separate heads to focus on different parts of the sequence.|
| `ffn_hidden`   | The number of units in the hidden layer of the position-wise feed-forward neural network.           | 2048          | The feed-forward network has a hidden layer with 2048 units.                                     |
| `drop_prob`    | The dropout probability used in various layers to prevent overfitting by randomly setting neurons to zero during training. | 0.1           | 10% of the neurons are randomly set to zero during each forward pass to prevent overfitting.     |


## Class Introductions with Source Code

### LayerNorm
The LayerNorm (Layer Normalization) class normalizes the input tensor along the last dimension, which is typically the feature dimension in a sequence. This helps in stabilizing the learning process and improves the convergence speed.


<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-19_at_4.24.42_PM.png" alt="Layer Normalization Diagram" width="400">
<img src="res/layer_norm.jpg" alt="Layer Normalization Diagram" width="300">

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
#### Summary:
- The input tensor `x` of shape `[batch_size, seq_len, d_model]` is normalized along the `d_model` dimension.
- The normalization process ensures that for each position in the sequence (for each `[batch_size, seq_len]`), the features (of length `d_model`) have a mean of 0 and a variance of 1.
- The learnable parameters `gamma` and `beta` then scale and shift these normalized values to allow the model to learn the optimal scaling and shifting for each feature dimension.

Resources
- **[Layer Normalization on Papers with Code](https://paperswithcode.com/method/layer-normalization)**
- **[Implementation of LayerNorm in PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#layernorm)**


### PositionwiseFeedForward
The PositionwiseFeedForward class applies a feed-forward neural network to each position of the input sequence independently. This is an essential component of the Transformer model, providing non-linearity and mixing the features after the self-attention mechanism.

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

#### Summary:
- Input: The input tensor x has shape [batch_size, seq_len, d_model].
- Linear1: The input is linearly transformed to shape [batch_size, seq_len, hidden].
- ReLU: The ReLU activation is applied, maintaining the shape [batch_size, seq_len, hidden].
- Dropout: Dropout is applied, maintaining the shape [batch_size, seq_len, hidden].
- Linear2: The output is linearly transformed back to shape [batch_size, seq_len, d_model].
- Output: The output tensor has the same shape as the input, [batch_size, seq_len, d_model].

This feed-forward network is applied independently to each position of the sequence, enabling the model to learn complex transformations and representations for each token in the sequence.

### ScaleDotProductAttention
The ScaleDotProductAttention class performs the scaled dot-product attention mechanism, a fundamental part of the Transformer model. This mechanism calculates the attention weights and applies them to the values to produce the output.

The self-attention mechanism can be mathematically described by the following formula:
Scaled Dot-Product Attention:
![](res/mha_formula.jpg)

Where:
- \( Q \) (Query) is a matrix of query vectors.
- \( K \) (Key) is a matrix of key vectors.
- \( V \) (Value) is a matrix of value vectors.
- \( d_k \) is the dimensionality of the key vectors (typically equal to the dimensionality of the queries and values).

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

#### Summary:
- Input: The input tensors q, k, and v have the shape [batch_size, n_head, seq_len, d_tensor].
- Transpose: The key tensor k is transposed to shape [batch_size, n_head, d_tensor, seq_len].
- Matrix Multiplication: Calculate the attention scores using the scaled dot-product, resulting in a score tensor of shape [batch_size, n_head, seq_len, - seq_len].
- Masking: Optionally mask certain positions in the score tensor.
- Softmax: Apply the softmax function to convert the scores to probabilities, maintaining the shape [batch_size, n_head, seq_len, seq_len].
- Output Calculation: Multiply the score tensor by the value tensor v to get the final output tensor of shape [batch_size, n_head, seq_len, d_tensor].
- Output: Return the output tensor and the attention scores tensor.


### MultiHeadAttention
The MultiHeadAttention class implements the multi-head attention mechanism used in the Transformer model. This mechanism allows the model to jointly attend to information from different representation subspaces at different positions.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        """
        Initializes the MultiHeadAttention module.

        Args:
            d_model (int): Dimension of token embeddings.
            n_head (int): Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        
        # Why Do We Need Linear Projections?
        # Dimensionality Transformation: The linear projections allow the model to transform the input tensor into different subspaces (query, key, and value spaces).
        # Learnable Parameters: By using linear layers for projections, the model introduces learnable parameters that can adapt during training. 
        # Handling Multiple Heads: When using multiple heads in multi-head attention, linear projections allow each head to have its own set of parameters for queries, keys, and values. 
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        Forward pass of the MultiHeadAttention module.

        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, d_model].
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, d_model].
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, d_model].
            mask (torch.Tensor, optional): Mask tensor of shape [batch_size, 1, seq_len, seq_len].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        # Linear Projections: 
        # The input tensors q, k, and v are projected into the query, key, and value spaces using linear transformations.
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)  # [batch_size, seq_len, d_model]

        # Split into Multiple Heads: 
        # The projected tensors are split into n_head heads. Each head has a dimensionality of d_tensor = d_model // n_head.
        q, k, v = self.split(q), self.split(k), self.split(v)  # [batch_size, n_head, seq_len, d_tensor]

        # Scaled Dot-Product Attention:
        # The attention method applies scaled dot-product attention to the split tensors.
        out, attention = self.attention(q, k, v, mask=mask)  # [batch_size, n_head, seq_len, d_tensor]

        # Concatenate Heads:
        # The outputs of the multiple heads are concatenated back into a single tensor.
        out = self.concat(out)  # [batch_size, seq_len, d_model]

        # Final Linear Transformation:
        # The concatenated tensor is passed through a final linear transformation. 
        out = self.w_concat(out)  # [batch_size, seq_len, d_model]

        return out

    def split(self, tensor):
        """
        Splits the input tensor into multiple heads.

        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Tensor split into heads of shape [batch_size, n_head, seq_len, d_tensor].
        """
        batch_size, length, d_model = tensor.size()

        # d_tensor is calculated as the integer division of 'dimension of token embeddings' by 'number of attention heads'.
        d_tensor = d_model // self.n_head

        # The tensor variable is then reshaped using the view method.
        # This method swaps two dimensions of the tensor with transpose method.
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        """
        Concatenates the multi-head tensor back into a single tensor.

        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, n_head, seq_len, d_tensor].

        Returns:
            torch.Tensor: Concatenated tensor of shape [batch_size, seq_len, d_model].
        """
        batch_size, head, length, d_tensor = tensor.size()

        # d_model is calculated as the product of 'number of attention heads' and d_tensor
        d_model = head * d_tensor

        # The transpose method is called on the tensor to swap the second and third dimensions:
        #   [batch_size, head, length, d_tensor] => [batch_size, length, head, d_tensor]
        # The contiguous method is called to ensure that the tensor is stored in a contiguous block of memory.
        # The view method is used to reshape the tensor (d_model == head * d_tensor):
        #   [batch_size, length, head, d_tensor] => [batch_size, length, d_model] 
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
```

#### Summary:
- Input: Tensors q, k, v with shape [batch_size, seq_len, d_model].
- Linear Projections: Project to shape [batch_size, seq_len, d_model].
- Split: Split into n_head heads, shape [batch_size, n_head, seq_len, d_tensor].
- Attention: Apply scaled dot-product attention, shape [batch_size, n_head, seq_len, d_tensor].
- Concat: Concatenate heads, shape [batch_size, seq_len, d_model].
- Linear Transformation: Apply final linear transformation, shape [batch_size, seq_len, d_model].
- Output: Return tensor of shape [batch_size, seq_len, d_model].


### TokenEmbedding
The TokenEmbedding class is a subclass of nn.Embedding from PyTorch. It is used to convert token indices into dense vectors of a fixed size (d_model). This is typically the first layer in a neural network that processes sequences of tokens, such as words in a sentence.

Reference: [Definition of nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

```python
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        """
        Initializes the TokenEmbedding module.

        Args:
            vocab_size (int): The number of unique tokens in the vocabulary.
            d_model (int): Dimensionality of the embeddings.
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
```

Example:
1. Vocabulary and Embedding Dimension:
    - Suppose the `vocab_size` is 10,000 (i.e., there are 10,000 unique tokens).
    - Suppose `d_model` is 512 (i.e., each token is represented by a 512-dimensional vector).
2. Input Tensor:
    - Assume the input tensor is `x`, with shape `(batch_size, seq_len)`, where:
        - `batch_size` is the number of sequences (e.g., sentences) in the batch.
        - `seq_len` is the length of each sequence.
    ```python
    # Example input tensor with batch_size=2 and seq_len=3
    x = torch.tensor([[2, 4, 1], [3, 5, 1]])
    ```
3. Embedding Lookup:
    - The `TokenEmbedding` layer processes the input tensor `x` and looks up the embedding vectors for each token index.
    - The output tensor will have the shape `(batch_size, seq_len, d_model)`.
    ```python
    # Assuming the embedding layer is initialized
    embedding_layer = TokenEmbedding(vocab_size=10000, d_model=512)

    # Forward pass
    embeddings = embedding_layer(x)
    ```
    - If `x` is `[[2, 4, 1], [3, 5, 1]]` and `d_model` is 512:
        - The embedding for token `2` is a 512-dimensional vector.
        - The embedding for token `4` is a 512-dimensional vector.
        - The embedding for token `1` (padding) is a 512-dimensional zero vector.
    - The output `embeddings` will have the shape `(2, 3, 512)`, where:
        - The first dimension is the batch size (2).
        - The second dimension is the sequence length (3).
        - The third dimension is the embedding size (512).

#### Summary:
The `TokenEmbedding` class transforms token indices into dense vectors (embeddings) using an embedding matrix. The embedding matrix is learned during training, allowing the model to capture semantic information about the tokens. The process involves looking up each token index in the embedding matrix and returning the corresponding embedding vector. This is essential for converting discrete tokens into continuous representations that can be processed by neural networks.


### PositionalEncoding
The PositionalEncoding class adds positional information to the token embeddings. This is important in transformer models because, unlike recurrent neural networks, transformers do not inherently capture the order of the input tokens. Positional encoding helps the model understand the position of each token in the sequence.

Positional Encoding Visualization:
<br><img src="res/transformer_pos_enc1.jpg" alt="Positional Encoding 1" width="500">
<br><img src="res/transformer_pos_enc2.jpg" alt="Positional Encoding 2" width="800">

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): Dimensionality of the model's hidden states.
            max_len (int): Maximum length of the input sequences.
            device (torch.device): Device to run the model on (CPU or GPU).
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        # The pos variable is a tensor that contains the positions, which are just the integers from 0 to max_len - 1. 
        # The unsqueeze method is called to add an extra dimension to this tensor, turning it from a 1D tensor into a 2D tensor.
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        # The _2i variable is a tensor that contains the even integers from 0 to d_model - 1. 
        # This is used to create different positional encodings for the even and odd positions.
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # [:, 0::2] line is applying a sine function to the positions and storing the result in the even columns of self.encoding.
        # [:, 1::2] line is doing the same with a cosine function for the odd columns. 
        # The positions are divided by (10000 ** (_2i / d_model)) to ensure that the positional encoding varies smoothly and slowly with the position.
        # This division operation results in different frequencies of the sine and cosine functions for different dimensions, 
        # which allows the model to learn to attend to relative positions.
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        """
        Forward pass of the PositionalEncoding module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Positional encoding tensor of shape [seq_len, d_model].
        """
        batch_size, seq_len = x.size()

        # The [:seq_len, :] part is slicing the tensor to select only the first seq_len rows.
        # This operation ensures that the encoding is the same length as the sequences.
        # The unsqueeze(0) method is called to add an extra dimension to the tensor at the beginning. 
        # The expand(batch_size, -1, -1) method is called to repeat the tensor along the specified dimensions.
        return self.encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
```

References on Positional Encoding in Transformer Models:
- [A Gentle Introduction to Positional Encoding in Transformer Models (Part 1)](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)
- [Understanding Positional Encoding in Transformers](https://towardsdatascience.com/understanding-positional-encoding-in-transformers-dc6bafc021ab)


### TransformerEmbedding
The TransformerEmbedding class combines token embeddings and positional encodings to produce the input embeddings for the transformer model. This class ensures that the model can incorporate both the semantic information of the tokens and their positional information within the sequence.

```python
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        Initializes the TransformerEmbedding module.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimensionality of the token embeddings.
            max_len (int): Maximum length of the input sequences.
            drop_prob (float): Dropout probability.
            device (torch.device): Device to run the model on (CPU or GPU).
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        Forward pass of the TransformerEmbedding module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Embedded tensor of shape [batch_size, seq_len, d_model].
        """
        # convert the token IDs into token embeddings
        # generate positional embeddings
        tok_emb = self.tok_emb(x)       
        pos_emb = self.pos_emb(x)       

        # The tok_emb + pos_emb operation is adding the token embeddings and the positional embeddings together. 
        # This is done to combine the information about the tokens and their positions into a single representation. 
        # This combined representation is then returned as the output of this code block.
        return self.drop_out(tok_emb + pos_emb)
```
#### Summary:
1. Token Embeddings:
    - The input tensor `x` of shape `(batch_size, seq_len)` contains token indices.
    - These indices are converted into dense vectors by the `TokenEmbedding` layer, resulting in a tensor of shape `(batch_size, seq_len, d_model)`.
2. Positional Encodings:
    - The positional encoding matrix is precomputed during initialization and has the shape `(max_len, d_model)`.
    - During the forward pass, the relevant slice of this matrix is extracted to match the sequence length (`seq_len`), resulting in a tensor of shape `(seq_len, d_model)`.
    - This slice is then expanded to match the batch size, resulting in a tensor of shape `(batch_size, seq_len, d_model)`.
3. Combining and Dropout:
    - The token embeddings and positional encodings are added together element-wise.
    - The combined tensor is then passed through a dropout layer to prevent overfitting.
    - The final output tensor has the shape `(batch_size, seq_len, d_model)`.


### EncoderLayer
The EncoderLayer class implements a single layer of the transformer encoder. Each encoder layer consists of a multi-head self-attention mechanism followed by a position-wise feed-forward network. Layer normalization and dropout are applied at various points to ensure stability and prevent overfitting.

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """
        Initializes the EncoderLayer module.

        Args:
            d_model (int): Dimensionality of the model's hidden states.
            ffn_hidden (int): Number of hidden units in the feed-forward network.
            n_head (int): Number of attention heads.
            drop_prob (float): Dropout probability.
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        """
        Forward pass of the EncoderLayer module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].
            src_mask (torch.Tensor): Source mask tensor of shape [batch_size, 1, 1, src_seq_len].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        # Save the input tensor for the residual connection
        _x = x

        # Apply multi-head attention
        # Apply dropout after attention
        # Apply residual connection and layer normalization
        x = self.attention(q=x, k=x, v=x, mask=src_mask)  # [batch_size, seq_len, d_model]
        x = self.dropout1(x)                              # [batch_size, seq_len, d_model]
        x = self.norm1(x + _x)                            # [batch_size, seq_len, d_model]

        # Save the intermediate output for the next residual connection
        _x = x
        
        # Apply position-wise feed-forward network
        # Apply dropout after feed-forward network
        # Apply residual connection and layer normalization
        x = self.ffn(x)         # [batch_size, seq_len, d_model]
        x = self.dropout2(x)    # [batch_size, seq_len, d_model]
        x = self.norm2(x + _x)  # [batch_size, seq_len, d_model]
        return x
```
#### Summary:
By processing the input tensor through these steps, the EncoderLayer effectively captures both local and global dependencies within the sequence, stabilizes training with normalization, and prevents overfitting with dropout. This process is repeated for each layer in the encoder stack, allowing the model to build increasingly complex representations of the input data.

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





