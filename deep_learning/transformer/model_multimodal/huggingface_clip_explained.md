# Inside CLIP: Understanding Its Core Mechanisms

## Architecture of CLIP
The architecture of CLIP (Contrastive Language-Image Pre-training) comprises two main components: the Vision Encoder and the Text Encoder. These components work together to map images and text into a shared feature space where their similarities can be directly compared.

### 1. Vision Encoder
The Vision Encoder in CLIP can be either a modified ResNet or a Vision Transformer (ViT). Both architectures are designed to extract meaningful features from images.

**Modified ResNet:**

- **Stem**: The stem consists of three convolutional layers with an average pooling layer, instead of the single convolutional layer with max pooling found in standard ResNets.
- **Residual Blocks**: The network uses Bottleneck blocks with anti-aliasing strided convolutions, where an average pooling layer precedes convolutions with stride > 1.
- **Attention Pooling**: The final pooling layer is replaced by an Attention Pooling layer, which uses QKV (query, key, value) attention mechanism to pool the spatial features.

**Vision Transformer (ViT):**

- **Patch Embedding**: The input image is divided into fixed-size patches, and each patch is linearly embedded.
- **Positional Embedding**: Positional embeddings are added to the patch embeddings to retain spatial information.
- **Transformer Encoder**: The transformer encoder consists of multiple layers of multi-head self-attention and feed-forward neural networks. Each layer is equipped with residual connections and layer normalization.
- **Classification Token**: A special classification token is prepended to the sequence of embedded patches, and its output representation is used for classification tasks.

### 2. Text Encoder
The Text Encoder is a Transformer model that processes the input text and converts it into a feature representation.
- **Tokenization**: The input text is tokenized using a Byte Pair Encoding (BPE) tokenizer.
- **Token Embedding**: Each token is converted into an embedding vector.
- **Positional Embedding**: Positional embeddings are added to the token embeddings to encode the position information of each token in the sequence.
- **Transformer Encoder**: Similar to the Vision Transformer's encoder, it consists of multiple layers of multi-head self-attention and feed-forward neural networks, with residual connections and layer normalization.
- **Projection Layer**: The output representation of the [CLS] token (classification token) is projected into the shared feature space.

### 3. Contrastive Loss
During training, CLIP uses a contrastive loss to align the visual and textual representations in a shared feature space.
- **Normalization**: The output features from both the vision and text encoders are normalized to unit length.
- **Similarity Calculation**: The cosine similarity between all pairs of image and text features in a batch is computed.
- **Contrastive Loss**: A contrastive loss is applied, encouraging the similarity of matching image-text pairs while minimizing the similarity of non-matching pairs.


## 2. Main Classes of CLIP
Learning these main classes is crucial for a deep understanding of CLIP because they encapsulate its core architecture and functionality. The `CLIP` class integrates vision and text encoders, enabling image-text similarity computations. Understanding `ModifiedResNet` and `VisionTransformer` reveals how CLIP processes images using different architectures, while the `Transformer` class shows how text is handled through sequence processing. Key components like `ResidualAttentionBlock` detail the internal workings of multi-head attention and feed-forward networks.

Supporting classes such as `LayerNorm` and `QuickGELU` are vital for understanding normalization and activation within the model, ensuring stability and efficiency. The `Bottleneck` class illustrates convolutional operations, and `AttentionPool2d` highlights advanced spatial feature pooling techniques. Finally, the `SimpleTokenizer` is essential for processing textual input. Together, these classes provide a comprehensive view of how CLIP integrates visual and textual data, leveraging advanced neural network architectures for tasks like zero-shot learning and image-text retrieval.

Here's a table introducing the main classes of CLIP and their descriptions:
| **Class Name**               | **Description**                                                                                 |
|------------------------------|-------------------------------------------------------------------------------------------------|
| `SimpleTokenizer`            | A tokenizer for processing input text, used for converting text into tokenized sequences.        |
| `QuickGELU`                  | A custom activation function used in the feed-forward network of `ResidualAttentionBlock`.       |
| `LayerNorm`                  | A subclass of PyTorch's `LayerNorm` that handles fp16 precision.                                 |
| `Bottleneck`                 | A bottleneck block used in `ModifiedResNet`, consisting of convolutions, batch normalization, and ReLU activations. |
| `AttentionPool2d`            | A 2D attention pooling layer used in `ModifiedResNet` for downsampling features.                 |
| `ResidualAttentionBlock`     | A building block for the `Transformer` class, consisting of multi-head attention and feed-forward layers with residual connections. |
| `Transformer`                | The transformer architecture used for the text encoder in CLIP, also used in the VisionTransformer. |
| `ModifiedResNet`             | A modified version of ResNet used as the vision encoder, featuring multiple "stem" convolutions and anti-aliasing.  |
| `VisionTransformer`          | The Vision Transformer (ViT) used as an alternative vision encoder, processing image patches with a transformer architecture. |
| `CLIP`                       | The main class representing the CLIP model, combining vision and text encoders, and performing the forward pass. |

This annotated structure highlights the hierarchical dependencies and roles of each class within the CLIP architecture, illustrating how they contribute to the overall functionality of the model.
```
CLIP
├── ModifiedResNet
│   ├── Bottleneck
│   └── AttentionPool2d
├── VisionTransformer
│   └── Transformer
│       └── ResidualAttentionBlock
│           ├── LayerNorm
│           └── QuickGELU
└── SimpleTokenizer
```

### class SimpleTokenizer
A tokenizer class for processing input text into tokenized sequences using Byte Pair Encoding (BPE).

```python
class SimpleTokenizer(object):
    """
    A simple tokenizer that uses byte pair encoding (BPE) to tokenize text.

    Attributes:
        byte_encoder: A dictionary that maps bytes to unicode characters.
        byte_decoder: A dictionary that maps unicode characters to bytes.
        encoder: A dictionary that maps tokens to their BPE encoded integer values.
        decoder: A dictionary that maps BPE encoded integer values to their corresponding tokens.
        bpe_ranks: A dictionary that maps BPE merge pairs to their rank.
        cache: A cache for storing BPE encoded tokens.
        pat: A regex pattern for tokenizing text.
    """

    def __init__(self, bpe_path: str = default_bpe()):
        """
        Initializes the SimpleTokenizer with the provided BPE file.

        Args:
            bpe_path (str): The path to the BPE file.
        """
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Read BPE merges from the file and parse them
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]

        # Initialize vocabulary with unicode characters and their BPE variants
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'': '', '<|startoftext|>': '', '<|endoftext|>': ''}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        """
        Applies byte pair encoding (BPE) to a token.

        Args:
            token (str): The input token to be BPE encoded.

        Returns:
            str: The BPE encoded token.
        """
        if token in self.cache:
            return self.cache[token]

        # Split token into pairs and add end of word token
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            # Find the lowest rank BPE merge pair
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """
        Encodes the input text into BPE tokens.

        Args:
            text (str): The input text to be encoded.

        Returns:
            List[int]: A list of BPE encoded integer tokens.
        """
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        """
        Decodes a list of BPE tokens back into a string.

        Args:
            tokens (List[int]): A list of BPE encoded integer tokens.

        Returns:
            str: The decoded string.
        """
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

# Helper functions used in SimpleTokenizer (assuming they are defined elsewhere in the code)

def default_bpe():
    # Placeholder function to get the default BPE path
    pass

def bytes_to_unicode():
    # Placeholder function to get the byte to unicode mappings
    pass

def get_pairs(word):
    # Placeholder function to get pairs of tokens from a word
    pass

def basic_clean(text):
    # Placeholder function to clean text
    pass

def whitespace_clean(text):
    # Placeholder function to clean whitespace from text
    pass
```

### class QuickGELU
A custom activation function used in the feed-forward networks, applying the GELU activation more quickly.

```python
class QuickGELU(nn.Module):
    """
    A custom implementation of the GELU (Gaussian Error Linear Unit) activation function.

    This class implements a faster approximation of the GELU activation function,
    which is commonly used in transformer models. The approximation uses the sigmoid function
    for computational efficiency.
    """

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the QuickGELU activation function.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input, where the QuickGELU function has been applied element-wise.
        """
        # Apply the QuickGELU function: x * sigmoid(1.702 * x)
        return x * torch.sigmoid(1.702 * x)  # Shape: same as input [*]
```

### class LayerNorm
A subclass of PyTorch's LayerNorm to handle fp16 precision, ensuring numerical stability during normalization.

```python
class LayerNorm(nn.LayerNorm):
    """
    A subclass of PyTorch's LayerNorm to handle fp16 precision.

    This class ensures that the input tensor is cast to float32 before applying layer normalization,
    and then cast back to the original data type (fp16) after normalization. This helps to avoid
    precision issues that can arise when using fp16 for layer normalization.
    """

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the LayerNorm.

        Args:
            x (torch.Tensor): Input tensor of shape [*, C] where * is any number of additional dimensions.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input.
        """
        # Store the original data type of the input tensor
        orig_type = x.dtype  # e.g., torch.float16

        # Cast the input tensor to float32 before applying layer normalization
        ret = super().forward(x.type(torch.float32))  # Shape: [*, C]

        # Cast the normalized tensor back to the original data type
        return ret.type(orig_type)  # Shape: [*, C]
```

### class Bottleneck
A residual block used in ResNet architectures, consisting of convolutional layers, batch normalization, and ReLU activations, with an optional downsampling layer.

```python
class Bottleneck(nn.Module):
    """
    A bottleneck block used in ResNet architectures.

    This block consists of three convolutional layers, each followed by batch normalization and ReLU activation.
    An average pooling layer is optionally applied for downsampling when stride > 1. The block also includes a 
    downsampling layer if needed to match the dimensions for residual connections.

    Attributes:
        expansion (int): The expansion factor for the output channels.
        conv1 (nn.Conv2d): The first convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization for the first convolutional layer.
        relu1 (nn.ReLU): ReLU activation for the first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization for the second convolutional layer.
        relu2 (nn.ReLU): ReLU activation for the second convolutional layer.
        avgpool (Union[nn.AvgPool2d, nn.Identity]): Average pooling layer for downsampling.
        conv3 (nn.Conv2d): The third convolutional layer.
        bn3 (nn.BatchNorm2d): Batch normalization for the third convolutional layer.
        relu3 (nn.ReLU): ReLU activation for the third convolutional layer.
        downsample (nn.Sequential, optional): Downsampling layer to match dimensions for residual connection.
        stride (int): Stride for the block, affects downsampling.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        """
        Initializes the Bottleneck block.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels for the first two convolutions.
            stride (int): Stride for the block, affects downsampling. Default is 1.
        """
        super().__init__()

        # First convolutional layer with batch normalization and ReLU activation
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # Shape: [N, planes, H, W]
        self.bn1 = nn.BatchNorm2d(planes)  # Shape: [N, planes, H, W]
        self.relu1 = nn.ReLU(inplace=True)  # Shape: [N, planes, H, W]

        # Second convolutional layer with batch normalization and ReLU activation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)  # Shape: [N, planes, H, W]
        self.bn2 = nn.BatchNorm2d(planes)  # Shape: [N, planes, H, W]
        self.relu2 = nn.ReLU(inplace=True)  # Shape: [N, planes, H, W]

        # Optional average pooling layer for downsampling
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()  # Shape: [N, planes, H//stride, W//stride]

        # Third convolutional layer with batch normalization and ReLU activation
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)  # Shape: [N, planes*expansion, H, W]
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)  # Shape: [N, planes*expansion, H, W]
        self.relu3 = nn.ReLU(inplace=True)  # Shape: [N, planes*expansion, H, W]

        self.downsample = None
        self.stride = stride

        # Downsampling layer if needed to match dimensions for residual connection
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),  # Shape: [N, inplanes, H//stride, W//stride]
                ("0", nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=1, bias=False)),  # Shape: [N, planes*expansion, H, W]
                ("1", nn.BatchNorm2d(planes * self.expansion))  # Shape: [N, planes*expansion, H, W]
            ]))

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Bottleneck block.

        Args:
            x (torch.Tensor): Input tensor with shape [N, inplanes, H, W].

        Returns:
            torch.Tensor: Output tensor with shape [N, planes*expansion, H, W].
        """
        identity = x  # Shape: [N, inplanes, H, W]

        out = self.relu1(self.bn1(self.conv1(x)))  # Shape: [N, planes, H, W]
        out = self.relu2(self.bn2(self.conv2(out)))  # Shape: [N, planes, H, W]
        out = self.avgpool(out)  # Shape: [N, planes, H//stride, W//stride]
        out = self.bn3(self.conv3(out))  # Shape: [N, planes*expansion, H//stride, W//stride]

        if self.downsample is not None:
            identity = self.downsample(x)  # Shape: [N, planes*expansion, H//stride, W//stride]

        out += identity  # Shape: [N, planes*expansion, H//stride, W//stride]
        out = self.relu3(out)  # Shape: [N, planes*expansion, H//stride, W//stride]
        return out
```

### class AttentionPool2d
A 2D attention pooling layer used for downsampling features by applying multi-head attention and incorporating positional embeddings.

```python
class AttentionPool2d(nn.Module):
    """
    A 2D attention pooling layer.

    This layer applies multi-head attention to pool spatial features into a fixed-size output embedding.
    It first flattens the input feature map, adds positional embeddings, and then applies multi-head attention.

    Attributes:
        positional_embedding (nn.Parameter): Positional embedding for the flattened spatial dimensions.
        k_proj (nn.Linear): Linear projection layer for keys.
        q_proj (nn.Linear): Linear projection layer for queries.
        v_proj (nn.Linear): Linear projection layer for values.
        c_proj (nn.Linear): Linear projection layer for the output.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        """
        Initializes the AttentionPool2d layer.

        Args:
            spacial_dim (int): Spatial dimension of the input feature map (H or W).
            embed_dim (int): Embedding dimension of the input feature map.
            num_heads (int): Number of attention heads.
            output_dim (int, optional): Output embedding dimension. If None, defaults to embed_dim.
        """
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        """
        Forward pass of the AttentionPool2d layer.

        Args:
            x (torch.Tensor): Input tensor with shape [N, C, H, W].

        Returns:
            torch.Tensor: Output tensor with shape [N, output_dim or embed_dim].
        """
        # Flatten the input feature map and permute dimensions for attention
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # Shape: [N, C, H, W] -> [N, C, HW] -> [HW, N, C]

        # Add a mean pooled token and positional embedding
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # Shape: [HW, N, C] -> [(HW+1), N, C]

        # Add positional embeddings
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # Shape: [(HW+1), N, C]

        # Apply multi-head attention
        x, _ = F.multi_head_attention_forward(
            query=x[:1],                         # Shape: [1, N, C] for the query (using the first token)
            key=x,                               # Shape: [(HW+1), N, C] for the key
            value=x,                             # Shape: [(HW+1), N, C] for the value
            embed_dim_to_check=x.shape[-1],      # Embedding dimension to check (C)
            num_heads=self.num_heads,            # Number of attention heads
            q_proj_weight=self.q_proj.weight,    # Query projection weight
            k_proj_weight=self.k_proj.weight,    # Key projection weight
            v_proj_weight=self.v_proj.weight,    # Value projection weight
            in_proj_weight=None,                 # In-projection weight (None since separate projection weights are used)
            in_proj_bias=torch.cat([             # In-projection bias for query, key, and value (concatenated)
                self.q_proj.bias,                # Query projection bias
                self.k_proj.bias,                # Key projection bias
                self.v_proj.bias                 # Value projection bias
            ]),
            bias_k=None,                         # Optional bias for key (None)
            bias_v=None,                         # Optional bias for value (None)
            add_zero_attn=False,                 # Whether to add a batch of zeros to the key and value sequences at index 1 (False)
            dropout_p=0,                         # Dropout probability (0 for no dropout)
            out_proj_weight=self.c_proj.weight,  # Output projection weight
            out_proj_bias=self.c_proj.bias,      # Output projection bias
            use_separate_proj_weight=True,       # Whether to use separate projection weights for query, key, and value (True)
            training=self.training,              # Whether the model is in training mode
            need_weights=False                   # Whether to return attention weights (False)
        )

        return x.squeeze(0)  # Shape: [1, N, C] -> [N, C]
```

### class ResidualAttentionBlock
A building block for the transformer, consisting of multi-head attention and feed-forward layers, each followed by layer normalization and residual connections.

```python
class ResidualAttentionBlock(nn.Module):
    """
    A residual attention block used in transformer architectures.

    This block consists of a multi-head attention layer followed by a feed-forward network (MLP),
    with layer normalization and residual connections applied to both sub-layers.
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        """
        Initializes the ResidualAttentionBlock.

        Args:
            d_model (int): Dimension of the model (number of features).
            n_head (int): Number of attention heads.
            attn_mask (torch.Tensor, optional): Attention mask to apply. Default is None.
        """
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)  # Multi-head attention layer
        self.ln_1 = LayerNorm(d_model)  # Layer normalization before attention
        self.mlp = nn.Sequential(OrderedDict([  # Feed-forward network (MLP)
            ("c_fc", nn.Linear(d_model, d_model * 4)),  # Fully connected layer
            ("gelu", QuickGELU()),  # GELU activation
            ("c_proj", nn.Linear(d_model * 4, d_model))  # Projection layer
        ]))
        self.ln_2 = LayerNorm(d_model)  # Layer normalization before MLP
        self.attn_mask = attn_mask  # Optional attention mask

    def attention(self, x: torch.Tensor):
        """
        Applies multi-head attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [L, N, C] where L is the sequence length, N is the batch size, and C is the number of features.

        Returns:
            torch.Tensor: Output tensor of shape [L, N, C] after applying multi-head attention.
        """
        # Apply the attention mask if provided
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]  # Shape: [L, N, C]

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the ResidualAttentionBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [L, N, C].

        Returns:
            torch.Tensor: Output tensor of shape [L, N, C] after applying the residual attention block.
        """
        # Apply layer normalization and attention, then add the residual connection
        x = x + self.attention(self.ln_1(x))  # Shape: [L, N, C]

        # Apply layer normalization and MLP, then add the residual connection
        x = x + self.mlp(self.ln_2(x))  # Shape: [L, N, C]
        
        return x
```

### class Transformer
A transformer model composed of multiple ResidualAttentionBlock layers, used to encode text sequences.

```python
class Transformer(nn.Module):
    """
    A transformer model consisting of multiple residual attention blocks.

    This class implements a transformer with a specified number of layers, each containing a residual attention block.
    The transformer can optionally use an attention mask.
    """

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        """
        Initializes the Transformer.

        Args:
            width (int): Dimension of the model (number of features).
            layers (int): Number of residual attention blocks (layers).
            heads (int): Number of attention heads in each block.
            attn_mask (torch.Tensor, optional): Attention mask to apply. Default is None.
        """
        super().__init__()
        self.width = width  # Model dimension
        self.layers = layers  # Number of layers

        # Create a sequence of residual attention blocks
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )  # Shape: [L, N, C] -> [L, N, C] through each block

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape [L, N, C] where L is the sequence length, N is the batch size, and C is the number of features.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input, after passing through all layers.
        """
        return self.resblocks(x)  # Shape: [L, N, C] -> [L, N, C] through each block
```

### class ModifiedResNet
A modified version of ResNet used as a vision encoder, featuring multiple "stem" convolutions and anti-aliasing, and utilizing attention pooling instead of average pooling.

```python
class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1.
    - The final pooling layer is a QKV attention instead of an average pool.
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        """
        Initializes the ModifiedResNet.

        Args:
            layers (list): Number of blocks in each layer.
            output_dim (int): Dimension of the output features.
            heads (int): Number of attention heads.
            input_resolution (int): Input image resolution. Default is 224.
            width (int): Width of the ResNet. Default is 64.
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # The 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)  # Shape: [N, width//2, H//2, W//2]
        self.bn1 = nn.BatchNorm2d(width // 2)  # Shape: [N, width//2, H//2, W//2]
        self.relu1 = nn.ReLU(inplace=True)     # Shape: [N, width//2, H//2, W//2]

        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)  # Shape: [N, width//2, H//2, W//2]
        self.bn2 = nn.BatchNorm2d(width // 2)  # Shape: [N, width//2, H//2, W//2]
        self.relu2 = nn.ReLU(inplace=True)     # Shape: [N, width//2, H//2, W//2]

        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)  # Shape: [N, width, H//2, W//2]
        self.bn3 = nn.BatchNorm2d(width)    # Shape: [N, width, H//2, W//2]
        self.relu3 = nn.ReLU(inplace=True)  # Shape: [N, width, H//2, W//2]

        self.avgpool = nn.AvgPool2d(2)  # Shape: [N, width, H//4, W//4]

        # Residual layers
        self._inplanes = width  # This is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])                # Shape: [N, width*4, H//4, W//4]
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)  # Shape: [N, width*8, H//8, W//8]
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)  # Shape: [N, width*16, H//16, W//16]
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)  # Shape: [N, width*32, H//32, W//32]

        embed_dim = width * 32  # The ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)  # Shape: [N, output_dim]

    def _make_layer(self, planes, blocks, stride=1):
        """
        Creates a residual layer.

        Args:
            planes (int): Number of output channels.
            blocks (int): Number of blocks in the layer.
            stride (int): Stride for the first block. Default is 1.

        Returns:
            nn.Sequential: A sequential container of the blocks.
        """
        layers = [Bottleneck(self._inplanes, planes, stride)]  # First block with possibly different stride

        self._inplanes = planes * Bottleneck.expansion         # Update inplanes for the next blocks
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))  # Subsequent blocks

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ModifiedResNet.

        Args:
            x (torch.Tensor): Input tensor with shape [N, 3, H, W].

        Returns:
            torch.Tensor: Output tensor with shape [N, output_dim].
        """
        def stem(x):
            """
            Forward pass through the stem layers.

            Args:
                x (torch.Tensor): Input tensor with shape [N, 3, H, W].

            Returns:
                torch.Tensor: Output tensor with shape [N, width, H//4, W//4].
            """
            x = self.relu1(self.bn1(self.conv1(x)))  # Shape: [N, width//2, H//2, W//2]
            x = self.relu2(self.bn2(self.conv2(x)))  # Shape: [N, width//2, H//2, W//2]
            x = self.relu3(self.bn3(self.conv3(x)))  # Shape: [N, width, H//2, W//2]
            x = self.avgpool(x)                      # Shape: [N, width, H//4, W//4]
            return x

        x = x.type(self.conv1.weight.dtype)  # Ensure input tensor is of the same type as conv1's weights
        x = stem(x)  # Pass through the stem layers
        x = self.layer1(x)    # Shape: [N, width*4, H//4, W//4]
        x = self.layer2(x)    # Shape: [N, width*8, H//8, W//8]
        x = self.layer3(x)    # Shape: [N, width*16, H//16, W//16]
        x = self.layer4(x)    # Shape: [N, width*32, H//32, W//32]
        x = self.attnpool(x)  # Shape: [N, output_dim]

        return x
```

### class VisionTransformer
A vision transformer model that processes images by dividing them into patches, projecting these patches into embeddings, and passing them through a transformer model.

```python
class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model.

    This class implements a Vision Transformer, which processes an input image by dividing it into patches,
    projecting these patches into a sequence of embeddings, and passing them through a transformer model.
    """

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        """
        Initializes the VisionTransformer.

        Args:
            input_resolution (int): Resolution of the input image.
            patch_size (int): Size of each image patch.
            width (int): Width of the transformer (number of features).
            layers (int): Number of transformer layers.
            heads (int): Number of attention heads in each layer.
            output_dim (int): Dimension of the output features.
        """
        super().__init__()
        self.input_resolution = input_resolution  # Input image resolution
        self.output_dim = output_dim  # Output feature dimension

        # Convolutional layer to divide the image into patches
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)  # Shape: [N, width, grid, grid]

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # Class embedding
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))  # Positional embedding
        self.ln_pre = LayerNorm(width)  # Layer normalization before transformer

        self.transformer = Transformer(width, layers, heads)  # Transformer model

        self.ln_post = LayerNorm(width)  # Layer normalization after transformer
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))  # Projection layer for the output

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the VisionTransformer.

        Args:
            x (torch.Tensor): Input tensor with shape [N, 3, H, W].

        Returns:
            torch.Tensor: Output tensor with shape [N, output_dim].
        """
        x = self.conv1(x)  # Shape: [N, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # Shape: [N, width, grid ** 2]
        x = x.permute(0, 2, 1)  # Shape: [N, grid ** 2, width]

        # Add class embedding and positional embeddings
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # Shape: [N, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)  # Add positional embeddings
        x = self.ln_pre(x)  # Apply layer normalization

        # Apply transformer
        x = x.permute(1, 0, 2)  # Shape: [N, grid ** 2 + 1, width] -> [grid ** 2 + 1, N, width]
        x = self.transformer(x)  # Shape: [grid ** 2 + 1, N, width]
        x = x.permute(1, 0, 2)  # Shape: [grid ** 2 + 1, N, width] -> [N, grid ** 2 + 1, width]

        # Apply final layer normalization and projection
        x = self.ln_post(x[:, 0, :])  # Take the class token [N, width]

        if self.proj is not None:
            x = x @ self.proj  # Shape: [N, output_dim]

        return x  # Shape: [N, output_dim]
```

### class CLIP
The main CLIP model class that combines the vision and text encoders, aligning these modalities in a shared embedding space using contrastive learning.

```python
class CLIP(nn.Module):
    """
    CLIP (Contrastive Language-Image Pretraining) model.

    This class implements the CLIP model, which consists of separate encoders for vision and text, and performs
    contrastive learning to align these modalities in a shared embedding space.
    """

    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):
        """
        Initializes the CLIP model.

        Args:
            embed_dim (int): Dimension of the shared embedding space.
            image_resolution (int): Resolution of the input images.
            vision_layers (Union[Tuple[int, int, int, int], int]): Configuration of the vision encoder layers.
            vision_width (int): Width of the vision encoder.
            vision_patch_size (int): Patch size for the vision transformer.
            context_length (int): Maximum context length for the text encoder.
            vocab_size (int): Size of the vocabulary for the text encoder.
            transformer_width (int): Width of the transformer model for text.
            transformer_heads (int): Number of attention heads in the text transformer.
            transformer_layers (int): Number of layers in the text transformer.
        """
        super().__init__()

        self.context_length = context_length

        # Initialize the vision encoder
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        # Initialize the text encoder
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # Shape: [vocab_size, transformer_width]
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))  # Shape: [context_length, transformer_width]
        self.ln_final = LayerNorm(transformer_width)  # Layer normalization for the output of the text transformer

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))  # Projection to the shared embedding space
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # Logit scaling factor

        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initializes the model parameters.
        """
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        """
        Builds the causal attention mask for the text transformer.

        Returns:
            torch.Tensor: Attention mask tensor of shape [context_length, context_length].
        """
        # Lazily create causal attention mask, with full attention between the vision tokens
        # PyTorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # Zero out the lower diagonal
        return mask  # Shape: [context_length, context_length]

    @property
    def dtype(self):
        """
        Returns the data type of the model weights.

        Returns:
            torch.dtype: Data type of the model weights.
        """
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        """
        Encodes an input image using the vision encoder.

        Args:
            image (torch.Tensor): Input image tensor of shape [N, 3, H, W].

        Returns:
            torch.Tensor: Encoded image features of shape [N, embed_dim].
        """
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        """
        Encodes input text using the text encoder.

        Args:
            text (torch.Tensor): Input text tensor of shape [N, context_length].

        Returns:
            torch.Tensor: Encoded text features of shape [N, embed_dim].
        """
        x = self.token_embedding(text).type(self.dtype)  # Shape: [N, context_length, transformer_width]

        x = x + self.positional_embedding.type(self.dtype)  # Add positional embeddings
        x = x.permute(1, 0, 2)  # Shape: [N, context_length, transformer_width] -> [context_length, N, transformer_width]
        x = self.transformer(x)  # Shape: [context_length, N, transformer_width]
        x = x.permute(1, 0, 2)  # Shape: [context_length, N, transformer_width] -> [N, context_length, transformer_width]
        x = self.ln_final(x).type(self.dtype)  # Apply final layer normalization

        # Take features from the end-of-text (eot) embedding
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection  # Shape: [N, embed_dim]

        return x

    def forward(self, image, text):
        """
        Forward pass of the CLIP model.

        Args:
            image (torch.Tensor): Input image tensor of shape [N, 3, H, W].
            text (torch.Tensor): Input text tensor of shape [N, context_length].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits for image-text and text-image pairs.
        """
        image_features = self.encode_image(image)  # Shape: [N, embed_dim]
        text_features = self.encode_text(text)  # Shape: [N, embed_dim]

        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()  # Shape: [N, N]
        logits_per_text = logits_per_image.t()  # Shape: [N, N]

        return logits_per_image, logits_per_text  # Shape: [global_batch_size, global_batch_size]
```

## 3. Main functions of CLIP
Knowing the main functions of CLIP is crucial for effectively using the model. The function **clip.available_models()** helps users identify available pre-trained models, while **clip.load()** enables proper loading of these models onto the desired device, ensuring readiness for inference or further training. **clip.tokenize()** is essential for correctly preprocessing text inputs, making them compatible with the model's requirements.

Understanding these functions optimizes workflow efficiency and reduces errors. It allows users to fully leverage CLIP's capabilities for tasks like zero-shot learning and image-text retrieval, ensuring flexibility and adaptability in various applications, from research to production deployment.

This table provides a concise overview of the main functions provided by the CLIP module, along with their descriptions, parameters, and return values.

Here is a table explaining the functions with their descriptions:

| **Function**                | **Description**                                                                                         |
|-----------------------------|---------------------------------------------------------------------------------------------------------|
| `build_model()` | Builds and returns a CLIP model from the provided state dictionary. It initializes the model, determines the type of vision encoder (ViT or ResNet), extracts necessary parameters, constructs the model, and loads the state dictionary into it. |
| `clip.available_models()`        | Returns the names of available CLIP models. It retrieves the keys from the `_MODELS` dictionary, representing the names of the pre-trained CLIP models that can be downloaded and used. |
| `clip.load()` | Loads a specified CLIP model and returns the model along with the necessary preprocessing transform. It supports both JIT and non-JIT models and handles downloading the model if necessary. |
| `clip.tokenize()` | Returns the tokenized representation of given input string(s). This function converts text into tokenized sequences that can be fed into the CLIP model, ensuring each sequence fits within the specified context length. |

### build_model()
The build_model function constructs and initializes a CLIP model from a provided state dictionary containing model parameters. It determines whether the model uses a Vision Transformer (ViT) or ResNet architecture, extracts the necessary parameters such as vision width, vision layers, and image resolution, and then initializes the CLIP model with these parameters. The function also converts the model weights to fp16 if applicable and removes unnecessary keys from the state dictionary before loading it into the model. Finally, the model is returned in evaluation mode.

```python
def build_model(state_dict: dict):
    """
    Builds and returns a CLIP model from the provided state dictionary.

    This function initializes a CLIP model based on the provided state_dict, which contains the model's parameters.
    It determines the type of vision encoder (ViT or ResNet), extracts necessary parameters, constructs the model,
    and loads the state_dict into the model.

    Args:
        state_dict (dict): A dictionary containing the model parameters.

    Returns:
        model (CLIP): The constructed and initialized CLIP model.
    """
    # Check if the model is a Vision Transformer (ViT) by looking for a specific key in the state_dict
    vit = "visual.proj" in state_dict

    if vit:
        # Extract vision transformer parameters
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        # Extract ResNet parameters
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    # Extract common parameters for both vision and text encoders
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # Initialize the CLIP model with the extracted parameters
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    # Remove unnecessary keys from the state_dict
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # Convert the model weights to fp16 (if applicable) and load the state_dict into the model
    convert_weights(model)
    model.load_state_dict(state_dict)
    
    # Return the model in evaluation mode
    return model.eval()
```

### clip.available_models()
The available_models function retrieves and returns the names of available pre-trained CLIP models. It accesses a predefined dictionary containing model names and their corresponding URLs for downloading model weights. The function returns a list of the model names, which can be used to load specific models using the clip.load function.

```python
def available_models() -> List[str]:
    """
    Returns the names of available CLIP models.

    This function retrieves the keys from the _MODELS dictionary, which represent the names of the pre-trained CLIP models
    that can be downloaded and used.

    Returns:
        List[str]: A list containing the names of available CLIP models.
    """
    # Dictionary containing available model names and their corresponding URLs for downloading the model weights
    _MODELS = {
        "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
        "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
        "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
        "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
        "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
        "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
    }

    # Return the list of available model names (keys of the _MODELS dictionary)
    return list(_MODELS.keys())
```

### clip.load()
The load function loads a specified CLIP model and returns the model along with the necessary preprocessing transform. It can load models by name from a list of available models or from a local checkpoint. The function allows specifying the device (CPU or CUDA) and whether to load a JIT-optimized model. It handles downloading the model if necessary and patches device names in JIT models. For non-JIT models, it builds the model from the state dictionary. The function returns the loaded model and a preprocessing transform that converts a PIL image into a tensor suitable for the model.

```python
def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """
    Load a CLIP model.

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict.

    device : Union[str, torch.device]
        The device to put the loaded model.

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default: False).

    download_root : str
        Path to download the model files; by default, it uses "~/.cache/clip".

    Returns
    -------
    model : torch.nn.Module
        The CLIP model.

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input.
    """
    # Check if the model name is in the available models and download the model if necessary
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        # If a local checkpoint is provided, use it
        model_path = name
    else:
        # Raise an error if the model name is not found
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # Try to load the JIT-compiled model
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # If loading JIT fails, fall back to loading the state dictionary
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        # Build the model using the state dictionary and move it to the specified device
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            # Use float32 precision on CPU
            model.float()
        return model, _transform(model.visual.input_resolution)

    # Patch the device names in the JIT model
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """
        Gets attributes of a node which is polymorphic over return type.
        
        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        """
        Patch the device of the given module.

        Args:
            module: The module to patch the device for.
        """
        # Patch the device attribute in the JIT model
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # Patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            """
            Patch the float function of a module. This function takes a module as input and patches its float function. 
            The patched float function will modify the behavior of the module's float method.

            Args:
                module: The module to be patched.
            """
            # Patch the data type attribute to float32 in the JIT model
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    # Return the model and the preprocessing transform
    return model, _transform(model.input_resolution.item())
```

Here is the calling sequence of clip.load() represented in a tree view:
```
clip.load()
├── _download() [if model name is in _MODELS]
│   ├── os.makedirs()
│   ├── urllib.request.urlopen()
│   ├── open()
│   └── hashlib.sha256()
├── open()
│   ├── torch.jit.load() [try block]
│   └── torch.load() [except block]
├── build_model() [if not jit]
│   ├── Extract parameters from state_dict
│   ├── CLIP.__init__()
│   ├── convert_weights()
│   └── model.load_state_dict()
├── torch.jit.trace() [if jit]
│   ├── lambda function to create tensor
│   └── device_node = device_holder.graph.findAllNodes()
├── patch_device() [if jit]
│   ├── module.graph
│   └── node.copyAttributes()
├── patch_float() [if device is CPU and jit]
│   ├── float_holder.graph.findNode()
│   ├── node.copyAttributes()
│   ├── model.apply()
│   ├── patch_float(model.encode_image)
│   └── patch_float(model.encode_text)
├── _transform()
└── return model
```

### clip.tokenize()
The tokenize function converts input text or a list of texts into tokenized sequences that can be fed into the CLIP model. Each sequence is padded or truncated to a specified context length. The function handles single and multiple text inputs, adds start-of-text and end-of-text tokens, and ensures compatibility with different PyTorch versions by selecting the appropriate tensor type. It populates a tensor with tokenized sequences, truncating if necessary based on the truncate parameter, and raises an error if truncation is not allowed and the input exceeds the context length. The function returns the tokenized sequences as a tensor.

```python
# We will introduce class SimpleTokenizer later
# _tokenizer = SimpleTokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s).

    This function converts input text or a list of texts into tokenized sequences that can be fed into the CLIP model. 
    Each tokenized sequence is padded or truncated to the specified context length.

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize.

    context_length : int
        The context length to use; all CLIP models use 77 as the context length.

    truncate : bool
        Whether to truncate the text in case its encoding is longer than the context length.

    Returns
    -------
    Union[torch.IntTensor, torch.LongTensor]
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    # Ensure the input is a list of strings
    if isinstance(texts, str):
        texts = [texts]

    # Define the start of text (sot) and end of text (eot) tokens
    sot_token = _tokenizer.encoder[""]
    eot_token = _tokenizer.encoder[""]

    # Tokenize each text and add sot and eot tokens
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    # Determine the appropriate tensor type based on the PyTorch version
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    # Populate the result tensor with tokenized sequences
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                # Truncate tokens to fit the context length
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                # Raise an error if tokens exceed the context length and truncation is not allowed
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
```

Here is the calling sequence of clip.tokenize() represented in a tree view:
```
clip.tokenize()
├── isinstance(texts, str)
│   └── texts = [texts]  [if texts is a string]
├── _tokenizer.encoder[""]
│   ├── sot_token
│   └── eot_token
├── _tokenizer.encode(text)  [for each text in texts]
│   └── all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
├── version.parse(torch.__version__) < version.parse("1.8.0")
│   ├── result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)  [if True]
│   └── result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)  [if False]
├── for i, tokens in enumerate(all_tokens)
│   ├── len(tokens) > context_length
│   │   ├── truncate  [if True]
│   │   │   ├── tokens = tokens[:context_length]
│   │   │   └── tokens[-1] = eot_token
│   │   └── raise RuntimeError  [if False and len(tokens) > context_length]
│   └── result[i, :len(tokens)] = torch.tensor(tokens)
└── return result
```

## 4. CLIP Model Inference Process

### Essential Steps of the CLIP Model Inference Process

1. **`clip.load`**: Loads the CLIP model and the preprocessing function for the specified model architecture.
2. **`preprocess`**: Preprocesses the input image by resizing, cropping, and normalizing it.
3. **`clip.tokenize`**: Tokenizes input text into a format suitable for the CLIP model.
4. **`model.encode_image`**: Encodes the preprocessed image into feature vectors.
5. **`model.encode_text`**: Encodes the tokenized text into feature vectors.
6. **`model.forward`**: Computes the similarity logits between image and text features.
7. **Softmax**: Converts logits to probabilities, indicating how well each text description matches the image.

### Breaking Down the CLIP Model Inference Process

1. **Load the Model**:
   - **Function**: `clip.load`

```python
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
```

2. **Image Processing**:
   - **Code**: Preprocessing the image

```python
# Preprocess the input image and add a batch dimension
image = preprocess(Image.open("path/to/image.png")).unsqueeze(0).to(device)
```

3. **Text Processing**:
   - **Function**: `clip.tokenize`

```python
# Tokenize the input text and move it to the specified device
text = clip.tokenize(["a description of the image"]).to(device)
```

4. **Feature Extraction**:
   - **Function**: `model.encode_image` and `model.encode_text`

```python
with torch.no_grad():
    # Encode the image and text to obtain their feature representations
    image_features = model.encode_image(image)  # Shape: [batch_size, 512]
    text_features = model.encode_text(text)    # Shape: [num_texts, 512]
```

5. **Normalization**:
   - **Code**: Normalizing the features

```python
    # Normalize the features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
```

6. **Similarity Calculation**:
   - **Function**: `model.forward` (implicitly called)

```python
    # Calculate similarity between image and text features
    logits_per_image, logits_per_text = model(image, text)  # Shape: [1, 3] and [3, 1]
```

7. **Softmax to Get Probabilities**:
   - **Code**: Apply softmax to the logits

```python
    # Apply softmax to the logits to obtain probabilities
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # Shape: [1, 3]
```

### Complete Inference Process Code

```python
import torch
import clip
from PIL import Image

# Load the model and preprocessing function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 1. Image Processing
image = preprocess(Image.open("path/to/image.png")).unsqueeze(0).to(device)

# 2. Text Processing
text = clip.tokenize(["a description of the image", "another possible description", "yet another description"]).to(device)

# 3. Feature Extraction and Inference
with torch.no_grad():
    # Encode the image and text to obtain their feature representations
    image_features = model.encode_image(image)  # Shape: [1, 512]
    text_features = model.encode_text(text)    # Shape: [3, 512]

    # Normalize the features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity between image and text features
    logits_per_image, logits_per_text = model(image, text)  # Shape: [1, 3] and [3, 1]

    # Apply softmax to the logits to obtain probabilities
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # Shape: [1, 3]

# Print the label probabilities
print("Label probs:", probs)  # Example output: [[0.9927937  0.00421068 0.00299572]]
```


## 5. Examples of Using CLIP

### Example 1: Basic Image-Text Matching with CLIP
This example demonstrates the fundamental use of the CLIP model for image and text matching. It loads the CLIP model, preprocesses an input image, and tokenizes input text. The model then encodes both the image and text to obtain their feature representations. It calculates the similarity between the image and text using the model's forward pass, resulting in probabilities that indicate how well the text descriptions match the image. This example highlights the core functionality of CLIP for image-text similarity tasks.

```python
import torch
import clip
from PIL import Image

# Determine the device to use for computation (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and the preprocessing function for the specified model
model, preprocess = clip.load("ViT-B/32", device=device)

# Preprocess the input image and add a batch dimension
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)  # Shape: [1, 3, H, W]

# Tokenize the input text and move it to the specified device
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)  # Shape: [3, 77]

# Disable gradient calculation for inference
with torch.no_grad():
    # Encode the image and text to obtain their feature representations
    image_features = model.encode_image(image)  # Shape: [1, 512]
    text_features = model.encode_text(text)  # Shape: [3, 512]
    
    # Get the logits for image-text similarity
    logits_per_image, logits_per_text = model(image, text)  # Shape: [1, 3] and [3, 1]

    # Apply softmax to the logits to obtain probabilities
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # Shape: [1, 3]

# Print the label probabilities
print("Label probs:", probs)  # Prints: [[0.9927937  0.00421068 0.00299572]]
```

### Example 2: Zero-Shot Prediction with CIFAR-100
This example showcases zero-shot prediction using CLIP. It loads the CLIP model and the CIFAR-100 dataset. An input image and all possible class names are prepared by preprocessing and tokenizing, respectively. The model encodes the image and text to extract their feature representations. It then calculates the similarity between the image features and each class's text features, applying softmax to get probabilities. The top 5 class predictions are printed, demonstrating how CLIP can make predictions without any fine-tuning on the specific dataset.

```python
import os
import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model and preprocessing function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# Download the CIFAR-100 dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]  # Select a specific image and its class ID
image_input = preprocess(image).unsqueeze(0).to(device)  # Preprocess the image and add a batch dimension, Shape: [1, 3, H, W]
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)  # Tokenize all class names, Shape: [100, 77]

# Calculate features
with torch.no_grad():  # Disable gradient calculation for inference
    image_features = model.encode_image(image_input)  # Encode the image, Shape: [1, 512]
    text_features = model.encode_text(text_inputs)  # Encode the text inputs, Shape: [100, 512]

# Normalize the features
image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize image features, Shape: [1, 512]
text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize text features, Shape: [100, 512]

# Calculate similarity between image and text features
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)  # Calculate softmax similarity, Shape: [1, 100]
values, indices = similarity[0].topk(5)  # Get top 5 most similar labels, values Shape: [5], indices Shape: [5]

# Print the top predictions
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")  # Print class name and confidence
```

### Example 3: Linear-Probe Evaluation with Logistic Regression
This example illustrates a linear-probe evaluation using logistic regression on features extracted by the CLIP model. The CIFAR-100 dataset is loaded and preprocessed. Features are extracted for both the training and test sets using the CLIP model. Logistic regression is then performed on the extracted features to classify the images. The accuracy of the logistic regression classifier is evaluated on the test set, showcasing how CLIP's feature representations can be used for downstream tasks like classification with traditional machine learning models.

```python
import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

# Load the CLIP model and preprocessing function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# Load the CIFAR-100 dataset with preprocessing applied
root = os.path.expanduser("~/.cache")
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)

def get_features(dataset):
    """
    Extracts features and labels from the dataset using the CLIP model.

    Args:
        dataset: The dataset from which to extract features.

    Returns:
        Tuple containing:
            - all_features (np.ndarray): Array of extracted features.
            - all_labels (np.ndarray): Array of corresponding labels.
    """
    all_features = []
    all_labels = []

    # Disable gradient calculation for feature extraction
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))  # Shape: [batch_size, 512]

            all_features.append(features)
            all_labels.append(labels)

    # Concatenate all features and labels into single arrays
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features for the training and test sets
train_features, train_labels = get_features(train)  # train_features Shape: [num_train_samples, 512], train_labels Shape: [num_train_samples]
test_features, test_labels = get_features(test)  # test_features Shape: [num_test_samples, 512], test_labels Shape: [num_test_samples]

# Perform logistic regression on the extracted features
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate the logistic regression classifier on the test set
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.  # Calculate accuracy
print(f"Accuracy = {accuracy:.3f}")  # Print the accuracy
```


## 6. Conclusion
In this document, we have explored the architecture, functionality, and practical applications of CLIP (Contrastive Language-Image Pre-training). CLIP is a powerful model developed by OpenAI that leverages the synergy between images and text to achieve state-of-the-art performance on various vision and language tasks. We delved into the main classes and functions that constitute CLIP, providing an in-depth understanding of its inner workings. Through practical examples, we demonstrated how CLIP can be used for image-text matching, zero-shot prediction, and linear-probe evaluation. These examples highlight the versatility and efficacy of CLIP in performing complex tasks without the need for extensive fine-tuning. As a versatile tool in the field of AI, CLIP opens up new possibilities for seamless integration of visual and textual data, paving the way for innovative applications across diverse domains.


## 7. References
- OpenAI. CLIP: Connecting Vision and Language. Retrieved from [OpenAI](https://openai.com/index/clip/)
- Hugging Face. CLIP ViT-Large-Patch14 Model. Retrieved from [Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14)
- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv preprint arXiv:2103.00020. Retrieved from [arXiv](https://arxiv.org/abs/2103.00020)
- OpenAI. CLIP GitHub Repository. Retrieved from [GitHub](https://github.com/openai/CLIP/tree/main/clip)

