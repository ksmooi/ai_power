# An Introduction to Sentence Transformer (Part 1)

## Introduction to the Sentence Transformer
The Sentence Transformer, also known as a bi-encoder model, is a powerful tool designed to generate fixed-size vector representations (embeddings) from texts or images. These embeddings are essential for various natural language processing (NLP) tasks due to their efficient calculation and fast similarity computation. Sentence Transformers are widely applicable in tasks such as semantic textual similarity, semantic search, clustering, classification, and paraphrase mining.

A key feature of Sentence Transformer models is their role in two-step retrieval processes. Initially, a bi-encoder is used to quickly retrieve top-k results based on embedding similarity. Subsequently, a Cross-Encoder (or reranker) model re-ranks these top-k results for enhanced accuracy and relevance. This combination leverages the efficiency of bi-encoders and the detailed comparison capabilities of Cross-Encoders, making Sentence Transformers highly effective for complex NLP applications.

### Overview of class SentenceTransformer
The `SentenceTransformer` class provides several important member functions that facilitate embedding text, handling multi-process encoding, and managing the model. Below is a table listing the main member functions, their purpose, and their return types.

| Function                              | Purpose                    | Return Type                        |
|---------------------------------------|----------------------------|------------------------------------|
| `__init__`                            | Initializes the SentenceTransformer model with various configurations and parameters.           | `None`                             |
| `encode`                              | Computes sentence embeddings for a list of sentences.                                           | `Union[List[Tensor], ndarray, Tensor]` |
| `start_multi_process_pool`            | Starts a multi-process pool for encoding with multiple GPUs or CPUs.                             | `Dict[Literal["input", "output", "processes"], Any]` |
| `stop_multi_process_pool`             | Stops all processes started with `start_multi_process_pool`.                                    | `None`                             |
| `encode_multi_process`                | Encodes a list of sentences using multiple processes and GPUs.                                  | `np.ndarray`                       |
| `similarity`                          | Computes the similarity between two collections of embeddings.                                  | `Tensor`                           |
| `similarity_pairwise`                 | Computes the similarity between each pair of embeddings from two collections.                   | `Tensor`                           |
| `tokenize`                            | Tokenizes a list of texts.                                                                      | `Dict[str, Tensor]`                |
| `get_sentence_features`               | Extracts sentence features from the tokenized input.                                            | `Dict[Literal["sentence_embedding"], torch.Tensor]` |
| `get_sentence_embedding_dimension`    | Returns the number of dimensions in the output of `encode`.                                      | `Optional[int]`                    |
| `save`                                | Saves the model and its configuration files to a specified directory.                            | `None`                             |
| `save_pretrained`                     | Saves the model and its configuration files to a directory for loading later.                    | `None`                             |
| `push_to_hub`                         | Uploads the model to the Hugging Face Hub repository.                                            | `str`                              |
| `load`                                | Loads a SentenceTransformer model from a specified path.                                         | `SentenceTransformer`              |
| `evaluate`                            | Evaluates the model using a specified evaluator.                                                 | `Union[Dict[str, float], float]`   |
| `set_pooling_include_prompt`          | Sets the `include_prompt` attribute in the pooling layer, useful for INSTRUCTOR models.          | `None`                             |
| `truncate_sentence_embeddings`        | Context manager to truncate sentence embeddings to a specified dimension during encoding.        | `Iterator[None]`                   |


## Examples of Sentence Transformer

### class SentenceTransformer
The primary purpose of this example is to demonstrate the flexibility and customization options available when initializing and using the `SentenceTransformer` class from the `sentence_transformers` library. The example covers various scenarios, including loading a pre-trained model by name, loading a model from a local path, specifying the device (e.g., GPU) for computation, and utilizing custom prompts and similarity functions. It also shows how to configure the model for specific use cases, such as setting a cache folder, loading a specific model revision, and using Hugging Face authentication tokens for accessing private models. Additionally, the example highlights how to adjust model and tokenizer configurations, such as setting the data type and enabling fast tokenization.

In the final comprehensive example, multiple parameters are combined to illustrate the full extent of customization possible with the `SentenceTransformer` class. After setting up the model with various configurations, the example proceeds to encode a list of sentences, demonstrating the practical application of the configured model. The resulting embeddings are printed, showcasing the model's ability to convert sentences into fixed-size vectors suitable for various downstream tasks like semantic search, clustering, and similarity computations. This example serves as a practical guide for users to tailor the `SentenceTransformer` class to meet their specific requirements in different natural language processing applications.

```python
from sentence_transformers import SentenceTransformer

# Load a pre-trained model by name
# ----------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load a model from a local path
# ----------------------------------------------------------
model = SentenceTransformer("/path/to/local/model")

# Load model and specify device as GPU
# ----------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# Define custom prompts
# Load model with prompts
# ----------------------------------------------------------
prompts = {
    "query": "query: ",
    "passage": "passage: "
}
model = SentenceTransformer("all-MiniLM-L6-v2", prompts=prompts, default_prompt_name="query")

# Load model with custom similarity function
# ----------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.cosine)

# Specify cache folder for storing models
# ----------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/path/to/cache")

# Load a specific model revision
# ----------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2", revision="v1.0")

# Load model only from local files
# ----------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

# Load model with Hugging Face authentication token
# ----------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2", token="your_huggingface_token")

# Load model with truncated dimension for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2", truncate_dim=256)

# Custom model and tokenizer configuration parameters
# Load model with custom configurations
# ----------------------------------------------------------
model_kwargs = {"torch_dtype": "float16"}
tokenizer_kwargs = {"use_fast": True}
model = SentenceTransformer("all-MiniLM-L6-v2", model_kwargs=model_kwargs, tokenizer_kwargs=tokenizer_kwargs)

# Comprehensive example with multiple parameters
# ----------------------------------------------------------
model_kwargs = {"torch_dtype": "float16"}
tokenizer_kwargs = {"use_fast": True}
model = SentenceTransformer(
    model_name_or_path="all-MiniLM-L6-v2",
    device="cuda",
    prompts={"query": "query: ", "passage": "passage: "},
    default_prompt_name="query",
    similarity_fn_name="cosine",
    cache_folder="/path/to/cache",
    revision="v1.0",
    local_files_only=False,
    token="your_huggingface_token",
    truncate_dim=256,
    model_kwargs={"torch_dtype": "float16"},
    tokenizer_kwargs={"use_fast": True}
)

# Example sentences
sentences = ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]

# Calculate embeddings
embeddings = model.encode(sentences)
print(embeddings.shape)  # Output: (3, 384)
```

### SentenceTransformer.encode()
The `encode` method of the `SentenceTransformer` class computes embeddings for a given list of sentences. This method is highly versatile and allows for various customizations, such as specifying prompts, controlling batch sizes, choosing output formats, and more. The resulting embeddings can be used for tasks like semantic search, clustering, and similarity computations.

The table below provides detailed descriptions of the parameters used in the encode method of the SentenceTransformer class. Each parameter is listed with its type, default value, and a brief description of its function.
| Parameter             | Type                                                                 | Default    | Description                                                                                                                 |
|-----------------------|----------------------------------------------------------------------|------------|-----------------------------------------------------------------------------------------------------------------------------|
| `sentences`           | `Union[str, List[str]]`                                              | Required   | The sentences to embed.                                                                                                     |
| `prompt_name`         | `Optional[str]`                                                      | `None`     | The name of the prompt to use for encoding. Must be a key in the prompts dictionary. Ignored if `prompt` is set.            |
| `prompt`              | `Optional[str]`                                                      | `None`     | The prompt to use for encoding. Overrides `prompt_name`.                                                                    |
| `batch_size`          | `int`                                                                | `32`       | The batch size used for the computation.                                                                                    |
| `show_progress_bar`   | `Optional[bool]`                                                     | `None`     | Whether to output a progress bar when encoding sentences.                                                                   |
| `output_value`        | `Optional[Literal["sentence_embedding", "token_embeddings"]]`        | `sentence_embedding` | The type of embeddings to return: "sentence_embedding" for sentence embeddings, "token_embeddings" for wordpiece token embeddings. |
| `precision`           | `Literal["float32", "int8", "uint8", "binary", "ubinary"]`           | `float32`  | The precision to use for the embeddings. Quantized embeddings (non-float32) are smaller and faster but may have lower accuracy. |
| `convert_to_numpy`    | `bool`                                                               | `True`     | Whether the output should be a list of NumPy vectors. If `False`, the output is a list of PyTorch tensors.                  |
| `convert_to_tensor`   | `bool`                                                               | `False`    | Whether the output should be one large tensor. Overwrites `convert_to_numpy`.                                               |
| `device`              | `Optional[str]`                                                      | `None`     | Which `torch.device` to use for the computation.                                                                            |
| `normalize_embeddings`| `bool`                                                               | `False`    | Whether to normalize returned vectors to have length 1. If `True`, enables faster dot-product similarity instead of cosine similarity. |

```python
from sentence_transformers import SentenceTransformer

# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sentences to encode
sentences = ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]

# Calculate embeddings with default settings
# ----------------------------------------------------------
embeddings = model.encode(sentences)

# Using a prompt for encoding
# ----------------------------------------------------------
embeddings = model.encode(sentences, prompt="query: ")

# Specifying a different batch size
# ----------------------------------------------------------
embeddings = model.encode(sentences, batch_size=16)

# Showing a progress bar during encoding
# ----------------------------------------------------------
embeddings = model.encode(sentences, show_progress_bar=True)

# Getting token embeddings instead of sentence embeddings
# ----------------------------------------------------------
embeddings = model.encode(sentences, output_value="token_embeddings")

# Using quantized embeddings for smaller size and faster computation
# ----------------------------------------------------------
embeddings = model.encode(sentences, precision="int8")

# Returning the output as a PyTorch tensor
# ----------------------------------------------------------
embeddings = model.encode(sentences, convert_to_tensor=True)

# Specifying the device for computation (e.g., "cuda" for GPU)
# ----------------------------------------------------------
embeddings = model.encode(sentences, device="cuda")

# Normalizing embeddings to have length 1
# ----------------------------------------------------------
embeddings = model.encode(sentences, normalize_embeddings=True)

# Comprehensive example with multiple parameters
# ----------------------------------------------------------
embeddings = model.encode(
    sentences,
    prompt="query: ",
    batch_size=16,
    show_progress_bar=True,
    output_value="sentence_embedding",
    precision="float32",
    convert_to_numpy=True,
    convert_to_tensor=False,
    device="cuda",
    normalize_embeddings=True
)

print(embeddings.shape)  # Output: (3, 384)
```

### SentenceTransformer.encode_multi_process()
The `encode_multi_process` method of the `SentenceTransformer` class is designed for efficiently encoding large sets of sentences using multiple processes and GPUs. This method divides the sentences into smaller chunks and distributes them to individual processes, allowing parallel computation across different GPUs or CPUs. It is particularly useful for scenarios requiring high-throughput text encoding, such as semantic search, large-scale clustering, and corpus analysis.

The table below provides detailed descriptions of the parameters used in the encode_multi_process method of the SentenceTransformer class. Each parameter is listed with its type, default value, and a brief description of its function.
| Parameter            | Type                                                                 | Default    | Description                                                                                                                 |
|----------------------|----------------------------------------------------------------------|------------|-----------------------------------------------------------------------------------------------------------------------------|
| `sentences`          | `List[str]`                                                          | Required   | A list of sentences to encode.                                                                                              |
| `pool`               | `Dict[Literal["input", "output", "processes"], Any]`                 | Required   | A pool of workers started with `SentenceTransformer.start_multi_process_pool`.                                              |
| `prompt_name`        | `Optional[str]`                                                      | `None`     | The name of the prompt to use for encoding. Must be a key in the prompts dictionary. If `prompt` is also set, this is ignored. |
| `prompt`             | `Optional[str]`                                                      | `None`     | The prompt to use for encoding. If set, `prompt_name` is ignored.                                                           |
| `batch_size`         | `int`                                                                | `32`       | The batch size for encoding sentences.                                                                                      |
| `chunk_size`         | `Optional[int]`                                                      | `None`     | The chunk size for distributing sentences to individual processes. Determines a sensible size if `None`.                    |
| `precision`          | `Literal["float32", "int8", "uint8", "binary", "ubinary"]`           | `'float32'`| The precision to use for the embeddings. Quantized embeddings (non-float32) are smaller and faster but may have lower accuracy. |
| `normalize_embeddings`| `bool`                                                              | `False`    | Whether to normalize returned vectors to have length 1. Enables faster dot-product similarity if set to `True`.             |

```python
from sentence_transformers import SentenceTransformer, util

# Load model and define prompts
model = SentenceTransformer("all-MiniLM-L6-v2")
prompts = {"query": "query: ", "passage": "passage: "}
model.prompts = prompts
model.default_prompt_name = "query"

# Sentences to encode
sentences = ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]

# Start a multi-process pool
pool = model.start_multi_process_pool(['cuda:0', 'cuda:1'])  # Example with two GPUs

# Encode sentences using multiple processes
# ----------------------------------------------------------
embeddings = model.encode_multi_process(sentences, pool)
print(embeddings.shape)  # Output: (3, 384)

# Encode with prompt
# ----------------------------------------------------------
embeddings_with_prompt = model.encode_multi_process(sentences, pool, prompt="query: ")
print(embeddings_with_prompt.shape)  # Output: (3, 384)

# Encode with specified batch size
# ----------------------------------------------------------
embeddings_batch = model.encode_multi_process(sentences, pool, batch_size=16)
print(embeddings_batch.shape)  # Output: (3, 384)

# Encode with int8 precision for smaller and faster embeddings
# ----------------------------------------------------------
embeddings_int8 = model.encode_multi_process(sentences, pool, precision="int8")
print(embeddings_int8.shape)  # Output: (3, 384)

# Encode with normalized embeddings
# ----------------------------------------------------------
normalized_embeddings = model.encode_multi_process(sentences, pool, normalize_embeddings=True)
print(normalized_embeddings.shape)  # Output: (3, 384)

# Encode sentences with multiple parameters
# ----------------------------------------------------------
complex_embeddings = model.encode_multi_process(
    sentences,
    pool,
    prompt="query: ",
    batch_size=16,
    chunk_size=100,
    precision="float32",
    normalize_embeddings=True
)
print(complex_embeddings.shape)  # Output: (3, 384)

# Close the multi-process pool
model.stop_multi_process_pool(pool)
```

### util.paraphrase_mining()
The paraphrase_mining function in the sentence_transformers library is used to identify and rank pairs of sentences or texts that are semantically similar within a given list. It compares all sentences against each other and returns a list of pairs that have the highest similarity scores based on a specified similarity function, typically cosine similarity.

Below is a table that lists the parameters of the `paraphrase_mining` function along with their descriptions:
| Parameter            | Description                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------------|
| `show_progress_bar`  | Enables the display of a progress bar to visualize the processing progress.                     |
| `batch_size`         | Specifies the number of texts encoded simultaneously, affecting memory usage and processing time.|
| `query_chunk_size`   | Specifies the number of queries processed simultaneously. Decreasing this value reduces memory usage but increases runtime. |
| `corpus_chunk_size`  | Specifies the number of corpus entries processed simultaneously. Decreasing this value reduces memory usage but increases runtime. |
| `max_pairs`          | Specifies the maximum number of text pairs returned by the function.                            |
| `top_k`              | Specifies the number of top similar sentences retrieved for each sentence, determining the number of similar pairs returned. |
| `score_function`     | Allows using a custom score function (e.g., dot product) for computing similarity scores. By default, cosine similarity is used. |

```python
from sentence_transformers import SentenceTransformer, util

# Load a pretrained SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define a list of sentences
sentences = [
    "The weather is lovely today.",
    "It's a beautiful day.",
    "He drove to the stadium.",
    "She went to the park.",
    "The weather is nice today."
]

# Define a custom score function (e.g., dot product)
def dot_score(query_embeddings, corpus_embeddings):
    return torch.matmul(query_embeddings, corpus_embeddings.T)

# Perform paraphrase mining with default settings
# ----------------------------------------------------------
paraphrases = util.paraphrase_mining(model, sentences)

# Perform paraphrase mining and show progress bar
# ----------------------------------------------------------
paraphrases = util.paraphrase_mining(model, sentences, show_progress_bar=True)

# Perform paraphrase mining with a specified batch size
# ----------------------------------------------------------
paraphrases = util.paraphrase_mining(model, sentences, batch_size=16)

# Perform paraphrase mining with a specified query chunk size
# ----------------------------------------------------------
paraphrases = util.paraphrase_mining(model, sentences, query_chunk_size=2000)

# Perform paraphrase mining with a specified corpus chunk size
# ----------------------------------------------------------
paraphrases = util.paraphrase_mining(model, sentences, corpus_chunk_size=50000)

# Perform paraphrase mining with a specified max pairs limit
# ----------------------------------------------------------
paraphrases = util.paraphrase_mining(model, sentences, max_pairs=10)

# Perform paraphrase mining with a specified top-k limit
# ----------------------------------------------------------
paraphrases = util.paraphrase_mining(model, sentences, top_k=2)

# Perform paraphrase mining with the custom score function
# ----------------------------------------------------------
paraphrases = util.paraphrase_mining(model, sentences, score_function=dot_score)

# Perform paraphrase mining with multiple parameters
# ----------------------------------------------------------
paraphrases = util.paraphrase_mining(
    model,
    sentences,
    show_progress_bar=True,
    batch_size=16,
    query_chunk_size=2000,
    corpus_chunk_size=50000,
    max_pairs=10,
    top_k=2,
    score_function=dot_score
)

print(paraphrases)
# Output: [[0.95, 0, 4], [0.88, 0, 1], [0.75, 2, 3], ...]
```


## Use Cases
### Semantic Textual Similarity
Below example is to demonstrate how to use the SentenceTransformer model to compute embeddings for pairs of sentences and then calculate the semantic similarity between them. By comparing the embeddings using cosine similarity, we can determine how similar different pairs of sentences are, which is useful for tasks like semantic textual similarity (STS), paraphrase detection, and more.

```python
from sentence_transformers import SentenceTransformer

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Two lists of sentences for comparison
sentences1 = [
    "The new movie is awesome",
    "The cat sits outside",
    "A man is playing guitar",
]

sentences2 = [
    "The dog plays in the garden",
    "The new movie is so great",
    "A woman watches TV",
]

# Compute embeddings for the first list of sentences
embeddings1 = model.encode(sentences1)

# Compute embeddings for the second list of sentences
embeddings2 = model.encode(sentences2)

# Compute cosine similarities between embeddings of the two lists
similarities = model.similarity(embeddings1, embeddings2)

# Output the pairs with their similarity score
for idx_i, sentence1 in enumerate(sentences1):
    print(sentence1)
    for idx_j, sentence2 in enumerate(sentences2):
        # Print each pair of sentences with their cosine similarity score
        print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")
```

### Semantic Search
The main purpose of this example is to demonstrate how to use the SentenceTransformer model to perform semantic search. Semantic search aims to improve search accuracy by understanding the semantic meaning of the search query and the corpus, allowing it to find relevant results even if there are synonyms, abbreviations, or misspellings. This example shows how to encode a corpus of sentences and a set of query sentences to find the most semantically similar sentences from the corpus for each query.

```python
"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""

import torch
from sentence_transformers import SentenceTransformer

# Load a pre-trained Sentence Transformer model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Corpus with example sentences
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]
# Encode the corpus sentences into embeddings
# Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences
queries = [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field.",
]

# Find the closest 5 sentences in the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    # Encode the query sentence into an embedding
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # Compute cosine similarities between the query embedding and all corpus embeddings
    similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
    # Find the top 5 highest scores (most similar sentences)
    scores, indices = torch.topk(similarity_scores, k=top_k)

    # Print the query and the top 5 most similar sentences from the corpus
    print("\nQuery:", query)
    print("Top 5 most similar sentences in corpus:")

    for score, idx in zip(scores, indices):
        print(corpus[idx], "(Score: {:.4f})".format(score))

    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarity + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]  # Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """
```

### Paraphrase Mining
The main purpose of this example is to demonstrate how to use the SentenceTransformer model to perform paraphrase mining. Paraphrase mining is the task of finding texts with identical or similar meanings within a large corpus of sentences. This example shows how to efficiently find paraphrases in a large collection of sentences using the `paraphrase_mining()` function, which is optimized for scalability.

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Single list of sentences - Possible tens of thousands of sentences
sentences = [
    "The cat sits outside",
    "A man is playing guitar",
    "I love pasta",
    "The new movie is awesome",
    "The cat plays in the garden",
    "A woman watches TV",
    "The new movie is so great",
    "Do you like pizza?",
]

# Perform paraphrase mining to find similar sentences in the list
paraphrases = paraphrase_mining(model, sentences)

# Print the top 10 paraphrase pairs with their similarity scores
for paraphrase in paraphrases[0:10]:
    score, i, j = paraphrase
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))
```

**Code Explanation:**
- **Loading the Model**: The SentenceTransformer model is loaded with the pre-trained "all-MiniLM-L6-v2" model.
- **List of Sentences**: A list of sentences is defined, which can potentially contain tens of thousands of sentences.
- **Paraphrase Mining**: The `paraphrase_mining()` function is called with the model and the list of sentences. This function efficiently finds similar sentences in the list.
- **Output**: The top 10 paraphrase pairs are printed along with their similarity scores. The pairs are formatted to show the two similar sentences and their corresponding similarity score.


## Conclusion
In conclusion, Sentence Transformers provide a robust framework for generating embeddings that can be utilized in a variety of NLP tasks. Their ability to efficiently calculate and compare embeddings makes them suitable for applications requiring semantic understanding and similarity measurement. By integrating both bi-encoders and Cross-Encoders, Sentence Transformers offer a balanced approach combining speed and accuracy, making them indispensable tools in modern NLP.

Exploring further into the various classes and methods provided by the Sentence Transformers library, such as `SentenceTransformer.encode`, `encode_multi_process`, and `util.paraphrase_mining`, can greatly enhance one's ability to implement advanced NLP solutions. This introduction serves as a starting point for understanding and leveraging the full potential of Sentence Transformers in practical applications.


## References
1. [Sentence Transformers Quickstart](https://sbert.net/docs/quickstart.html)
2. [Sentence Transformers Usage](https://sbert.net/docs/sentence_transformer/usage/usage.html)
3. [Sentence Transformers Package Reference](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html)
