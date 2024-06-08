# An Introduction to Sentence Transformer (Part 2)

## Introduction to the CrossEncoder API
Cross Encoder models, also known as rerankers, are designed to compute similarity scores for given pairs of texts. Unlike Sentence Transformers (bi-encoders), which generate embeddings for individual texts, Cross Encoders evaluate each pair of texts directly, resulting in a similarity score. This approach often yields superior performance compared to bi-encoders, particularly in tasks requiring detailed comparison between text pairs.

However, the detailed pairwise comparison of Cross Encoders comes with a trade-off: they are generally slower than Sentence Transformer models. This is because the computation needs to be repeated for each text pair, rather than just for each individual text. Due to their performance and computational characteristics, Cross Encoders are commonly used to re-rank the top-k results initially retrieved by a faster Sentence Transformer model. This combination leverages the efficiency of bi-encoders for initial retrieval and the accuracy of Cross Encoders for final ranking, making it a powerful approach in various natural language processing (NLP) applications.

### Overview of class CrossEncoder
The `CrossEncoder` class is a specialized model designed for comparing pairs of texts. Unlike Sentence Transformers (bi-encoders) that generate embeddings for individual texts, the CrossEncoder directly computes a similarity score or label for a given pair of texts. This makes CrossEncoders highly effective for tasks requiring fine-grained comparisons between text pairs, such as determining the semantic similarity of two sentences or ranking documents based on their relevance to a query.

Key characteristics of the `CrossEncoder` include:
- **Input Requirement**: It requires exactly two texts as input for each prediction.
- **Performance**: Generally provides superior performance compared to bi-encoders for pairwise comparison tasks.
- **Computation**: Slower than bi-encoders because it computes scores for each text pair individually rather than generating embeddings that can be reused.
- **Usage**: Often used as a second stage in a two-step retrieval process, where a bi-encoder first retrieves top-k results, and the CrossEncoder re-ranks these results for better accuracy and relevance.

The table below provides a summary of the main member functions of the `CrossEncoder` class, detailing their purpose and key parameters:
| Function                  | Description                                                                                             | Key Parameters                                                                                                                           |
|---------------------------|---------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `__init__`                | Initializes the CrossEncoder model.                                                                     | `model_name`, `num_labels`, `max_length`, `device`, `tokenizer_args`, `automodel_args`, `trust_remote_code`, `revision`, `local_files_only`, `default_activation_function`, `classifier_dropout`  |
| `smart_batching_collate`  | Prepares batches of input examples for training or evaluation.                                          | `batch` (List of InputExamples)                                                                                                          |
| `smart_batching_collate_text_only` | Prepares batches of text-only input examples.                                                              | `batch` (List of InputExamples)                                                                                                          |
| `fit`                     | Trains the model using the provided data loader and training parameters.                                | `train_dataloader`, `evaluator`, `epochs`, `loss_fct`, `activation_fct`, `scheduler`, `warmup_steps`, `optimizer_class`, `optimizer_params`, `weight_decay`, `evaluation_steps`, `output_path`, `save_best_model`, `max_grad_norm`, `use_amp`, `callback`, `show_progress_bar`  |
| `predict`                 | Performs predictions on the given sentence pairs.                                                       | `sentences`, `batch_size`, `show_progress_bar`, `num_workers`, `activation_fct`, `apply_softmax`, `convert_to_numpy`, `convert_to_tensor`|
| `rank`                    | Ranks the given query and documents, returning a sorted list of document indices and scores.            | `query`, `documents`, `top_k`, `return_documents`, `batch_size`, `show_progress_bar`, `num_workers`, `activation_fct`, `apply_softmax`, `convert_to_numpy`, `convert_to_tensor`  |
| `_eval_during_training`   | Runs evaluation during training.                                                                        | `evaluator`, `output_path`, `save_best_model`, `epoch`, `steps`, `callback`                                                              |
| `save`                    | Saves the model and tokenizer to the specified path.                                                    | `path`, `safe_serialization`, `kwargs`                                                                                                   |
| `save_pretrained`         | Saves the model and tokenizer to the specified path (alias for `save`).                                 | `path`, `safe_serialization`, `kwargs`                                                                                                   |
| `push_to_hub`             | Pushes the model to the HuggingFace Hub.                                                                | `repo_id`, `commit_message`, `private`, `safe_serialization`, `tags`, `kwargs`                                                           |



## Examples of CrossEncoder

### class CrossEncoder
The primary purpose of this example is to demonstrate the various ways to initialize and customize the `CrossEncoder` class from the `sentence_transformers` library for different use cases. The example covers a wide range of configurations, such as loading a pre-trained model, specifying the number of labels for classification tasks, setting the maximum length for input sequences, and choosing the computational device (e.g., GPU). It also highlights how to pass custom arguments to the tokenizer and AutoModel, enabling features like fast tokenization and attention output. Additionally, the example shows how to configure a custom activation function and adjust the classifier dropout rate, which can be critical for fine-tuning the model's performance on specific tasks.

In the comprehensive example, multiple parameters are combined to showcase the full potential of customization with the `CrossEncoder` class. After setting up the model with various configurations, the example demonstrates how to use the model to predict scores for pairs of sentences. By passing a list of sentence pairs to the `predict` method, the example illustrates how the `CrossEncoder` evaluates the semantic similarity or relevance of each pair, outputting scores that indicate their relationship. This practical application serves as a guide for users to tailor the `CrossEncoder` class to meet their specific needs in natural language processing tasks such as semantic similarity, paraphrase detection, and relevance ranking.

```python
from sentence_transformers import CrossEncoder

# Load a pretrained CrossEncoder model
# ----------------------------------------------------------
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Load a CrossEncoder model with specific number of labels
# ----------------------------------------------------------
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=3)

# Load a CrossEncoder model with a specified max length for input sequences
# ---------------------------------------------------------------------------
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=128)

# Load a CrossEncoder model and specify the device as GPU
# ----------------------------------------------------------
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")

# Load a CrossEncoder model with custom tokenizer arguments
# ----------------------------------------------------------
tokenizer_args = {"use_fast": True}
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", tokenizer_args=tokenizer_args)

# Load a CrossEncoder model with custom AutoModel arguments
# ----------------------------------------------------------
automodel_args = {"output_attentions": True}
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", automodel_args=automodel_args)

# Load a CrossEncoder model with a custom activation function
# -------------------------------------------------------------
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation_function=nn.Softmax(dim=1))

# Load a CrossEncoder model with a specified classifier dropout
# ---------------------------------------------------------------
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", classifier_dropout=0.1)

# Comprehensive example with multiple parameters
# ----------------------------------------------------------
tokenizer_args = {"use_fast": True}
automodel_args = {"output_attentions": True}
model = CrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    num_labels=3,
    max_length=128,
    device="cuda",
    tokenizer_args=tokenizer_args,
    automodel_args=automodel_args,
    default_activation_function=nn.Softmax(dim=1),
    classifier_dropout=0.1
)

# Define sentence pairs
sentence_pairs = [("The weather is lovely today.", "It's a beautiful day."),
                  ("He drove to the stadium.", "She went to the park.")]

# Predict scores
scores = model.predict(sentence_pairs)
print(scores)  # Output: [0.7, 0.2]
```

### CrossEncoder.predict()
The `predict` method of the `CrossEncoder` class performs predictions on pairs of sentences, determining their relationship based on the pre-trained model. This method outputs scores or labels for each pair, which can be used for tasks like semantic similarity, relevance ranking, or paraphrase identification. The output format can be customized to be a list, NumPy array, or PyTorch tensor.

The table below details the parameters used in the predict method of the CrossEncoder class. Each parameter is listed with its type, default value, and a brief description.
| Parameter             | Type                                            | Default     | Description                                                                                                                 |
|-----------------------|-------------------------------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------|
| `sentences`           | `List[List[str]]`                               | Required    | A list of sentence pairs, where each pair is a list of two strings `[[Sent1, Sent2], [Sent3, Sent4], ...]`.                  |
| `batch_size`          | `int`                                           | `32`        | The batch size used for processing the sentence pairs.                                                                      |
| `show_progress_bar`   | `Optional[bool]`                                | `None`      | If set to `True`, a progress bar will be displayed during the prediction process.                                            |
| `num_workers`         | `int`                                           | `0`         | The number of worker threads to use for tokenization. Increasing this value may speed up processing at the cost of memory.   |
| `activation_fct`      | `Optional[callable]`                            | `None`      | Activation function applied on the logits output. If `None`, `nn.Sigmoid()` is used if `num_labels=1`, otherwise `nn.Identity()`. |
| `apply_softmax`       | `bool`                                          | `False`     | If set to `True` and the model has more than 2 dimensions, softmax is applied on the logits output.                          |
| `convert_to_numpy`    | `bool`                                          | `True`      | If set to `True`, the output is converted to a NumPy array.                                                                  |
| `convert_to_tensor`   | `bool`                                          | `False`     | If set to `True`, the output is converted to a PyTorch tensor.                                                               |

```python
from sentence_transformers import CrossEncoder

# Load a pretrained CrossEncoder model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Define sentence pairs
sentence_pairs = [["The weather is lovely today.", "It's a beautiful day."],
                  ["He drove to the stadium.", "She went to the park."]]

# Predict scores with default settings
# ----------------------------------------------------------
scores = model.predict(sentence_pairs)

# Predict scores with a specified batch size
# ----------------------------------------------------------
scores = model.predict(sentence_pairs, batch_size=16)

# Predict scores and show progress bar
# ----------------------------------------------------------
scores = model.predict(sentence_pairs, show_progress_bar=True)

# Predict scores using multiple workers for tokenization
# ----------------------------------------------------------
scores = model.predict(sentence_pairs, num_workers=4)

# Predict scores with a custom activation function
# ----------------------------------------------------------
scores = model.predict(sentence_pairs, activation_fct=nn.Softmax(dim=1))

# Predict scores and apply softmax
# ----------------------------------------------------------
scores = model.predict(sentence_pairs, apply_softmax=True)

# Predict scores and convert output to numpy array
# ----------------------------------------------------------
scores = model.predict(sentence_pairs, convert_to_numpy=True)

# Predict scores and convert output to tensor
# ----------------------------------------------------------
scores = model.predict(sentence_pairs, convert_to_tensor=True)

# Comprehensive example with multiple parameters
# ----------------------------------------------------------
scores = model.predict(
    sentence_pairs,
    batch_size=16,
    show_progress_bar=True,
    num_workers=4,
    activation_fct=nn.Softmax(dim=1),
    apply_softmax=True,
    convert_to_numpy=True,
    convert_to_tensor=False
)

print(scores)
```

### CrossEncoder.rank()
The `rank` method of the `CrossEncoder` class is used to rank a list of documents based on their relevance to a given query. It takes a query and a list of documents as input and returns a sorted list of the document indices and scores, indicating how relevant each document is to the query. This method is particularly useful for information retrieval and relevance ranking tasks.

The table below provides a detailed description of the parameters used in the rank method of the CrossEncoder class. Each parameter is listed with its type, default value, and a brief description.
| Parameter            | Type                                            | Default     | Description                                                                                                                 |
|----------------------|-------------------------------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------|
| `query`              | `str`                                           | Required    | A single query string.                                                                                                      |
| `documents`          | `List[str]`                                     | Required    | A list of documents to be ranked based on the query.                                                                        |
| `top_k`              | `Optional[int]`                                 | `None`      | The number of top documents to return. If `None`, all documents are returned.                                               |
| `return_documents`   | `bool`                                          | `False`     | If `True`, the documents are also returned along with their indices and scores.                                             |
| `batch_size`         | `int`                                           | `32`        | The batch size for encoding.                                                                                                |
| `show_progress_bar`  | `Optional[bool]`                                | `None`      | If `True`, a progress bar is displayed during the ranking process.                                                          |
| `num_workers`        | `int`                                           | `0`         | The number of worker threads to use for tokenization. Increasing this value may speed up processing at the cost of memory.  |
| `activation_fct`     | `Optional[callable]`                            | `None`      | The activation function applied to the logits output. If `None`, `nn.Sigmoid()` is used if `num_labels=1`, otherwise `nn.Identity()`. |
| `apply_softmax`      | `bool`                                          | `False`     | If `True` and the model has more than 2 dimensions, softmax is applied on the logits output.                                |
| `convert_to_numpy`   | `bool`                                          | `True`      | If `True`, the output is converted to a NumPy array.                                                                        |
| `convert_to_tensor`  | `bool`                                          | `False`     | If `True`, the output is converted to a PyTorch tensor.                                                                     |

```python
from sentence_transformers import CrossEncoder

# Load a pretrained CrossEncoder model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Define query and documents
query = "What is the capital of France?"
documents = ["Paris is the capital of France.", "Berlin is the capital of Germany.", "Madrid is the capital of Spain."]

# Rank documents with default settings
# ----------------------------------------------------------
ranking = model.rank(query, documents)

# Rank documents and return top-2 documents
# ----------------------------------------------------------
ranking = model.rank(query, documents, top_k=2)

# Rank documents and return both documents and scores
# ----------------------------------------------------------
ranking = model.rank(query, documents, return_documents=True)

# Rank documents with a specified batch size
# ----------------------------------------------------------
ranking = model.rank(query, documents, batch_size=16)

# Rank documents and show progress bar
# ----------------------------------------------------------
ranking = model.rank(query, documents, show_progress_bar=True)

# Rank documents using multiple workers for tokenization
# ----------------------------------------------------------
ranking = model.rank(query, documents, num_workers=4)

# Rank documents with a custom activation function
# ----------------------------------------------------------
ranking = model.rank(query, documents, activation_fct=nn.Softmax(dim=1))

# Rank documents and apply softmax
# ----------------------------------------------------------
ranking = model.rank(query, documents, apply_softmax=True)

# Rank documents and convert output to numpy array
# ----------------------------------------------------------
ranking = model.rank(query, documents, convert_to_numpy=True)

# Rank documents and convert output to tensor
# ----------------------------------------------------------
ranking = model.rank(query, documents, convert_to_tensor=True)

# Comprehensive example with multiple parameters
# ----------------------------------------------------------
ranking = model.rank(
    query,
    documents,
    top_k=2,
    return_documents=True,
    batch_size=16,
    show_progress_bar=True,
    num_workers=4,
    activation_fct=nn.Softmax(dim=1),
    apply_softmax=True,
    convert_to_numpy=True,
    convert_to_tensor=False
)

print(ranking)
```

### util.semantic_search()
The `semantic_search` function in the `sentence_transformers` library performs a cosine similarity search between a list of query embeddings and a list of corpus embeddings. This function is particularly useful for information retrieval and semantic search in large corpora, capable of handling up to about 1 million entries. It returns a list of the top-k most similar entries from the corpus for each query.

The table below outlines the parameters for the semantic_search function in the sentence_transformers library. Each parameter is listed with its type, default value, and a brief description.
| Parameter            | Type                           | Default    | Description                                                                                                                |
|----------------------|--------------------------------|------------|----------------------------------------------------------------------------------------------------------------------------|
| `query_embeddings`   | `torch.Tensor`                 | Required   | A 2-dimensional tensor containing the embeddings for the queries.                                                          |
| `corpus_embeddings`  | `torch.Tensor`                 | Required   | A 2-dimensional tensor containing the embeddings for the corpus.                                                           |
| `query_chunk_size`   | `int`                          | `100`      | Number of queries processed simultaneously. Increasing this value increases speed but requires more memory.                 |
| `corpus_chunk_size`  | `int`                          | `500000`   | Number of corpus entries processed at a time. Increasing this value increases speed but requires more memory.               |
| `top_k`              | `int`                          | `10`       | Number of top matching entries to retrieve for each query.                                                                 |
| `score_function`     | `Callable[[torch.Tensor, torch.Tensor], torch.Tensor]` | `cos_sim` | Function for computing similarity scores. By default, cosine similarity is used.                                           |

```python
from sentence_transformers import SentenceTransformer, util
import torch

# Load a pretrained SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define the query and documents
query = "What is the capital of France?"
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy.",
    "London is the capital of the United Kingdom."
]

# Define a custom score function (e.g., dot product)
def dot_score(query_embeddings, corpus_embeddings):
    return torch.matmul(query_embeddings, corpus_embeddings.T)

# Encode the query and documents
query_embedding = model.encode(query, convert_to_tensor=True)
document_embeddings = model.encode(documents, convert_to_tensor=True)

# Perform semantic search with default settings
# ----------------------------------------------------------
hits = util.semantic_search(query_embedding, document_embeddings)

# Perform semantic search and retrieve top-2 documents
# ----------------------------------------------------------
hits = util.semantic_search(query_embedding, document_embeddings, top_k=2)

# Perform semantic search with a specified query chunk size
# ----------------------------------------------------------
hits = util.semantic_search(query_embedding, document_embeddings, query_chunk_size=50)

# Perform semantic search with a specified corpus chunk size
# ------------------------------------------------------------
hits = util.semantic_search(query_embedding, document_embeddings, corpus_chunk_size=100000)

# Perform semantic search with the custom score function
# ----------------------------------------------------------
hits = util.semantic_search(query_embedding, document_embeddings, score_function=dot_score)

# Perform semantic search with multiple parameters
# ----------------------------------------------------------
hits = util.semantic_search(
    query_embedding,
    document_embeddings,
    query_chunk_size=50,
    corpus_chunk_size=100000,
    top_k=3,
    score_function=dot_score
)

print(hits)
```


## Use Cases
### Retrieve & Re-Rank
This is an example of how to use a Bi-Encoder (SentenceTransformer) for retrieval and a Cross-Encoder (CrossEncoder) for re-ranking. This combination is commonly used to efficiently retrieve relevant documents and then re-rank them for better accuracy.

**Step-by-Step Implementation:**
1. **Retrieve top-k documents using a Bi-Encoder.**
2. **Re-rank the top-k documents using a Cross-Encoder.**

Here's the complete code with comments explaining each step:
```python
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# Load a pretrained Bi-Encoder model
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Load a pretrained Cross-Encoder model
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Define the query
query = "What is the capital of France?"

# Define the corpus of documents
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy.",
    "London is the capital of the United Kingdom."
]

# Step 1: 
# Retrieval using Bi-Encoder
# Encode the query and documents using the Bi-Encoder
query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
document_embeddings = bi_encoder.encode(documents, convert_to_tensor=True)

# Perform semantic search to retrieve top-k documents (e.g., top-3)
top_k = 3
hits = util.semantic_search(query_embedding, document_embeddings, top_k=top_k)
hits = hits[0]  # Retrieve the top-k hits for the given query

# Step 2: 
# Re-Ranking using Cross-Encoder
# Prepare the sentence pairs for the Cross-Encoder
sentence_pairs = [[query, documents[hit['corpus_id']]] for hit in hits]

# Predict scores using the Cross-Encoder
scores = cross_encoder.predict(sentence_pairs)

# Combine the scores with the original hits and re-rank them
for idx, score in enumerate(scores):
    hits[idx]['score'] = score

# Sort hits based on the Cross-Encoder scores
hits = sorted(hits, key=lambda x: x['score'], reverse=True)

# Print the re-ranked results
for hit in hits:
    print(f"Document: {documents[hit['corpus_id']]} (Score: {hit['score']})")

# Output Example:
# Document: Paris is the capital of France. (Score: 0.95)
# Document: Rome is the capital of Italy. (Score: 0.30)
# Document: Berlin is the capital of Germany. (Score: 0.25)
```

**Explanation:**
1. **Load Models**:
   - The `SentenceTransformer` model is loaded for the Bi-Encoder.
   - The `CrossEncoder` model is loaded for the Cross-Encoder.
2. **Define Query and Documents**:
   - A query and a list of documents are defined.
3. **Retrieval using Bi-Encoder**:
   - The query and documents are encoded into embeddings using the Bi-Encoder.
   - `util.semantic_search` is used to find the top-k documents most relevant to the query based on cosine similarity.
4. **Re-Ranking using Cross-Encoder**:
   - Sentence pairs (query, document) are created for the top-k retrieved documents.
   - The Cross-Encoder is used to predict relevance scores for each query-document pair.
   - The original hits are updated with the Cross-Encoder scores.
   - The hits are re-ranked based on the Cross-Encoder scores.
5. **Output**:
   - The re-ranked documents are printed with their corresponding scores.


## Conclusion
Sentence Transformers and Cross Encoders offer complementary strengths in the realm of natural language processing. Sentence Transformers are highly efficient for generating embeddings and performing tasks such as semantic search and clustering. Cross Encoders, while slower, provide superior accuracy for tasks requiring detailed comparison of text pairs, making them ideal for re-ranking top-k results from Sentence Transformers. By leveraging both models, one can achieve a balance between efficiency and accuracy in complex NLP applications.

By integrating the strengths of Sentence Transformers and Cross Encoders in a Retrieve & Re-Rank pipeline, one can create a robust system for various information retrieval tasks, combining the efficiency of initial retrieval with the precision of detailed re-ranking. This approach ensures that the most relevant results are identified and presented, providing a powerful solution for complex search and question-answering applications.


## References
1. [Sentence Transformers Quickstart](https://sbert.net/docs/quickstart.html)
2. [CrossEncoder Usage](https://sbert.net/docs/cross_encoder/usage/usage.html)
3. [CrossEncoder API Reference](https://sbert.net/docs/package_reference/cross_encoder/index.html)
