
# Understanding HuggingFace Auto Classes (Part 1)

## Introduction

### Auto Classes of HuggingFace
Often, the architecture you want to use can be inferred from the name or path of the pretrained model you provide to the `from_pretrained()` method. Auto Classes are designed to simplify this process, enabling you to automatically obtain the relevant model, configuration, or tokenizer based on the provided name or path to the pretrained weights, configuration, or vocabulary.

Using Auto Classes like `AutoConfig`, `AutoModel`, and `AutoTokenizer`, you can directly create an instance of the required architecture without specifying it manually. This approach streamlines your workflow and ensures uniformity across various models and tasks.

Behind the scenes, `AutoModelForSequenceClassification` and `AutoTokenizer` collaborate to power the `pipeline()` function you used earlier. An AutoClass is a convenient shortcut that automatically identifies the architecture of a pretrained model based on its name or path. Simply choose the appropriate AutoClass for your task and its related preprocessing class, and you're ready to go.

### Overview of Auto Classes
| **Category** | **Class Name**                           | **Description** |
|--------------|------------------------------------------|-----------------|
| **Basic**    | `AutoConfig`                             | Loads the model configuration. |
|              | `AutoTokenizer`                          | Loads the tokenizer for text processing. |
|              | `AutoFeatureExtractor`                   | Loads feature extractor for non-text data. |
|              | `AutoImageProcessor`                     | Loads image processing utilities. |
|              | `AutoProcessor`                          | Loads the appropriate processor for the task. |
| **Model**    | `AutoModel`                              | Loads the base model architecture. |
|              | `AutoModelForPreTraining`                | Loads a model for pretraining tasks. |

These Auto Classes cover a wide range of tasks in natural language processing, computer vision, and audio processing, making it easier to leverage HuggingFace's extensive model hub.

## Examples in Action

### AutoConfig
This example demonstrates how to load and use configurations for models from HuggingFace's model hub.

```python
from transformers import AutoConfig

# Download configuration from huggingface.co and cache it.
config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")

# Download user-uploaded configuration from huggingface.co and cache it.
config = AutoConfig.from_pretrained("dbmdz/bert-base-german-cased")

# Load configuration from a local directory (e.g., saved using save_pretrained('./test/saved_model/')).
config = AutoConfig.from_pretrained("./test/bert_saved_model/")

# Load a specific configuration file from a local directory.
config = AutoConfig.from_pretrained("./test/bert_saved_model/my_configuration.json")

# Change some config attributes when loading a pretrained config.
config = AutoConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
config.output_attentions

# Load configuration and get unused arguments.
config, unused_kwargs = AutoConfig.from_pretrained(
    "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
)
config.output_attentions

# Print unused arguments.
unused_kwargs
```

### AutoTokenizer
This example shows how to load tokenizers for different models from HuggingFace's model hub.

```python
from transformers import AutoTokenizer

# Download vocabulary from huggingface.co and cache it.
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

# Download user-uploaded vocabulary from huggingface.co and cache it.
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

# Load vocabulary files from a local directory (e.g., tokenizer was saved using save_pretrained('./test/saved_model/')).
# tokenizer = AutoTokenizer.from_pretrained("./test/bert_saved_model/")

# Download vocabulary from huggingface.co and define model-specific arguments.
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", add_prefix_space=True)

# Prepare some input text
input_text = "Once upon a time"

# Tokenize the input text
encoded_input = tokenizer(input_text, return_tensors="pt")
print("Tokenized input:", encoded_input)

# Decode the token IDs back to text
decoded_text = tokenizer.decode(encoded_input["input_ids"][0])
print("Decoded text:", decoded_text)
```


### AutoFeatureExtractor
This example explains how to load feature extractors for audio models from HuggingFace's model hub.

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch

# Download feature extractor from huggingface.co and cache it.
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

# Load feature extractor files from a local directory (e.g., feature extractor was saved using save_pretrained('./test/saved_model/')).
# feature_extractor = AutoFeatureExtractor.from_pretrained("./test/saved_model/")

# Example audio data (replace with actual data)
audio_input = [0.1, 0.2, 0.3, 0.4]
sampling_rate = 16000

# Extract features from the audio data
inputs = feature_extractor(audio_input, sampling_rate=sampling_rate, return_tensors="pt")

# Load the audio classification model from huggingface.co
model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base-960h")

# Perform inference with the model
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted class
predicted_class_id = torch.argmax(logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_id]

print(predicted_label)
```


### AutoImageProcessor
This example illustrates how to load image processors for vision models from HuggingFace's model hub.

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image

# Download image processor from huggingface.co and cache it.
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Load image processor files from a local directory (e.g., image processor was saved using save_pretrained('./test/saved_model/')).
# image_processor = AutoImageProcessor.from_pretrained("./test/saved_model/")

# Load the image classification model from huggingface.co
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")

# Load and preprocess the image
image = Image.open("path/to/your/image.jpg")
inputs = image_processor(images=image, return_tensors="pt")

# Perform inference with the model
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted class
predicted_class_id = torch.argmax(logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_id]

print(predicted_label)
```


### AutoProcessor
This example shows how to load processors for multimodal models from HuggingFace's model hub.

```python
from transformers import AutoProcessor, AutoModelForCTC
import torch

# Download processor from huggingface.co and cache it.
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

# Load processor files from a local directory (e.g., processor was saved using save_pretrained('./test/saved_model/')).
# processor = AutoProcessor.from_pretrained("./test/saved_model/")

# Example audio data (replace with actual data)
audio_input = [0.1, 0.2, 0.3, 0.4]
sampling_rate = 16000

# Process the audio data
inputs = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt")

# Load the ASR model from huggingface.co
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Perform inference with the model
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted IDs and decode them to text
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

print(transcription)
```


### AutoModel
This example demonstrates how to load a model from a configuration object.

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch

# Download configuration from huggingface.co and cache it.
config = AutoConfig.from_pretrained("google-bert/bert-base-cased")

# Load a model from the configuration.
model = AutoModel.from_config(config)

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Prepare some input text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

print(last_hidden_states)
```

This example shows how to load a pretrained model from HuggingFace's model hub.

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch

# Download model and configuration from huggingface.co and cache it.
model = AutoModel.from_pretrained("google-bert/bert-base-cased")

# Update configuration during loading
model = AutoModel.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
print(model.config.output_attentions)

# Load from a TensorFlow checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
model = AutoModel.from_pretrained(
    "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
)

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Prepare some input text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

print(last_hidden_states)
```


### AutoModelForPreTraining
This example demonstrates how to load a pretraining model from a configuration object.

```python
from transformers import AutoConfig, AutoModelForPreTraining, AutoTokenizer
import torch

# Download configuration from huggingface.co and cache it.
config = AutoConfig.from_pretrained("google-bert/bert-base-cased")

# Load a model for pretraining from the configuration.
model = AutoModelForPreTraining.from_config(config)

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Prepare some input text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    prediction_logits = outputs.prediction_logits

print(prediction_logits)
```

This example shows how to load a pretrained model for pretraining from HuggingFace's model hub.

```python
from transformers import AutoConfig, AutoModelForPreTraining, AutoTokenizer
import torch

# Download model and configuration from huggingface.co and cache it.
model = AutoModelForPreTraining.from_pretrained("google-bert/bert-base-cased")

# Update configuration during loading.
model = AutoModelForPreTraining.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
print(model.config.output_attentions)

# Load from a TensorFlow checkpoint file instead of a PyTorch model (slower).
config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
model = AutoModelForPreTraining.from_pretrained(
    "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
)

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Prepare some input text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    prediction_logits = outputs.prediction_logits

print(prediction_logits)
```


## References
For further reading and to deepen your understanding of HuggingFace Auto Classes, refer to the following resources:

- **[HuggingFace Documentation on Auto Classes](https://huggingface.co/docs/transformers/model_doc/auto) :** Provides comprehensive details and examples on utilizing Auto Classes for various models and tasks.
- **[HuggingFace AutoClass Tutorial](https://huggingface.co/docs/transformers/autoclass_tutorial) :** An in-depth tutorial on how to effectively use Auto Classes in your projects, ensuring you leverage their full potential.
- **[YouTube Video on HuggingFace Transformers](https://www.youtube.com/watch?v=IXxv7d0IHsA) :** A visual guide explaining the application of Auto Classes within the HuggingFace Transformers library.

These resources will provide you with detailed instructions, practical examples, and visual aids to enhance your proficiency with HuggingFace Auto Classes.


