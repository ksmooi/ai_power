
# Understanding HuggingFace Auto Classes (Part 2)

## Introduction

### Overview of Auto Classes
The table below provides a comprehensive overview of the Auto Classes available in the HuggingFace Transformers library. Each Auto Class is designed to load a specific type of model or configuration for various tasks in Natural Language Processing (NLP), Vision, Audio, and Multimodal applications. These classes automate the process of retrieving the appropriate model architecture from the model name or path, making it easier for users to work with different pretrained models and their respective tasks.

| **Category** | **Class Name**                           | **Description** |
|--------------|------------------------------------------|-----------------|
| **NLP**      | `AutoModelForCausalLM`                   | Loads a model for causal language modeling. |
|              | `AutoModelForMaskedLM`                   | Loads a model for masked language modeling. |
|              | `AutoModelForMaskGeneration`             | Loads a model for mask generation tasks. |
|              | `AutoModelForSeq2SeqLM`                  | Loads a model for sequence-to-sequence tasks. |
|              | `AutoModelForSequenceClassification`     | Loads a model for text classification tasks. |
|              | `AutoModelForMultipleChoice`             | Loads a model for multiple-choice tasks. |
|              | `AutoModelForNextSentencePrediction`     | Loads a model for next sentence prediction tasks. |
|              | `AutoModelForTokenClassification`        | Loads a model for token classification tasks. |
|              | `AutoModelForQuestionAnswering`          | Loads a model for question answering tasks. |
|              | `AutoModelForTextEncoding`               | Loads a model for text encoding tasks. |
| **Vision**   | `AutoModelForDepthEstimation`            | Loads a model for depth estimation tasks. |
|              | `AutoModelForImageClassification`        | Loads a model for image classification tasks. |
|              | `AutoModelForVideoClassification`        | Loads a model for video classification tasks. |
|              | `AutoModelForKeypointDetection`          | Loads a model for keypoint detection in images. |
|              | `AutoModelForMaskedImageModeling`        | Loads a model for masked image modeling tasks. |
|              | `AutoModelForObjectDetection`            | Loads a model for object detection in images. |
|              | `AutoModelForImageSegmentation`          | Loads a model for segmenting images. |
|              | `AutoModelForImageToImage`               | Loads a model for image-to-image tasks. |
|              | `AutoModelForSemanticSegmentation`       | Loads a model for semantic segmentation of images. |
|              | `AutoModelForInstanceSegmentation`       | Loads a model for instance segmentation of images. |
|              | `AutoModelForUniversalSegmentation`      | Loads a model for universal segmentation of images. |
|              | `AutoModelForZeroShotImageClassification`| Loads a model for zero-shot image classification. |
|              | `AutoModelForZeroShotObjectDetection`    | Loads a model for zero-shot object detection. |
| **Audio**    | `AutoModelForAudioClassification`        | Loads a model for audio classification tasks. |
|              | `AutoModelForAudioFrameClassification`   | Loads a model for frame-wise audio classification tasks. |
|              | `AutoModelForCTC`                        | Loads a model for Connectionist Temporal Classification. |
|              | `AutoModelForSpeechSeq2Seq`              | Loads a model for sequence-to-sequence speech tasks. |
|              | `AutoModelForAudioXVector`               | Loads a model for audio x-vector tasks. |
|              | `AutoModelForTextToSpectrogram`          | Loads a model for text-to-spectrogram tasks. |
|              | `AutoModelForTextToWaveform`             | Loads a model for text-to-waveform tasks. |
| **Multimodal**| `AutoModelForTableQuestionAnswering`    | Loads a model for question answering from tables. |
|              | `AutoModelForDocumentQuestionAnswering`  | Loads a model for question answering from documents. |
|              | `AutoModelForVisualQuestionAnswering`    | Loads a model for visual question answering. |
|              | `AutoModelForVision2Seq`                 | Loads a model for vision-to-sequence tasks. |

These Auto Classes cover a wide range of tasks in natural language processing, computer vision, and audio processing, making it easier to leverage HuggingFace's extensive model hub.

## Examples in Action

### AutoModelForCausalLM
This example demonstrates how to load and use a model for causal language modeling, including performing inference with the model.

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

# Download configuration from huggingface.co and cache.
config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForCausalLM.from_config(config)

# Download model and configuration from huggingface.co and cache.
model = AutoModelForCausalLM.from_pretrained("google-bert/bert-base-cased")

# Update configuration during loading
model = AutoModelForCausalLM.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
print(model.config.output_attentions)

# Loading from a TensorFlow checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
model = AutoModelForCausalLM.from_pretrained(
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
    logits = outputs.logits

# Generate text
predicted_ids = torch.argmax(logits, dim=-1)
predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

print(predicted_text)
```


### AutoModelForSeq2SeqLM
This example demonstrates how to load and use a model for sequence-to-sequence tasks using HuggingFace's Auto Classes.

```python
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Download configuration from huggingface.co and cache.
config = AutoConfig.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_config(config)

# Download model and configuration from huggingface.co and cache.
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

# Update configuration during loading
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base", output_attentions=True)
print(model.config.output_attentions)

# Loading from a TensorFlow checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_pretrained("./tf_model/t5_tf_model_config.json")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "./tf_model/t5_tf_checkpoint.ckpt.index", from_tf=True, config=config
)

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

# Prepare some input text
input_text = "Translate English to French: How are you?"
inputs = tokenizer(input_text, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model.generate(**inputs)

# Decode the generated text
predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(predicted_text)
```


### AutoModelForSequenceClassification
This example demonstrates how to load and use a model for sequence classification, including performing inference with the model.

```python
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch

# Download configuration from huggingface.co and cache.
config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForSequenceClassification.from_config(config)

# Download model and configuration from huggingface.co and cache.
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")

# Update configuration during loading
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
print(model.config.output_attentions)

# Loading from a TensorFlow checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
model = AutoModelForSequenceClassification.from_pretrained(
    "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
)

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Prepare some input text
input_text = "This movie was amazing!"
inputs = tokenizer(input_text, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get predicted class
predicted_class_id = torch.argmax(logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_id]

print(predicted_label)
```



### AutoModelForQuestionAnswering
This example demonstrates how to load and use a model for question answering, including performing inference with the model.

```python
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Download configuration from huggingface.co and cache.
config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForQuestionAnswering.from_config(config)

# Download model and configuration from huggingface.co and cache.
model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-base-cased")

# Update configuration during loading
model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
print(model.config.output_attentions)

# Loading from a TensorFlow checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
model = AutoModelForQuestionAnswering.from_pretrained(
    "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
)

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Prepare the context and question
context = "HuggingFace Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge."
question = "Where is HuggingFace based?"

# Tokenize inputs
inputs = tokenizer(question, context, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

# Get the most likely beginning and end of the answer span
all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

print(answer)
```


### AutoModelForImageClassification
This example demonstrates how to load and use a model for image classification, including performing inference with the model.

```python
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image

# Download configuration from huggingface.co and cache.
config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForImageClassification.from_config(config)

# Download model and configuration from huggingface.co and cache.
model = AutoModelForImageClassification.from_pretrained("google-bert/bert-base-cased")

# Update configuration during loading
model = AutoModelForImageClassification.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
print(model.config.output_attentions)

# Loading from a TensorFlow checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
model = AutoModelForImageClassification.from_pretrained(
    "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
)

# Download the image processor
processor = AutoImageProcessor.from_pretrained("google-bert/bert-base-cased")

# Load and preprocess the image
image = Image.open("path/to/your/image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted class
predicted_class_id = torch.argmax(logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_id]

print(predicted_label)
```


### AutoModelForObjectDetection
This example demonstrates how to load and use a model for object detection, including performing inference with the model.

```python
from transformers import AutoConfig, AutoModelForObjectDetection, AutoImageProcessor
import torch
from PIL import Image

# Download configuration from huggingface.co and cache.
config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForObjectDetection.from_config(config)

# Download model and configuration from huggingface.co and cache.
model = AutoModelForObjectDetection.from_pretrained("google-bert/bert-base-cased")

# Update configuration during loading
model = AutoModelForObjectDetection.from_pretrained("google-bert/bert-base-cased", output_attentions=True)
print(model.config.output_attentions)

# Loading from a TensorFlow checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_pretrained("./tf_model/bert_tf_model_config.json")
model = AutoModelForObjectDetection.from_pretrained(
    "./tf_model/bert_tf_checkpoint.ckpt.index", from_tf=True, config=config
)

# Download the image processor
processor = AutoImageProcessor.from_pretrained("google-bert/bert-base-cased")

# Load and preprocess the image
image = Image.open("path/to/your/image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Process the outputs to get the detected objects
# Note: Actual object detection output processing might be more complex and involve additional steps.
predicted_boxes = outputs.pred_boxes
predicted_scores = outputs.scores
predicted_labels = [model.config.id2label[label_id] for label_id in torch.argmax(logits, dim=-1).tolist()]

print(predicted_boxes, predicted_scores, predicted_labels)
```



### AutoModelForAudioClassification
This example demonstrates how to load and use a model for audio classification, including performing inference with the model.

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch

# Download configuration from huggingface.co and cache.
#config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
#model = AutoModelForAudioClassification.from_config(config)

# Load the feature extractor from huggingface.co
feature_extractor = AutoFeatureExtractor.from_pretrained("stevhliu/my_awesome_minds_model")

# Example audio data (replace with actual data)
dataset = [{"audio": {"array": [0.1, 0.2, 0.3, 0.4], "sampling_rate": 16000}}]

# Extract features from the audio data
inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=dataset[0]["audio"]["sampling_rate"], return_tensors="pt")

# Load the audio classification model from huggingface.co
model = AutoModelForAudioClassification.from_pretrained("stevhliu/my_awesome_minds_model")

# Perform inference with the model
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted class
predicted_class_ids = torch.argmax(logits).item()
predicted_label = model.config.id2label[predicted_class_ids]

print(predicted_label)
```


### AutoModelForCTC
This example demonstrates how to load and use a model for Connectionist Temporal Classification (CTC), including performing inference with the model.

```python
from transformers import AutoProcessor, AutoModelForCTC
import torch

# Download configuration from huggingface.co and cache.
#config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
#model = AutoModelForCTC.from_config(config)

# Load the processor from huggingface.co
processor = AutoProcessor.from_pretrained("stevhliu/my_awesome_asr_mind_model")

# Example audio data (replace with actual data)
dataset = [{"audio": {"array": [0.1, 0.2, 0.3, 0.4], "sampling_rate": 16000}}]

# Process the audio data
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=dataset[0]["audio"]["sampling_rate"], return_tensors="pt")

# Load the CTC model from huggingface.co
model = AutoModelForCTC.from_pretrained("stevhliu/my_awesome_asr_mind_model")

# Perform inference with the model
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted IDs and decode them to text
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

print(transcription)
```


### AutoModelForTableQuestionAnswering
This example demonstrates how to load and use a model for table question answering, including performing inference with the model.

```python
from transformers import AutoConfig, AutoModelForTableQuestionAnswering, AutoTokenizer

# Download configuration from huggingface.co and cache.
config = AutoConfig.from_pretrained("google/tapas-base-finetuned-wtq")
model = AutoModelForTableQuestionAnswering.from_config(config)

# Load the model from huggingface.co
model = AutoModelForTableQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

# Update configuration during loading
model = AutoModelForTableQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq", output_attentions=True)
print(model.config.output_attentions)

# Loading from a TensorFlow checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_pretrained("./tf_model/tapas_tf_model_config.json")
model = AutoModelForTableQuestionAnswering.from_pretrained(
    "./tf_model/tapas_tf_checkpoint.ckpt.index", from_tf=True, config=config
)

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")

# Example table and question
table = {
    "rows": [
        {"name": "New York", "population": "8.4M"},
        {"name": "Los Angeles", "population": "3.9M"},
    ]
}
question = "What is the population of New York?"

# Tokenize inputs
inputs = tokenizer(table=table, query=question, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Process the outputs to get the answer (simplified example)
answer_ids = torch.argmax(logits, dim=-1).tolist()
answer = tokenizer.decode(answer_ids[0])

print(answer)
```


### AutoModelForDocumentQuestionAnswering
This example demonstrates how to load and use a model for document question answering, including performing inference with the model.

```python
from transformers import AutoConfig, AutoModelForDocumentQuestionAnswering, AutoProcessor
import torch

# Download configuration from huggingface.co and cache.
config = AutoConfig.from_pretrained("impira/layoutlm-document-qa", revision="52e01b3")
model = AutoModelForDocumentQuestionAnswering.from_config(config)

# Load the model from huggingface.co
model = AutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="52e01b3")

# Update configuration during loading
model = AutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="52e01b3", output_attentions=True)
print(model.config.output_attentions)

# Loading from a TensorFlow checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_pretrained("./tf_model/layoutlm_tf_model_config.json")
model = AutoModelForDocumentQuestionAnswering.from_pretrained(
    "./tf_model/layoutlm_tf_checkpoint.ckpt.index", from_tf=True, config=config
)

# Download the processor
processor = AutoProcessor.from_pretrained("impira/layoutlm-document-qa", revision="52e01b3")

# Example document image and question
image_path = "path/to/document/image.jpg"
question = "What is the date on the document?"

# Process the document image and question
image = processor(image=image_path, question=question, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**image)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

# Get the most likely beginning and end of the answer span
all_tokens = processor.tokenizer.convert_ids_to_tokens(image["input_ids"].numpy()[0])
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

print(answer)
```


### AutoModelForVisualQuestionAnswering
This example demonstrates how to load and use a model for visual question answering, including performing inference with the model.

```python
from transformers import AutoConfig, AutoModelForVisualQuestionAnswering, AutoProcessor
import torch

# Download configuration from huggingface.co and cache.
config = AutoConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = AutoModelForVisualQuestionAnswering.from_config(config)

# Load the model from huggingface.co
model = AutoModelForVisualQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Update configuration during loading
model = AutoModelForVisualQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa", output_attentions=True)
print(model.config.output_attentions)

# Loading from a TensorFlow checkpoint file instead of a PyTorch model (slower)
config = AutoConfig.from_pretrained("./tf_model/vilt_tf_model_config.json")
model = AutoModelForVisualQuestionAnswering.from_pretrained(
    "./tf_model/vilt_tf_checkpoint.ckpt.index", from_tf=True, config=config
)

# Download the processor
processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Example image and question
image_path = "path/to/image.jpg"
question = "What is the object in the center?"

# Process the image and question
inputs = processor(image=image_path, question=question, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted answer (simplified example)
predicted_answer_id = torch.argmax(logits, dim=-1).item()
predicted_answer = model.config.id2label[predicted_answer_id]

print(predicted_answer)
```


## References
For further reading and to deepen your understanding of HuggingFace Auto Classes, refer to the following resources:

- **[HuggingFace Documentation on Auto Classes](https://huggingface.co/docs/transformers/model_doc/auto) :** Provides comprehensive details and examples on utilizing Auto Classes for various models and tasks.
- **[HuggingFace AutoClass Tutorial](https://huggingface.co/docs/transformers/autoclass_tutorial) :** An in-depth tutorial on how to effectively use Auto Classes in your projects, ensuring you leverage their full potential.
- **[YouTube Video on HuggingFace Transformers](https://www.youtube.com/watch?v=IXxv7d0IHsA) :** A visual guide explaining the application of Auto Classes within the HuggingFace Transformers library.

These resources will provide you with detailed instructions, practical examples, and visual aids to enhance your proficiency with HuggingFace Auto Classes.


