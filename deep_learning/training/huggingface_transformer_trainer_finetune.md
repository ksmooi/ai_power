# Exploring the HuggingFace Transformer Trainer

## Introduction of Transformer Trainer
The Transformer Trainer is a core component of the HuggingFace Transformers library designed to simplify the process of training and fine-tuning transformer models. It abstracts away much of the complexity involved in setting up and managing the training loop, making it easier for researchers and practitioners to focus on developing and refining their models. The Trainer provides a high-level interface that integrates seamlessly with popular deep learning frameworks like PyTorch, enabling efficient training and evaluation of transformer models on a wide range of tasks.

### Major Features of HuggingFace Transformer Trainer
1. **Support Dataset Preprocessing**
   - The Trainer integrates seamlessly with the datasets library, allowing for straightforward loading, preprocessing, and management of datasets. This includes tokenization, data augmentation, and batching, which are crucial steps for preparing data before training a model.
2. **Support Rich Training Hyperparameters**
   - The Trainer allows users to specify and manage a wide range of hyperparameters that control various aspects of the training process. These include learning rate, batch size, number of epochs, weight decay, gradient accumulation steps, and more. This flexibility ensures that users can fine-tune their training configurations to optimize model performance.
3. **Support Parameter-Efficient Fine-Tuning (PEFT) Methods**
   - PEFT methods enable efficient training by updating only a subset of model parameters instead of the entire model. This reduces computational requirements and speeds up training, making it feasible to fine-tune large models on limited resources. The Trainer supports various PEFT techniques, such as LoRA (Low-Rank Adaptation), which helps achieve high performance with fewer trainable parameters.
4. **Supports Both 8bit and 4bit Precision Data Types**
   - The Trainer supports mixed precision training, allowing models to be trained using 8bit or 4bit precision. This reduces memory usage and speeds up training without significantly impacting model accuracy. This feature is particularly useful for training large models on hardware with limited memory capacity.
5. **Built-in Function for Computing and Reporting Metrics**
   - The Trainer includes built-in functionality to compute and report evaluation metrics during training. This helps users monitor model performance on validation datasets in real-time, enabling early stopping, hyperparameter tuning, and identification of potential overfitting. Metrics can be logged to various platforms like TensorBoard, Weights & Biases, and more.
6. **Simplifies the Training Loop**
   - The Trainer abstracts the complexity of the training loop, including forward and backward passes, gradient computation, optimization steps, and checkpointing. This allows users to focus on higher-level tasks such as model architecture and data preparation, rather than the intricacies of training procedures. By providing a high-level interface, the Trainer makes it easier to implement and manage the entire training workflow.

These features make the HuggingFace Transformer Trainer a powerful and user-friendly tool for training and fine-tuning transformer models, catering to both beginners and advanced users in the field of natural language processing.


## Examples of Transformer Trainer
## Overview of Key Classes

Below are brief introductions to three key classes used in the HuggingFace Transformers library, along with a summary of their major features and use cases.

| **Class**             | **Description**                               |
|-----------------------|-----------------------------------------------|
| `LoraConfig`          | A configuration class to store parameters for Low-Rank Adaptation (LoRA) of transformer models. It supports specifying target modules, alpha parameter for scaling, dropout probability, and other settings for fine-tuning models. |
| `TrainingArguments`   | A class that defines a wide range of training hyperparameters and settings used by the `Trainer`. This includes settings for batch size, learning rate, number of epochs, evaluation strategy, logging, and more.                   |
| `Trainer`             | A high-level API that simplifies the training and evaluation of transformer models. It integrates seamlessly with datasets and manages the training loop, evaluation metrics, and model checkpointing.                              |

## class LoraConfig
The `LoraConfig` class is designed to store the configuration of a `LoraModel`. It contains various parameters that control the behavior and settings of the Lora layers applied to a model.

### Overview of LoraConfig

The table below provides a description of the key parameters in the `LoraConfig` class, explaining their purpose and usage.

| **Parameter** | **Description** |
|---------------|-----------------|
| `r` | Lora attention dimension (the "rank"). |
| `target_modules` | The names of the modules to apply the adapter to. If specified, only the modules with the specified names will be replaced. Supports string (regex match) or list of strings (exact match or suffix match). Special value 'all-linear' matches all linear/Conv1D modules except the output layer. |
| `lora_alpha` | The alpha parameter for Lora scaling. |
| `lora_dropout` | The dropout probability for Lora layers. |
| `fan_in_fan_out` | Set this to True if the layer to replace stores weight like (fan_in, fan_out). |
| `bias` | Bias type for LoRA. Can be 'none', 'all' or 'lora_only'. |
| `use_rslora` | When set to True, uses Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`. |
| `modules_to_save` | List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint. |
| `init_lora_weights` | Method to initialize the weights of the adapter layers. Can be True (default), 'gaussian', 'pissa', 'pissa_niter_[number of iters]', or 'loftq'. |
| `layers_to_transform` | The layer indices to transform. Can be a list of ints or a single integer. |
| `layers_pattern` | The layer pattern name, used only if `layers_to_transform` is specified. |
| `rank_pattern` | Mapping from layer names or regexp expression to ranks different from the default rank specified by `r`. |
| `alpha_pattern` | Mapping from layer names or regexp expression to alphas different from the default alpha specified by `lora_alpha`. |
| `megatron_config` | TransformerConfig arguments for Megatron, used to create LoRA's parallel linear layer. |
| `megatron_core` | Core module from Megatron to use. Defaults to 'megatron.core'. |
| `loftq_config` | Configuration of LoftQ. If specified, LoftQ will be used to quantize the backbone weights and initialize Lora layers. |
| `use_dora` | Enable 'Weight-Decomposed Low-Rank Adaptation' (DoRA), which decomposes the updates of the weights into magnitude and direction. |
| `layer_replication` | Build a new stack of layers by stacking the original model layers according to the specified ranges, allowing model expansion without duplicating base weights. |


### Examples of Using LoraConfig

**1. Basic Configuration**

Set up a basic configuration with essential parameters.
```python
from dataclasses import dataclass, field
from typing import List, Optional, Union
from peft import LoraConfig

# Basic LoraConfig setup
config = LoraConfig(
    r=16,                       # Set the rank of the Lora attention dimension
    target_modules=['q', 'v'],  # Apply Lora to specific modules
    lora_alpha=32,              # Set the alpha parameter for scaling
    lora_dropout=0.1,           # Apply dropout with 10% probability
    fan_in_fan_out=True,        # Indicate that the layer stores weight like (fan_in, fan_out)
    bias='all',                 # Update all biases during training
    use_rslora=True             # Use Rank-Stabilized LoRA
)
```

**2. Advanced Configuration with Custom Initialization**

Set up a more advanced configuration using custom initialization and saving specific modules.
```python
from peft import LoraConfig

# Advanced LoraConfig setup
config = LoraConfig(
    r=32,                                       # Set a higher rank for better performance
    target_modules='.*decoder.*(SelfAttention|EncDecAttention).*',  # Use regex to match target modules
    lora_alpha=64,                              # Increase alpha for better scaling
    lora_dropout=0.2,                           # Apply higher dropout
    fan_in_fan_out=False,                       # Indicate the layer stores weight like (fan_out, fan_in)
    bias='lora_only',                           # Update only Lora biases
    use_rslora=False,                           # Use the original Lora scaling
    modules_to_save=['classifier', 'score'],    # Save additional modules
    init_lora_weights='gaussian'                # Initialize Lora weights using Gaussian initialization
)
```

**3. Configuration for Specific Layers and Patterns**

Target specific layers and use custom patterns for transformation.
```python
from peft import LoraConfig

# Configuration targeting specific layers and patterns
config = LoraConfig(
    r=8,  # Set a lower rank
    target_modules='all-linear',                    # Apply to all linear/Conv1D layers except the output layer
    lora_alpha=16,                                  # Lower alpha for scaling
    lora_dropout=0.05,                              # Apply lower dropout
    bias='none',                                    # Do not update any biases
    use_rslora=True,                                # Use Rank-Stabilized LoRA
    layers_to_transform=[0, 1, 2],                  # Transform specific layers
    layers_pattern='pattern_name',                  # Use custom layer pattern name
    rank_pattern={'model.layer.0.attention': 4},    # Set specific rank for a layer
    alpha_pattern={'model.layer.0.attention': 10}   # Set specific alpha for a layer
)
```

**4. Configuration with LoftQ and DoRA**

Use LoftQ for quantization and enable DoRA for weight decomposition.
```python
from peft import LoraConfig

# Configuration with LoftQ and DoRA
config = LoraConfig(
    r=16,                                   # Set the rank
    target_modules='all-linear',            # Apply to all linear/Conv1D layers except the output layer
    lora_alpha=32,                          # Set alpha for scaling
    lora_dropout=0.1,                       # Apply dropout
    fan_in_fan_out=True,                    # Indicate the layer stores weight like (fan_in, fan_out)
    bias='all',                             # Update all biases
    use_rslora=False,                       # Use the original Lora scaling
    init_lora_weights='loftq',              # Use LoftQ initialization
    loftq_config={'quantization_level': 8}, # Configuration for LoftQ
    use_dora=True,                          # Enable DoRA for weight decomposition
    layer_replication=[(0, 4), (2, 5)]      # Replicate layers for model expansion
)
```

## class TrainingArguments
The `TrainingArguments` class is used to define and manage the various hyperparameters and configurations required for training, evaluation, and testing of models using the Hugging Face `transformers` library.

### Overview of TrainingArguments

This table provides an explanation of the key parameters used in the TrainingArguments constructor. These parameters allow you to customize and control different aspects of the training and evaluation process, ensuring flexibility and efficiency during model training.

| **Parameter**                   | **Description**                                                         |
|---------------------------------|-------------------------------------------------------------------------|
| `output_dir`                    | The directory where the model predictions and checkpoints will be written.                                                                                    |
| `do_train`                      | Whether to run training.                                                                                                                                      |
| `do_eval`                       | Whether to run evaluation on the validation set.                                                                                                              |
| `do_predict`                    | Whether to run predictions on the test set.                                                                                                                   |
| `evaluation_strategy`           | The evaluation strategy to adopt during training. Options are "no", "steps", "epoch".                                                                          |
| `prediction_loss_only`          | Whether to only return the loss when doing evaluation or predictions.                                                                                         |
| `per_device_train_batch_size`   | The batch size per GPU/TPU core/CPU for training.                                                                                                             |
| `per_device_eval_batch_size`    | The batch size per GPU/TPU core/CPU for evaluation.                                                                                                           |
| `gradient_accumulation_steps`   | Number of updates steps to accumulate before performing a backward/update pass.                                                                                |
| `eval_accumulation_steps`       | Number of prediction steps to accumulate before moving the results to the CPU.                                                                                 |
| `learning_rate`                 | The initial learning rate for the optimizer.                                                                                                                  |
| `weight_decay`                  | The weight decay to apply (if any) to all layers except all bias and LayerNorm weights.                                                                        |
| `adam_beta1`                    | The beta1 parameter for the Adam optimizer.                                                                                                                   |
| `adam_beta2`                    | The beta2 parameter for the Adam optimizer.                                                                                                                   |
| `adam_epsilon`                  | The epsilon parameter for the Adam optimizer.                                                                                                                 |
| `max_grad_norm`                 | Maximum norm of gradients for gradient clipping.                                                                                                              |
| `num_train_epochs`              | Total number of training epochs to perform.                                                                                                                   |
| `max_steps`                     | If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.                                                      |
| `lr_scheduler_type`             | The scheduler type to use.                                                                                                                                     |
| `warmup_steps`                  | Number of steps used for a linear warmup from 0 to `learning_rate`.                                                                                            |
| `warmup_ratio`                  | Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.                                                                              |
| `logging_dir`                   | The directory where the logs will be written.                                                                                                                 |
| `logging_strategy`              | The logging strategy to adopt during training. Options are "no", "steps", "epoch".                                                                             |
| `logging_steps`                 | Number of update steps between two logs if `logging_strategy` is "steps".                                                                                     |
| `save_strategy`                 | The checkpoint save strategy to adopt during training. Options are "no", "epoch", "steps".                                                                    |
| `save_steps`                    | Number of updates steps before two checkpoint saves if `save_strategy` is "steps".                                                                             |
| `save_total_limit`              | If set, will limit the total amount of checkpoints. Deletes the older checkpoints in `output_dir`.                                                            |
| `seed`                          | Random seed for initialization.                                                                                                                               |
| `dataloader_drop_last`          | Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size).                                                   |
| `dataloader_num_workers`        | Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.                                                      |
| `dataloader_pin_memory`         | Whether to pin memory in data loaders or not.                                                                                                                 |
| `load_best_model_at_end`        | Whether to load the best model found at the end of training.                                                                                                   |
| `metric_for_best_model`         | The metric to use to compare two different models.                                                                                                            |
| `greater_is_better`             | Whether the `metric_for_best_model` should be maximized or not.                                                                                               |
| `ignore_data_skip`              | When resuming training, whether or not to skip the epochs and batches to get the data loading at the same stage as in the previous training.                   |
| `optim`                         | The optimizer to use. Options include "adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adamw_anyprecision", or "adafactor".               |
| `report_to`                     | The list of integrations to report the results and logs to. Supported platforms include "azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", and more. |
| `resume_from_checkpoint`        | The path to a folder with a valid checkpoint for your model.                                                                                                  |
| `push_to_hub`                   | Whether to push the model to the HuggingFace Model Hub.                                                                                                        |

### Examples of Using TrainingArguments

**1. Basic Training Configuration**

This example demonstrates a simple training setup with essential parameters.
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",                    # Directory to save model checkpoints and predictions
    do_train=True,                             # Enable training
    do_eval=True,                              # Enable evaluation
    evaluation_strategy="steps",               # Evaluate every N steps
    per_device_train_batch_size=8,             # Batch size for training
    per_device_eval_batch_size=16,             # Batch size for evaluation
    learning_rate=5e-5,                        # Initial learning rate
    num_train_epochs=3,                        # Number of training epochs
    logging_dir="./logs",                      # Directory to save logs
    logging_steps=500,                         # Log every N steps
    save_steps=1000                            # Save checkpoints every N steps
)
```

**2. Advanced Training Configuration**

This example demonstrates an advanced setup with gradient accumulation and learning rate scheduler.
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",                    # Directory to save model checkpoints and predictions
    do_train=True,                             # Enable training
    do_eval=True,                              # Enable evaluation
    evaluation_strategy="epoch",               # Evaluate at the end of each epoch
    per_device_train_batch_size=8,             # Batch size for training
    per_device_eval_batch_size=16,             # Batch size for evaluation
    learning_rate=3e-5,                        # Initial learning rate
    num_train_epochs=5,                        # Number of training epochs
    gradient_accumulation_steps=4,             # Accumulate gradients over 4 steps
    lr_scheduler_type="linear",                # Use linear learning rate scheduler
    warmup_steps=500,                          # Warm up learning rate over 500 steps
    logging_dir="./logs",                      # Directory to save logs
    logging_steps=200,                         # Log every N steps
    save_strategy="epoch",                     # Save checkpoint at the end of each epoch
    save_total_limit=3,                        # Only keep the last 3 checkpoints
    load_best_model_at_end=True,               # Load the best model at the end of training
    metric_for_best_model="accuracy",          # Use accuracy to determine the best model
    greater_is_better=True                     # Higher accuracy is better
)
```

**3. Distributed Training and Mixed Precision**

This example shows a setup for distributed training and mixed precision training.
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",                    # Directory to save model checkpoints and predictions
    do_train=True,                             # Enable training
    do_eval=True,                              # Enable evaluation
    evaluation_strategy="steps",               # Evaluate every N steps
    per_device_train_batch_size=16,            # Batch size for training
    per_device_eval_batch_size=32,             # Batch size for evaluation
    learning_rate=2e-5,                        # Initial learning rate
    num_train_epochs=10,                       # Number of training epochs
    logging_dir="./logs",                      # Directory to save logs
    logging_steps=100,                         # Log every N steps
    save_strategy="epoch",                     # Save checkpoint at the end of each epoch
    save_total_limit=2,                        # Only keep the last 2 checkpoints
    load_best_model_at_end=True,               # Load the best model at the end of training
    metric_for_best_model="f1",                # Use F1 score to determine the best model
    greater_is_better=True,                    # Higher F1 score is better
    seed=42,                                   # Set random seed for reproducibility
    dataloader_drop_last=True,                 # Drop the last incomplete batch
    dataloader_num_workers=4,                  # Number of subprocesses for data loading
    dataloader_pin_memory=True,                # Pin memory in data loaders
    optim="adamw_torch",                       # Optimizer type
    report_to="wandb",                         # Report results to Weights & Biases
    push_to_hub=True                           # Push the model to the Hugging Face Hub
)
```

**4. Custom Logging and Checkpointing**

This example demonstrates custom logging and checkpointing settings.
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",                    # Directory to save model checkpoints and predictions
    do_train=True,                             # Enable training
    do_eval=True,                              # Enable evaluation
    evaluation_strategy="epoch",               # Evaluate at the end of each epoch
    prediction_loss_only=True,                 # Only log the loss during evaluation
    per_device_train_batch_size=32,            # Batch size for training
    per_device_eval_batch_size=64,             # Batch size for evaluation
    learning_rate=1e-4,                        # Initial learning rate
    num_train_epochs=3,                        # Number of training epochs
    logging_dir="./logs",                      # Directory to save logs
    logging_strategy="epoch",                  # Log at the end of each epoch
    logging_steps=50,                          # Log every N steps (ignored if logging_strategy is 'epoch')
    save_strategy="no",                        # Do not save checkpoints
    seed=123,                                  # Set random seed for reproducibility
    dataloader_drop_last=False,                # Do not drop the last incomplete batch
    dataloader_num_workers=2,                  # Number of subprocesses for data loading
    dataloader_pin_memory=False,               # Do not pin memory in data loaders
    optim="adamw_hf",                          # Optimizer type
    report_to="tensorboard",                   # Report results to TensorBoard
    resume_from_checkpoint=True                # Resume training from the last checkpoint
)
```

## class Trainer
The `Trainer` class is a comprehensive training interface for training models with the Hugging Face library. It supports various training setups, including distributed training, mixed precision, and integration with hyperparameter search frameworks.

### Overview of Trainer
The `Trainer` class provides various methods for managing the training process, evaluating models, saving and loading checkpoints, and more. Below is a description of its methods:

| **Method** | **Description** |
|------------|-----------------|
| `train` | Main training entry point. |
| `training_step` | Perform a training step on a batch of inputs. |
| `compute_loss` | How the loss is computed by Trainer. By default, all models return the loss in the first element. |
| `is_local_process_zero` | Whether or not this process is the local main process. |
| `is_world_process_zero` | Whether or not this process is the global main process. |
| `save_model` | Will save the model, so you can reload it using `from_pretrained()`. |
| `store_flos` | Store the number of floating-point operations that went into the model. |
| `evaluate` | Run evaluation and return metrics. |
| `predict` | Run prediction and return predictions and potential metrics. |
| `evaluation_loop` | Prediction/evaluation loop shared by `Trainer.evaluate()` and `Trainer.predict()`. |
| `prediction_step` | Perform an evaluation step on `model` using `inputs`. |
| `floating_point_ops` | Compute the number of floating point operations for every backward + forward pass. |
| `init_hf_repo` | Initialize a git repo in `self.args.hub_model_id`. |
| `create_model_card` | Create a draft of a model card using the information available to the `Trainer`. |
| `push_to_hub` | Upload `self.model` and `self.tokenizer` to the 🤗 model hub. |
| `prediction_loop` | Deprecated prediction/evaluation loop. |
| `create_accelerator_and_postprocess` | Create accelerator object and post-process the setup. |
| `propagate_args_to_deepspeed` | Set values in the deepspeed plugin based on the Trainer args. |
| `_save_tpu` | Save model checkpoint to TPU. |
| `_save` | Save the model checkpoint to a specified directory. |
| `_sorted_checkpoints` | Get a list of sorted checkpoints. |
| `_rotate_checkpoints` | Rotate checkpoints to manage the number of saved checkpoints. |
| `_nested_gather` | Gather value of tensors and convert them to numpy before concatenating them. |
| `_push_from_checkpoint` | Push from a checkpoint. |
| `_finish_current_push` | Wait for the current checkpoint push to be finished. |
| `_gather_and_numpify` | Gather value of tensors and convert them to numpy before concatenating them. |
| `_add_sm_patterns_to_gitignore` | Add SageMaker Checkpointing patterns to .gitignore file. |
| `_fsdp_qlora_plugin_updates` | Update FSDP and QLora plugins based on the model configuration. |

### Examples of Using Trainer

**1. Basic Training Example**

This example demonstrates how to initiate the training process using the `train` method with essential parameters.
```python
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",             # Directory where the model checkpoints and logs will be saved
    num_train_epochs=3,                 # Number of training epochs
    per_device_train_batch_size=16,     # Batch size for training on each device
    per_device_eval_batch_size=16,      # Batch size for evaluation on each device
    warmup_steps=500,                   # Number of steps to perform learning rate warmup
    weight_decay=0.01,                  # Weight decay for the optimizer to prevent overfitting
    logging_dir="./logs",               # Directory for storing logs for monitoring training
)

# Initialize Trainer
trainer = Trainer(
    model=model,                        # The instantiated HuggingFace Transformers model to be trained
    args=training_args,                 # Training arguments, defined above
    train_dataset=train_dataset,        # The training dataset
    eval_dataset=eval_dataset,          # The evaluation dataset
)

# Start training
trainer.train()
```

**2. Evaluation Example**

This example shows how to evaluate a model using the `evaluate` method.
```python
# Evaluate the model
evaluation_results = trainer.evaluate(
    eval_dataset=eval_dataset,  # the evaluation dataset
    metric_key_prefix="eval"    # prefix for evaluation metrics
)

print(evaluation_results)
```

**3. Prediction Example**

This example demonstrates how to run predictions on a test dataset using the `predict` method.
```python
# Predict using the model
predictions = trainer.predict(
    test_dataset=test_dataset,  # the test dataset
    metric_key_prefix="test"    # prefix for test metrics
)

print(predictions)
```

**4. Save Model Example**

This example illustrates how to save the trained model using the `save_model` method.
```python
# Save the model
trainer.save_model(
    output_dir="./saved_model"  # directory to save the model
)
```

**5. Custom Training Step Example**

This example shows how to customize the training step using the `training_step` method.
```python
class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        # Custom training step logic
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        return loss

# Initialize CustomTrainer
custom_trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Start training with custom training step
custom_trainer.train()
```

**6. Compute Loss Example**

This example demonstrates how to customize the loss computation using the `compute_loss` method.
```python
# Import necessary libraries
from transformers import Trainer, TrainingArguments

# Define a custom loss function
def custom_loss_function(logits, labels):
    # Custom logic for computing loss between model's predictions (logits) and the true labels
    return ...  # Replace with actual implementation

# Define a custom Trainer class that inherits from the HuggingFace Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")               # Extract labels from the inputs
        outputs = model(**inputs)                   # Get model outputs using the remaining inputs
        logits = outputs.logits                     # Extract logits from the model outputs
        loss = custom_loss_function(logits, labels) # Compute custom loss using the logits and labels

        # Return the loss (and optionally the outputs)
        return (loss, outputs) if return_outputs else loss

# Initialize the CustomTrainer with model, training arguments, and datasets
custom_trainer = CustomTrainer(
    model=model,                        # The model to be trained
    args=training_args,                 # Training arguments defined earlier
    train_dataset=train_dataset,        # The dataset to be used for training
    eval_dataset=eval_dataset,          # The dataset to be used for evaluation
)

# Start training with custom loss computation
custom_trainer.train()
```

**7. Push to Hub Example**

This example shows how to push the model to the Hugging Face Hub using the `push_to_hub` method.
```python
# Push the model to the Hugging Face Hub
trainer.push_to_hub(
    commit_message="End of training",   # commit message for the push
    blocking=True,                      # whether to wait for the push to complete
)
```

**8. Create Model Card Example**

This example illustrates how to create a model card using the `create_model_card` method.
```python
# Create a model card
trainer.create_model_card(
    language="en",                      # language of the model
    license="apache-2.0",               # license for the model
    tags=["example", "transformers"],   # tags for the model card
    model_name="example-model",         # name of the model
    finetuned_from="bert-base-uncased", # base model used for fine-tuning
    tasks=["text-classification"],      # tasks the model is used for
    dataset_tags=["imdb"],              # dataset tags
    dataset="imdb",                     # dataset used
    dataset_args="train",               # dataset arguments
)
```

**9. Hyperparameter Search Example**

This example demonstrates how to perform hyperparameter search using the `hyperparameter_search` method.
```python
def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 3),
    }

# Perform hyperparameter search
best_run = trainer.hyperparameter_search(
    hp_space=my_hp_space,   # hyperparameter space
    direction="minimize",   # optimization direction
    n_trials=10             # number of trials
)

print(best_run)
```

**10. Evaluation Loop Example**

This example illustrates how to use the evaluation loop directly with custom configurations.
```python
# Custom evaluation loop
eval_results = trainer.evaluation_loop(
    dataloader=eval_dataloader,     # evaluation data loader
    description="Evaluation",       # description for the evaluation loop
    prediction_loss_only=False,     # whether to return only the loss
    metric_key_prefix="eval"        # prefix for evaluation metrics
)

print(eval_results)
```


## Use Cases

### Fine-tuning Meta LLaMA Model
This example demonstrates the process of fine-tuning a pretrained model using HuggingFace's Trainer class, incorporating Low-Rank Adaptation (LoRA) adapters, and utilizing 4-bit precision for efficient training. The example covers preparing a dataset, tokenizing it, configuring and adding a LoRA adapter to a model, setting up training arguments, and using the Trainer to handle the training and evaluation processes. The model is then saved and reloaded for future use, showcasing a complete workflow for fine-tuning transformer models with advanced techniques.

```python
# Import necessary libraries
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig

# Prepare the dataset
data = {
    "train": [
        {"text": "I love this product!", "label": 1},                       # Positive sentiment
        {"text": "This is the worst thing I've ever bought.", "label": 0},  # Negative sentiment
    ],
    "validation": [
        {"text": "Absolutely fantastic!", "label": 1},                  # Positive sentiment
        {"text": "Terrible quality, very disappointed.", "label": 0},   # Negative sentiment
    ],
}

# Convert data to a DatasetDict, which contains the training and validation datasets
dataset = DatasetDict(
    {
        "train": Dataset.from_dict(data["train"]),              # Training dataset
        "validation": Dataset.from_dict(data["validation"]),    # Validation dataset
    }
)

# Tokenize the data
tokenizer = AutoTokenizer.from_pretrained("huggingface/llama-13b")

# Function to tokenize the examples in the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply the tokenize_function to each example in the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load the model and add LoRA adapter
# https://huggingface.co/docs/transformers/v4.34.0/main_classes/quantization
#   load_in_8bit (bool, default False) — enable 8-bit quantization with LLM.int8().
#   load_in_4bit (bool, default False) — enable 4-bit quantization with FP4/NF4 layers from bitsandbytes.
peft_model_id = "huggingface/llama-13b-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id, load_in_4bit=True, device_map="auto")

# Define the configuration for the LoRA adapter
lora_config = LoraConfig(
    lora_alpha=16,                          # LoRA alpha parameter
    lora_dropout=0.1,                       # Dropout rate for LoRA
    r=64,                                   # Rank of the low-rank matrices in LoRA
    target_modules=["q_proj", "k_proj"],    # Target modules for LoRA
)

# Add the LoRA adapter to the model
model.add_adapter(lora_config, adapter_name="lora_adapter")

# Enable the adapter
model.set_adapter("lora_adapter")

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",         # Directory to save the model and checkpoints
    evaluation_strategy="epoch",    # Evaluate the model at the end of each epoch
    learning_rate=2e-5,             # Learning rate for training
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=4,   # Batch size for evaluation
    num_train_epochs=3,             # Number of training epochs
    weight_decay=0.01,              # Weight decay for optimization
)

# Initialize the Trainer with the model, training arguments, and datasets
trainer = Trainer(
    model=model,                                    # The model to be trained. In this case, it's the Whisper model with the LoRA adapter loaded in 4-bit precision.
    args=training_args,                             # Training arguments that specify the configurations for training such as learning rate, batch size, etc.
    train_dataset=tokenized_datasets["train"],      # The dataset to be used for training the model.
    eval_dataset=tokenized_datasets["validation"],  # The dataset to be used for evaluating the model during training.
    tokenizer=tokenizer,                            # The tokenizer to be used for encoding the text data.
                                                    # This ensures that the text data is tokenized in the same way as the pre-trained model was.
)

# Start training the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the model with the adapter
model.save_pretrained("path_to_save_adapter")

# Load the model with the adapter for future use
model = AutoModelForCausalLM.from_pretrained("path_to_save_adapter", load_in_4bit=True, device_map="auto")
```

### Fine-tuning OpenAI Whisper Model
This example showcases the process of fine-tuning the OpenAI Whisper model using Low-Rank Adaptation (LoRA) adapters with 4-bit precision for efficient training. It involves loading and preprocessing the Common Voice dataset, adding a LoRA adapter to the model, setting up training arguments, and utilizing the Trainer class to manage the training and evaluation workflow. The example demonstrates how to save and reload the fine-tuned model along with the LoRA adapter, providing a comprehensive guide to fine-tuning transformer models with advanced techniques.

```python
# Import necessary libraries
from transformers import WhisperTokenizer, WhisperForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict
from peft import LoraConfig
import torch
from torchaudio.transforms import MelSpectrogram, Resample

# Define the model and tokenizer
peft_model_id = "openai/whisper-large-lora"
tokenizer = WhisperTokenizer.from_pretrained(peft_model_id)

# Load the model in 4-bit precision
# https://huggingface.co/docs/transformers/v4.34.0/main_classes/quantization
model = WhisperForConditionalGeneration.from_pretrained(peft_model_id, load_in_4bit=True, device_map="auto")

# Load and preprocess the dataset
# Here we use the Common Voice dataset as an example
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split={"train": "train", "validation": "validation"})

# Display a sample from the dataset
print(dataset["train"][0])
# Example output of raw dataset
# {
#     "client_id": "0c7171cd62a64596e198f093c05b1c9b",
#     "path": "path/to/audio/file.wav",
#     "audio": {
#         "array": [0.0, 0.1, -0.1, ...],
#         "sampling_rate": 48000,
#         "duration": 1.23,
#         "path": "path/to/audio/file.wav"
#     },
#     "sentence": "I love programming.",
#     "age": "thirties",
#     "gender": "male",
#     "accent": "us",
#     ...
# }

# Preprocess the audio data
def preprocess_audio(batch):
    # Resample audio to 16kHz
    resample = Resample(orig_freq=batch["audio"]["sampling_rate"], new_freq=16000)
    batch["audio"]["array"] = resample(torch.tensor(batch["audio"]["array"]))
    
    # Compute mel spectrogram
    mel_spectrogram = MelSpectrogram(
        sample_rate=16000,  # The sample rate of the audio to be transformed
        n_fft=400,          # The size of the FFT window
        win_length=400,     # The size of the window used for FFT
        hop_length=160,     # The number of audio samples between adjacent STFT columns
        n_mels=80           # The number of Mel bands to generate
    )
    # Apply mel spectrogram transformation and transpose the result
    batch["input_features"] = mel_spectrogram(batch["audio"]["array"]).transpose(0, 1).numpy()
    
    # Tokenize the target text
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# Apply the preprocessing to the dataset
dataset = dataset.map(preprocess_audio, remove_columns=["audio", "sentence"], batched=True)

# Display a sample from the preprocessed dataset
print(dataset["train"][0])
# Example output of preprocessed dataset
# {
#    "client_id": "0c7171cd62a64596e198f093c05b1c9b",
#    "path": "path/to/audio/file.wav",
#    "input_features": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
#    "labels": [2, 345, 1234, ...],
#    "age": "thirties",
#    "gender": "male",
#    "accent": "us",
#    ...
# }

# Define the LoRA configuration
lora_config = LoraConfig(
    lora_alpha=16,                  # LoRA alpha parameter
    lora_dropout=0.1,               # Dropout rate for LoRA
    r=64,                           # Rank of the low-rank matrices in LoRA
    target_modules=[
        "decoder.layers.*.self_attn.q_proj", 
        "decoder.layers.*.self_attn.k_proj"
    ],                              # Target modules for LoRA
)

# Add the LoRA adapter to the model
model.add_adapter(lora_config, adapter_name="lora_adapter")

# Enable the adapter
model.set_adapter("lora_adapter")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",         # Directory to save the model and checkpoints
    evaluation_strategy="epoch",    # Evaluate the model at the end of each epoch
    learning_rate=1e-5,             # Learning rate for training
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,   # Batch size for evaluation
    num_train_epochs=3,             # Number of training epochs
    weight_decay=0.01,              # Weight decay for optimization
    logging_dir='./logs',           # Directory for storing logs
    logging_steps=10,               # Log every 10 steps
    save_total_limit=3,             # Limit the total amount of checkpoints
    save_steps=500,                 # Save checkpoint every 500 steps
    eval_steps=500,                 # Evaluate every 500 steps
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                        # The model to be trained. In this case, it's the Whisper model with the LoRA adapter loaded in 4-bit precision.
    args=training_args,                 # Training arguments that specify the configurations for training such as learning rate, batch size, etc.
    train_dataset=dataset["train"],     # The dataset to be used for training the model. 
    eval_dataset=dataset["validation"], # The dataset to be used for evaluating the model during training.
    tokenizer=tokenizer,                # The tokenizer to be used for encoding the text data. 
                                        # This ensures that the text data is tokenized in the same way as the pre-trained model was.
)

# Start training the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the fine-tuned model
model.save_pretrained("path_to_save_finetuned_model")
tokenizer.save_pretrained("path_to_save_finetuned_model")


# Save the model with the adapter
model.save_pretrained("path_to_save_adapter")
tokenizer.save_pretrained("path_to_save_adapter")

# Load the model with the adapter for future use
model = WhisperForConditionalGeneration.from_pretrained("path_to_save_adapter", load_in_4bit=True, device_map="auto")
tokenizer = WhisperTokenizer.from_pretrained("path_to_save_adapter")
```

### Fine-tuning OpenAI CLIP Model
This example demonstrates the process of fine-tuning the OpenAI CLIP model using Low-Rank Adaptation (LoRA) adapters and 4-bit precision to optimize for efficiency. The process involves loading and preprocessing the COCO dataset, configuring and integrating LoRA adapters into the model, defining training parameters, and leveraging the Trainer class to manage the training and evaluation pipeline. The example showcases how to save and reload the fine-tuned model along with its adapters, providing a comprehensive guide for fine-tuning transformer models with advanced techniques.

```python
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer
from peft import LoraConfig
import torch

# Load a sample dataset - in this case, we use the 'coco' dataset from the Hugging Face hub.
dataset = load_dataset("coco", split="train[:1%]")

print(dataset)
# Example output of raw dataset
# {
#   'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x7F74E44790D0>,
#   'caption': 'A bicycle replica with a clock as the front wheel.'
# }

# Initialize the CLIP processor
peft_model_id = "openai/clip-vit-large-patch14-lora"
processor = CLIPProcessor.from_pretrained(peft_model_id)

# Define a preprocessing function
def preprocess_data(examples):
    # The processor will resize and normalize images, and tokenize the text
    return processor(text=examples["caption"], images=examples["image"], return_tensors="pt", padding=True)

# Apply the preprocessing to the dataset
dataset = dataset.map(preprocess_data, batched=True, remove_columns=["image", "caption"])

# Print a sample from the preprocessed dataset
print(dataset[0])
# Example output of preprocessed dataset
# {
#   'input_ids': tensor([[49406, 320, 1125, 17469, 1580, 438, 320, 2600, 1364, 329, 259, 320, 1125, 1125, 481, 49407]]),
#   'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
#   'pixel_values': tensor([[[[0.7500, 0.7608, 0.7843,  ..., 0.5843, 0.6118, 0.6500],
#                             ...,
#                             [0.4784, 0.4863, 0.5059,  ..., 0.4941, 0.4824, 0.4667]],
#                            ...
#                            [[0.7804, 0.8000, 0.8157,  ..., 0.6588, 0.6941, 0.7333],
#                             ...,
#                             [0.5608, 0.5686, 0.5882,  ..., 0.5725, 0.5608, 0.5451]]]])
# }

# Load the CLIP model with 4-bit precision (QLoRA)
# https://huggingface.co/docs/transformers/v4.34.0/main_classes/quantization
model = CLIPModel.from_pretrained(peft_model_id, load_in_4bit=True, device_map="auto")

# Define the LoRA configuration
lora_config = LoraConfig(
    lora_alpha=16,                  # LoRA alpha parameter
    lora_dropout=0.1,               # Dropout rate for LoRA
    r=64,                           # Rank of the low-rank matrices in LoRA
    target_modules=[
        "vision_model.encoder.layers.*.self_attn.q_proj", 
        "vision_model.encoder.layers.*.self_attn.k_proj"
    ],                              # Target modules for LoRA
)

# Add the LoRA adapter to the model
model.add_adapter(lora_config, adapter_name="lora_adapter")

# Enable the adapter
model.set_adapter("lora_adapter")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",         # Directory to save the model and checkpoints
    evaluation_strategy="epoch",    # Evaluate the model at the end of each epoch
    learning_rate=5e-5,             # Learning rate for training
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=4,   # Batch size for evaluation
    num_train_epochs=3,             # Number of training epochs
    weight_decay=0.01,              # Weight decay for optimization
    logging_dir='./logs',           # Directory for storing logs
    logging_steps=10,               # Log every 10 steps
    save_total_limit=3,             # Limit the total amount of checkpoints
    save_steps=500,                 # Save checkpoint every 500 steps
    eval_steps=500,                 # Evaluate every 500 steps
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                        # The model to be trained. In this case, it's the Whisper model with the LoRA adapter loaded in 4-bit precision.
    args=training_args,                 # Training arguments that specify the configurations for training such as learning rate, batch size, etc.
    train_dataset=dataset["train"],     # The dataset to be used for training the model.
    eval_dataset=dataset["validation"], # The dataset to be used for evaluating the model during training.
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the model with the adapter
model.save_pretrained("path_to_save_adapter")

# Load the model with the adapter for future use
model = CLIPModel.from_pretrained("path_to_save_adapter", load_in_4bit=True, device_map="auto")
```


## Conclusion
The HuggingFace Transformer Trainer is an invaluable tool for both researchers and practitioners in the field of natural language processing. By abstracting the complexities involved in the training loop, it allows users to focus on the more critical aspects of model development, such as architecture design and data preparation. Its seamless integration with popular deep learning frameworks like PyTorch, along with robust support for dataset preprocessing, rich training hyperparameters, and parameter-efficient fine-tuning methods, makes it an essential component for efficiently training and fine-tuning transformer models.

Additionally, the support for mixed precision training and built-in functionality for computing and reporting metrics further enhances the Trainer's utility. These features not only reduce memory usage and training time but also provide real-time insights into model performance, facilitating early stopping and hyperparameter tuning. Overall, the HuggingFace Transformer Trainer simplifies and optimizes the training process, making it accessible to a broader audience, from beginners to experts, and enabling the development of high-performance models with ease.


## References
- [HuggingFace Transformers Training Documentation](https://huggingface.co/docs/transformers/training)
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/transformers/peft)
- [LoRA Configuration in HuggingFace PEFT](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py)
- [TrainingArguments Class in Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)
- [Trainer Class in Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py)

