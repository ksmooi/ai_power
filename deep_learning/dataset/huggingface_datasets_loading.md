# Introducing the Dataset Loading Feature of HuggingFace Datasets

## Introduction
The HuggingFace Datasets library is a powerful tool designed to simplify the process of loading, processing, and utilizing datasets for various machine learning tasks, particularly in natural language processing (NLP), computer vision, and audio. Here’s an overview of the key features and functionalities for loading datasets:

### **Data Storage Locations**
Datasets can be stored in multiple locations, and HuggingFace Datasets can help you load them regardless of their storage location:
- **Local Machine:** Datasets stored on your disk.
- **GitHub Repositories:** Public or private repositories.
- **In-Memory Data Structures:** Python dictionaries or Pandas DataFrames.

### **Loading Methods**
HuggingFace Datasets offers several methods to load datasets:

**1. The Hugging Face Hub**

The Hugging Face Hub is a community-driven platform that hosts a wide array of datasets. You can easily discover and load datasets from the Hub without requiring a dataset loading script. Simply use the `load_dataset` function:

```python
from datasets import load_dataset

# Load the rotten_tomatoes dataset
dataset = load_dataset("rotten_tomatoes")
# DatasetDict({
#     train: Dataset({
#         features: ['text', 'label'],
#         num_rows: 8530
#     })
#     validation: Dataset({
#         features: ['text', 'label'],
#         num_rows: 1066
#     })
#     test: Dataset({
#         features: ['text', 'label'],
#         num_rows: 1066
#     })
# })

# Load a specific split with the split parameter
dataset = load_dataset("rotten_tomatoes", split="train")
# Dataset({
#     features: ['text', 'label'],
#     num_rows: 8530
# })
```

**2. Local Loading Script**

If you have a local dataset loading script, you can load the dataset by passing the script’s path to `load_dataset`:

```python
# Load a dataset using a local loading script
dataset = load_dataset("path/to/local/loading_script/loading_script.py", split="train", trust_remote_code=True)
```

**3. Local and Remote Files**

You can load datasets stored as CSV, JSON, SQL, or WebDataset files, either locally or from remote URLs:

```python
# Load a CSV file
dataset = load_dataset("csv", data_files="my_file.csv")

# Load a JSON file
dataset = load_dataset("json", data_files="my_file.json")

# load entire SQL table
dataset = Dataset.from_sql("data_table_name", con="sqlite:///sqlite_file.db")

# load from SQL query
dataset = Dataset.from_sql("SELECT text FROM table WHERE length(text) > 100 LIMIT 10", con="sqlite:///sqlite_file.db")

# load from WebDataset
path = "path/to/train/*.tar"
dataset = load_dataset("webdataset", data_files={"train": path}, split="train", streaming=True)
```

**4. In-Memory Data**

You can create datasets directly from in-memory data structures:

```python
from datasets import Dataset
import pandas as pd

# From a Python dictionary
my_dict = {"a": [1, 2, 3]}
dataset = Dataset.from_dict(my_dict)

# From a Pandas DataFrame
df = pd.DataFrame({"a": [1, 2, 3]})
dataset = Dataset.from_pandas(df)
```

### **Offline Mode**
If you’ve downloaded a dataset from the Hub before, it’s cached locally, allowing you to load it offline. Set the `HF_DATASETS_OFFLINE` environment variable to `1` to enable full offline mode.

### **Slice Splits**
You can load specific slices of a split using string instructions or the ReadInstruction API. This allows for flexible data selection, including cross-validated splits.

```python
# Select specific rows
train_10_20_ds = load_dataset("bookcorpus", split="train[10:20]")

# Select a percentage of the split
train_10pct_ds = load_dataset("bookcorpus", split="train[:10%]")
```

### **Custom Features**
When creating a dataset from local files, you can define custom features using the `Features` class:

```python
from datasets import Features, Value, ClassLabel

class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
emotion_features = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})

dataset = load_dataset('csv', data_files="file.csv", features=emotion_features)
```


## Loading Methods of HuggingFace Datasets
The following table provides an overview of key methods in the HuggingFace Datasets library. Each method is described with its primary functionality, helping users understand how to load, access, and retrieve information about datasets from both local storage and the Hugging Face Hub.

| **Method**              | **Description**                                       |
|-------------------------|-------------------------------------------------------|
| `load_dataset`          | Loads a dataset from the Hugging Face Hub, or a local dataset. It can handle data in various formats such as JSON, CSV, Parquet, and text. This function downloads and processes the dataset, caches it, and returns the dataset splits. It supports both local and remote datasets, streaming, multiprocessing, and custom features.                                      |
| `load_from_disk`        | Loads a dataset that was previously saved using `save_to_disk` from a local directory or a remote filesystem. This method is useful for reloading datasets that have been saved locally or on a remote storage service like S3. It supports loading both individual datasets and dataset dictionaries.                                                        |
| `get_dataset_infos`     | Retrieves the meta information about a dataset, returned as a dictionary mapping configuration names to `DatasetInfoDict`. It provides details such as description, features, and splits available for the dataset. It supports fetching information for datasets hosted on the Hugging Face Hub or local datasets with processing scripts.                      |
| `get_dataset_config_names` | Gets the list of available configuration names for a particular dataset. This method is used to find all the configurations that a dataset supports, which can include different versions or variations of the dataset. It works for both local datasets with processing scripts and datasets hosted on the Hugging Face Hub.                                      |
| `get_dataset_default_config_name` | Gets the default configuration name for a particular dataset. This method returns the default configuration if the dataset has multiple configurations, or `None` if there is no default configuration. It is useful for identifying the primary configuration to use when multiple configurations are available.                                             |
| `get_dataset_config_info` | Retrieves the meta information (DatasetInfo) about a dataset for a particular configuration. It provides detailed information about the dataset such as its features, splits, and other metadata. This method supports both local datasets with processing scripts and datasets hosted on the Hugging Face Hub.                                                   |
| `get_dataset_split_names` | Gets the list of available splits for a particular dataset configuration. This method returns the names of the splits available in the dataset, such as train, validation, and test splits. It is useful for understanding the structure of the dataset and how it is divided into different subsets for training and evaluation purposes.                           |

By leveraging these methods, HuggingFace Datasets makes it easy to load and manipulate datasets from diverse sources, ensuring flexibility and efficiency in your machine learning workflows.

### load_dataset()
Here are some examples using various parameters of the `load_dataset` function, with comments to explain each step:

**Loading a Dataset from the Hugging Face Hub**

This example shows how to load a dataset from the Hugging Face Hub using a specific dataset name and split. It also demonstrates specifying a revision to use a particular version of the dataset.

```python
from datasets import load_dataset

# Load the 'rotten_tomatoes' dataset from the Hugging Face Hub, using the 'train' split and specifying a revision
dataset = load_dataset('rotten_tomatoes', split='train', revision='main')

# Print the first few rows to check the data
print(dataset.head())
```

**Loading a Dataset with Specific Data Files**

This example demonstrates how to load a dataset by specifying the paths to the data files. This is useful when the dataset is stored in multiple files that need to be mapped to different splits.

```python
from datasets import load_dataset

# Define the data files for different splits
data_files = {'train': 'path/to/train.csv', 'test': 'path/to/test.csv'}

# Load the dataset using the specified data files
dataset = load_dataset('csv', data_files=data_files)

# Print the first few rows of the train split to verify
print(dataset['train'].head())
```

**Loading a Dataset from Local Directory**

This example shows how to load a dataset from a local directory containing a CSV file.

```python
from datasets import load_dataset

# Load the dataset from a local directory containing CSV files
dataset = load_dataset('csv', data_files='path/to/local/data.csv')

# Print the first few rows to check the data
print(dataset.head())
```

**Loading a Streaming Dataset**

This example demonstrates how to load a dataset in streaming mode. Streaming mode is useful for large datasets that cannot fit into memory.

```python
from datasets import load_dataset

# Load the 'rotten_tomatoes' dataset in streaming mode
dataset = load_dataset('rotten_tomatoes', split='train', streaming=True)

# Iterate over the first few examples in the streaming dataset
for example in dataset.take(5):
    print(example)
```

**Loading a Dataset with Multiprocessing**

This example shows how to load a dataset using multiple processes to speed up the downloading and preparation of the dataset.

```python
from datasets import load_dataset

# Load the 'imagenet-1k' dataset using 8 processes
dataset = load_dataset('imagenet-1k', split='train', num_proc=8)

# Print the first few rows to check the data
print(dataset.head())
```

**Loading a Dataset with Custom Features**

This example demonstrates how to load a dataset and specify custom features using the `Features` class. This is useful when you want to enforce a specific schema for the dataset.

```python
from datasets import load_dataset, Features, Value, ClassLabel

# Define custom features
custom_features = Features({
    'text': Value('string'),
    'label': ClassLabel(names=['negative', 'positive'])
})

# Load the dataset with the specified custom features
dataset = load_dataset('csv', data_files='path/to/data.csv', features=custom_features)

# Print the features of the dataset to verify
print(dataset.features)
```

**Loading a Dataset with Storage Options**

This example shows how to load a dataset with specific storage options. This is useful when working with remote filesystems or other special storage configurations.

```python
from datasets import load_dataset

# Define storage options for a remote filesystem (e.g., S3)
storage_options = {
    'anon': True,
    'default_fill_cache': False,
    'default_cache_type': 'memory'
}

# Load the dataset using the specified storage options
dataset = load_dataset('s3://bucket-name/dataset', split='train', storage_options=storage_options)

# Print the first few rows to check the data
print(dataset.head())
```

### load_from_disk()
Here are some examples using various parameters of the `load_from_disk` function, with comments to explain each step:

**Loading a Dataset from a Local Directory**

This example shows how to load a dataset that was previously saved to a local directory.

```python
from datasets import load_from_disk

# Load a dataset from a local directory
dataset = load_from_disk('path/to/dataset/directory')

# Print the first few rows to check the data
print(dataset.head())
```

**Loading a Dataset Dictionary from a Local Directory**

In this directory structure, each split (train, validation, test) has its own subdirectory containing the necessary files: dataset.arrow, dataset_info.json, and state.json.
```
/path/to/dataset_dict/directory/
│
├── train/
│   ├── dataset.arrow
│   ├── dataset_info.json
│   └── state.json
│
├── validation/
│   ├── dataset.arrow
│   ├── dataset_info.json
│   └── state.json
│
└── test/
    ├── dataset.arrow
    ├── dataset_info.json
    └── state.json
```

This example demonstrates how to load a dataset dictionary (which includes multiple splits) from a local directory.
```python
from datasets import load_from_disk

# Load a dataset dictionary from a local directory
dataset_dict = load_from_disk('path/to/dataset_dict/directory')

# Print the splits and the first few rows of each split to check the data
for split, dataset in dataset_dict.items():
    print(f"Split: {split}")
    print(dataset.head())
```

The output might look something like this:
```
Split: train
   text  label
0  This movie is great!      1
1  I didn't like this film.  0
2  It was an amazing experience. 1
3  Terrible movie, would not recommend. 0
4  Loved every minute of it. 1

Split: validation
   text  label
0  Not my type of movie.  0
1  Fantastic story and characters.  1
2  It was just okay.  0
3  Really enjoyed it.  1
4  Could have been better.  0

Split: test
   text  label
0  A waste of time.  0
1  Brilliant cinematography.  1
2  I fell asleep halfway through.  0
3  Would watch again.  1
4  Not worth the hype.  0
```

**Loading a Dataset from a Remote URI**

This example demonstrates how to load a dataset from a remote URI, such as an S3 bucket, using specific storage options.

```python
from datasets import load_from_disk

# Define storage options for an S3 bucket
storage_options = {
    'anon': True,  # Use anonymous access
    'default_fill_cache': False,
    'default_cache_type': 'memory'
}

# Load the dataset from a remote URI using the specified storage options
dataset = load_from_disk('s3://my-bucket/dataset/train', storage_options=storage_options)

# Print the first few rows to check the data
print(dataset.head())
```

**Loading a Dataset and Keeping it in Memory**

This example shows how to load a dataset and keep it in memory for faster access. This is useful for smaller datasets that fit into memory.

```python
from datasets import load_from_disk

# Load a dataset from a local directory and keep it in memory
dataset = load_from_disk('path/to/dataset/directory', keep_in_memory=True)

# Print the first few rows to check the data
print(dataset.head())
```

### get_dataset_infos()
Here are some examples using various parameters of the `get_dataset_infos` function, with comments to explain each step:

**Getting Dataset Info from the Hugging Face Hub**

This example shows how to get the metadata information of a dataset hosted on the Hugging Face Hub.

```python
from datasets import get_dataset_infos

# Get metadata information for the 'rotten_tomatoes' dataset from the Hugging Face Hub
dataset_infos = get_dataset_infos('rotten_tomatoes')

# Print the retrieved dataset info
print(dataset_infos)
```

**Getting Dataset Info with a Specific Version**

This example demonstrates how to get the metadata information for a specific version of the dataset by specifying the revision parameter.

```python
from datasets import get_dataset_infos

# Get metadata information for the 'glue' dataset, specifying the 'main' branch as the revision
dataset_infos = get_dataset_infos('glue', revision='main')

# Print the retrieved dataset info
print(dataset_infos)
```

**Getting Dataset Info with Specific Data Files**

This example shows how to get the metadata information of a dataset by specifying particular data files.

```python
from datasets import get_dataset_infos

# Define the data files to use for the dataset
data_files = {
    'train': 'path/to/train.csv',
    'validation': 'path/to/validation.csv',
    'test': 'path/to/test.csv'
}

# Get metadata information for a dataset using the specified data files
dataset_infos = get_dataset_infos('csv', data_files=data_files)

# Print the retrieved dataset info
print(dataset_infos)
```

**Getting Dataset Info with Download Configuration**

This example demonstrates how to get the metadata information for a dataset with specific download configuration parameters.

```python
from datasets import get_dataset_infos, DownloadConfig

# Define the download configuration
download_config = DownloadConfig(
    cache_dir='/path/to/cache',
    force_download=True
)

# Get metadata information for the 'squad' dataset using the specified download configuration
dataset_infos = get_dataset_infos('squad', download_config=download_config)

# Print the retrieved dataset info
print(dataset_infos)
```

**Getting Dataset Info with Custom Configurations**

This example shows how to get metadata information for a dataset with additional custom configurations.

```python
from datasets import get_dataset_infos

# Define custom configurations
config_kwargs = {
    'field': 'data',
    'script_version': '1.0.0'
}

# Get metadata information for the 'squad' dataset with custom configurations
dataset_infos = get_dataset_infos('squad', **config_kwargs)

# Print the retrieved dataset info
print(dataset_infos)
```

### get_dataset_config_names()
Here are some examples using various parameters of the `get_dataset_config_names` function, with comments to explain each step:

**Getting Config Names for a Dataset from the Hugging Face Hub**

This example shows how to get the list of available configuration names for a dataset hosted on the Hugging Face Hub.

```python
from datasets import get_dataset_config_names

# Get the list of available configuration names for the 'glue' dataset from the Hugging Face Hub
config_names = get_dataset_config_names('glue')

# Print the retrieved configuration names
print(config_names)
```

**Getting Config Names for a Specific Version of a Dataset**

This example demonstrates how to get the list of available configuration names for a specific version of a dataset by specifying the revision parameter.

```python
from datasets import get_dataset_config_names

# Get the list of available configuration names for the 'glue' dataset, specifying the 'main' branch as the revision
config_names = get_dataset_config_names('glue', revision='main')

# Print the retrieved configuration names
print(config_names)
```

**Getting Config Names with Specific Data Files**

This example shows how to get the list of available configuration names for a dataset by specifying particular data files.

```python
from datasets import get_dataset_config_names

# Define the data files to use for the dataset
data_files = {
    'train': 'path/to/train.csv',
    'validation': 'path/to/validation.csv',
    'test': 'path/to/test.csv'
}

# Get the list of available configuration names for a dataset using the specified data files
config_names = get_dataset_config_names('csv', data_files=data_files)

# Print the retrieved configuration names
print(config_names)
```

**Getting Config Names with Download Configuration**

This example demonstrates how to get the list of available configuration names for a dataset with specific download configuration parameters.

```python
from datasets import get_dataset_config_names, DownloadConfig

# Define the download configuration
download_config = DownloadConfig(
    cache_dir='/path/to/cache',
    force_download=True
)

# Get the list of available configuration names for the 'squad' dataset using the specified download configuration
config_names = get_dataset_config_names('squad', download_config=download_config)

# Print the retrieved configuration names
print(config_names)
```

**Getting Config Names with Custom Configurations**

This example shows how to get the list of available configuration names for a dataset with additional custom configurations.

```python
from datasets import get_dataset_config_names

# Define custom configurations
config_kwargs = {
    'field': 'data',
    'script_version': '1.0.0'
}

# Get the list of available configuration names for the 'squad' dataset with custom configurations
config_names = get_dataset_config_names('squad', **config_kwargs)

# Print the retrieved configuration names
print(config_names)
```

### get_dataset_config_info()
Here are some examples using various parameters of the `get_dataset_config_info` function, with comments to explain each step:

**Getting Dataset Info for a Specific Configuration**

This example shows how to get the metadata information for a specific configuration of a dataset hosted on the Hugging Face Hub.

```python
from datasets import get_dataset_config_info

# Get metadata information for the 'squad' dataset with the 'plain_text' configuration
config_info = get_dataset_config_info('squad', config_name='plain_text')

# Print the retrieved dataset info
print(config_info)
```

**Getting Dataset Info with Specific Data Files**

This example demonstrates how to get the metadata information for a dataset by specifying particular data files.

```python
from datasets import get_dataset_config_info

# Define the data files to use for the dataset
data_files = {
    'train': 'path/to/train.csv',
    'validation': 'path/to/validation.csv',
    'test': 'path/to/test.csv'
}

# Get metadata information for a dataset using the specified data files
config_info = get_dataset_config_info('csv', config_name='my_dataset_config', data_files=data_files)

# Print the retrieved dataset info
print(config_info)
```

**Getting Dataset Info with Download Configuration**

This example shows how to get the metadata information for a dataset with specific download configuration parameters.

```python
from datasets import get_dataset_config_info, DownloadConfig

# Define the download configuration
download_config = DownloadConfig(
    cache_dir='/path/to/cache',
    force_download=True
)

# Get metadata information for the 'squad' dataset using the specified download configuration
config_info = get_dataset_config_info('squad', download_config=download_config)

# Print the retrieved dataset info
print(config_info)
```

**Getting Dataset Info with a Specific Version**

This example demonstrates how to get the metadata information for a specific version of a dataset by specifying the revision parameter.

```python
from datasets import get_dataset_config_info

# Get metadata information for the 'glue' dataset, specifying the 'main' branch as the revision
config_info = get_dataset_config_info('glue', config_name='cola', revision='main')

# Print the retrieved dataset info
print(config_info)
```

**Getting Dataset Info with Custom Configurations**

This example shows how to get the metadata information for a dataset with additional custom configurations.

```python
from datasets import get_dataset_config_info

# Define custom configurations
config_kwargs = {
    'field': 'data',
    'script_version': '1.0.0'
}

# Get metadata information for the 'squad' dataset with custom configurations
config_info = get_dataset_config_info('squad', config_name='plain_text', **config_kwargs)

# Print the retrieved dataset info
print(config_info)
```

### get_dataset_split_names()
Here are some examples using various parameters of the `get_dataset_split_names` function, with comments to explain each step:

**Getting Split Names for a Dataset from the Hugging Face Hub**

This example shows how to get the list of available splits for a dataset hosted on the Hugging Face Hub.

```python
from datasets import get_dataset_split_names

# Get the list of available splits for the 'rotten_tomatoes' dataset from the Hugging Face Hub
split_names = get_dataset_split_names('rotten_tomatoes')

# Print the retrieved split names
print(split_names)
```

**Getting Split Names for a Specific Configuration of a Dataset**

This example demonstrates how to get the list of available splits for a specific configuration of a dataset.

```python
from datasets import get_dataset_split_names

# Get the list of available splits for the 'squad' dataset with the 'plain_text' configuration
split_names = get_dataset_split_names('squad', config_name='plain_text')

# Print the retrieved split names
print(split_names)
```

**Getting Split Names with Specific Data Files**

This example shows how to get the list of available splits for a dataset by specifying particular data files.

```python
from datasets import get_dataset_split_names

# Define the data files to use for the dataset
data_files = {
    'train': 'path/to/train.csv',
    'validation': 'path/to/validation.csv',
    'test': 'path/to/test.csv'
}

# Get the list of available splits for a dataset using the specified data files
split_names = get_dataset_split_names('csv', config_name='my_dataset_config', data_files=data_files)

# Print the retrieved split names
print(split_names)
```

**Getting Split Names with Download Configuration**

This example shows how to get the list of available splits for a dataset with specific download configuration parameters.

```python
from datasets import get_dataset_split_names, DownloadConfig

# Define the download configuration
download_config = DownloadConfig(
    cache_dir='/path/to/cache',
    force_download=True
)

# Get the list of available splits for the 'squad' dataset using the specified download configuration
split_names = get_dataset_split_names('squad', download_config=download_config)

# Print the retrieved split names
print(split_names)
```

**Getting Split Names for a Specific Version of a Dataset**

This example demonstrates how to get the list of available splits for a specific version of a dataset by specifying the revision parameter.

```python
from datasets import get_dataset_split_names

# Get the list of available splits for the 'glue' dataset, specifying the 'main' branch as the revision
split_names = get_dataset_split_names('glue', config_name='cola', revision='main')

# Print the retrieved split names
print(split_names)
```

**Getting Split Names with Custom Configurations**

This example shows how to get the list of available splits for a dataset with additional custom configurations.

```python
from datasets import get_dataset_split_names

# Define custom configurations
config_kwargs = {
    'field': 'data',
    'script_version': '1.0.0'
}

# Get the list of available splits for the 'squad' dataset with custom configurations
split_names = get_dataset_split_names('squad', config_name='plain_text', **config_kwargs)

# Print the retrieved split names
print(split_names)
```


## Conclusion
The HuggingFace Datasets library offers a comprehensive suite of tools for managing datasets, enabling seamless loading, processing, and utilization for machine learning tasks. With support for multiple data storage locations and formats, along with a range of methods to access and retrieve detailed dataset information, this library simplifies the workflow for data scientists and machine learning practitioners. By providing robust functionalities and ease of use, HuggingFace Datasets helps streamline the process of working with data, allowing users to focus more on model development and less on data handling.


## References
- [HuggingFace Datasets Quickstart](https://huggingface.co/docs/datasets/quickstart)
- [HuggingFace Datasets Package Reference: Loading Methods](https://huggingface.co/docs/datasets/package_reference/loading_methods)
- [HuggingFace Datasets Source Code](https://github.com/huggingface/datasets/blob/2.19.0/src/datasets/load.py)


