# Introducing the Main Classes of HuggingFace Datasets

## Introduction
HuggingFace Datasets is a powerful library designed for seamless integration and manipulation of large datasets, particularly in the field of Natural Language Processing (NLP). This library provides a wide array of tools and classes to facilitate efficient data handling, allowing researchers and developers to focus on model building and experimentation.

### Key Features
- **Scalability**: Efficiently handle large datasets without compromising on performance.
- **Flexibility**: Support for various data formats, including CSV, JSON, and Parquet.
- **Interoperability**: Easily integrate with popular machine learning frameworks like TensorFlow, PyTorch, and HuggingFace Transformers.

### Core Classes
1. **DatasetInfo**: Provides metadata about the dataset, such as the dataset's description, citation, license, and features.
2. **Dataset**: Represents a single split of a dataset (e.g., train, test, validation) and provides methods for data manipulation, such as filtering, mapping, and shuffling.
3. **DatasetDict**: A dictionary-like structure that holds multiple `Dataset` objects, representing different splits of the dataset.
4. **IterableDataset**: Designed for datasets that can be iterated over, useful for streaming large datasets that don't fit in memory.
5. **IterableDatasetDict**: Similar to `DatasetDict`, but specifically for `IterableDataset` objects, allowing for efficient handling of large, streaming datasets.
6. **Features**: Defines the internal structure of a dataset, specifying the types and properties of each column, such as `Value`, `ClassLabel`, `Sequence`, and more.

These classes form the backbone of the HuggingFace Datasets library, providing the necessary tools to preprocess, analyze, and transform datasets effectively. Whether you are working with small, in-memory datasets or large-scale, streaming datasets, these classes ensure a seamless and efficient workflow.

## Examples of Sentence Transformer

## class DatasetInfo
The `DatasetInfo` class encapsulates the metadata and descriptive information about a dataset. This includes essential details like the dataset's name, version, features, and various attributes necessary for understanding and utilizing the dataset effectively.

**Overview of DatasetInfo**

The `DatasetInfo` class provides a comprehensive structure for documenting a dataset, including its description, citation, license, and features. It supports loading and saving this metadata, merging information from multiple sources, and aligning task templates with dataset features.

| Method                  | Description                                                                                                                    |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `write_to_directory()`  | Writes the dataset information and license to a specified directory in JSON format.                                           |
| `from_merge()`          | Merges information from multiple `DatasetInfo` instances, ensuring consistent and combined metadata.                          |
| `from_directory()`      | Creates a `DatasetInfo` instance from a JSON file in a specified directory.                                                    |
| `from_dict()`           | Creates a `DatasetInfo` instance from a dictionary representation.                                                            |
| `update()`              | Updates the current `DatasetInfo` with values from another instance, optionally ignoring `None` values.                       |
| `copy()`                | Creates a deep copy of the current `DatasetInfo` instance.                                                                     |
| `__post_init__()`       | Initializes the dataset info, converting dictionary representations to the correct classes and aligning task templates.        |
| `_to_yaml_dict()`       | Converts the dataset info to a dictionary formatted for YAML serialization.                                                   |
| `_from_yaml_dict()`     | Creates a `DatasetInfo` instance from a dictionary formatted for YAML serialization.                                           |
| `_dump_info()`          | Dumps the dataset info into a file-like object in bytes mode.                                                                 |
| `_dump_license()`       | Dumps the dataset license into a file-like object in bytes mode.                                                              |


**1. Updating DatasetInfo**

Demonstrates how to update an existing `DatasetInfo` instance with new metadata.

```python
from datasets import DatasetInfo

# Existing dataset info
info1 = DatasetInfo(description="Initial description", citation="Initial citation")

# New dataset info to update with
info2 = DatasetInfo(description="Updated description", homepage="http://example.com")

# Update info1 with info2
info1.update(info2)

print(info1.description)  # Output: "Updated description"
print(info1.citation)     # Output: "Initial citation"
print(info1.homepage)     # Output: "http://example.com"
```

**2. Copying DatasetInfo**

Shows how to create a deep copy of a `DatasetInfo` instance.

```python
from datasets import DatasetInfo

# Original dataset info
info = DatasetInfo(description="Original description", citation="Original citation")

# Create a deep copy
info_copy = info.copy()

# Modify the copy
info_copy.description = "Modified description"

print(info.description)     # Output: "Original description"
print(info_copy.description)  # Output: "Modified description"
```

**3. Writing DatasetInfo to Directory**

Illustrates how to save `DatasetInfo` metadata to a directory in JSON format.

```python
from datasets import DatasetInfo
import os

# Create dataset info
info = DatasetInfo(description="Dataset description", citation="Dataset citation")

# Directory to save the info
directory = "./dataset_info"
os.makedirs(directory, exist_ok=True)

# Write to directory
info.write_to_directory(directory, pretty_print=True)

# The JSON file is saved in the specified directory
```

**4. Merging Multiple DatasetInfo Instances**

Demonstrates how to merge information from multiple `DatasetInfo` instances.

```python
from datasets import DatasetInfo

# Dataset info instances
info1 = DatasetInfo(description="Description 1", citation="Citation 1")
info2 = DatasetInfo(description="Description 2", homepage="http://example.com")

# Merge the dataset info
merged_info = DatasetInfo.from_merge([info1, info2])

print(merged_info.description)  # Output: "Description 1\n\nDescription 2"
print(merged_info.citation)     # Output: "Citation 1"
print(merged_info.homepage)     # Output: "http://example.com"
```

**5. Loading DatasetInfo from Directory**

Shows how to create a `DatasetInfo` instance from a JSON file in a directory.

```python
from datasets import DatasetInfo
import os

# Directory containing the dataset info JSON file
directory = "./dataset_info"

# Load dataset info from the directory
loaded_info = DatasetInfo.from_directory(directory)

print(loaded_info.description)  # Output: "Dataset description"
print(loaded_info.citation)     # Output: "Dataset citation"
```

**6. Creating DatasetInfo from Dictionary**

Illustrates how to create a `DatasetInfo` instance from a dictionary.

```python
from datasets import DatasetInfo

# Dictionary representing dataset info
info_dict = {
    "description": "Dataset from dictionary",
    "citation": "Dictionary citation",
    "homepage": "http://example.com"
}

# Create DatasetInfo from dictionary
info = DatasetInfo.from_dict(info_dict)

print(info.description)  # Output: "Dataset from dictionary"
print(info.citation)     # Output: "Dictionary citation"
print(info.homepage)     # Output: "http://example.com"
```

## class Dataset
class Dataset representing a dataset backed by an Arrow table, providing various methods for data manipulation, transformation, and export. It extends `DatasetInfoMixin`, `IndexableMixin`, and `TensorflowDatasetMixin`.

**Overview of Dataset**

The Dataset class in the HuggingFace Datasets library provides methods for efficiently handling, transforming, and exporting datasets. Below is an overview of the available methods and their descriptions:

| **Method** | **Description** |
|------------|-----------------|
| `filter` | Applies a filter function to the dataset, keeping only the examples that meet the criteria defined by the function. |
| `flatten_indices` | Flattens the indices mapping to create a new dataset. |
| `select` | Selects rows based on provided indices and returns a new dataset. |
| `skip` | Creates a new dataset by skipping the first `n` elements. |
| `take` | Creates a new dataset with only the first `n` elements. |
| `sort` | Sorts the dataset based on specified columns. |
| `shuffle` | Shuffles the dataset. |
| `train_test_split` | Splits the dataset into train and test subsets. |
| `shard` | Returns a shard (subset) of the dataset. |
| `to_csv` | Exports the dataset to a CSV file. |
| `to_dict` | Converts the dataset to a dictionary. |
| `to_list` | Converts the dataset to a list. |
| `to_json` | Exports the dataset to a JSON file. |
| `to_pandas` | Converts the dataset to a `pandas.DataFrame`. |
| `to_polars` | Converts the dataset to a `polars.DataFrame`. |
| `to_parquet` | Exports the dataset to a Parquet file. |
| `to_sql` | Exports the dataset to a SQL database. |
| `to_iterable_dataset` | Converts the dataset to an `IterableDataset`. |
| `push_to_hub` | Pushes the dataset to the Hugging Face Hub as a Parquet dataset. |
| `add_column` | Adds a new column to the dataset. |
| `add_faiss_index` | Adds a dense index using Faiss for fast retrieval. |
| `add_faiss_index_from_external_arrays` | Adds a dense index using external arrays with Faiss for fast retrieval. |
| `add_elasticsearch_index` | Adds a text index using Elasticsearch for fast retrieval. |
| `add_item` | Adds a new item (row) to the dataset. |
| `align_labels_with_mapping` | Aligns the dataset's label ID and label name mapping with an input `label2id` mapping. |
| `_new_dataset_with_indices` | Returns a new dataset with provided indices. |
| `_estimate_nbytes` | Estimates the number of bytes of the dataset. |
| `_push_parquet_shards_to_hub` | Pushes dataset shards as Parquet files to the Hugging Face Hub. |
| `_generate_tables_from_shards` | Generates tables from dataset shards. |
| `_generate_tables_from_cache_file` | Generates tables from a cache file. |


**Examples of Dataset**

**1. Sorting the Dataset**

Sort the dataset based on a specific column in descending order.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Sort the dataset by the 'label' column in descending order
sorted_ds = ds.sort(column_names="label", reverse=True)
print(sorted_ds[:10])  # Print the first 10 examples to verify sorting
```

**2. Shuffling the Dataset**

Shuffle the dataset to randomize the order of the rows.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Shuffle the dataset with a specific seed for reproducibility
shuffled_ds = ds.shuffle(seed=42)
print(shuffled_ds[:10])  # Print the first 10 examples to verify shuffling
```

**3. Selecting Specific Rows**

Select specific rows based on indices.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Select rows with indices 0, 1, and 2
selected_ds = ds.select(indices=[0, 1, 2])
print(selected_ds)  # Print the selected rows
```

**4. Splitting the Dataset**

Split the dataset into train and test sets.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Split the dataset into 80% train and 20% test
split_ds = ds.train_test_split(test_size=0.2)
print(split_ds)  # Print the resulting splits
```

**5. Sharding the Dataset**

Create a shard of the dataset.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Shard the dataset into 3 parts and select the second shard (index=1)
sharded_ds = ds.shard(num_shards=3, index=1)
print(sharded_ds)  # Print the shard
```

**6. Renaming a Column**

Rename a column in the dataset.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Rename the 'text' column to 'review'
renamed_ds = ds.rename_column(original_column_name="text", new_column_name="review")
print(renamed_ds.column_names)  # Print the column names to verify renaming
```

**7. Removing a Column**

Remove a column from the dataset.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Remove the 'label' column
removed_column_ds = ds.remove_columns(column_names="label")
print(removed_column_ds.column_names)  # Print the column names to verify removal
```

**8. Casting a Column to a Different Type**

Cast a column to a different data type.
```python
from datasets import load_dataset, Features, Value

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Define the new feature types
new_features = Features({"text": Value("string"), "label": Value("int8")})

# Cast the dataset
casted_ds = ds.cast(new_features)
print(casted_ds.features)  # Print the feature types to verify casting
```

**9. Flattening Nested Columns**

Flatten nested columns in the dataset.
```python
from datasets import load_dataset

# Load a dataset with nested columns
ds = load_dataset("super_glue", "record", split="train")

# Flatten the dataset
flattened_ds = ds.flatten()
print(flattened_ds.column_names)  # Print the column names to verify flattening
```

**10. Mapping a Function Over the Dataset**

Apply a transformation to each element in the dataset.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Define a transformation function
def add_exclamation(example):
    example["text"] = example["text"] + "!"
    return example

# Apply the transformation
mapped_ds = ds.map(add_exclamation)
print(mapped_ds[:3])  # Print the first 3 examples to verify the transformation
```

**11. Setting the Format of the Dataset**

Change the format of the dataset to 'numpy'.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Set the format to 'numpy'
numpy_ds = ds.set_format(type="numpy")
print(numpy_ds.format)  # Print the format to verify
```

**12. With Format Context Manager**

Use a context manager to temporarily change the format of the dataset.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Use 'with_format' to temporarily change the format
with ds.with_format("pandas"):
    pandas_df = ds[:5]
    print(pandas_df)  # Print the first 5 examples in pandas format

# Verify the format is reverted back
print(ds.format)
```

**13. Saving the Dataset to Disk**

Save the dataset to disk and then load it back.
```python
from datasets import load_dataset, Dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Save the dataset to disk
ds.save_to_disk("saved_dataset")

# Load the dataset back from disk
loaded_ds = Dataset.load_from_disk("saved_dataset")
print(loaded_ds)  # Print the loaded dataset
```

**14. Exporting the Dataset to CSV**

Export the dataset to a CSV file.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Export the dataset to CSV
ds.to_csv("dataset.csv")
```

**15. Exporting the Dataset to JSON**

Export the dataset to a JSON file.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Export the dataset to JSON
ds.to_json("dataset.json")
```

**16. Exporting the Dataset to SQL**

Export the dataset to a SQL database.
```python
from datasets import load_dataset
import sqlite3

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Create a SQLite database connection
con = sqlite3.connect("dataset.db")

# Export the dataset to the SQL database
ds.to_sql(name="rotten_tomatoes", con=con)
```

**17. Converting the Dataset to a Pandas DataFrame**

Convert the dataset to a Pandas DataFrame.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Convert the dataset to a Pandas DataFrame
df = ds.to_pandas()
print(df.head())  # Print the first few rows of the DataFrame
```

**18. Converting the Dataset to a Dictionary**

Convert the dataset to a dictionary.
```python
from datasets import load_dataset

# Load a dataset
ds = load_dataset("rotten_tomatoes", split="validation")

# Convert the dataset to a dictionary
dict_data = ds.to_dict()
print(dict_data.keys())  # Print the dictionary keys to verify conversion
```

## class DatasetDict
A dictionary (dict of str: datasets.Dataset) with dataset transforms methods (map, filter, etc.)

**Overview of DatasetDict**

This table provides an overview of the methods available in the DatasetDict class, which enable various dataset transformations and manipulations.

| **Method**              | **Description**                                                                                          |
|-------------------------|----------------------------------------------------------------------------------------------------------|
| `filter`                | Applies a filter function to the dataset, keeping only the examples that meet the criteria defined by the function. |
| `flatten_indices`       | Flattens the indices mapping to create a new dataset.                                                    |
| `map`                   | Applies a function to all elements in the dataset, updating the dataset based on the function's output.  |
| `shuffle`               | Creates a new dataset where the rows are shuffled.                                                       |
| `sort`                  | Creates a new dataset sorted according to specified columns.                                             |
| `cast`                  | Casts the dataset to a new set of features.                                                              |
| `remove_columns`        | Removes specified columns from each split in the dataset.                                                |
| `rename_column`         | Renames a column in the dataset, updating the associated features.                                       |
| `select_columns`        | Selects specified columns from each split in the dataset.                                                |
| `class_encode_column`   | Casts the given column as `ClassLabel` and updates the tables.                                           |
| `save_to_disk`          | Saves the dataset to a specified directory on disk.                                                      |
| `load_from_disk`        | Loads a dataset from a directory on disk where it was previously saved.                                  |
| `from_csv`              | Creates a `DatasetDict` from CSV file(s).                                                                |
| `from_json`             | Creates a `DatasetDict` from JSON Lines file(s).                                                         |
| `from_parquet`          | Creates a `DatasetDict` from Parquet file(s).                                                            |
| `from_text`             | Creates a `DatasetDict` from text file(s).                                                               |
| `with_format`           | Sets the format for `__getitem__` return type and columns.                                               |
| `with_transform`        | Sets a transform to apply to batches when `__getitem__` is called.                                       |
| `push_to_hub`           | Pushes the dataset to the Hugging Face Hub as a Parquet dataset.                                         |
| `align_labels_with_mapping` | Aligns dataset labels with a given mapping.                                                      |
| `prepare_for_task`      | Prepares the dataset for a specific task.                                                                |
| `cleanup_cache_files`   | Cleans up all cache files in the dataset cache directory, except the currently used cache file.          |
| `set_format`            | Sets the format for `__getitem__` return type and columns.                                               |
| `reset_format`          | Resets `__getitem__` return format to python objects and all columns.                                    |
| `set_transform`         | Sets a transform for formatting the dataset.                                                             |
| `cast_column`           | Casts a specified column to a new feature type.                                                          |
| `rename_columns`        | Renames several columns in the dataset, updating the associated features.                                |


**Examples of DatasetDict**

**1. Sorting the DatasetDict**

Sorting a dataset by a specific column in ascending order.
```python
from datasets import Dataset, DatasetDict

# Creating example datasets
data = {'text': ['Example1', 'Example2', 'Example3'], 'label': [1, 2, 0]}
dataset = Dataset.from_dict(data)
dataset_dict = DatasetDict({"train": dataset, "test": dataset})

# Sorting the dataset by the 'label' column
sorted_dataset_dict = dataset_dict.sort(column_names='label')

print(sorted_dataset_dict['train']['label'])  # Output: [0, 1, 2]
```

**2. Shuffling the DatasetDict**

Shuffling the dataset with a specified seed for reproducibility.
```python
from datasets import Dataset, DatasetDict

# Creating example datasets
data = {'text': ['Example1', 'Example2', 'Example3'], 'label': [1, 2, 0]}
dataset = Dataset.from_dict(data)
dataset_dict = DatasetDict({"train": dataset, "test": dataset})

# Shuffling the dataset with a seed
shuffled_dataset_dict = dataset_dict.shuffle(seed=42)

print(shuffled_dataset_dict['train']['label'])
```

**3. Casting DatasetDict Columns**

Changing the feature types of the dataset columns.
```python
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value

# Creating example datasets
data = {'text': ['Example1', 'Example2', 'Example3'], 'label': [1, 2, 0]}
dataset = Dataset.from_dict(data)
dataset_dict = DatasetDict({"train": dataset, "test": dataset})

# Define new features with different types
new_features = Features({'text': Value('string'), 'label': ClassLabel(names=['Class0', 'Class1', 'Class2'])})

# Casting the dataset to new features
casted_dataset_dict = dataset_dict.cast(new_features)

print(casted_dataset_dict['train'].features)
```

**4. Flattening Indices in DatasetDict**

Flattening the indices mapping to create a new dataset.
```python
from datasets import Dataset, DatasetDict

# Creating example datasets
data = {'text': ['Example1', 'Example2', 'Example3'], 'label': [1, 2, 0]}
dataset = Dataset.from_dict(data)
dataset_dict = DatasetDict({"train": dataset, "test": dataset})

# Flattening indices
flattened_dataset_dict = dataset_dict.flatten_indices()

print(flattened_dataset_dict)
```

**5. Filtering the DatasetDict**

Applying a filter function to keep only certain rows.
```python
from datasets import Dataset, DatasetDict

# Creating example datasets
data = {'text': ['Example1', 'Example2', 'Example3'], 'label': [1, 2, 0]}
dataset = Dataset.from_dict(data)
dataset_dict = DatasetDict({"train": dataset, "test": dataset})

# Filtering the dataset to keep only rows where 'label' is 1
filtered_dataset_dict = dataset_dict.filter(lambda x: x['label'] == 1)

print(filtered_dataset_dict['train'])
```

**6. Mapping a Function Over DatasetDict**

Applying a function to all elements in the dataset.
```python
from datasets import Dataset, DatasetDict

# Creating example datasets
data = {'text': ['Example1', 'Example2', 'Example3'], 'label': [1, 2, 0]}
dataset = Dataset.from_dict(data)
dataset_dict = DatasetDict({"train": dataset, "test": dataset})

# Function to add a prefix to the text
def add_prefix(example):
    example['text'] = 'Review: ' + example['text']
    return example

# Applying the function using map
mapped_dataset_dict = dataset_dict.map(add_prefix)

print(mapped_dataset_dict['train']['text'])
```

**7. Removing Columns from DatasetDict**

Removing specified columns from the dataset.
```python
from datasets import Dataset, DatasetDict

# Creating example datasets
data = {'text': ['Example1', 'Example2', 'Example3'], 'label': [1, 2, 0]}
dataset = Dataset.from_dict(data)
dataset_dict = DatasetDict({"train": dataset, "test": dataset})

# Removing the 'label' column
modified_dataset_dict = dataset_dict.remove_columns('label')

print(modified_dataset_dict['train'])
```

**8. Selecting Columns from DatasetDict**

Selecting specific columns from the dataset.
```python
from datasets import Dataset, DatasetDict

# Creating example datasets
data = {'text': ['Example1', 'Example2', 'Example3'], 'label': [1, 2, 0]}
dataset = Dataset.from_dict(data)
dataset_dict = DatasetDict({"train": dataset, "test": dataset})

# Selecting only the 'text' column
selected_dataset_dict = dataset_dict.select_columns('text')

print(selected_dataset_dict['train'])
```

**9. Setting Format for DatasetDict**

Setting the format for the dataset's output.
```python
from datasets import Dataset, DatasetDict

# Creating example datasets
data = {'text': ['Example1', 'Example2', 'Example3'], 'label': [1, 2, 0]}
dataset = Dataset.from_dict(data)
dataset_dict = DatasetDict({"train": dataset, "test": dataset})

# Setting the format to return numpy arrays
dataset_dict.set_format(type='numpy', columns=['text', 'label'])

print(dataset_dict['train'][:])
```

**10. Using With Format for DatasetDict**

Temporarily setting the format for the dataset's output within a context.
```python
from datasets import Dataset, DatasetDict

# Creating example datasets
data = {'text': ['Example1', 'Example2', 'Example3'], 'label': [1, 2, 0]}
dataset = Dataset.from_dict(data)
dataset_dict = DatasetDict({"train": dataset, "test": dataset})

# Using with_format to temporarily set the format to pandas
with dataset_dict.with_format('pandas'):
    print(dataset_dict['train'][:])
```

**11. Using With Transform for DatasetDict**

Setting a transform to apply to batches when `__getitem__` is called.
```python
from datasets import Dataset, DatasetDict

# Creating example datasets
data = {'text': ['Example1', 'Example2', 'Example3'], 'label': [1, 2, 0]}
dataset = Dataset.from_dict(data)
dataset_dict = DatasetDict({"train": dataset, "test": dataset})

# Defining a transform function
def transform(example):
    return {'text': example['text'].upper(), 'label': example['label']}

# Using with_transform to apply the transform
transformed_dataset_dict = dataset_dict.with_transform(transform)

print(transformed_dataset_dict['train'][0])
```

**12. Saving DatasetDict to Disk**

Saving the dataset to a specified directory on disk.
```python
from datasets import Dataset, DatasetDict

# Creating example datasets
data = {'text': ['Example1', 'Example2', 'Example3'], 'label': [1, 2, 0]}
dataset = Dataset.from_dict(data)
dataset_dict = DatasetDict({"train": dataset, "test": dataset})

# Saving the dataset to disk
dataset_dict.save_to_disk('path/to/dataset_directory')
```

**13. Loading DatasetDict from CSV**

Creating a `DatasetDict` from CSV file(s).
```python
from datasets import DatasetDict

# Loading a DatasetDict from CSV files
dataset_dict = DatasetDict.from_csv({'train': 'path/to/train.csv', 'test': 'path/to/test.csv'})

print(dataset_dict)
```

**14. Loading DatasetDict from JSON**

Creating a `DatasetDict` from JSON Lines file(s).
```python
from datasets import DatasetDict

# Loading a DatasetDict from JSON Lines files
dataset_dict = DatasetDict.from_json({'train': 'path/to/train.jsonl', 'test': 'path/to/test.jsonl'})

print(dataset_dict)
```

## class IterableDataset
The `IterableDataset` class represents a dataset backed by an iterable, allowing for efficient, streaming access to data. This is particularly useful for handling large datasets that cannot be loaded into memory at once.

**Overview of IterableDataset**

The table below provides an overview of the methods available in the `IterableDataset` class, including their descriptions and main functionalities.

| **Method**        | **Description**                                                                                                           |
|-------------------|---------------------------------------------------------------------------------------------------------------------------|
| `__init__`        | Initializes the `IterableDataset` with various configurations like shuffling, formatting, and distributed settings.       |
| `__repr__`        | Returns a string representation of the `IterableDataset`, including features and the number of shards.                    |
| `__getstate__`    | Prepares the `IterableDataset` for pickling by returning its state.                                                       |
| `__setstate__`    | Restores the state of the `IterableDataset` from a pickled state.                                                         |
| `__iter__`        | Provides the main iteration logic over the dataset, handling formatting and example conversion.                           |
| `n_shards`        | Returns the number of shards in the dataset, adjusted for distributed settings.                                           |
| `iter`            | Iterates through the dataset in batches of a specified size.                                                              |
| `from_generator`  | Creates an `IterableDataset` from a generator function.                                                                    |
| `from_spark`      | Creates an `IterableDataset` from a Spark DataFrame.                                                                       |
| `from_file`       | Instantiates an `IterableDataset` from an Arrow table stored in a file.                                                   |
| `with_format`     | Returns a dataset with the specified format, supporting formats like "arrow" and "torch".                                 |
| `map`             | Applies a function to all examples in the dataset, optionally in batches.                                                 |
| `filter`          | Applies a filter function to the dataset, keeping only the examples that meet the criteria defined by the function.       |
| `shuffle`         | Randomly shuffles the elements of the dataset using a specified buffer size and seed.                                      |
| `set_epoch`       | Sets the current epoch for the dataset, affecting the shuffling behavior.                                                 |
| `skip`            | Creates a new `IterableDataset` that skips the first `n` elements.                                                        |
| `take`            | Creates a new `IterableDataset` with only the first `n` elements.                                                         |
| `column_names`    | Returns the names of the columns in the dataset.                                                                          |
| `add_column`      | Adds a new column to the dataset.                                                                                         |
| `rename_column`   | Renames a column in the dataset.                                                                                          |
| `rename_columns`  | Renames several columns in the dataset.                                                                                   |
| `remove_columns`  | Removes specified columns from the dataset.                                                                               |
| `select_columns`  | Selects specified columns from the dataset, removing all others.                                                          |
| `cast_column`     | Casts a column to a specified feature type.                                                                               |
| `cast`            | Casts the entire dataset to a new set of features.                                                                        |
| `_iter_pytorch`   | Handles iteration over the dataset for PyTorch `IterableDataset`, including sharding and formatting.                      |
| `_is_main_process` | Determines if the current process is the main process in a distributed setting.                                          |
| `_prepare_ex_iterable_for_iteration` | Prepares the example iterable for iteration, including shuffling and sharding.                           |
| `_head`           | Returns the first `n` examples from the dataset.                                                                          |
| `_effective_generator` | Returns the effective generator for shuffling based on the epoch and shuffling configuration.                        |
| `_step`           | Creates a new `IterableDataset` with elements selected in steps (e.g., every nth element).                                |
| `_resolve_features` | Resolves and infers features of the dataset if they are not explicitly provided.                                         |


**Examples of IterableDataset**

**1. Creating an IterableDataset from a Generator**

This example demonstrates how to create an `IterableDataset` from a generator function.

```python
from datasets import IterableDataset

# Define a generator function
def gen():
    yield {"text": "Good", "label": 0}
    yield {"text": "Bad", "label": 1}

# Create an IterableDataset from the generator function
dataset = IterableDataset.from_generator(gen)

# Iterate through the dataset and print examples
for example in dataset:
    print(example)

# Output:
# {'text': 'Good', 'label': 0}
# {'text': 'Bad', 'label': 1}
```

**2. Creating an IterableDataset from a File**

This example demonstrates how to create an `IterableDataset` from an Arrow file.

```python
from datasets import IterableDataset

# Create an IterableDataset from an Arrow file
dataset = IterableDataset.from_file('path/to/your/file.arrow')

# Iterate through the dataset and print examples
for example in dataset:
    print(example)
```

**3. Selecting Specific Columns**

This example demonstrates how to select specific columns from the dataset.

```python
from datasets import IterableDataset

# Assume `dataset` is an already created IterableDataset
dataset = IterableDataset.from_generator(gen)

# Select specific columns from the dataset
selected_dataset = dataset.select_columns(["text"])

# Iterate through the selected dataset and print examples
for example in selected_dataset:
    print(example)
```

**4. Renaming Columns**

This example demonstrates how to rename columns in the dataset.

```python
from datasets import IterableDataset

# Assume `dataset` is an already created IterableDataset
dataset = IterableDataset.from_generator(gen)

# Rename columns in the dataset
renamed_dataset = dataset.rename_columns({"text": "review_text", "label": "review_label"})

# Iterate through the renamed dataset and print examples
for example in renamed_dataset:
    print(example)

# Output:
# {'review_text': 'Good', 'review_label': 0}
# {'review_text': 'Bad', 'review_label': 1}
```

**5. Removing Columns**

This example demonstrates how to remove specific columns from the dataset.

```python
from datasets import IterableDataset

# Assume `dataset` is an already created IterableDataset
dataset = IterableDataset.from_generator(gen)

# Remove specific columns from the dataset
reduced_dataset = dataset.remove_columns(["label"])

# Iterate through the reduced dataset and print examples
for example in reduced_dataset:
    print(example)
```

**6. Iterating in Batches**

This example demonstrates how to iterate through the dataset in batches.

```python
from datasets import IterableDataset

# Assume `dataset` is an already created IterableDataset
dataset = IterableDataset.from_generator(gen)

# Iterate through the dataset in batches
for batch in dataset.iter(batch_size=2):
    print(batch)
```

**7. Shuffling the Dataset**

This example demonstrates how to shuffle the dataset with a specified buffer size and seed.

```python
from datasets import IterableDataset

# Assume `dataset` is an already created IterableDataset
dataset = IterableDataset.from_generator(gen)

# Shuffle the dataset
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10)

# Iterate through the shuffled dataset and print examples
for example in shuffled_dataset:
    print(example)
```

**8. Filtering Examples**

This example demonstrates how to filter examples in the dataset based on a condition.

```python
from datasets import IterableDataset

# Assume `dataset` is an already created IterableDataset
dataset = IterableDataset.from_generator(gen)

# Filter examples where the label is 1
filtered_dataset = dataset.filter(lambda x: x["label"] == 1)

# Iterate through the filtered dataset and print examples
for example in filtered_dataset:
    print(example)
```

**9. Mapping a Function**

This example demonstrates how to apply a function to all examples in the dataset.

```python
from datasets import IterableDataset

# Assume `dataset` is an already created IterableDataset
dataset = IterableDataset.from_generator(gen)

# Define a function to add a prefix to the text
def add_prefix(example):
    example["text"] = "Review: " + example["text"]
    return example

# Apply the function to all examples in the dataset
mapped_dataset = dataset.map(add_prefix)

# Iterate through the mapped dataset and print examples
for example in mapped_dataset:
    print(example)
```

**10. Adding a Column**

This example demonstrates how to add a new column to the dataset.

```python
from datasets import IterableDataset

# Assume `dataset` is an already created IterableDataset
dataset = IterableDataset.from_generator(gen)

# Define a new column
new_column = ["positive", "negative"]

# Add the new column to the dataset
updated_dataset = dataset.add_column("sentiment", new_column)

# Iterate through the updated dataset and print examples
for example in updated_dataset:
    print(example)
```

## class IterableDatasetDict
The `IterableDatasetDict` class is a dictionary that contains multiple `IterableDataset` objects. This class provides methods to manipulate all datasets within the dictionary in a uniform way, such as renaming columns, filtering, shuffling, and applying transformations.

**Overview of IterableDatasetDict**

The following table describes the methods available in the `IterableDatasetDict` class:

| **Method**        | **Description**                                                                                                                                                      |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `with_format`     | Sets a specified format (e.g., "torch") for all datasets in the dictionary.                                                                                          |
| `map`             | Applies a function to all examples in the datasets (individually or in batches), transforming them as specified.                                                     |
| `filter`          | Filters examples in the datasets according to a specified function, keeping only the examples that meet the criteria.                                               |
| `shuffle`         | Randomly shuffles the examples in the datasets.                                                                                                                      |
| `rename_column`   | Renames a specified column in all datasets.                                                                                                                          |
| `rename_columns`  | Renames multiple columns in all datasets according to a provided mapping.                                                                                            |
| `remove_columns`  | Removes one or more columns from all datasets.                                                                                                                       |
| `select_columns`  | Selects and keeps only specified columns in all datasets.                                                                                                            |
| `cast_column`     | Casts a specified column to a new feature type for decoding in all datasets.                                                                                         |
| `cast`            | Casts all datasets to a new set of features.                                                                                                                         |

**Examples of IterableDatasetDict**

**1. Setting a format for all datasets in the dictionary**

The main purpose of this example is to set the format of all datasets in the dictionary to "torch" so they can be used with PyTorch DataLoader.

```python
from datasets import load_dataset

# Load a dataset with streaming enabled
ds_dict = load_dataset("rotten_tomatoes", streaming=True)

# Convert to IterableDatasetDict
iterable_ds_dict = IterableDatasetDict(ds_dict)

# Set the format to "torch" for all datasets in the dictionary
torch_formatted_ds_dict = iterable_ds_dict.with_format("torch")

# Print the type of the datasets to confirm the format
for split, dataset in torch_formatted_ds_dict.items():
    print(f"{split} dataset format: {type(dataset)}")
```

**2. Applying a function to all examples in the datasets**

The main purpose of this example is to apply a function that adds a prefix to the "text" column for all datasets in the dictionary.

```python
from datasets import load_dataset

# Load a dataset with streaming enabled
ds_dict = load_dataset("rotten_tomatoes", streaming=True)

# Convert to IterableDatasetDict
iterable_ds_dict = IterableDatasetDict(ds_dict)

# Define a function to add a prefix to the "text" column
def add_prefix(example):
    example["text"] = "Review: " + example["text"]
    return example

# Apply the function to all datasets in the dictionary
mapped_ds_dict = iterable_ds_dict.map(add_prefix)

# Print some examples to verify the function was applied
for example in mapped_ds_dict["train"].take(3):
    print(example)

# Output:
# {'label': 1, 'text': 'Review: the rock is destined to be the 21st century ...'}
# {'label': 1, 'text': 'Review: the gorgeously elaborate continuation of ...'}
# {'label': 1, 'text': 'Review: effective but too-tepid ...'}
```

**3. Filtering examples in the datasets**

The main purpose of this example is to filter the examples in the datasets, keeping only those with a "label" of 0.

```python
from datasets import load_dataset

# Load a dataset with streaming enabled
ds_dict = load_dataset("rotten_tomatoes", streaming=True)

# Convert to IterableDatasetDict
iterable_ds_dict = IterableDatasetDict(ds_dict)

# Apply a filter to keep only examples with a label of 0
filtered_ds_dict = iterable_ds_dict.filter(lambda x: x["label"] == 0)

# Print some examples to verify the filter was applied
for example in filtered_ds_dict["train"].take(3):
    print(example)
```

**4. Shuffling the examples in the datasets**

The main purpose of this example is to shuffle the examples in the datasets.

```python
from datasets import load_dataset

# Load a dataset with streaming enabled
ds_dict = load_dataset("rotten_tomatoes", streaming=True)

# Convert to IterableDatasetDict
iterable_ds_dict = IterableDatasetDict(ds_dict)

# Shuffle the examples in the datasets
shuffled_ds_dict = iterable_ds_dict.shuffle(seed=42, buffer_size=1000)

# Print some examples to verify the shuffling was applied
for example in shuffled_ds_dict["train"].take(3):
    print(example)
```

**5. Selecting specific columns in the datasets**

The main purpose of this example is to select only the "text" column in the datasets.

```python
from datasets import load_dataset

# Load a dataset with streaming enabled
ds_dict = load_dataset("rotten_tomatoes", streaming=True)

# Convert to IterableDatasetDict
iterable_ds_dict = IterableDatasetDict(ds_dict)

# Select only the "text" column in the datasets
selected_columns_ds_dict = iterable_ds_dict.select_columns(["text"])

# Print some examples to verify the column selection
for example in selected_columns_ds_dict["train"].take(3):
    print(example)
```

**6. Renaming columns in the datasets**

The main purpose of this example is to rename the "text" column to "review_text" and the "label" column to "review_label" in the datasets.

```python
from datasets import load_dataset

# Load a dataset with streaming enabled
ds_dict = load_dataset("rotten_tomatoes", streaming=True)

# Convert to IterableDatasetDict
iterable_ds_dict = IterableDatasetDict(ds_dict)

# Rename columns in the datasets
renamed_columns_ds_dict = iterable_ds_dict.rename_columns({"text": "review_text", "label": "review_label"})

# Print some examples to verify the column renaming
for example in renamed_columns_ds_dict["train"].take(3):
    print(example)
```

**7. Removing columns in the datasets**

The main purpose of this example is to remove the "label" column from the datasets.

```python
from datasets import load_dataset

# Load a dataset with streaming enabled
ds_dict = load_dataset("rotten_tomatoes", streaming=True)

# Convert to IterableDatasetDict
iterable_ds_dict = IterableDatasetDict(ds_dict)

# Remove the "label" column from the datasets
removed_columns_ds_dict = iterable_ds_dict.remove_columns("label")

# Print some examples to verify the column removal
for example in removed_columns_ds_dict["train"].take(3):
    print(example)
```

**8. Casting columns in the datasets**

The main purpose of this example is to cast the "label" column to a new feature type in the datasets.

```python
from datasets import load_dataset, ClassLabel

# Load a dataset with streaming enabled
ds_dict = load_dataset("rotten_tomatoes", streaming=True)

# Convert to IterableDatasetDict
iterable_ds_dict = IterableDatasetDict(ds_dict)

# Cast the "label" column to a new feature type
new_features = ds_dict["train"].features.copy()
new_features["label"] = ClassLabel(names=["negative", "positive"])

casted_ds_dict = iterable_ds_dict.cast(new_features)

# Print some examples to verify the column casting
for example in casted_ds_dict["train"].take(3):
    print(example)

# Output:
# The first three examples from the "train" split of the "rotten_tomatoes" dataset, 
# with the "label" column cast to a ClassLabel type with the names "negative" and "positive".
# {'label': 1, 'text': 'Review: the rock is destined to be the 21st century ...'}
# {'label': 1, 'text': 'Review: the gorgeously elaborate continuation of ...'}
# {'label': 1, 'text': 'Review: effective but too-tepid ...'}
```

## class Features
The `Features` class is a special dictionary that defines the internal structure of a dataset. It specifies the types of the columns in a dataset and supports various data types, including nested fields, lists, and multidimensional arrays.

**Overview of Features**

The table below provides an overview of the methods available in the `Features` class:

| **Method** | **Description** |
|------------|-----------------|
| `from_arrow_schema` | Construct `Features` from an Arrow schema, including support for Hugging Face Datasets features. |
| `from_dict` | Construct `Features` from a dictionary, regenerating the nested feature object from a deserialized dict. |
| `to_dict` | Convert the `Features` object to a dictionary. |
| `encode_example` | Encode a single example into a format suitable for Arrow. |
| `encode_column` | Encode an entire column into a format suitable for Arrow. |
| `encode_batch` | Encode a batch of data into a format suitable for Arrow. |
| `decode_example` | Decode a single example with custom feature decoding. |
| `decode_column` | Decode an entire column with custom feature decoding. |
| `decode_batch` | Decode a batch of data with custom feature decoding. |
| `copy` | Make a deep copy of the `Features` object. |
| `reorder_fields_as` | Reorder `Features` fields to match the field order of another `Features` object. |
| `flatten` | Flatten the features, replacing dictionary columns with their subfields. |

**Examples of Features**

**1. Creating Features from a Dictionary**

This example demonstrates how to create a `Features` object from a dictionary.
```python
from datasets import Features, Value, ClassLabel

# Define the features using a dictionary
features_dict = {
    "text": Value("string"),
    "label": ClassLabel(names=["negative", "positive"])
}

# Create a Features object from the dictionary
features = Features.from_dict(features_dict)

# Print the Features object to verify
print(features)

# Output:
# {
#   'text': Value(dtype='string', id=None), 
#   'label': ClassLabel(num_classes=2, names=['negative', 'positive'], id=None)
# }
```

**2. Converting Features to a Dictionary**

This example demonstrates how to convert a `Features` object back to a dictionary.
```python
from datasets import Features, Value, ClassLabel

# Define the features using a dictionary
features_dict = {
    "text": Value("string"),
    "label": ClassLabel(names=["negative", "positive"])
}

# Create a Features object from the dictionary
features = Features.from_dict(features_dict)

# Convert the Features object to a dictionary
features_as_dict = features.to_dict()

# Print the dictionary to verify
print(features_as_dict)
```

**3. Encoding a Single Example**

This example demonstrates how to encode a single example using the `encode_example` method.
```python
from datasets import Features, Value, ClassLabel

# Define the features
features_dict = {
    "text": Value("string"),
    "label": ClassLabel(names=["negative", "positive"])
}
features = Features.from_dict(features_dict)

# Define an example
example = {"text": "Good", "label": 1}

# Encode the example
encoded_example = features.encode_example(example)

# Print the encoded example
print(encoded_example)
```

**4. Encoding an Entire Column**

This example demonstrates how to encode an entire column using the `encode_column` method.
```python
from datasets import Features, Value, ClassLabel

# Define the features
features_dict = {
    "text": Value("string"),
    "label": ClassLabel(names=["negative", "positive"])
}
features = Features.from_dict(features_dict)

# Define a column
column = ["Good", "Bad"]

# Encode the column
encoded_column = features.encode_column(column, column_name="text")

# Print the encoded column
print(encoded_column)
```

**5. Encoding a Batch of Data**

This example demonstrates how to encode a batch of data using the `encode_batch` method.
```python
from datasets import Features, Value, ClassLabel

# Define the features
features_dict = {
    "text": Value("string"),
    "label": ClassLabel(names=["negative", "positive"])
}
features = Features.from_dict(features_dict)

# Define a batch of data
batch = {
    "text": ["Good", "Bad"],
    "label": [1, 0]
}

# Encode the batch
encoded_batch = features.encode_batch(batch)

# Print the encoded batch
print(encoded_batch)

# Output:
# {
#   'text': [b'Good', b'Bad'], 
#   'label': [1, 0]
# }
```

**6. Decoding a Single Example**

This example demonstrates how to decode a single example using the `decode_example` method.
```python
from datasets import Features, Value, ClassLabel

# Define the features
features_dict = {
    "text": Value("string"),
    "label": ClassLabel(names=["negative", "positive"])
}
features = Features.from_dict(features_dict)

# Define an encoded example
encoded_example = {"text": "Good", "label": 1}

# Decode the example
decoded_example = features.decode_example(encoded_example)

# Print the decoded example
print(decoded_example)
```

**7. Decoding an Entire Column**

This example demonstrates how to decode an entire column using the `decode_column` method.
```python
from datasets import Features, Value, ClassLabel

# Define the features
features_dict = {
    "text": Value("string"),
    "label": ClassLabel(names=["negative", "positive"])
}
features = Features.from_dict(features_dict)

# Define an encoded column
encoded_column = ["Good", "Bad"]

# Decode the column
decoded_column = features.decode_column(encoded_column, column_name="text")

# Print the decoded column
print(decoded_column)
```

**8. Decoding a Batch of Data**

This example demonstrates how to decode a batch of data using the `decode_batch` method.
```python
from datasets import Features, Value, ClassLabel

# Define the features
features_dict = {
    "text": Value("string"),
    "label": ClassLabel(names=["negative", "positive"])
}
features = Features.from_dict(features_dict)

# Define an encoded batch
encoded_batch = {
    "text": ["Good", "Bad"],
    "label": [1, 0]
}

# Decode the batch
decoded_batch = features.decode_batch(encoded_batch)

# Print the decoded batch
print(decoded_batch)
```

**9. Reordering Fields to Match Another Features Object**

This example demonstrates how to reorder the fields of a `Features` object to match another `Features` object.
```python
from datasets import Features, Value, ClassLabel

# Define the original features
original_features_dict = {
    "text": Value("string"),
    "label": ClassLabel(names=["negative", "positive"])
}
original_features = Features.from_dict(original_features_dict)

# Define the new features with a different order
new_features_dict = {
    "label": ClassLabel(names=["negative", "positive"]),
    "text": Value("string")
}
new_features = Features.from_dict(new_features_dict)

# Reorder the original features to match the new features
reordered_features = original_features.reorder_fields_as(new_features)

# Print the reordered features
print(reordered_features)
```

**10. Flattening Features**

This example demonstrates how to flatten nested features using the `flatten` method.
```python
from datasets import Features, Value, Sequence

# Define nested features
nested_features_dict = {
    "text": Value("string"),
    "metadata": {
        "author": Value("string"),
        "length": Value("int32")
    }
}
nested_features = Features.from_dict(nested_features_dict)

# Flatten the nested features
flattened_features = nested_features.flatten()

# Print the flattened features
print(flattened_features)

# Output:
# {
#   'text': Value(dtype='string', id=None),
#   'metadata.author': Value(dtype='string', id=None),
#   'metadata.length': Value(dtype='int32', id=None)
# }
```


## Conclusion
The HuggingFace Datasets library offers a comprehensive set of tools for efficient dataset handling and manipulation, particularly tailored for NLP tasks. By leveraging classes such as `DatasetInfo`, `Dataset`, `DatasetDict`, `IterableDataset`, `IterableDatasetDict`, and `Features`, users can easily load, preprocess, and transform datasets for their machine learning and data science projects. These classes provide the flexibility and scalability needed to work with both small and large datasets, ensuring seamless integration with popular ML frameworks like TensorFlow and PyTorch.

Throughout this guide, we have explored the functionalities of these core classes, showcasing practical examples to illustrate their usage. By understanding and utilizing these tools, you can streamline your data processing workflows, allowing you to focus more on model development and experimentation.

Whether you are a researcher, data scientist, or developer, HuggingFace Datasets equips you with the necessary capabilities to handle diverse datasets efficiently, making it an indispensable tool in your data science toolkit.


## References
- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/package_reference/main_classes)
- [HuggingFace Datasets GitHub - info.py](https://github.com/huggingface/datasets/blob/2.19.0/src/datasets/info.py)
- [HuggingFace Datasets GitHub - arrow_dataset.py](https://github.com/huggingface/datasets/blob/2.19.0/src/datasets/arrow_dataset.py)
- [HuggingFace Datasets GitHub - dataset_dict.py](https://github.com/huggingface/datasets/blob/2.19.0/src/datasets/dataset_dict.py)
- [HuggingFace Datasets GitHub - iterable_dataset.py](https://github.com/huggingface/datasets/blob/2.19.0/src/datasets/iterable_dataset.py)
- [HuggingFace Datasets GitHub - features.py](https://github.com/huggingface/datasets/blob/2.19.0/src/datasets/features/features.py)

