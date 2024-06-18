# SFT: Advanced Data Handling Techniques for Robust Model Training

## Introduction

In deep learning, effective data handling is crucial for training robust and accurate models. Techniques like data augmentation, stratified sampling, and synthetic data generation enhance model performance, especially with imbalanced or limited datasets. This article explores these advanced methods, offering practical examples to help data scientists and machine learning practitioners improve their training processes. By using tools like Albumentations, SMOTE, MixUp, and CutMix, we can create more diverse and balanced training datasets, leading to models that generalize better and perform more reliably on unseen data.


## Applying Advanced Data Augmentation Techniques with Albumentations

Albumentations is a powerful library for image augmentation. It provides various advanced augmentation techniques that can improve model robustness by generating diverse training data.

**Main Benefits:**
1. **Enhanced Model Robustness**: By applying a wide variety of augmentations, Albumentations helps create diverse training data, which improves the model's ability to generalize to new, unseen data.
2. **Efficient and Fast**: Albumentations is optimized for performance and can apply complex augmentations quickly, making it suitable for large datasets and real-time applications.
3. **Rich Set of Transformations**: Provides a comprehensive collection of augmentation techniques, including geometric, color, and pixel-level transformations, allowing for extensive and varied augmentation.
4. **Easy Integration**: Seamlessly integrates with PyTorch, TensorFlow, and other deep learning frameworks, simplifying the augmentation process within existing workflows.

**How It Works Briefly:**
- Albumentations applies a series of transformations to images to create augmented versions. These transformations can include simple techniques like flipping and cropping, as well as more complex ones like elastic transformations, grid distortions, and advanced color manipulations.

[Albumentations GitHub](https://github.com/albumentations-team/albumentations)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """
    Custom dataset class for loading images and their corresponding labels, with optional data augmentation.
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize the dataset with image paths, labels, and optional transformations.

        Args:
            image_paths (list): List of paths to the image files.
            labels (list): List of labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get the image and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is a transformed tensor and label is the corresponding label.
        """
        image = cv2.imread(self.image_paths[idx])                # Read the image from file
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)           # Convert image to RGB
        label = self.labels[idx]                                 # Get the corresponding label

        if self.transform:
            image = self.transform(image=image)['image']         # Apply transformations if provided

        return image, label

# Define Albumentations transform
transform = A.Compose([
    A.RandomCrop(width=256, height=256),      # Randomly crop the image to 256x256 pixels
    A.HorizontalFlip(p=0.5),                  # Apply horizontal flip with a probability of 0.5
    A.RandomBrightnessContrast(p=0.2),        # Randomly change brightness and contrast with a probability of 0.2
    A.Normalize(mean=(0.485, 0.456, 0.406), 
	            std=(0.229, 0.224, 0.225)),   # Normalize the image
    ToTensorV2()                              # Convert the image to a PyTorch tensor
])

# Example usage of the dataset and dataloader
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
labels = [0, 1]

dataset = CustomDataset(image_paths=image_paths, labels=labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)     # Create a DataLoader with batch size of 2

for images, labels in dataloader:
    # Batch of 2 images, each with 3 channels and size 256x256
    print(images.shape)  # Output: torch.Size([2, 3, 256, 256])  

    # Corresponding labels
    print(labels)        # Output: tensor([0, 1])                
```


## Implementing Stratified Sampling for Imbalanced Datasets

Stratified sampling ensures that each class is equally represented in each batch, which is particularly useful for imbalanced datasets.

**Main Benefits:**
1. **Balanced Representation**: Ensures that each class is equally represented in each batch, preventing the model from becoming biased towards the majority classes.
2. **Improved Model Performance**: By maintaining a balanced representation of classes in each batch, the model learns more robust features for all classes, leading to improved performance, especially for underrepresented classes.
3. **Better Generalization**: Stratified sampling promotes better generalization by exposing the model to a balanced variety of samples during training.
4. **Stabilized Training**: Reduces the variance in gradient updates by providing a consistent class distribution, leading to more stable and effective training.

**For Examples:**
1. **Calculate Class Weights**: If class 0 has 80 samples and class 1 has 20 samples, the weights could be 1/80 for class 0 and 1/20 for class 1.
2. **Assign Weights to Samples**: Each sample in class 0 gets a weight of 1/80, and each sample in class 1 gets a weight of 1/20.
3. **Create a Sampler**: Use `WeightedRandomSampler` with these weights to sample data.
4. **Use in DataLoader**: Generate batches with `DataLoader` that use this sampler, ensuring balanced representation in each batch.

```python
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

class CustomDataset(Dataset):
    """
    A custom dataset class for handling data and labels.
    """
    def __init__(self, data, labels):
        """
        Initialize the dataset with data and labels.

        Args:
            data (torch.Tensor): Tensor containing the data samples.
            labels (torch.Tensor): Tensor containing the corresponding labels.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the data and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (data, label) where data is a tensor and label is the corresponding label.
        """
        return self.data[idx], self.labels[idx]

# Example data and labels
data = torch.randn(100, 3, 32, 32)       # Example data (100 samples, 3 channels, 32x32)
labels = torch.randint(0, 2, (100,))     # Example labels (binary classification, 100 labels)

# Calculate class weights
class_counts = torch.bincount(labels)    # Count the number of samples for each class
class_weights = 1. / class_counts        # Compute the weight for each class
sample_weights = class_weights[labels]   # Assign a weight to each sample based on its class

# Create a sampler
#
# sample_weights: 
# This is a sequence where each element corresponds to the weight of a sample in the dataset. 
# These weights are typically calculated based on the inverse frequency of the class labels to 
# address class imbalance.
#
# num_samples=len(sample_weights): 
# This sets the number of samples drawn in each epoch to the total number of samples in the dataset. 
# This ensures that the DataLoader generates a batch of samples with the specified weights in every epoch.
#
# replacement=True: 
# This means that samples are drawn with replacement. In the context of imbalanced datasets, 
# this allows for more frequent sampling of minority class examples, 
# which helps in balancing the class distribution in each batch.
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Create a DataLoader with the sampler
# Create DataLoader with batch size of 10
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

# Example usage of the dataloader
for batch_data, batch_labels in dataloader:
    # Batch of 10 samples, each with 3 channels and size 32x32
    print(batch_data.shape)  # Output: torch.Size([10, 3, 32, 32])
	
	# Corresponding labels for the batch
    print(batch_labels)      # Output: tensor([...])
```


## Applying Synthetic Data Generation for Rare Classes

Generating synthetic data for rare classes can help balance the dataset and improve model performance on underrepresented classes.

**Main Benefits:**
1. **Balancing the Dataset**: Generating synthetic data for rare classes helps to balance the class distribution in the dataset, preventing the model from being biased towards the majority classes.
2. **Improved Model Performance**: By providing more samples for underrepresented classes, the model can learn better and more generalized features for these classes, leading to improved performance and higher accuracy.
3. **Enhanced Robustness**: A balanced dataset ensures that the model is exposed to a variety of scenarios and features, making it more robust and capable of handling diverse inputs.
4. **Reduced Overfitting**: With more synthetic samples for rare classes, the model is less likely to overfit to the limited real samples available, improving its generalization to new data.

**For Examples:**
1. **Select a Minority Sample**: Choose a sample from the minority class.
2. **Find Nearest Neighbors**: Identify the k-nearest neighbors of this sample within the minority class.
3. **Create Synthetic Samples**: For each neighbor, create a synthetic sample by interpolating between the original sample and the neighbor, based on a random weight.
4. **Add to Dataset**: Add the newly generated synthetic samples to the dataset.

This process increases the number of samples for the minority classes, balancing the dataset and helping the model learn better representations for these underrepresented classes.

[SMOTE API](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

```python
from imblearn.over_sampling import SMOTE
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    """
    A custom dataset class for handling data and labels.
    """
    def __init__(self, data, labels):
        """
        Initialize the dataset with data and labels.

        Args:
            data (torch.Tensor): Tensor containing the data samples.
            labels (torch.Tensor): Tensor containing the corresponding labels.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the data and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (data, label) where data is a tensor and label is the corresponding label.
        """
        return self.data[idx], self.labels[idx]

# Example data and labels
data = np.random.randn(100, 20)        # Example data (100 samples, 20 features)
labels = np.random.randint(0, 2, 100)  # Example labels (binary classification, 100 labels)

# NOTE:
# Synthetic Minority Over-sampling Technique (SMOTE)
# SMOTE is a popular technique used to address class imbalance in datasets. 
# It generates synthetic samples for the minority class by interpolating between existing minority class examples. 
# This helps to balance the dataset and improve model performance on underrepresented classes.
smote = SMOTE()

# Fitting and Resampling the Data
# 1. Identify Minority and Majority Classes: 
#    SMOTE first identifies the minority and majority classes in the provided labels.
# 2. Generate Synthetic Samples: 
#    For each sample in the minority class, SMOTE selects one or more of its nearest neighbors from the same class. 
#    It then creates synthetic samples by interpolating between the feature values of the selected sample and its neighbors.
# 3. Combine Original and Synthetic Samples: 
#    The original samples from both classes and the synthetic samples generated for the minority class are 
#    combined to form the resampled dataset.
data_resampled, labels_resampled = smote.fit_resample(data, labels)

# Convert the resampled data to PyTorch tensors
data_resampled = torch.tensor(data_resampled, dtype=torch.float32)   # Resampled data (num_samples, 20)
labels_resampled = torch.tensor(labels_resampled, dtype=torch.long)  # Resampled labels (num_samples,)

# Example usage in a DataLoader
# Create DataLoader with batch size of 10
dataset = CustomDataset(data_resampled, labels_resampled)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)  

for batch_data, batch_labels in dataloader:
    # Batch of 10 samples, each with 20 features
    print(batch_data.shape)  # Output: torch.Size([10, 20])  
        
    # Corresponding labels for the batch
    print(batch_labels)      # Output: tensor([...])         
```

## Using MixUp and CutMix for Enhanced Generalization

MixUp and CutMix are data augmentation techniques that create new training examples by mixing or cutting and pasting parts of images and labels, which can enhance generalization.

**Main Benefits:**
- **Enhanced Generalization**: By generating new training examples, MixUp and CutMix help the model learn more robust features and reduce overfitting, which leads to better performance on unseen data.
- **Improved Class Boundaries**: These techniques create smoother decision boundaries between classes, helping the model generalize better across different classes.
- **Regularization Effect**: Both techniques act as a form of data augmentation, providing a regularization effect that improves the model's ability to generalize.

Both MixUp and CutMix effectively augment the dataset, creating more diverse training samples that help the model learn better representations and improve its generalization ability.
- **MixUp**: It creates new training examples by linearly interpolating between two randomly selected samples and their corresponding labels. Specifically, it forms a new sample by taking a weighted combination of two images and their labels.
- **CutMix**: It creates new training examples by cutting and pasting a patch from one image onto another image, while also mixing their labels proportionally to the area of the patch.

```python
import numpy as np
import torch
import torch.nn.functional as F

# MixUp function
def mixup_data(x, y, alpha=1.0):
    """
    Apply MixUp augmentation to the input data and labels.

    Args:
        x (torch.Tensor): Input data of shape (batch_size, channels, height, width).
        y (torch.Tensor): Input labels of shape (batch_size,).
        alpha (float): Parameter for the beta distribution. Default is 1.0.

    Returns:
        tuple: Mixed inputs, paired labels, and the lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]                         # Get batch size
    index = torch.randperm(batch_size).cuda()        # Generate a random permutation of indices

    mixed_x = lam * x + (1 - lam) * x[index, :]      # Mix the inputs
    y_a, y_b = y, y[index]                           # Pair the labels

    # Return Values:
    # 1. mixed_x (torch.Tensor):
    #    - This is the augmented batch of input data created by mixing pairs of original inputs.
    #    - Each input in the batch is a linear combination of two original inputs, weighted by `lam`.
    # 2. y_a (torch.Tensor):
    #    - This is the original set of labels for the first set of mixed samples.
    #    - These labels correspond to the original inputs `x`.
    # 3. y_b (torch.Tensor):
    #    - This is the original set of labels for the second set of mixed samples.
    #    - These labels correspond to the inputs selected by the random permutation of indices (`x[index, :]`).
    # 4. lam (float):
    #    - This is the lambda value used to mix the inputs and labels.
    #    - It is typically drawn from a Beta distribution with parameter `alpha`.
    #    - `lam` determines the proportion of the mix between the two inputs and their corresponding labels.    
    return mixed_x, y_a, y_b, lam

# CutMix function
def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation to the input data and labels.

    Args:
        x (torch.Tensor): Input data of shape (batch_size, channels, height, width).
        y (torch.Tensor): Input labels of shape (batch_size,).
        alpha (float): Parameter for the beta distribution. Default is 1.0.

    Returns:
        tuple: Mixed inputs, paired labels, and the lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size, _, H, W = x.size()                   # Get batch size, height, and width
    index = torch.randperm(batch_size).cuda()        # Generate a random permutation of indices

    bbx1, bby1, bbx2, bby2 = rand_bbox(W, H, lam)    # Get bounding box coordinates
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]  # Apply CutMix

    y_a, y_b = y, y[index]                           # Pair the labels

    # Return Values:
    # 1. x (torch.Tensor):
    #    - This is the augmented batch of input data created by applying CutMix.
    #    - Each input in the batch has a patch from another input pasted onto it, creating a mixed sample.
    # 2. y_a (torch.Tensor):
    #    - This is the original set of labels for the first set of samples.
    #    - These labels correspond to the unmodified parts of the inputs.
    # 3. y_b (torch.Tensor):
    #    - This is the original set of labels for the second set of samples.
    #    - These labels correspond to the patches that were cut and pasted onto the first set of inputs.
    # 4. lam (float):
    #    - This is the lambda value representing the area ratio of the patch applied.
    #    - It determines the proportion of the mix between the two sets of inputs and their corresponding labels.
    #    - Like in MixUp, `lam` is typically drawn from a Beta distribution with parameter `alpha`.
    return x, y_a, y_b, lam

# Bounding box function for CutMix
def rand_bbox(W, H, lam):
    """
    Generate a random bounding box for CutMix.

    Args:
        W (int): Width of the image.
        H (int): Height of the image.
        lam (float): Lambda value from the beta distribution.

    Returns:
        tuple: Coordinates of the bounding box (x1, y1, x2, y2).
    """
    cut_rat = np.sqrt(1. - lam)                      # Calculate the cut ratio
    cut_w = np.int(W * cut_rat)                      # Calculate the cut width
    cut_h = np.int(H * cut_rat)                      # Calculate the cut height

    cx = np.random.randint(W)                        # Random center x
    cy = np.random.randint(H)                        # Random center y

    bbx1 = np.clip(cx - cut_w // 2, 0, W)            # Calculate x1
    bby1 = np.clip(cy - cut_h // 2, 0, H)            # Calculate y1
    bbx2 = np.clip(cx + cut_w // 2, 0, W)            # Calculate x2
    bby2 = np.clip(cy + cut_h // 2, 0, H)            # Calculate y2

    return bbx1, bby1, bbx2, bby2

# Example usage in training loop
for batch_data, batch_labels in dataloader:
    # Apply MixUp
    inputs, targets_a, targets_b, lam = mixup_data(batch_data, batch_labels)
    
    # Apply CutMix
    # inputs, targets_a, targets_b, lam = cutmix_data(batch_data, batch_labels)
    
    outputs = model(inputs)                                 # Forward pass
    loss = lam * F.cross_entropy(outputs, targets_a) + 
           (1 - lam) * F.cross_entropy(outputs, targets_b)  # Calculate mixed loss
    loss.backward()                                         # Backward pass
    optimizer.step()                                        # Update model parameters
```

## Conclusion

Advanced data handling techniques are vital for building high-performing models. Albumentations provides a rich set of augmentation transformations, enhancing model robustness by creating diverse training data. Stratified sampling ensures balanced class representation, effectively addressing class imbalance. Synthetic data generation, especially for rare classes, balances datasets and improves model performance. MixUp and CutMix offer powerful augmentation methods that enhance generalization and prevent overfitting. Implementing these techniques significantly improves the training process, resulting in models that are more resilient and accurate in real-world applications.