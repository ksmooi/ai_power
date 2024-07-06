# SFT: Advanced Training Strategies for Deep Learning Models

## Introduction

In the rapidly evolving field of deep learning, optimizing model performance while maintaining computational efficiency is a key challenge. Advanced training strategies have emerged as powerful tools to enhance the capabilities of deep learning models, particularly in the context of transfer learning and fine-tuning pre-trained networks. Techniques such as layer-wise adaptive learning rate adjustment, cyclical learning rates, mixed precision training, freezing batch normalization layers, and progressive layer unfreezing offer targeted approaches to improve training stability, efficiency, and overall model performance. This article delves into these advanced strategies, providing insights into their benefits and practical implementation methods. By leveraging these techniques, practitioners can achieve superior results in various deep learning applications.


## Layer-wise Adaptive Learning Rate Adjustment

Layer-wise adaptive learning rate adjustment involves setting different learning rates for different layers of a neural network. This can help in fine-tuning specific layers while keeping others stable. This technique allows for fine-tuning certain layers more aggressively while keeping others stable, which is particularly useful when dealing with pre-trained models. By doing so, it helps optimize the learning process and achieve better performance, especially when some layers require more significant updates than others.

**Main Benefits:**
1. **Optimized Fine-Tuning**:
    - **Targeted Learning Rates**: Different layers in a neural network, especially in pre-trained models, may require different levels of fine-tuning. Layer-wise adaptive learning rate adjustment allows setting higher learning rates for layers that need more significant updates, and lower learning rates for layers that only need minor adjustments. This targeted approach ensures that each layer is updated optimally.
2. **Improved Performance**:
    - **Enhanced Accuracy**: By fine-tuning specific layers more aggressively, the model can achieve better performance. Layers that have learned generic features in the pre-trained model can be fine-tuned more precisely to adapt to the new task, improving the overall accuracy of the model.
    - **Stable Training**: Lower learning rates for already well-tuned layers prevent disruptive updates, maintaining the stability of the model. This balance helps in achieving a stable and efficient training process.
3. **Efficient Use of Pre-trained Models**:
    - **Preserving Pre-trained Knowledge**: Pre-trained models contain valuable knowledge learned from large datasets. Layer-wise adaptive learning rate adjustment helps preserve this knowledge by avoiding unnecessary large updates to layers that already have useful features, ensuring that the model retains its pre-trained strengths while adapting to new tasks.
    - **Faster Convergence**: By fine-tuning only the necessary layers, the model can converge faster than if all layers were updated with a uniform learning rate. This efficiency reduces the overall training time.
4. **Flexibility and Control**:
    - **Customizable Learning Rates**: This technique provides the flexibility to set custom learning rates for each layer or group of layers. It allows for precise control over the training process, enabling the adjustment of learning rates based on the specific needs of the task and the characteristics of the layers.
    - **Layer-specific Adjustments**: Different layers may require different levels of fine-tuning based on their depth and the type of features they have learned. Layer-wise learning rate adjustment accommodates these differences, ensuring that each layer is tuned appropriately.


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset

# Define a model class for ResNet50
class SimpleCNN(nn.Module):
    """
    A ResNet50 model with a custom final layer to match the number of classes and support for cyclical learning rates.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
                  
        # Load the pre-trained ResNet50 model
        # Modify the final layer
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.model(x)

# Instantiate the model
num_classes = 10
model = SimpleCNN(num_classes=num_classes).cuda()

# Create cross entropy loss function
loss_fn = nn.CrossEntropyLoss()

# Define layer-wise learning rates
layer_learning_rates = [
    {'params': model.layer1.parameters(), 'lr': 0.001},
    {'params': model.layer2.parameters(), 'lr': 0.005},
    {'params': model.layer3.parameters(), 'lr': 0.01},
    {'params': model.layer4.parameters(), 'lr': 0.05},
    {'params': model.fc.parameters(), 'lr': 0.1},
]

# Define the optimizer with layer-wise learning rates
optimizer = optim.AdamW(layer_learning_rates)

# Generate random data for demonstration purposes
input_data = torch.randn(100, 3, 224, 224)  # 100 samples of 3x224x224 images
target_data = torch.randint(0, num_classes, (100,)) # 100 labels in range 0-9

# Create a DataLoader with batch size of 5
dataset = TensorDataset(input_data, target_data)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Example training step with batch size of 5
for data, target in dataloader:      # Iterate over batches of data and target from the dataloader
    optimizer.zero_grad()            # Zero the parameter gradients to prepare for the next backward pass
    output = model(data)             # Forward pass: compute model output for the current batch of data
    loss = loss_fn(output, target)   # Compute loss (Cross Entropy Loss) between model output and target labels
    loss.backward()                  # Backward pass: compute the gradients of the loss with respect to model parameters
    optimizer.step()                 # Update model parameters using the computed gradients

# Example input tensor for a forward pass
input_tensor = torch.randn(5, 3, 224, 224)  # Batch of 5 images, each 3x224x224
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Output: torch.Size([5, 10])
```


## Learning Rate Decay Adjustment with ExponentialLR
The ExponentialLR scheduler is best suited for scenarios where a smooth and continuous decay of the learning rate is desired throughout the entire training process. A continuous decay can help the model converge more steadily, avoiding abrupt changes that could destabilize the training. This is particularly useful in training large, complex models where small, consistent adjustments to the learning rate are beneficial.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

# Instantiate the model
num_classes = 10
model = SimpleCNN(num_classes=num_classes).cuda()

# Create MSE loss function
loss_fn = nn.MSELoss()

# Define an optimizer (Stochastic Gradient Descent) with an initial learning rate
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Define the ExponentialLR scheduler with gamma=0.9
# This means the learning rate will be multiplied by 0.9 every epoch
scheduler = ExponentialLR(optimizer, gamma=0.9)

# Training loop
for epoch in range(10):
    # Dummy input and target for illustration purposes
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)
    
    optimizer.zero_grad()               # Zero the parameter gradients
    outputs = model(inputs)             # Forward pass
    loss = loss_fn(outputs, targets)    # Compute loss (mean squared error)
    loss.backward()                     # Computes the gradients of the loss with respect to each parameter in the model
    optimizer.step()                    # Uses these gradients to update the parameters according to the optimization algorithm
    scheduler.step()                    # Step the scheduler to update the learning rate
    
    # Print the learning rate for each epoch
    print(f"Epoch {epoch+1}: learning rate is {scheduler.get_last_lr()[0]}")
```

In this example:
1. A simple neural network model with a single fully connected layer is defined.
2. An `SGD` optimizer is instantiated with an initial learning rate of 0.1.
3. The `ExponentialLR` scheduler is defined with a `gamma` value of 0.9, meaning the learning rate will be reduced by 10% every epoch.
4. The training loop runs for 10 epochs. In each epoch:
   - Dummy inputs and targets are created for illustration.
   - The model's gradients are zeroed.
   - A forward pass is performed, and the loss is computed.
   - A backward pass is performed, and the optimizer updates the model parameters.
   - The scheduler updates the learning rate.
   - The current learning rate is printed.


## Stepwise Learning Rate Adjustment with MultiStepLR
The MultiStepLR scheduler is ideal for scenarios where you expect to hit certain key milestones during training and want to reduce the learning rate abruptly at those points. Abrupt changes in the learning rate can help the model jump out of local minima and converge more effectively, especially after significant phases of training. This is useful when certain epochs are known to mark a shift in the training dynamics.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

# Instantiate the model
num_classes = 10
model = SimpleCNN(num_classes=num_classes).cuda()

# Create MSE loss function
loss_fn = nn.MSELoss()

# Define an optimizer (Stochastic Gradient Descent) with an initial learning rate
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Define the MultiStepLR scheduler with milestones and gamma
# Milestones are [30, 60, 90], meaning the learning rate will be adjusted at these epochs
# Gamma is 0.1, meaning the learning rate will be multiplied by 0.1 at each milestone
scheduler = MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)

# Training loop
for epoch in range(10):
    # Dummy input and target for illustration purposes
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)
    
    optimizer.zero_grad()               # Zero the parameter gradients
    outputs = model(inputs)             # Forward pass
    loss = loss_fn(outputs, targets)    # Compute loss (mean squared error)
    loss.backward()                     # Computes the gradients of the loss with respect to each parameter in the model
    optimizer.step()                    # Uses these gradients to update the parameters according to the optimization algorithm
    scheduler.step()                    # Step the scheduler to update the learning rate if the current epoch is a milestone
    
    # Print the learning rate for each epoch
    print(f"Epoch {epoch+1}: learning rate is {scheduler.get_last_lr()[0]}")
```

In this example:
1. A simple neural network model with a single fully connected layer is defined.
2. An `SGD` optimizer is instantiated with an initial learning rate of 0.1.
3. The `MultiStepLR` scheduler is defined with milestones `[3, 6, 9]` and a `gamma` value of 0.1. This means the learning rate will be reduced by 90% at epochs 3, 6, and 9.
4. The training loop runs for 10 epochs. In each epoch:
   - Dummy inputs and targets are created for illustration.
   - The model's gradients are zeroed.
   - A forward pass is performed, and the loss is computed.
   - A backward pass is performed, and the optimizer updates the model parameters.
   - The scheduler updates the learning rate if the current epoch is one of the milestones.
   - The current learning rate is printed.


## Implementing Cyclical Learning Rates

Cyclical Learning Rates (CLR) vary the learning rate between a minimum and maximum value cyclically. This can help in escaping local minima and potentially achieve better performance. By periodically increasing and decreasing the learning rate, CLR helps the optimizer escape local minima and potentially find better overall solutions. This dynamic adjustment can improve model performance and training efficiency by encouraging exploration of the loss landscape.

**Main Benefits:**
1. **Escaping Local Minima**:
    - **Dynamic Learning Rate Adjustment**: CLR varies the learning rate cyclically between a minimum and maximum value. This periodic adjustment helps the optimizer escape local minima by allowing the learning rate to increase periodically, giving the optimizer the momentum to jump out of local minima and potentially find better, more optimal solutions.
2. **Faster Convergence**:
    - **Efficient Training**: CLR can lead to faster convergence by maintaining an optimal balance between exploration and exploitation. The periodic increase in the learning rate allows the optimizer to make more significant updates when necessary, speeding up the training process.
    - **Adaptive Learning**: As the learning rate decreases cyclically, the optimizer can make finer adjustments, ensuring that the model parameters converge efficiently towards the optimal solution.
3. **Simplified Hyperparameter Tuning**:
    - **Reduced Need for Manual Tuning**: With CLR, the need for manually tuning the learning rate is reduced. Instead of searching for a single best learning rate, CLR allows the learning rate to vary within a specified range, simplifying the process and saving time.
    - **Automatic Adjustment**: The cyclic nature of CLR means that the learning rate is automatically adjusted during training, providing a self-regulating mechanism that adapts to the training dynamics.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset

# Instantiate the model
num_classes = 10
model = SimpleCNN(num_classes=num_classes).cuda()

# Define the optimizer with AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# Define the CLR scheduler
scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=2000, mode='triangular')

# Generate random data for demonstration purposes
input_data = torch.randn(100, 3, 224, 224)  # 100 samples of 3x224x224 images
target_data = torch.randint(0, num_classes, (100,))  # 100 labels in range 0-9

# Create a DataLoader with batch size of 5
dataset = TensorDataset(input_data, target_data)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Example training loop with CLR
epochs = 10
for epoch in range(epochs):
    for data, target in dataloader:
        optimizer.zero_grad()                        # Clear gradients
        output = model(data)                         # Forward pass
        loss = nn.CrossEntropyLoss()(output, target) # Compute loss
        loss.backward()                              # Backward pass
        optimizer.step()                             # Update parameters
        scheduler.step()                             # Update the learning rate

# Example input tensor for a forward pass
input_tensor = torch.randn(5, 3, 224, 224)  # Batch of 5 images, each 3x224x224
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Output: torch.Size([5, 10])
```


## Progressive Layer Unfreezing

Progressive layer unfreezing is a training technique used to fine-tune pre-trained neural network models. Initially, most or all layers of the model are frozen, meaning their weights are not updated during training. Only a subset of layers, typically the final layers, are unfrozen and trained. As training progresses, additional layers are gradually unfrozen, allowing their weights to be updated. This approach helps in stabilizing the training process, preventing large updates to the earlier layers' weights, which could disrupt the pre-trained features. It allows the model to learn new tasks while preserving the useful features learned during pre-training. This technique is particularly effective when fine-tuning large models on new datasets with limited data.

**Main Benefits:**
1. **Stabilized Training Process**:
    - **Controlled Updates**: By initially freezing most layers and only gradually unfreezing them, progressive layer unfreezing ensures that the model does not undergo large, disruptive updates. This controlled approach helps maintain the stability of the pre-trained features.
    - **Smooth Transition**: The gradual unfreezing allows the model to smoothly transition from leveraging pre-trained features to adapting to the new task. This helps in avoiding sudden shifts in the learned representations that could destabilize training.
2. **Preservation of Pre-trained Features**:
    - **Retaining Useful Features**: Initially freezing the earlier layers ensures that the useful features learned during pre-training are preserved. These features often capture generic patterns that are beneficial across various tasks.
    - **Focused Fine-Tuning**: By focusing the initial training on the final layers, the model can quickly adapt to the new task without losing the valuable features captured in the earlier layers.
3. **Enhanced Model Performance**:
    - **Optimized Adaptation**: The progressive unfreezing allows the model to adapt more effectively to the new task by fine-tuning each layer in a controlled manner. This often leads to better overall performance compared to fine-tuning all layers simultaneously.
    - **Layer-wise Learning**: Each layer can be fine-tuned with the appropriate learning rate and training duration, ensuring that the adaptation process is optimized for each part of the network.
4. **Reduced Risk of Overfitting**:
    - **Gradual Learning**: The gradual approach reduces the risk of overfitting to the new dataset, as it prevents the model from making drastic changes too quickly. This is especially important when dealing with small or noisy datasets.
    - **Conservation of Pre-trained Knowledge**: By preserving the pre-trained knowledge in the earlier layers, the model maintains a strong foundation, reducing the likelihood of overfitting to the specific characteristics of the new data.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset

# Function to progressively unfreeze layers
def unfreeze_layers(model, layers):
    """
    Unfreezes specified layers of the model by setting their parameters to require gradients.

    Args:
        model (nn.Module): The neural network model.
        layers (list): List of layers to unfreeze.
    """
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = True


# Instantiate the model
num_classes = 10
model = SimpleCNN(num_classes=num_classes).cuda()

# Initially freeze all layers
for param in model.parameters():
    param.requires_grad = False
	
# Unfreeze specific layers progressively
# Unfreeze the last residual block and fully connected layer
unfreeze_layers(model, [model.model.layer4, model.model.fc])

# Define the optimizer with unfrozen layers
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

# Generate random data for demonstration purposes
input_data = torch.randn(100, 3, 224, 224)           # 100 samples of 3x224x224 images
target_data = torch.randint(0, num_classes, (100,))  # 100 labels in range 0-9

# Create a DataLoader with batch size of 5
dataset = TensorDataset(input_data, target_data)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Example training step
for data, target in dataloader:
    optimizer.zero_grad()                         # Clear gradients
    output = model(data)                          # Forward pass
    loss = nn.CrossEntropyLoss()(output, target)  # Compute loss
    loss.backward()                               # Backward pass
    optimizer.step()                              # Update parameters

# Example input tensor for a forward pass
input_tensor = torch.randn(5, 3, 224, 224).cuda() # Batch of 5 images, each 3x224x224, on GPU
output_tensor = model(input_tensor)
print(output_tensor.shape)                        # Output: torch.Size([5, 10])
```


## Utilizing Mixed Precision Training

Mixed precision training involves using both 16-bit and 32-bit floating-point types, which can significantly speed up training and reduce memory usage. This approach significantly speeds up training and reduces memory usage without sacrificing model accuracy. By performing computations in 16-bit precision and maintaining important variables in 32-bit precision, it leverages the faster processing capabilities of modern GPUs while ensuring numerical stability and precision in critical operations.

**Main Benefits:**
1. **Speed Up Training**:
    - **Faster Computations**: Mixed precision training leverages the faster computational capabilities of modern GPUs when using 16-bit (half-precision) floating-point operations. This results in significantly faster training times compared to using only 32-bit (single-precision) floating-point operations.
    - **Increased Throughput**: With the reduced precision of 16-bit operations, more computations can be processed simultaneously, increasing the overall throughput and efficiency of the training process.
2. **Reduced Memory Usage**:
    - **Lower Memory Footprint**: Using 16-bit precision for most of the operations significantly reduces the amount of memory required to store model parameters and intermediate activations. This allows for training larger models or using larger batch sizes within the same hardware constraints.
    - **Enhanced GPU Utilization**: The reduced memory usage enables better utilization of GPU resources, allowing more of the GPU's memory to be dedicated to active computations rather than data storage.
3. **Maintained Model Accuracy**:
    - **Preserved Precision for Critical Operations**: Mixed precision training maintains the accuracy of the model by keeping important variables, such as loss and certain key gradients, in 32-bit precision. This ensures that the numerical stability and precision of critical operations are not compromised.
    - **Dynamic Scaling**: Techniques like dynamic loss scaling are used to prevent underflow and overflow issues that can arise with 16-bit precision. This helps maintain the stability and accuracy of the training process.
4. **Improved Energy Efficiency**:
    - **Lower Power Consumption**: The reduced computational and memory requirements of 16-bit operations lead to lower power consumption, making mixed precision training more energy-efficient. This is particularly beneficial for large-scale training environments and data centers.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

# Instantiate the model
num_classes = 10
model = SimpleCNN(num_classes=num_classes).cuda()

# Define the optimizer with AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# Create a GradScaler for mixed precision training
scaler = GradScaler()

# Generate random data for demonstration purposes
input_data = torch.randn(100, 3, 224, 224)          # 100 samples of 3x224x224 images
target_data = torch.randint(0, num_classes, (100,)) # 100 labels in range 0-9

# Create a DataLoader with batch size of 5
dataset = TensorDataset(input_data, target_data)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Example training loop with mixed precision
epochs = 10
for epoch in range(epochs):
    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()  # Move data to GPU

        optimizer.zero_grad()     # Clear gradients
        with autocast():          # Enable mixed precision
            output = model(data)  # Forward pass
            loss = nn.CrossEntropyLoss()(output, target)  # Compute loss
        
        # Scaling the Loss: 
        # In this step, the GradScaler scales the loss by a certain factor. 
        # This scaled loss is then used in the backward pass to compute the gradients. 
        # By scaling the loss, the gradients are also scaled, making them larger and less likely to underflow. 
        # The backward() function computes the gradients of the loss with respect to the model parameters.
        
        # Updating the Model Parameters: 
        # After computing the scaled gradients, the scaler.step(optimizer) function is called. 
        # This function first unscales the gradients back to their original scale and then passes them to the optimizer 
        # to update the model parameters. If any of the gradients are found to be NaN or Inf after unscaling, 
        # the optimizer step is skipped to maintain stability.
        
        # Updating the Scaler: 
        # Finally, the scaler.update() function updates the scale factor used by the GradScaler. 
        # If gradients were too small and caused underflow, the scaler will increase the scale factor for the next 
        # iteration to avoid this issue. Conversely, if no such issues were encountered, the scaler might decrease 
        # the scale factor to improve training efficiency. This dynamic adjustment helps maintain the balance between 
        # training speed and numerical stability.
        
        scaler.scale(loss).backward() # Scaling the Loss
        scaler.step(optimizer)        # Updating the Model Parameters
        scaler.update()               # Updating the Scaler

# Example input tensor for a forward pass
input_tensor = torch.randn(5, 3, 224, 224).cuda()  # Batch of 5 images, each 3x224x224, on GPU
output_tensor = model(input_tensor)
print(output_tensor.shape)                         # Output: torch.Size([5, 10])
```

## Freezing Batch Normalization Layers While Training

Freezing Batch Normalization (BN) layers can be useful during fine-tuning to prevent the statistics of pre-trained BN layers from being updated, which might degrade the performance. This is useful because the pre-trained BN layers' statistics are well-tuned to the original training data. Updating these statistics on a new, smaller dataset can degrade performance by making the model less stable. Freezing BN layers helps maintain the benefits of pre-training, ensuring consistent normalization and better transfer learning performance.

**Main Benefits:**
1. **Maintaining Stable Performance**:
    - **Preservation of Pre-trained Statistics**: Freezing BN layers ensures that the running statistics (mean and variance) collected during the pre-training phase are preserved. These statistics are well-tuned to the original training data and contribute to the model's stability.
    - **Avoiding Degradation**: When fine-tuning on a new, typically smaller dataset, updating these BN statistics can lead to instability and degrade the model's performance. Freezing BN layers prevents this degradation, maintaining the benefits of the pre-trained model.
2. **Improved Transfer Learning**:
    - **Consistent Normalization**: By keeping the BN statistics unchanged, the model can consistently normalize the inputs, leading to better performance on the new task. This consistency helps in transferring the learned features more effectively.
    - **Enhanced Generalization**: Preserving the pre-trained statistics helps the model generalize better to new data, as the normalization process remains aligned with the broader, original dataset.
3. **Reduced Risk of Overfitting**:
    - **Stability in Smaller Datasets**: Fine-tuning on smaller datasets can lead to overfitting, especially if BN layers start updating their statistics based on limited data. Freezing BN layers reduces this risk, as the normalization process remains governed by the robust statistics from the larger pre-training dataset.
    - **Focus on Fine-Tuning**: Freezing BN layers allows the model to focus on fine-tuning the weights of other layers without the added variability from changing normalization parameters, leading to a more stable and controlled fine-tuning process.
4. **Simplified Training Process**:
    - **Less Computational Overhead**: Freezing BN layers simplifies the training process by reducing the number of parameters that need to be updated. This can lead to faster convergence and reduced computational overhead during fine-tuning.
    - **Ease of Implementation**: Implementing BN layer freezing is straightforward and can be easily integrated into existing training pipelines, providing a quick and effective way to enhance model performance during fine-tuning.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset

# Function to freeze batch normalization layers
def set_bn_eval(m):
    """
    Sets batch normalization layers to evaluation mode to freeze their statistics.

    Args:
        m (torch.nn.Module): A module in the neural network.
    """
    if isinstance(m, nn.BatchNorm2d):
        m.eval()


# Instantiate the model
num_classes = 10
model = SimpleCNN(num_classes=num_classes).cuda()

# Apply the freeze function to batch normalization layers
model.apply(set_bn_eval)

# Define the optimizer with AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# Generate random data for demonstration purposes
input_data = torch.randn(100, 3, 224, 224)          # 100 samples of 3x224x224 images
target_data = torch.randint(0, num_classes, (100,)) # 100 labels in range 0-9

# Create a DataLoader with batch size of 5
dataset = TensorDataset(input_data, target_data)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Example training step
for data, target in dataloader:
    optimizer.zero_grad()                         # Clear gradients
    output = model(data)                          # Forward pass
    loss = nn.CrossEntropyLoss()(output, target)  # Compute loss
    loss.backward()                               # Backward pass
    optimizer.step()                              # Update parameters

# Example input tensor for a forward pass
input_tensor = torch.randn(5, 3, 224, 224).cuda() # Batch of 5 images, each 3x224x224, on GPU
output_tensor = model(input_tensor)
print(output_tensor.shape)                        # Output: torch.Size([5, 10])
```


## Conclusion

Advanced training strategies are crucial for the successful deployment of deep learning models, especially when fine-tuning pre-trained networks for new tasks. Each technique—layer-wise adaptive learning rate adjustment, cyclical learning rates, mixed precision training, freezing batch normalization layers, and progressive layer unfreezing—offers unique advantages that enhance training efficiency, stability, and accuracy.

By selectively applying these methods, practitioners can optimize their models for better performance on specific tasks, reduce training times, and manage computational resources more effectively. Embracing these advanced strategies enables the creation of robust, high-performance models that can adapt to diverse challenges in the ever-growing landscape of artificial intelligence.

