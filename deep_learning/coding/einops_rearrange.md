# Introduction to `rearrange()` in `einops`

## Introduction

In the world of data science and machine learning, manipulating tensors is a common task. Whether you are working with NumPy arrays, PyTorch tensors, or TensorFlow tensors, reshaping and rearranging these structures efficiently is crucial. The `einops` library offers a powerful and intuitive way to perform these operations. In this article, we will introduce the `rearrange()` function from `einops` and demonstrate its versatility with six practical examples.

## What is `rearrange()`?

`rearrange()` is a function in the `einops` library that allows you to change the shape and order of axes of a tensor in a simple and readable way. By using a concise notation, you can specify complex transformations without having to write cumbersome and error-prone code.

## Why Use `rearrange()`?

The `rearrange()` function is beneficial because it:
- Simplifies tensor operations with a clear and readable syntax.
- Reduces the chance of errors in tensor reshaping.
- Enhances code maintainability and readability.
- Supports complex tensor manipulations with ease.

## Example 1: Transposing a Matrix

```python
import numpy as np
from einops import rearrange

# Original matrix of shape (2, 3)
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Transpose the matrix to shape (3, 2)
transposed_matrix = rearrange(matrix, 'h w -> w h')

print(transposed_matrix)
```

Output:
```
[[1 4]
 [2 5]
 [3 6]]
```

## Example 2: Flattening a Tensor

```python
import numpy as np
from einops import rearrange

# Original tensor of shape (2, 3, 4)
tensor = np.arange(24).reshape(2, 3, 4)

# Flatten the tensor to shape (6, 4)
flattened_tensor = rearrange(tensor, 'b c h -> (b c) h')

print(flattened_tensor)
```

Output:
```
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]
 [16 17 18 19]
 [20 21 22 23]]
```

## Example 3: Adding New Axes with Mask

```python
import numpy as np
from einops import rearrange

# Original mask of shape (2, 3, 4)
mask = np.arange(24).reshape(2, 3, 4)

# Add new axes to make it shape (2, 1, 3, 1, 4)
new_axes_mask = rearrange(mask, 'b c h -> b () c () h')

# Print the original mask shape and mask
print("Original mask shape:", mask.shape)
print(mask)

# Print the new mask shape and mask with added axes
print("New mask shape:", new_axes_mask.shape)
print(new_axes_mask)
```

Output:
```
Original mask shape: (2, 3, 4)
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]
 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]

New mask shape: (2, 1, 3, 1, 4)
[[[[[ 0  1  2  3]]
   [[ 4  5  6  7]]
   [[ 8  9 10 11]]]
  [[[12 13 14 15]]
   [[16 17 18 19]]
   [[20 21 22 23]]]]
```

## Example 4: Adding New Axes to a Tensor

```python
import numpy as np
from einops import rearrange

# Original tensor of shape (2, 3)
tensor = np.arange(6).reshape(2, 3)

# Add new axes to make it shape (2, 1, 3, 1)
new_axes_tensor = rearrange(tensor, 'b c -> b 1 c 1')

# Print the original tensor shape and tensor
print("Original tensor shape:", tensor.shape)
print(tensor)

# Print the new tensor shape and tensor with added axes
print("New tensor shape:", new_axes_tensor.shape)
print(new_axes_tensor)
```

Output:
```
Original tensor shape: (2, 3)
[[0 1 2]
 [3 4 5]]

New tensor shape: (2, 1, 3, 1)
[[[[0]
   [1]
   [2]]]
 [[[3]
   [4]
   [5]]]]
```

## Example 5: Splitting and Reshaping Axes

```python
import numpy as np
from einops import rearrange

# Original tensor of shape (2, 9)
tensor = np.arange(18).reshape(2, 9)

# Split the second axis into two axes of sizes 3 and 3
split_tensor = rearrange(tensor, 'a (b c) -> a b c', c=3)

# Print the original tensor shape and tensor
print("Original tensor shape:", tensor.shape)
print(tensor)

# Print the split tensor shape and tensor
print("Split tensor shape:", split_tensor.shape)
print(split_tensor)
```

Output:
```
Original tensor shape: (2, 9)
[[ 0  1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16 17]]

Split tensor shape: (2, 3, 3)
[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]]
 [[ 9 10 11]
  [12 13 14]
  [15 16 17]]]
```

## Example 6: Merging and Reshaping Axes

```python
import numpy as np
from einops import rearrange

# Original tensor of shape (8, 3)
tensor = np.arange(24).reshape(8, 3)

# Split the first axis into two axes of sizes 2 and 4
split_merge_tensor = rearrange(tensor, '(a b) c -> a b c', a=2, b=4)

# Print the original tensor shape and tensor
print("Original tensor shape:", tensor.shape)
print(tensor)

# Print the split and merged tensor shape and tensor
print("Split and merged tensor shape:", split_merge_tensor.shape)
print(split_merge_tensor)
```

Output:
```
Original tensor shape: (8, 3)
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]
 [12 13 14]
 [15 16 17]
 [18 19 20]
 [21 22 23]]

Split and merged tensor shape: (2, 4, 3)
[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]
  [ 9 10 11]]
 [[12 13 14]
  [15 16 17]
  [18 19 20]
  [21 22 23]]]
```

## Conclusion
The `rearrange()` function from `einops` provides a powerful and intuitive way to manipulate tensors. With its concise notation, it simplifies tensor operations, making your code more readable and maintainable. The examples provided here demonstrate just a few of the many ways you can use `rearrange()` to transform your data structures efficiently.


## References
For further reading and a deeper understanding of the `rearrange()` function and other features provided by the `einops` library, you can refer to the following resources:

1. **API Documentation for `rearrange()`**  
   The official API documentation provides detailed information about the `rearrange()` function, including its syntax, parameters, and usage examples. It is a valuable resource for understanding the full capabilities of this function.  
   [API Documentation for `rearrange()`](https://einops.rocks/api/rearrange/)

2. **Einops Basics**  
   This guide introduces the basics of `einops`, explaining the core concepts and functionalities. It covers various tensor operations, providing examples and explanations that are easy to follow for beginners.  
   [Einops Basics](https://einops.rocks/1-einops-basics/)

3. **Einops for Deep Learning**  
   This article explores how `einops` can be leveraged in deep learning applications. It demonstrates how to integrate `einops` with popular deep learning frameworks like PyTorch and TensorFlow, showcasing practical examples and use cases.  
   [Einops for Deep Learning](https://einops.rocks/2-einops-for-deep-learning/)
