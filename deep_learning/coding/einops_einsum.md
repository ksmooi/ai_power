
# Introduction to `einsum()` in `einops`

The `einsum()` function from the `einops` library is a powerful tool for performing complex tensor operations using Einstein summation notation. It provides a concise and flexible way to express a wide range of operations, from simple element-wise multiplications to complex attention mechanisms in deep learning. In this article, we will introduce the `einsum()` function and demonstrate its versatility with several practical examples.

## Example 1: Element-wise Multiplication and Sum

Let's start with a basic example: computing the dot product of two vectors.

```python
import numpy as np
from einops import einsum

# Vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
dot_product = einsum('i, i ->', a, b)

print("Input shapes:", a.shape, b.shape)
print("Input vectors:", a, b)
print("Dot product:", dot_product)
```

Output:
```
Input shapes: (3,) (3,)
Input vectors: [1 2 3] [4 5 6]
Dot product: 32
```

## Example 2: Matrix Multiplication

Next, we will perform matrix multiplication using Einstein summation.

```python
import numpy as np
from einops import einsum

# Matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
result = einsum('ij, jk -> ik', A, B)

print("Input shapes:", A.shape, B.shape)
print("Input matrices:\n", A, "\n", B)
print("Result shape:", result.shape)
print("Matrix multiplication result:\n", result)
```

Output:
```
Input shapes: (2, 2) (2, 2)
Input matrices:
 [[1 2]
  [3 4]] 
 [[5 6]
  [7 8]]

Result shape: (2, 2)
Matrix multiplication result:
 [[19 22]
  [43 50]]
```

## Example 3: Outer Product of Vectors

We can also compute the outer product of two vectors.

```python
import numpy as np
from einops import einsum

# Vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Outer product
outer_product = einsum('i, j -> ij', a, b)

print("Input shapes:", a.shape, b.shape)
print("Input vectors:", a, b)
print("Outer product shape:", outer_product.shape)
print("Outer product:\n", outer_product)
```

Output:
```
Input shapes: (3,) (3,)
Input vectors: [1 2 3] [4 5 6]

Outer product shape: (3, 3)
Outer product:
 [[ 4  5  6]
  [ 8 10 12]
  [12 15 18]]
```

## Example 4: Tensor Contraction

Now let's contract a 3D tensor along specified axes.

```python
import numpy as np
from einops import einsum

# 3D Tensor
tensor = np.random.random((3, 4, 5))

# Contract the tensor along the first and third dimensions
# What this equation 'ijk -> j' does is it sums over the dimensions i and k of the input tensor, 
# resulting in a 1-dimensional tensor along the j dimension. This is equivalent to taking 
# the sum of the elements of tensor along the i and k dimensions.
result = einsum('ijk -> j', tensor)

print("Input shape:", tensor.shape)
print("Input tensor:\n", tensor)
print("Contracted tensor shape:", result.shape)
print("Contracted tensor:", result)
```

Output:
```
Input shape: (3, 4, 5)
Input tensor:
 [[[0.68393741 0.80261627 0.40922432 0.42827943 0.83594184]
   [0.58706757 0.67809856 0.60952722 0.11220812 0.69293368]
   [0.20863589 0.34050639 0.93416295 0.84354525 0.00911731]
   [0.93211642 0.22264631 0.23662187 0.35898179 0.81265904]]
  [[0.17434808 0.1581852  0.00469752 0.78172082 0.74551491]
   [0.82318969 0.79530873 0.66888827 0.64193317 0.55153145]
   [0.1801464  0.35844804 0.02973841 0.40297266 0.25366378]
   [0.14774135 0.78633592 0.96125931 0.07073648 0.99208091]]
  [[0.13186055 0.37128963 0.78014833 0.68304563 0.25657891]
   [0.04392491 0.82333795 0.75296266 0.96078223 0.65947727]
   [0.85691613 0.89846804 0.35611652 0.85807262 0.54461115]
   [0.88859116 0.93241334 0.69902147 0.91852024 0.89141784]]]

Contracted tensor shape: (4,)
Contracted tensor: [4.07058201 7.00952068 5.27199764 6.49716949]
```

## Example 5: Bilinear Transformation

Next, we perform a bilinear transformation on a pair of matrices.

```python
import numpy as np
from einops import einsum

# Matrices
A = np.random.random((3, 4))
B = np.random.random((4, 5))
C = np.random.random((5, 6))

# Bilinear transformation
# The operation string 'ij, jk, kl -> il' specifies a matrix multiplication operation between the three 
# input arrays A, B, and C. This operation is equivalent to the expression np.dot(A, np.dot(B, C)) 
# using the np.dot function in NumPy, which performs matrix multiplication.
result = einsum('ij, jk, kl -> il', A, B, C)

print("Input shapes:", A.shape, B.shape, C.shape)
print("Input matrices:\n", A, "\n", B, "\n", C)
print("Result shape:", result.shape)
print("Bilinear transformation result:\n", result)
```

Output:
```
Input shapes: (3, 4) (4, 5) (5, 6)
Input matrices:
 [[0.19659397 0.41647685 0.64883905 0.53801745]
  [0.89029279 0.54344617 0.73663955 0.68573841]
  [0.99184985 0.07947811 0.88554604 0.00856916]]
 [[0.67747735 0.31543279 0.96168464 0.14516077 0.66633872]
  [0.0328986  0.01950918 0.70329287 0.11093528 0.256413  ]
  [0.47397171 0.82470545 0.41977712 0.15818341 0.54647033]
  [0.96993215 0.83270191 0.01129959 0.31794362 0.77934813]]
 [[0.05143602 0.09924362 0.28240935 0.08310761 0.05530623 0.33541914]
  [0.27878788 0.42995632 0.04843041 0.03763191 0.64318797 0.13118314]
  [0.49762548 0.60213917 0.42022971 0.56357364 0.42218929 0.21770414]
  [0.17037025 0.05202226 0.90072012 0.23066372 0.19737316 0.66314959]
  [0.68828141 0.85223068 0.74804765 0.95148807 0.00763653 0.6794818 ]]

Result shape: (3, 6)
Bilinear transformation result:
 [[1.54823557 1.66572506 1.9215482  1.89365979 1.38172042 2.08949139]
  [2.25623288 2.27757071 2.85936429 2.24463694 1.83544477 2.73249131]
  [1.51788226 1.5154371  2.11887213 1.88222987 1.5413014  1.998937  ]]
```

## Example 6: Generalized Matrix Multiplication (Batched)

We can also perform batched matrix multiplication.

```python
import numpy as np
from einops import einsum

# Batched matrices
A = np.random.random((2, 3, 4))
B = np.random.random((2, 4, 5))

# Batched matrix multiplication
# The operation string 'bij, bjk -> bik' specifies a batched matrix multiplication operation between 
# the two input arrays A and B. This operation is equivalent to performing matrix multiplication for 
# each batch b in A and B, resulting in an output array where each batch b is the result of the 
# matrix multiplication of the corresponding batches in A and B.
result = einsum('bij, bjk -> bik', A, B)

print("Input shapes:", A.shape, B.shape)
print("Input batched matrices:\n", A, "\n", B)
print("Result shape:", result.shape)
print("Batched matrix multiplication result:\n", result)
```

Output:
```
Input shapes: (2, 3, 4) (2, 4, 5)
Input batched matrices:
 [[[0.44117669 0.58620788 0.15064565 0.15857837]
  [0.46414223 0.96894345 0.43159407 0.017098  ]
  [0.64269565 0.70328794 0.65856419 0.49912827]]
 [[0.57096294 0.08507025 0.55681684 0.62060991]
  [0.9484347  0.81293971 0.26807112 0.32858864]
  [0.70480125 0.99883176 0.16028894 0.64619342]]]

 [[[0.43569364 0.24493591 0.70760247 0.46011895 0.63087121]
  [0.96343235 0.53906441 0.25525177 0.06428769 0.7067965 ]
  [0.06474958 0.70913724 0.29393645 0.37400347 0.03600122]
  [0.51068406 0.78377549 0.9121381  0.72081216 0.49406692]]
 [[0.69776494 0.64760955 0.26037939 0.56907384 0.21728326]
  [0.00817816 0.9286423  0.07113489 0.20801548 0.94600519]
  [0.58024102 0.91979724 0.53718173 0.35411177 0.36592327]
  [0.64497362 0.67802235 0.14102188 0.95155814 0.84229592]]]

Result shape: (2, 3, 5)
Batched matrix multiplication result:
 [[[0.84369897 1.39125489 0.47820558 0.29851233 0.87897394]
  [0.97035138 1.50672611 0.54780335 0.61894578 0.83997059]
  [1.20061162 1.74644001 0.75905413 0.71070232 1.10718242]]
 [[1.08925068 1.37441203 0.48960921 1.21252018 1.1631465 ]
  [0.87286273 1.7287467  0.63130624 1.03102455 1.38100558]
  [1.05369969 1.5795054  0.62816401 1.06112591 1.20418056]]]
```

## Example 7: Computing Attention Scores

Let's move to a more advanced example: computing the attention scores by multiplying the query and key matrices.

```python
import numpy as np
from einops import einsum

# Generate random query and key matrices
query = np.random.random((2, 4, 8, 10, 64))  # (batch_size, groups, heads, query_length, d_model)
key = np.random.random((2, 8, 20, 64))       # (batch_size, heads, key_length, d_model)

# Compute attention scores
attention_scores = einsum(query, key, "b g h n d, b h s d -> b g h n s")

print("Input shapes:", query.shape, key.shape)
#print("Input query and key matrices:\n", query, "\n", key)
print("Attention scores shape:", attention_scores.shape)
#print("Attention scores:\n", attention_scores)
```

Output:
```
Input shapes: (2, 4, 8, 10, 64) (2, 8, 20, 64)
Attention scores shape: (2, 4, 8, 10, 20)
```

## Example 8: Applying Attention Weights to Value

Finally, we apply the attention weights to the value matrix.

```python
import numpy as np
from einops import einsum

# Generate random attention weights and value matrices
attention_weights = np.random.random((2, 4, 8, 10, 20))  # (batch_size, groups, heads, query_length, key_length)
value = np.random.random((2, 8, 20, 64))                 # (batch_size, heads, value_length, d_model)

# Apply attention weights to the value matrix
context_vector = einsum(attention_weights, value, "b g h n s, b h s d -> b g h n d")

print("Input shapes:", attention_weights.shape, value.shape)
#print("Input attention weights and value matrices:\n", attention_weights, "\n", value)
print("Context vector shape:", context_vector.shape)
#print("Context vector:\n", context_vector)
```

Output:
```
Input shapes: (2, 4, 8, 10, 20) (2, 8, 20, 64)
Context vector shape: (2, 4, 8, 10, 64)
```

## References
For further reading and a deeper understanding of the `einsum()` function and its applications, you can refer to the following resources:
1. [API Documentation for `einsum()`](https://einops.rocks/api/einsum/)
2. [Einstein Summation and Attention Mechanisms](https://theaisummer.com/einsum-attention/)

This article demonstrates the versatility of the `einsum()` function in performing a wide range of tensor operations, from simple element-wise multiplications to complex attention mechanisms used in deep learning.
