# Micrograd Engine

## Overview

This is a minimalistic automatic differentiation (autograd) engine implemented in C. It's inspired by [Karpathy's micrograd in python](https://github.com/karpathy/micrograd). The engine supports basic scalar operations and their gradients, allowing for the construction and differentiation of computational graphs.

## Features

- Scalar-valued computational graph construction
- Automatic differentiation (reverse mode)
- Support for basic operations: addition, subtraction, multiplication, division, power, and ReLU activation
- Gradient clipping to prevent exploding gradients
- Topological sorting for correct gradient computation order

## Key Components

1. `Value`: The core struct representing a node in the computational graph.
2. `reverse`: Function to perform backpropagation through the graph.
3. Operators: Functions like `add`, `sub`, `mul`, `divide`, `pwr`, and `relu`.
4. Gradient computation: Separate functions for computing gradients of each operation.


## Building and Running

Compile the engine with your C compiler. For example, using gcc:

```
gcc -o micrograd engine.c -lm
```

Then run the executable:

```
./micrograd
```

## Extending the Engine

To add new operations:
1. Implement the forward pass function (e.g., `new_op`)
2. Implement the corresponding backward pass function (e.g., `new_op_reverse`)
3. Update the `reverse` function if necessary

## Limitations

- Only supports scalar operations
- Limited to a maximum graph size defined by `MAX_DAG_SIZE`

## Future Improvements

- Implement vector and matrix operations
- Add more activation functions and loss functions
- Improve memory management
- Implement simple neural network layers
