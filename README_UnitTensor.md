# UnitTensor

A simplified PyTorch tensor subclass that supports physical units and sampling metadata as properties.

## Overview

`UnitTensor` extends PyTorch's `Tensor` class to include:
- **Units**: Physical units stored as properties (e.g., "m", "kg*m/s^2")
- **Sampling**: Metadata about sampling parameters (e.g., {"dx": 0.1, "dt": 0.01})

This class is designed to be simple and reliable, focusing on storing metadata rather than complex unit propagation through operations.

## Features

- ✅ Store units as properties
- ✅ Store sampling metadata as properties
- ✅ Pretty printing with unit information
- ✅ Unit parsing from strings (e.g., "kg*m/s^2")
- ✅ Convenience constructors (zeros, ones, randn, etc.)
- ✅ Full PyTorch compatibility
- ✅ Gradient support
- ✅ Simple and reliable

## Installation

The `UnitTensor` class is part of the `scatterem2` package. Make sure you have PyTorch installed:

```bash
pip install torch
```

## Basic Usage

### Creating UnitTensors

```python
from scatterem2.utils.data.unittensor import UnitTensor
import torch

# Basic creation with units
position = UnitTensor([1.0, 2.0, 3.0], "m")
velocity = UnitTensor([10.0, 20.0, 30.0], "m/s")

# With sampling metadata
data = UnitTensor([1, 2, 3], "kg", sampling={"dx": 0.1, "dt": 0.01})

# Complex units
force = UnitTensor([5.0, 10.0], "kg*m/s^2")
```

### Accessing Properties

```python
print(position.units)  # ('m',)
print(position.sampling)  # None
print(data.sampling)  # {'dx': 0.1, 'dt': 0.01}
```

### Convenience Constructors

```python
# Create tensors with specific units
zeros = UnitTensor.zeros(3, 3, units="kg")
ones = UnitTensor.ones(2, 4, units="m")
random = UnitTensor.randn(5, units="s")
arange = UnitTensor.arange(10, units="m")
linspace = UnitTensor.linspace(0, 1, 100, units="s")
```

### Unit Operations

```python
# Check if tensors have the same units
same_units = position.has_same_units(velocity)  # False

# Unit multiplication and division
combined_units = position.unit_multiply(velocity.units)
ratio_units = position.unit_divide(velocity.units)
```

### Regular Tensor Operations

UnitTensor instances behave like regular PyTorch tensors for mathematical operations:

```python
# Scalar operations
result = position * 2.0  # Returns a regular tensor

# Element-wise operations
sum_tensor = position + velocity  # Returns a regular tensor

# Gradient support
x = UnitTensor([1.0, 2.0], "m", requires_grad=True)
y = x * x  # Regular tensor operation
loss = y.sum()
loss.backward()
print(x.grad)  # Regular tensor gradient
```

## Unit String Format

Units can be specified as strings with the following format:

- Simple units: `"m"`, `"kg"`, `"s"`
- Multiplication: `"kg*m"`, `"m*s"`
- Division: `"m/s"`, `"kg/m^3"`
- Exponents: `"m^2"`, `"s^-1"`
- Complex: `"kg*m/s^2"`, `"J/(kg*K)"`

## Examples

### Basic Example

```python
from scatterem2.utils.data.unittensor import UnitTensor

# Create tensors with units
position = UnitTensor([1.0, 2.0, 3.0], "m")
velocity = UnitTensor([10.0, 20.0, 30.0], "m/s", sampling={"dt": 0.01})

print(position)  # UnitTensor([1., 2., 3.])  # units=m
print(velocity)  # UnitTensor([10., 20., 30.])  # units=m/s, sampling={'dt': 0.01}

# Access properties
print(f"Position units: {position.units}")
print(f"Velocity sampling: {velocity.sampling}")
```

### Scientific Computing Example

```python
import torch
from scatterem2.utils.data.unittensor import UnitTensor

# Physical constants
G = UnitTensor(6.67430e-11, "m^3/(kg*s^2)")  # Gravitational constant
M_earth = UnitTensor(5.972e24, "kg")  # Earth mass
R_earth = UnitTensor(6.371e6, "m")  # Earth radius

# Calculate gravitational acceleration
g = G * M_earth / (R_earth * R_earth)
print(f"Gravitational acceleration: {g}")  # Regular tensor result

# Store with units for reference
g_with_units = UnitTensor(g, "m/s^2")
print(f"g with units: {g_with_units}")
```

## Testing

Run the test suite:

```bash
pytest scatterem2/scatterem2/utils/data/test_unittensor.py -v
```

## Design Philosophy

This simplified `UnitTensor` class prioritizes:

1. **Reliability**: No complex dispatch logic that could break
2. **Simplicity**: Easy to understand and maintain
3. **Compatibility**: Works seamlessly with existing PyTorch code
4. **Metadata Storage**: Focus on storing units and sampling information

The class does not attempt to automatically propagate units through mathematical operations, as this can be complex and error-prone. Instead, it provides a clean way to store and access unit information alongside tensor data.

## Limitations

- Mathematical operations return regular PyTorch tensors (not UnitTensors)
- No automatic unit validation or propagation
- Unit algebra is basic (no simplification of units like m^2/m → m)

For more advanced unit handling, consider using dedicated unit libraries like `pint` or `astropy.units` in combination with PyTorch tensors. 