# Individual Learning Rates for SingleSlicePtychography

This document describes the new functionality that allows fine-grained control over learning rates for different parameter types in the `SingleSlicePtychography` model.

## Overview

The `SingleSlicePtychography` model now supports individual learning rates for different parameter types:
- **Object parameters**: The object being reconstructed (pixels or hash encoding)
- **Probe parameters**: The electron probe wavefunction
- **DR parameters**: Position correction parameters (dr)

Additionally, you can specify **when** each parameter type should start being optimized during training by setting epoch-based start times.

## New Parameters

### Constructor Parameters

When initializing `SingleSlicePtychography`, you can now specify:

- `object_learning_rate` (float, optional): Learning rate for object parameters
- `probe_learning_rate` (float, optional): Learning rate for probe parameters  
- `dr_learning_rate` (float, optional): Learning rate for dr parameters
- `optimize_probe_start` (int, optional): Epoch when probe optimization should begin
- `optimize_dr_start` (int, optional): Epoch when dr optimization should begin

### Configuration File Parameters

You can also specify these in your YAML configuration file:

```yaml
ptychography:
  learning_rate: 1e-3  # Default learning rate for other parameters
  object_learning_rate: 5e-3  # Higher learning rate for object
  probe_learning_rate: 1e-4  # Lower learning rate for probe
  dr_learning_rate: 1e-5  # Very low learning rate for position corrections
  optimize_probe_start: 10  # Start optimizing probe at epoch 10
  optimize_dr_start: 20  # Start optimizing dr at epoch 20
```

## Usage Examples

### Example 1: Staged Optimization Strategy

```python
from scatterem2.models.single_slice_ptychography import SingleSlicePtychography

model = SingleSlicePtychography(
    meta4d=meta4d,
    learning_rate=1e-3,  # Default learning rate for other parameters
    object_learning_rate=5e-3,  # Higher learning rate for object
    probe_learning_rate=1e-4,  # Lower learning rate for probe
    dr_learning_rate=1e-5,  # Very low learning rate for position corrections
    optimize_probe_start=10,  # Start optimizing probe at epoch 10
    optimize_dr_start=20,  # Start optimizing dr at epoch 20
)
```

This creates a staged optimization strategy:
- **Epochs 0-9**: Only object parameters are optimized
- **Epochs 10-19**: Object and probe parameters are optimized
- **Epochs 20+**: All parameters (object, probe, dr) are optimized

### Example 2: Only Optimizing Object (Default Behavior)

```python
model = SingleSlicePtychography(
    meta4d=meta4d,
    learning_rate=1e-3,
    object_learning_rate=5e-3,
    # optimize_probe_start and optimize_dr_start are None by default
    # so probe and dr will never be optimized
)
```

### Example 3: Early Probe Optimization

```python
model = SingleSlicePtychography(
    meta4d=meta4d,
    object_model="hash_encoding",  # Use hash encoding instead of pixels
    learning_rate=1e-3,
    object_learning_rate=1e-2,  # Higher learning rate for hash encoding
    probe_learning_rate=1e-4,
    optimize_probe_start=5,  # Start optimizing probe early at epoch 5
    optimize_dr_start=15,  # Start optimizing dr later at epoch 15
)
```

## Monitoring Learning Rates and Optimization Status

You can monitor the current learning rates for each parameter group:

```python
# Get current learning rates
lr_dict = model.get_learning_rates()
for param_type, lr in lr_dict.items():
    print(f"{param_type}: {lr}")
```

The model also logs when probe and dr optimization starts:

```python
# During training, you'll see logs like:
# "probe_optimization_started": 1.0 (at epoch 10)
# "dr_optimization_started": 1.0 (at epoch 20)
```

## Implementation Details

### Parameter Groups

The optimizer is configured with parameter groups, each with its own learning rate:

1. **Object parameters**: Includes `self.forward_model.object` (either `nn.Parameter` or `MultiresolutionHashEncoding`)
2. **Probe parameters**: Includes all probe tensors (gradients enabled based on epoch)
3. **DR parameters**: Includes `self.forward_model.dr` (gradients enabled based on epoch)
4. **Other parameters**: Any remaining parameters not in the above groups

### Epoch-Based Gradient Control

The `_update_gradient_requirements()` method is called at the start of each training epoch to:

1. Check if the current epoch has reached `optimize_probe_start`
2. Check if the current epoch has reached `optimize_dr_start`
3. Set `requires_grad=True` for the appropriate parameters
4. Log when optimization starts for each parameter type

### Default Behavior

If no optimization start epochs are specified:
- All parameters use the default `learning_rate`
- Only object parameters are optimized (probe and dr have `requires_grad=False` throughout training)

## Benefits

1. **Staged optimization**: Start with stable parameters (object) and gradually add more sensitive ones
2. **Fine-grained control**: Different parameters can have different learning rates based on their characteristics
3. **Stability**: Lower learning rates for sensitive parameters (like probe and position corrections)
4. **Convergence**: Higher learning rates for robust parameters (like object)
5. **Flexibility**: Easy to control when each parameter type starts being optimized

## Best Practices

1. **Start with object only**: Begin training with only object parameters for the first few epochs
2. **Add probe later**: Start probe optimization after object has converged somewhat (epoch 5-15)
3. **Add dr last**: Start position correction optimization last (epoch 15-30)
4. **Conservative learning rates**: Use lower learning rates for probe (1e-4 to 1e-5) and dr (1e-5 to 1e-6)
5. **Monitor convergence**: Use the logging to verify when each parameter type starts being optimized

## Migration from Previous Versions

If you're upgrading from a previous version:

1. Your existing code will continue to work without changes
2. All parameters will use the default `learning_rate`
3. Only object parameters will be optimized (same as before)
4. To use the new functionality, add the new parameters to your constructor calls or configuration files 