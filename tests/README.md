# PITS-MRAS Tests

Unit and integration tests for the PITS-MRAS framework.

## Test Structure

```
tests/
├── test_models.py           # PITNN, decoders, attention
├── test_controllers.py      # MRAS controller tests
├── test_losses.py           # Physics, temporal, stability losses
├── test_stability.py        # Lyapunov validation
├── test_training.py         # Pre-training and co-training
├── test_inference.py        # Real-time inference engine
└── test_integration.py      # End-to-end system tests
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=pits_mras --cov-report=html

# Run specific test
pytest tests/test_stability.py::test_lyapunov_decrease
```

## Test Categories

### Unit Tests
- Individual component correctness
- Edge case handling
- Input validation

### Integration Tests
- Multi-component interactions
- End-to-end pipeline validation
- Performance benchmarks

### Stability Tests
- Lyapunov function decrease verification
- Energy conservation checks
- Parameter convergence validation

## Contributing Tests

When adding new features, please include:
1. Unit tests for new functions/classes
2. Integration tests for component interactions
3. Documentation of test cases
4. Expected behavior specifications

## Continuous Integration

Tests will be automatically run on:
- Pull requests
- Commits to main branch
- Scheduled nightly builds
