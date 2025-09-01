# ScatterEM2 Tests

This directory contains tests for the ScatterEM2 package.

## Test Files

- `test_single_slice_ptychography.py` - Tests for single slice ptychography functionality

## Running Tests

### Quick Tests (for pre-commit hooks)
```bash
pytest test/test_single_slice_ptychography.py::test_imports_and_basic_setup -v
```

### Full Tests (including slow tests)
```bash
pytest test/test_single_slice_ptychography.py -v
```

### All Tests
```bash
pytest test/ -v
```

## Pre-commit Hook

The tests are integrated into the pre-commit hook system. The pre-commit configuration runs a quick smoke test before each commit to ensure basic functionality.

To install pre-commit hooks:
```bash
pre-commit install
```

To run all pre-commit hooks manually:
```bash
pre-commit run --all-files
```

## Test Markers

- `@pytest.mark.slow` - Marks tests that take a long time to run
- `@pytest.mark.integration` - Marks integration tests

To run only fast tests:
```bash
pytest -m "not slow" -v
```

## Configuration

The test configuration is in `pytest.ini` at the project root. 