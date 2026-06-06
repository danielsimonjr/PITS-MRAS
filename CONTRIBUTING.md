# Contributing to PITS-MRAS

Thank you for your interest in contributing to PITS-MRAS! This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and constructive in all interactions with the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

### Suggesting Features

Feature requests are welcome! Please include:
- Use case description
- Proposed API or interface
- Potential implementation approach
- Why this feature would benefit the community

### Submitting Pull Requests

1. **Fork the repository** and create a new branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Write tests** for new functionality

4. **Update documentation** as needed

5. **Run tests and linters**
   ```bash
   ruff check .          # lint (add --fix to auto-fix)
   ruff format .         # format
   mypy src/pits_mras
   pytest
   ```

6. **Commit with descriptive messages**
   ```bash
   git commit -m "Add feature: brief description"
   ```

7. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/PITS-MRAS.git
cd PITS-MRAS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

## Coding Standards

### Python Style
- Follow PEP 8
- Use Ruff for linting and formatting (`ruff check` + `ruff format`)
- Maximum line length: 100 characters
- Use type hints for function signatures

### Documentation
- Docstrings for all public functions/classes (Google style)
- Update README.md for significant changes
- Add examples for new features

### Testing
- Maintain test coverage above 80%
- Unit tests for individual components
- Integration tests for component interactions
- Include edge cases and error handling

## Project Structure Guidelines

### Adding New Models
Place in `src/pits_mras/models/` with:
- Clear class/function names
- Comprehensive docstrings
- Type hints
- Unit tests in `tests/test_models.py`

### Adding New Loss Functions
Place in `src/pits_mras/losses/` with:
- Mathematical formulation in docstring
- Gradient verification tests
- Physical interpretation

### Adding Examples
Place in `examples/` with:
- Complete runnable script
- Configuration file
- README documentation
- Expected output/results

## Review Process

1. **Automated checks** run on all PRs (tests, linting)
2. **Maintainer review** for code quality and design
3. **Discussion** if changes needed
4. **Merge** once approved

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Academic publications (for significant contributions)

## Questions?

Feel free to:
- Open a discussion on GitHub
- Contact maintainers via email
- Join our community chat (link coming soon)

Thank you for helping make PITS-MRAS better!
