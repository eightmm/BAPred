# Contributing to BAPred

Thank you for your interest in contributing to BAPred! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/BAPred.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/eightmm/BAPred.git
cd BAPred

# Create virtual environment
conda env create -f env.yaml
conda activate BAPred

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov flake8 mypy black
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep line length under 100 characters
- Use meaningful variable names

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage

```bash
# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=bapred
```

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the requirements.txt if you add dependencies
3. Ensure all tests pass
4. Request review from maintainers

## Reporting Issues

- Use GitHub Issues to report bugs
- Include detailed description and steps to reproduce
- Include system information (OS, Python version, etc.)
- Add relevant labels

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.