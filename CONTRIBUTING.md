# Contributing to HyperFit

Thank you for your interest in contributing to HyperFit! This document provides guidelines and information for contributors.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hyperfit/hyperfit.git
   cd hyperfit
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests**:
   ```bash
   pytest tests/
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for all public functions and classes
- Use Black for code formatting: `black src/`
- Use flake8 for linting: `flake8 src/`

## Testing

- Write tests for all new functionality
- Ensure test coverage remains high
- Test both successful cases and error conditions
- Include integration tests for new models

## Adding New Material Models

1. Create a new file in `src/hyperfit/models/`
2. Inherit from `HyperelasticModel` base class
3. Implement all required abstract methods
4. Add the model to the registry in `models/__init__.py`
5. Write comprehensive tests
6. Update documentation

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests and documentation
5. Run the test suite
6. Commit your changes with clear messages
7. Push to your fork and submit a pull request

## Pull Request Guidelines

- Provide a clear description of changes
- Reference any related issues
- Ensure all tests pass
- Update documentation as needed
- Follow the existing code style

## Reporting Issues

- Use the GitHub issue tracker
- Provide a clear description of the problem
- Include steps to reproduce
- Specify your environment (OS, Python version, etc.)
- Include relevant code snippets or error messages

## Documentation

- Update docstrings for any changed functions
- Add examples for new features
- Update the README if needed
- Consider adding tutorial content

Thank you for contributing to HyperFit!
