# Release Process for BAPred

This document describes the release process for BAPred.

## Prerequisites

1. PyPI account (https://pypi.org/account/register/)
2. TestPyPI account (https://test.pypi.org/account/register/)
3. API tokens for both PyPI and TestPyPI
4. Configure `~/.pypirc` with your tokens (see `.pypirc.template`)

## Release Steps

### 1. Update Version

Update the version in `pyproject.toml`:
```toml
[project]
version = "1.0.1"  # New version
```

### 2. Update Changelog

Create or update `CHANGELOG.md` with release notes.

### 3. Run Tests

```bash
# Run all tests
pytest tests/

# Check code quality
flake8 bapred/
mypy bapred/
```

### 4. Build Package

```bash
# Use the build script
./scripts/build_package.sh

# Or manually
python -m build
twine check dist/*
```

### 5. Test on TestPyPI

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bapred
```

### 6. Create Git Tag

```bash
git tag -a v1.0.1 -m "Release version 1.0.1"
git push origin v1.0.1
```

### 7. Create GitHub Release

1. Go to https://github.com/eightmm/BAPred/releases
2. Click "Create a new release"
3. Choose the tag you just created
4. Add release notes
5. Upload the wheel and source files from `dist/`

### 8. Publish to PyPI

```bash
# Upload to PyPI
twine upload dist/*

# Verify installation
pip install bapred
```

## Post-Release

1. Verify the package on PyPI: https://pypi.org/project/bapred/
2. Test installation: `pip install bapred`
3. Update documentation if needed
4. Announce the release

## Versioning

We follow [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for new functionality (backwards compatible)
- PATCH version for backwards compatible bug fixes