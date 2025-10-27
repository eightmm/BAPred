#!/bin/bash
# Build script for BA-Pred PyPI package

set -e  # Exit on error

echo "🔧 Building BA-Pred package..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Install build tools
echo "Installing/updating build tools..."
pip install --upgrade pip setuptools wheel build twine

# Build the package
echo "Building package..."
python -m build

# Check the package
echo "Checking package with twine..."
twine check dist/*

echo "✅ Build complete! Package files in dist/"
echo ""
echo "To upload to TestPyPI (for testing):"
echo "  twine upload --repository testpypi dist/*"
echo ""
echo "To upload to PyPI (for production):"
echo "  twine upload dist/*"