#!/bin/bash
set -e

cd docs
make html

echo "Documentation built at docs/build/html/index.html"

DOC_INDEX="$(pwd)/build/html/index.html"

echo ""
echo "========================================"
echo "Documentation build completed."
echo "Open the documentation at:"
echo ""
echo "file://$DOC_INDEX"
echo "========================================"
