#!/bin/bash

# Exit immediately if a command fails
set -e
echo "Running tests with coverage..."

# Clean previous coverage data
coverage erase

# Run all tests in the 'tests/' folder with coverage, using custom configurations from .coveragerc
coverage run --rcfile=.coveragerc -m pytest tests/

# Print a report (with missing lines)
coverage report -m

# Generate an HTML coverage report
coverage html

echo "All tests completed."
echo "Open 'htmlcov/index.html' to view the coverage report."
