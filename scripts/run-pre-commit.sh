#!/bin/bash

# run pre-commits
poetry run pre-commit run --all-files --verbose --config .pre-commit-config.yaml
# $? stores exit value of the last command
if [ $? -ne 0 ]; then
 echo "Pre-commit hooks must pass before commit!"
 exit 1
fi