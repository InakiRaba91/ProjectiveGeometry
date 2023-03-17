#!/bin/bash

GIT_DIR=$(git rev-parse --git-dir)

pwd
echo "Installing hooks..."
# this command creates symlink to our pre-commit script
ln -s ../../scripts/run-pre-commit.sh $GIT_DIR/hooks/pre-commit
echo "Done"!