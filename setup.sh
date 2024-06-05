#!/bin/bash

# Clone the submodule
git submodule update --init --recursive

# Navigate to the submodule directory and install its dependencies if any
cd DiffDRR/diffdrr
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi
