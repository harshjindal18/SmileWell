#!/usr/bin/env bash
# Install system-level packages required for dlib
apt-get update && apt-get install -y cmake g++ make

# Continue with normal pip install
pip install -r requirements.txt
