#!/bin/bash

# Hardcoded file URL
FILE_URL="https://drive.google.com/file/d/1U8WGWExlU4vhTVZZfD1HVrgLTSOZ5-Zd/view?usp=sharing"

# Target directory
TARGET_DIR="./runs"

# Create directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Download file into the target directory
curl -L "$FILE_URL" -o "$TARGET_DIR/$(basename "$FILE_URL")"