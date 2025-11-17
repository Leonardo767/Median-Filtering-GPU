#!/bin/bash

# Build script for creating shared library wrapper for CUDA median filter
# This script compiles the CUDA code and wrapper into a shared library

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building CUDA median filter shared library...${NC}"

# Detect CUDA installation
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        CUDA_HOME="/usr/local/cuda"
    elif [ -d "/opt/cuda" ]; then
        CUDA_HOME="/opt/cuda"
    else
        echo -e "${RED}Error: CUDA_HOME not set and CUDA not found in standard locations${NC}"
        echo "Please set CUDA_HOME environment variable or install CUDA"
        exit 1
    fi
fi

echo -e "${GREEN}Using CUDA installation: $CUDA_HOME${NC}"

CUDA_LIB_PATH="$CUDA_HOME/lib64"
CUDA_INC_PATH="$CUDA_HOME/include"

# Check if CUDA libraries exist
if [ ! -d "$CUDA_LIB_PATH" ]; then
    echo -e "${YELLOW}Warning: CUDA lib64 directory not found at $CUDA_LIB_PATH${NC}"
    # Try lib instead of lib64
    if [ -d "$CUDA_HOME/lib" ]; then
        CUDA_LIB_PATH="$CUDA_HOME/lib"
        echo -e "${GREEN}Using $CUDA_LIB_PATH instead${NC}"
    else
        echo -e "${RED}Error: CUDA library directory not found${NC}"
        exit 1
    fi
fi

# Detect compute capability (try to query GPU, fallback to common values)
COMPUTE_CAP=""
if command -v nvidia-smi &> /dev/null; then
    # Try to get compute capability from nvidia-smi
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    if [[ "$GPU_NAME" == *"V100"* ]] || [[ "$GPU_NAME" == *"Titan V"* ]]; then
        COMPUTE_CAP="sm_70"
    elif [[ "$GPU_NAME" == *"RTX 30"* ]] || [[ "$GPU_NAME" == *"A100"* ]]; then
        COMPUTE_CAP="sm_80"
    elif [[ "$GPU_NAME" == *"RTX 40"* ]] || [[ "$GPU_NAME" == *"H100"* ]]; then
        COMPUTE_CAP="sm_89"
    elif [[ "$GPU_NAME" == *"H200"* ]]; then
        COMPUTE_CAP="sm_90"
    elif [[ "$GPU_NAME" == *"GTX 10"* ]] || [[ "$GPU_NAME" == *"GTX 16"* ]] || [[ "$GPU_NAME" == *"RTX 20"* ]]; then
        COMPUTE_CAP="sm_75"
    else
        # Default to a common modern architecture
        COMPUTE_CAP="sm_75"
    fi
    echo -e "${GREEN}Detected GPU: $GPU_NAME, using compute capability: $COMPUTE_CAP${NC}"
else
    # Fallback: use multiple architectures for compatibility
    COMPUTE_CAP="sm_75"
    echo -e "${YELLOW}nvidia-smi not found, using default compute capability: $COMPUTE_CAP${NC}"
    echo -e "${YELLOW}If build fails, you may need to specify the correct compute capability${NC}"
fi

# Clean previous builds
echo -e "${GREEN}Cleaning previous build artifacts...${NC}"
rm -f *.o libmedian_filter_gpu.so

# Compile CUDA code to object file
echo -e "${GREEN}Compiling medianFilter.cu...${NC}"
nvcc -arch=$COMPUTE_CAP -c medianFilter.cu -o medianFilter.o \
    -I. -I"$CUDA_INC_PATH" \
    -Xcompiler -fPIC \
    || {
    echo -e "${RED}Error: Failed to compile medianFilter.cu${NC}"
    echo -e "${YELLOW}Try specifying compute capability manually: nvcc -arch=sm_XX ...${NC}"
    exit 1
}

# Compile wrapper to object file (needs nvcc because it launches CUDA kernels)
echo -e "${GREEN}Compiling wrapper.cu...${NC}"
nvcc -arch=$COMPUTE_CAP -c wrapper.cu -o wrapper.o \
    -I. -I"$CUDA_INC_PATH" \
    -Xcompiler -fPIC -std=c++11 \
    || {
    echo -e "${RED}Error: Failed to compile wrapper.cu${NC}"
    exit 1
}

# Link into shared library
echo -e "${GREEN}Linking shared library...${NC}"
g++ -shared -o libmedian_filter_gpu.so medianFilter.o wrapper.o \
    -L"$CUDA_LIB_PATH" -lcudart \
    -Wl,-rpath,"$CUDA_LIB_PATH" \
    || {
    echo -e "${RED}Error: Failed to link shared library${NC}"
    exit 1
}

# Verify the library was created
if [ -f "libmedian_filter_gpu.so" ]; then
    echo -e "${GREEN}✓ Successfully built libmedian_filter_gpu.so${NC}"
    echo -e "${GREEN}Library location: $(pwd)/libmedian_filter_gpu.so${NC}"
    
    # Show library dependencies
    echo -e "${GREEN}Library dependencies:${NC}"
    ldd libmedian_filter_gpu.so | grep -E "(cuda|libc)" || true
else
    echo -e "${RED}Error: Shared library was not created${NC}"
    exit 1
fi

echo -e "${GREEN}Build complete!${NC}"

