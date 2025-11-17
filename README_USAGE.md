# Median Filter GPU - Usage Guide

## Quick Start

### 1. Create a test image

```bash
# Create a test image with noise
python3 create_test_image.py --width 512 --height 512 --noise 0.15

# This creates: inputs/test_image.bmp
```

### 2. Run the denoiser

```bash
# Using the example script
python3 run_median_filter.py inputs/test_image.bmp outputs/denoised.bmp

# Or using the test script (compares both kernels)
python3 test_denoiser.py inputs/test_image.bmp
```

### 3. Verify results

Check the `outputs/` directory for denoised images.

## Directory Structure

```
Median-Filtering-GPU/
├── inputs/          # Place input BMP images here
├── outputs/         # Denoised images are saved here
├── wrapper.cu       # C wrapper with CUDA integration
├── median_filter_gpu.py  # Python wrapper module
├── gpu_diagnostics.py    # GPU monitoring utilities
└── libmedian_filter_gpu.so  # Compiled shared library
```

## Using the Python API

```python
from median_filter_gpu import MedianFilterGPU, BitmapWrapper

# Initialize
filter_gpu = MedianFilterGPU()
filter_gpu.enable_logging(True)  # See detailed GPU operations

# Load image
input_img = BitmapWrapper.load(filter_gpu.lib, "inputs/test_image.bmp")

# Apply filter (regular kernel)
output_img, timing = filter_gpu.filter(input_img, use_shared_memory=False)
output_img.save("outputs/denoised.bmp")

# Print timing information
filter_gpu.print_timing_info(timing)
```

## Performance

The wrapper provides detailed timing:
- **Total Time**: Complete operation time
- **H2D Transfer**: Host-to-device memory transfer
- **Kernel Time**: Actual GPU computation time
- **D2H Transfer**: Device-to-host memory transfer

## GPU Verification

All GPU operations are logged. Look for `[GPU_WRAPPER]` messages showing:
- CUDA initialization
- Memory allocation
- Kernel launches
- Data transfers
- Timing results

## Troubleshooting

1. **Library not found**: Run `./build_wrapper.sh` to build the shared library
2. **CUDA errors**: Check GPU is accessible with `nvidia-smi`
3. **Image loading errors**: Ensure images are in BMP format (8-bit grayscale)

