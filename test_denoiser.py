#!/usr/bin/env python3
"""
Test script to verify the median filter denoiser works with input images.
"""

import sys
import os
from pathlib import Path

def test_denoiser(input_image_path, output_dir="outputs"):
    """
    Test the median filter on an input image.
    
    Args:
        input_image_path: Path to input BMP image
        output_dir: Directory to save output images
    """
    print("="*70)
    print(" Testing Median Filter Denoiser")
    print("="*70)
    
    # Import the wrapper
    try:
        from median_filter_gpu import MedianFilterGPU, BitmapWrapper
    except ImportError as e:
        print(f"Error: Failed to import median_filter_gpu: {e}")
        return False
    
    # Check input file
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found: {input_image_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get input filename for output naming
    input_name = Path(input_image_path).stem
    output_regular = os.path.join(output_dir, f"{input_name}_denoised_regular.bmp")
    output_shared = os.path.join(output_dir, f"{input_name}_denoised_shared.bmp")
    
    try:
        # Initialize GPU filter with logging
        print("\n1. Initializing GPU wrapper...")
        filter_gpu = MedianFilterGPU()
        filter_gpu.enable_logging(True)  # Enable detailed logging
        
        # Get device info
        device_info = filter_gpu.get_device_info()
        print(f"   Device: {device_info.get('name', 'Unknown')}")
        print(f"   Free Memory: {device_info.get('free_memory_mb', 0)} MB")
        
        # Load input image
        print(f"\n2. Loading input image: {input_image_path}")
        input_img = BitmapWrapper.load(filter_gpu.lib, input_image_path)
        print(f"   Image size: {input_img.width} x {input_img.height} pixels")
        print(f"   Total pixels: {input_img.width * input_img.height:,}")
        
        # Test regular kernel
        print(f"\n3. Running median filter (regular kernel)...")
        output_img_regular, timing_regular = filter_gpu.filter(
            input_img,
            use_shared_memory=False,
            collect_diagnostics=True
        )
        output_img_regular.save(output_regular)
        print(f"   ✓ Saved to: {output_regular}")
        filter_gpu.print_timing_info(timing_regular)
        
        # Test shared memory kernel (reload image to avoid resource handle issues)
        print(f"\n4. Running median filter (shared memory kernel)...")
        input_img_shared = BitmapWrapper.load(filter_gpu.lib, input_image_path)
        output_img_shared, timing_shared = filter_gpu.filter(
            input_img_shared,
            use_shared_memory=True,
            collect_diagnostics=True
        )
        output_img_shared.save(output_shared)
        print(f"   ✓ Saved to: {output_shared}")
        filter_gpu.print_timing_info(timing_shared)
        
        # Compare results
        print("\n" + "="*70)
        print(" Performance Comparison")
        print("="*70)
        print(f"{'Metric':<25} {'Regular':<20} {'Shared Memory':<20} {'Speedup':<10}")
        print("-"*70)
        print(f"{'Total Time (ms)':<25} {timing_regular['total_time_ms']:<20.4f} {timing_shared['total_time_ms']:<20.4f} {timing_regular['total_time_ms']/timing_shared['total_time_ms']:<10.2f}x")
        print(f"{'Kernel Time (ms)':<25} {timing_regular['kernel_time_ms']:<20.4f} {timing_shared['kernel_time_ms']:<20.4f} {timing_regular['kernel_time_ms']/timing_shared['kernel_time_ms']:<10.2f}x")
        print(f"{'H2D Transfer (ms)':<25} {timing_regular['h2d_time_ms']:<20.4f} {timing_shared['h2d_time_ms']:<20.4f} {'N/A':<10}")
        print(f"{'D2H Transfer (ms)':<25} {timing_regular['d2h_time_ms']:<20.4f} {timing_shared['d2h_time_ms']:<20.4f} {'N/A':<10}")
        
        # Add energy comparison if available
        if 'energy' in timing_regular and 'energy' in timing_shared:
            print(f"{'Energy (mJ)':<25} {timing_regular['energy']['total_energy_mj']:<20.3f} {timing_shared['energy']['total_energy_mj']:<20.3f} {timing_regular['energy']['total_energy_mj']/timing_shared['energy']['total_energy_mj']:<10.2f}x")
            print(f"{'Avg Power (W)':<25} {timing_regular['energy']['avg_power_watts']:<20.2f} {timing_shared['energy']['avg_power_watts']:<20.2f} {'N/A':<10}")
        elif 'energy' in timing_regular:
            print(f"{'Energy (mJ)':<25} {timing_regular['energy']['total_energy_mj']:<20.3f} {'N/A':<20} {'N/A':<10}")
            print(f"{'Avg Power (W)':<25} {timing_regular['energy']['avg_power_watts']:<20.2f} {'N/A':<20} {'N/A':<10}")
        
        print("\n" + "="*70)
        print(" ✓ Test completed successfully!")
        print("="*70)
        print(f"\nOutput images saved to:")
        print(f"  - Regular kernel: {output_regular}")
        print(f"  - Shared memory:  {output_shared}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the median filter denoiser')
    parser.add_argument('input', type=str, nargs='?', default='inputs/test_image.bmp',
                       help='Input image path (default: inputs/test_image.bmp)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory (default: outputs)')
    
    args = parser.parse_args()
    
    success = test_denoiser(args.input, args.output_dir)
    sys.exit(0 if success else 1)

