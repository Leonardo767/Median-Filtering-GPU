#!/usr/bin/env python3
"""
Simple test script to demonstrate energy tracking.
Uses only the regular kernel which works reliably.
"""

import sys
from median_filter_gpu import MedianFilterGPU, BitmapWrapper

def main():
    print("="*70)
    print(" Energy Consumption Test")
    print("="*70)
    
    # Initialize
    filter_gpu = MedianFilterGPU()
    filter_gpu.enable_logging(False)  # Cleaner output
    
    # Load image
    input_path = sys.argv[1] if len(sys.argv) > 1 else "inputs/test_image.bmp"
    print(f"\nLoading: {input_path}")
    img = BitmapWrapper.load(filter_gpu.lib, input_path)
    print(f"Image size: {img.width} x {img.height}")
    
    # Run filter with energy tracking
    print("\nRunning median filter with energy tracking...")
    output, timing = filter_gpu.filter(img, use_shared_memory=False, collect_diagnostics=True)
    
    # Save result
    output_path = "outputs/energy_test.bmp"
    output.save(output_path)
    print(f"✓ Saved to: {output_path}")
    
    # Display energy results
    print("\n" + "="*70)
    print(" ENERGY CONSUMPTION RESULTS")
    print("="*70)
    energy = timing['energy']
    print(f"Total Energy:      {energy['total_energy_joules']:.6f} J ({energy['total_energy_mj']:.3f} mJ)")
    print(f"Average Power:     {energy['avg_power_watts']:.2f} W")
    print(f"Max Power:         {energy['max_power_watts']:.2f} W")
    print(f"Min Power:         {energy['min_power_watts']:.2f} W")
    print(f"Duration:          {energy['duration_seconds']*1000:.4f} ms")
    print(f"\nTiming Breakdown:")
    print(f"  Kernel Time:     {timing['kernel_time_ms']:.4f} ms")
    print(f"  H2D Transfer:   {timing['h2d_time_ms']:.4f} ms")
    print(f"  D2H Transfer:   {timing['d2h_time_ms']:.4f} ms")
    print(f"  Total Time:     {timing['total_time_ms']:.4f} ms")
    
    # Calculate efficiency metrics
    if timing['kernel_time_ms'] > 0:
        energy_per_pixel = energy['total_energy_joules'] / (img.width * img.height)
        print(f"\nEfficiency Metrics:")
        print(f"  Energy per pixel: {energy_per_pixel*1e6:.6f} μJ/pixel")
        print(f"  Pixels processed: {img.width * img.height:,}")
    
    print("="*70)
    print("✓ Test completed successfully!")
    
if __name__ == "__main__":
    main()

