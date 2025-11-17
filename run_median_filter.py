#!/usr/bin/env python3
"""
Example script demonstrating GPU median filter usage with diagnostics.

This script:
1. Loads an input image
2. Runs GPU median filter (both regular and shared memory versions)
3. Displays comprehensive diagnostics
4. Saves output images
"""

import sys
import os
import argparse
from pathlib import Path

# Import the wrapper modules
try:
    from median_filter_gpu import MedianFilterGPU, BitmapWrapper
except ImportError as e:
    print(f"Error: Failed to import median_filter_gpu: {e}", file=sys.stderr)
    print("Make sure median_filter_gpu.py is in the Python path", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='GPU Median Filter with Diagnostics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter an image with default settings
  python run_median_filter.py input.bmp output.bmp
  
  # Use shared memory kernel
  python run_median_filter.py input.bmp output.bmp --shared
  
  # Compare both kernels
  python run_median_filter.py input.bmp output.bmp --compare
  
  # Specify library path
  python run_median_filter.py input.bmp output.bmp --library ./libmedian_filter_gpu.so
        """
    )
    
    parser.add_argument('input', type=str, help='Input image file (BMP format)')
    parser.add_argument('output', type=str, nargs='?', default=None,
                       help='Output image file (BMP format). If not specified, auto-generated.')
    parser.add_argument('--shared', action='store_true',
                       help='Use shared memory kernel (default: regular kernel)')
    parser.add_argument('--compare', action='store_true',
                       help='Run both kernels and compare performance')
    parser.add_argument('--library', type=str, default=None,
                       help='Path to libmedian_filter_gpu.so (auto-detected if not specified)')
    parser.add_argument('--no-diagnostics', action='store_true',
                       help='Disable GPU diagnostics collection')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Generate output filename if not provided
    if args.output is None:
        input_path = Path(args.input)
        suffix = "_shared" if args.shared else "_gpu"
        args.output = str(input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}")
        print(f"Output file not specified, using: {args.output}")
    
    print("="*70)
    print("GPU Median Filter - CUDA Implementation with Diagnostics")
    print("="*70)
    print(f"Input file:  {args.input}")
    print(f"Output file: {args.output}")
    print()
    
    try:
        # Initialize GPU filter
        print("Initializing CUDA...")
        filter_gpu = MedianFilterGPU(library_path=args.library)
        
        # Display device info
        device_info = filter_gpu.get_device_info()
        if device_info:
            print(f"\nCUDA Device: {device_info.get('name', 'Unknown')}")
            print(f"Compute Capability: {device_info.get('compute_capability', 'Unknown')}")
            print(f"Total Memory: {device_info.get('total_memory_mb', 0)} MB")
            print(f"Free Memory: {device_info.get('free_memory_mb', 0)} MB")
        
        # Load input image
        print(f"\nLoading image: {args.input}")
        input_image = BitmapWrapper.load(filter_gpu.lib, args.input)
        print(f"Image size: {input_image.width} x {input_image.height} pixels")
        
        # Run filter(s)
        if args.compare:
            # Compare both kernels
            print("\n" + "="*70)
            print("COMPARING KERNELS")
            print("="*70)
            
            # Regular kernel
            print("\n[1/2] Running regular kernel...")
            output_regular, timing_regular = filter_gpu.filter(
                input_image,
                use_shared_memory=False,
                collect_diagnostics=not args.no_diagnostics
            )
            output_path_regular = Path(args.output).parent / f"{Path(args.output).stem}_regular{Path(args.output).suffix}"
            output_regular.save(str(output_path_regular))
            print(f"Saved to: {output_path_regular}")
            filter_gpu.print_timing_info(timing_regular)
            
            # Shared memory kernel
            print("\n[2/2] Running shared memory kernel...")
            output_shared, timing_shared = filter_gpu.filter(
                input_image,
                use_shared_memory=True,
                collect_diagnostics=not args.no_diagnostics
            )
            output_path_shared = Path(args.output).parent / f"{Path(args.output).stem}_shared{Path(args.output).suffix}"
            output_shared.save(str(output_path_shared))
            print(f"Saved to: {output_path_shared}")
            filter_gpu.print_timing_info(timing_shared)
            
            # Comparison summary
            print("\n" + "="*70)
            print("PERFORMANCE COMPARISON")
            print("="*70)
            print(f"{'Metric':<25} {'Regular':<20} {'Shared Memory':<20} {'Speedup':<10}")
            print("-"*70)
            print(f"{'Total Time (ms)':<25} {timing_regular['total_time_ms']:<20.4f} {timing_shared['total_time_ms']:<20.4f} {timing_regular['total_time_ms']/timing_shared['total_time_ms']:<10.2f}x")
            print(f"{'Kernel Time (ms)':<25} {timing_regular['kernel_time_ms']:<20.4f} {timing_shared['kernel_time_ms']:<20.4f} {timing_regular['kernel_time_ms']/timing_shared['kernel_time_ms']:<10.2f}x")
            print(f"{'H2D Transfer (ms)':<25} {timing_regular['h2d_time_ms']:<20.4f} {timing_shared['h2d_time_ms']:<20.4f} {'N/A':<10}")
            print(f"{'D2H Transfer (ms)':<25} {timing_regular['d2h_time_ms']:<20.4f} {timing_shared['d2h_time_ms']:<20.4f} {'N/A':<10}")
            print("="*70)
            
        else:
            # Run single kernel
            kernel_type = "shared memory" if args.shared else "regular"
            print(f"\nRunning {kernel_type} kernel...")
            
            output_image, timing_info = filter_gpu.filter(
                input_image,
                use_shared_memory=args.shared,
                collect_diagnostics=not args.no_diagnostics
            )
            
            # Save output
            print(f"\nSaving output to: {args.output}")
            output_image.save(args.output)
            print("✓ Image saved successfully")
            
            # Display timing
            filter_gpu.print_timing_info(timing_info)
        
        print("\n✓ Processing complete!")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("\nMake sure to build the shared library first:", file=sys.stderr)
        print("  ./build_wrapper.sh", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

