#!/usr/bin/env python3
"""
Batch denoise video frames with energy consumption tracking.

Processes all noisy video frames through the GPU median filter
and tracks total energy consumption across the entire batch.
"""

import sys
import os
import argparse
from pathlib import Path
from median_filter_gpu import MedianFilterGPU, BitmapWrapper

def get_frame_number(filename: str) -> int:
    """Extract frame number from filename like 'frame_00001.bmp'."""
    try:
        base = Path(filename).stem  # 'frame_00001'
        num_str = base.split('_')[-1]  # '00001'
        return int(num_str)
    except:
        return 0

def main():
    parser = argparse.ArgumentParser(
        description='Batch denoise video frames with energy tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Denoise all frames in noisy_video/
  python3 denoise_video_batch.py
  
  # Denoise frames 1-50
  python3 denoise_video_batch.py --start 1 --end 50
  
  # Use shared memory kernel
  python3 denoise_video_batch.py --shared-memory
        """
    )
    
    parser.add_argument('--input-dir', type=str, default='noisy_video',
                       help='Input directory with noisy BMP files (default: noisy_video)')
    parser.add_argument('--output-dir', type=str, default='denoised_video',
                       help='Output directory for denoised BMP files (default: denoised_video)')
    parser.add_argument('--start', type=int, default=None,
                       help='Start frame number (default: first frame)')
    parser.add_argument('--end', type=int, default=None,
                       help='End frame number (default: last frame)')
    parser.add_argument('--shared-memory', action='store_true',
                       help='Use shared memory kernel (default: regular kernel)')
    parser.add_argument('--enable-logging', action='store_true',
                       help='Enable GPU wrapper logging')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all BMP files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return 1
    
    bmp_files = sorted(input_dir.glob('*.bmp'), key=lambda f: get_frame_number(str(f)))
    
    if not bmp_files:
        print(f"Error: No BMP files found in {input_dir}", file=sys.stderr)
        return 1
    
    print(f"Found {len(bmp_files)} BMP files in {input_dir}")
    
    # Filter by frame range if specified
    if args.start is not None or args.end is not None:
        filtered_files = []
        for f in bmp_files:
            frame_num = get_frame_number(str(f))
            if args.start is not None and frame_num < args.start:
                continue
            if args.end is not None and frame_num > args.end:
                continue
            filtered_files.append(f)
        bmp_files = filtered_files
        print(f"Processing frames {args.start or 'first'} to {args.end or 'last'}: {len(bmp_files)} files")
    
    # Initialize GPU filter
    print("\n" + "="*70)
    print(" Initializing GPU Median Filter")
    print("="*70)
    filter_gpu = MedianFilterGPU()
    filter_gpu.enable_logging(args.enable_logging)
    
    # Get device info
    device_info = filter_gpu.get_device_info()
    print(f"GPU: {device_info.get('name', 'Unknown')}")
    print(f"Compute Capability: {device_info.get('compute_capability', 'Unknown')}")
    print(f"Total Memory: {device_info.get('total_memory_gb', 0):.2f} GB")
    print("="*70)
    
    # Process frames
    print(f"\nProcessing {len(bmp_files)} frames...")
    print(f"Kernel: {'Shared Memory' if args.shared_memory else 'Regular'}")
    print(f"Output directory: {args.output_dir}/")
    print()
    
    successful = 0
    failed = 0
    total_energy_joules = 0.0
    total_kernel_time_ms = 0.0
    total_h2d_time_ms = 0.0
    total_d2h_time_ms = 0.0
    total_time_ms = 0.0
    
    energy_data = []
    
    for i, bmp_file in enumerate(bmp_files, 1):
        frame_num = get_frame_number(str(bmp_file))
        output_filename = f"frame_{frame_num:05d}.bmp"
        output_path = os.path.join(args.output_dir, output_filename)
        
        try:
            # Load noisy image
            noisy_img = BitmapWrapper.load(filter_gpu.lib, str(bmp_file))
            
            # Apply median filter with energy tracking
            denoised_img, timing = filter_gpu.filter(
                noisy_img,
                use_shared_memory=args.shared_memory,
                collect_diagnostics=True
            )
            
            # Save denoised result
            denoised_img.save(output_path)
            
            # Accumulate statistics
            if 'energy' in timing:
                energy = timing['energy']
                total_energy_joules += energy['total_energy_joules']
                energy_data.append({
                    'frame': frame_num,
                    'energy_mj': energy['total_energy_mj'],
                    'power_w': energy['avg_power_watts'],
                    'duration_ms': energy['duration_seconds'] * 1000
                })
            
            total_kernel_time_ms += timing.get('kernel_time_ms', 0)
            total_h2d_time_ms += timing.get('h2d_time_ms', 0)
            total_d2h_time_ms += timing.get('d2h_time_ms', 0)
            total_time_ms += timing.get('total_time_ms', 0)
            
            successful += 1
            
            if i % 10 == 0 or i == len(bmp_files):
                print(f"  [{i}/{len(bmp_files)}] Denoised frame {frame_num:05d}")
                
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(bmp_files)}] Error processing frame {frame_num:05d}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print(" BATCH DENOISING SUMMARY")
    print("="*70)
    print(f"Frames processed:  {successful} successful, {failed} failed")
    print(f"Kernel type:        {'Shared Memory' if args.shared_memory else 'Regular'}")
    
    if successful > 0:
        print(f"\nTiming Statistics:")
        print(f"  Total time:      {total_time_ms:.2f} ms ({total_time_ms/1000:.2f} s)")
        print(f"  Kernel time:     {total_kernel_time_ms:.2f} ms")
        print(f"  H2D transfer:   {total_h2d_time_ms:.2f} ms")
        print(f"  D2H transfer:   {total_d2h_time_ms:.2f} ms")
        print(f"  Avg per frame:   {total_time_ms/successful:.2f} ms")
        
        if total_energy_joules > 0:
            print(f"\nEnergy Statistics:")
            print(f"  Total energy:    {total_energy_joules:.6f} J ({total_energy_joules*1000:.3f} mJ)")
            print(f"  Avg per frame:   {total_energy_joules*1000/successful:.3f} mJ")
            
            if energy_data:
                avg_power = sum(e['power_w'] for e in energy_data) / len(energy_data)
                min_power = min(e['power_w'] for e in energy_data)
                max_power = max(e['power_w'] for e in energy_data)
                min_energy = min(e['energy_mj'] for e in energy_data)
                max_energy = max(e['energy_mj'] for e in energy_data)
                
                print(f"  Average power:   {avg_power:.2f} W")
                print(f"  Power range:     {min_power:.2f} - {max_power:.2f} W")
                print(f"  Energy range:    {min_energy:.3f} - {max_energy:.3f} mJ")
                
                # Calculate efficiency
                if noisy_img:
                    pixels_per_frame = noisy_img.width * noisy_img.height
                    total_pixels = pixels_per_frame * successful
                    energy_per_pixel = total_energy_joules / total_pixels
                    print(f"\nEfficiency Metrics:")
                    print(f"  Pixels per frame: {pixels_per_frame:,}")
                    print(f"  Total pixels:     {total_pixels:,}")
                    print(f"  Energy per pixel: {energy_per_pixel*1e6:.6f} μJ/pixel")
        
        print(f"\nOutput directory:  {args.output_dir}/")
    
    print("="*70)
    print("✓ Batch denoising completed!")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

