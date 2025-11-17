#!/usr/bin/env python3
"""
Process video frames: Convert PGM to BMP, add salt-and-pepper noise, and organize outputs.

This script:
1. Reads PGM files from input_video/ folder
2. Converts to 4-bit if needed, applies salt-and-pepper noise
3. Converts to 8-bit BMP format for compatibility with median filter wrapper
4. Organizes outputs into ground_truth_video/ and noisy_video/ folders
5. Optionally processes noisy frames through the denoiser
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional

# Image parameters
H, W = 256, 256
NUM_BITS = 4  # 4-bit greyscale
NUM_LAYERS = NUM_BITS


def add_salt_pepper_noise(image, noise_level=0.15):
    """
    Add salt and pepper noise to 4-bit image.

    Args:
        image: 4-bit image (values 0-15)
        noise_level: Probability of a pixel being corrupted (half salt, half pepper)

    Returns:
        Noisy image with salt (15) and pepper (0) noise
    """
    noisy = image.copy()
    
    # Generate random mask for noise
    noise_mask = np.random.random(image.shape) < noise_level
    
    # Split noise into salt and pepper
    salt_mask = noise_mask & (np.random.random(image.shape) < 0.5)
    pepper_mask = noise_mask & ~salt_mask
    
    # Apply salt (max value = 15 for 4-bit)
    noisy[salt_mask] = 15
    
    # Apply pepper (min value = 0)
    noisy[pepper_mask] = 0
    
    return noisy


def read_pgm(filename: str) -> Tuple[np.ndarray, int, int]:
    """
    Read PGM file and return image array, width, height.
    Handles both ASCII (P2) and binary (P5) formats.
    
    Args:
        filename: Path to PGM file
        
    Returns:
        Tuple of (image_array, width, height)
    """
    try:
        # Use PIL to read PGM (handles both formats)
        img = Image.open(filename)
        arr = np.array(img)
        return arr, img.width, img.height
    except Exception as e:
        print(f"Error reading {filename}: {e}", file=sys.stderr)
        raise


def convert_to_4bit(image_8bit: np.ndarray) -> np.ndarray:
    """
    Convert 8-bit image (0-255) to 4-bit (0-15).
    
    Args:
        image_8bit: 8-bit image array
        
    Returns:
        4-bit image array (0-15)
    """
    # Scale from 0-255 to 0-15
    image_4bit = np.round((image_8bit.astype(np.float32) / 255.0) * 15.0).astype(np.uint8)
    return np.clip(image_4bit, 0, 15)


def convert_to_8bit(image_4bit: np.ndarray) -> np.ndarray:
    """
    Convert 4-bit image (0-15) to 8-bit (0-255).
    Uses exact mapping: 0→0, 15→255, others scaled linearly.
    
    Args:
        image_4bit: 4-bit image array (0-15)
        
    Returns:
        8-bit image array (0-255)
    """
    # Scale from 0-15 to 0-255
    # Using: pixel_8bit = (pixel_4bit * 255) / 15
    image_8bit = np.round((image_4bit.astype(np.float32) / 15.0) * 255.0).astype(np.uint8)
    return np.clip(image_8bit, 0, 255)


def save_as_bmp(image_8bit: np.ndarray, filename: str):
    """
    Save 8-bit grayscale image as BMP file.
    
    Args:
        image_8bit: 8-bit image array (0-255)
        filename: Output BMP filename
    """
    img = Image.fromarray(image_8bit, mode='L')
    img.save(filename)


def process_frame(pgm_path: str, output_gt_path: str, output_noisy_path: str, 
                  noise_level: float = 0.15) -> bool:
    """
    Process a single frame: read PGM, convert to 4-bit, add noise, save as BMP.
    
    Args:
        pgm_path: Input PGM file path
        output_gt_path: Output path for ground truth BMP
        output_noisy_path: Output path for noisy BMP
        noise_level: Salt-and-pepper noise level (0.0-1.0)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read PGM file
        image_8bit, width, height = read_pgm(pgm_path)
        
        # Verify dimensions
        if width != W or height != H:
            print(f"Warning: {pgm_path} has dimensions {width}x{height}, expected {W}x{H}")
        
        # Convert to 4-bit
        image_4bit = convert_to_4bit(image_8bit)
        
        # Save ground truth (convert back to 8-bit for BMP)
        image_gt_8bit = convert_to_8bit(image_4bit)
        save_as_bmp(image_gt_8bit, output_gt_path)
        
        # Add salt-and-pepper noise in 4-bit space
        image_noisy_4bit = add_salt_pepper_noise(image_4bit, noise_level)
        
        # Convert noisy image to 8-bit for BMP
        image_noisy_8bit = convert_to_8bit(image_noisy_4bit)
        save_as_bmp(image_noisy_8bit, output_noisy_path)
        
        return True
        
    except Exception as e:
        print(f"Error processing {pgm_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def get_frame_number(filename: str) -> int:
    """Extract frame number from filename like 'frame_00001.pgm'."""
    try:
        # Extract number from filename
        base = Path(filename).stem  # 'frame_00001'
        num_str = base.split('_')[-1]  # '00001'
        return int(num_str)
    except:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Process video frames: convert PGM to BMP, add noise, and optionally denoise',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all frames with 15% noise
  python3 process_video_frames.py --noise 0.15
  
  # Process frames 1-50
  python3 process_video_frames.py --noise 0.15 --start 1 --end 50
  
  # Process and denoise
  python3 process_video_frames.py --noise 0.15 --denoise
        """
    )
    
    parser.add_argument('--input-dir', type=str, default='input_video',
                       help='Input directory with PGM files (default: input_video)')
    parser.add_argument('--noise', type=float, default=0.15,
                       help='Salt-and-pepper noise level (0.0-1.0, default: 0.15)')
    parser.add_argument('--start', type=int, default=None,
                       help='Start frame number (default: first frame)')
    parser.add_argument('--end', type=int, default=None,
                       help='End frame number (default: last frame)')
    parser.add_argument('--denoise', action='store_true',
                       help='Process noisy frames through median filter')
    parser.add_argument('--gt-dir', type=str, default='ground_truth_video',
                       help='Output directory for ground truth BMP files (default: ground_truth_video)')
    parser.add_argument('--noisy-dir', type=str, default='noisy_video',
                       help='Output directory for noisy BMP files (default: noisy_video)')
    parser.add_argument('--denoised-dir', type=str, default='denoised_video',
                       help='Output directory for denoised BMP files (default: denoised_video)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.gt_dir, exist_ok=True)
    os.makedirs(args.noisy_dir, exist_ok=True)
    if args.denoise:
        os.makedirs(args.denoised_dir, exist_ok=True)
    
    # Find all PGM files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return 1
    
    pgm_files = sorted(input_dir.glob('*.pgm'), key=get_frame_number)
    
    if not pgm_files:
        print(f"Error: No PGM files found in {input_dir}", file=sys.stderr)
        return 1
    
    print(f"Found {len(pgm_files)} PGM files in {input_dir}")
    
    # Filter by frame range if specified
    if args.start is not None or args.end is not None:
        filtered_files = []
        for f in pgm_files:
            frame_num = get_frame_number(str(f))
            if args.start is not None and frame_num < args.start:
                continue
            if args.end is not None and frame_num > args.end:
                continue
            filtered_files.append(f)
        pgm_files = filtered_files
        print(f"Processing frames {args.start or 'first'} to {args.end or 'last'}: {len(pgm_files)} files")
    
    # Process frames
    print(f"\nProcessing {len(pgm_files)} frames...")
    print(f"Noise level: {args.noise * 100:.1f}%")
    print(f"Output directories:")
    print(f"  Ground truth: {args.gt_dir}/")
    print(f"  Noisy: {args.noisy_dir}/")
    if args.denoise:
        print(f"  Denoised: {args.denoised_dir}/")
    print()
    
    successful = 0
    failed = 0
    
    for i, pgm_file in enumerate(pgm_files, 1):
        frame_num = get_frame_number(str(pgm_file))
        
        # Generate output filenames
        gt_filename = f"frame_{frame_num:05d}.bmp"
        noisy_filename = f"frame_{frame_num:05d}.bmp"
        denoised_filename = f"frame_{frame_num:05d}.bmp"
        
        gt_path = os.path.join(args.gt_dir, gt_filename)
        noisy_path = os.path.join(args.noisy_dir, noisy_filename)
        
        # Process frame
        if process_frame(str(pgm_file), gt_path, noisy_path, args.noise):
            successful += 1
            if i % 10 == 0 or i == len(pgm_files):
                print(f"  [{i}/{len(pgm_files)}] Processed frame {frame_num:05d}")
        else:
            failed += 1
            print(f"  [{i}/{len(pgm_files)}] Failed frame {frame_num:05d}")
    
    print(f"\n✓ Processing complete: {successful} successful, {failed} failed")
    
    # Denoise if requested
    if args.denoise and successful > 0:
        print(f"\nDenoising {successful} frames...")
        try:
            from median_filter_gpu import MedianFilterGPU, BitmapWrapper
            
            filter_gpu = MedianFilterGPU()
            filter_gpu.enable_logging(False)  # Cleaner output
            
            denoised_successful = 0
            total_energy = 0.0
            
            for i, pgm_file in enumerate(pgm_files, 1):
                frame_num = get_frame_number(str(pgm_file))
                noisy_path = os.path.join(args.noisy_dir, f"frame_{frame_num:05d}.bmp")
                denoised_path = os.path.join(args.denoised_dir, f"frame_{frame_num:05d}.bmp")
                
                if not os.path.exists(noisy_path):
                    continue
                
                try:
                    # Load noisy image
                    noisy_img = BitmapWrapper.load(filter_gpu.lib, noisy_path)
                    
                    # Apply median filter
                    denoised_img, timing = filter_gpu.filter(
                        noisy_img,
                        use_shared_memory=False,
                        collect_diagnostics=True
                    )
                    
                    # Save denoised result
                    denoised_img.save(denoised_path)
                    
                    # Accumulate energy
                    if 'energy' in timing:
                        total_energy += timing['energy']['total_energy_joules']
                    
                    denoised_successful += 1
                    
                    if i % 10 == 0 or i == len(pgm_files):
                        print(f"  [{i}/{len(pgm_files)}] Denoised frame {frame_num:05d}")
                        
                except Exception as e:
                    print(f"  Error denoising frame {frame_num:05d}: {e}", file=sys.stderr)
            
            print(f"\n✓ Denoising complete: {denoised_successful} frames processed")
            if total_energy > 0:
                print(f"  Total energy consumed: {total_energy * 1000:.3f} mJ")
                print(f"  Average energy per frame: {total_energy * 1000 / denoised_successful:.3f} mJ")
                
        except ImportError as e:
            print(f"Error: Could not import median filter wrapper: {e}", file=sys.stderr)
            print("Skipping denoising step", file=sys.stderr)
        except Exception as e:
            print(f"Error during denoising: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ All processing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
