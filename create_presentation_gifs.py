#!/usr/bin/env python3
"""
Create presentation GIFs: ground truth, denoised, and side-by-side comparison.

Generates animated GIFs from video frames for presentation purposes.
"""

import sys
import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

def get_frame_number(filename: str) -> int:
    """Extract frame number from filename like 'frame_00001.bmp'."""
    try:
        base = Path(filename).stem  # 'frame_00001'
        num_str = base.split('_')[-1]  # '00001'
        return int(num_str)
    except:
        return 0

def create_gif_from_frames(frame_dir: str, output_path: str, 
                          start_frame: int = None, end_frame: int = None,
                          duration: int = 100, loop: int = 0):
    """
    Create a GIF from BMP frames in a directory.
    
    Args:
        frame_dir: Directory containing BMP frames
        output_path: Output GIF file path
        start_frame: First frame number (None for all)
        end_frame: Last frame number (None for all)
        duration: Frame duration in milliseconds
        loop: Number of loops (0 = infinite)
    """
    frame_dir = Path(frame_dir)
    if not frame_dir.exists():
        raise FileNotFoundError(f"Directory not found: {frame_dir}")
    
    # Find all BMP files
    bmp_files = sorted(frame_dir.glob('*.bmp'), key=lambda f: get_frame_number(str(f)))
    
    if not bmp_files:
        raise ValueError(f"No BMP files found in {frame_dir}")
    
    # Filter by frame range
    if start_frame is not None or end_frame is not None:
        filtered_files = []
        for f in bmp_files:
            frame_num = get_frame_number(str(f))
            if start_frame is not None and frame_num < start_frame:
                continue
            if end_frame is not None and frame_num > end_frame:
                continue
            filtered_files.append(f)
        bmp_files = filtered_files
    
    if not bmp_files:
        raise ValueError(f"No frames found in range {start_frame} to {end_frame}")
    
    print(f"  Loading {len(bmp_files)} frames from {frame_dir}...")
    
    # Load all frames
    frames = []
    for bmp_file in bmp_files:
        img = Image.open(bmp_file)
        # Convert to RGB if needed (GIFs need RGB or palette mode)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        frames.append(img)
    
    # Save as GIF
    print(f"  Creating GIF: {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    
    print(f"  ✓ Created GIF: {output_path} ({len(frames)} frames, {duration}ms per frame)")
    return len(frames)

def create_side_by_side_gif(gt_dir: str, noisy_dir: str, denoised_dir: str, output_path: str,
                            start_frame: int = None, end_frame: int = None,
                            duration: int = 100, loop: int = 0, gap: int = 10):
    """
    Create a side-by-side comparison GIF with three videos: ground truth, noisy, and denoised.
    
    Args:
        gt_dir: Ground truth frames directory
        noisy_dir: Noisy frames directory
        denoised_dir: Denoised frames directory
        output_path: Output GIF file path
        start_frame: First frame number (None for all)
        end_frame: Last frame number (None for all)
        duration: Frame duration in milliseconds
        loop: Number of loops (0 = infinite)
        gap: Gap between images in pixels
    """
    gt_dir = Path(gt_dir)
    noisy_dir = Path(noisy_dir)
    denoised_dir = Path(denoised_dir)
    
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    if not noisy_dir.exists():
        raise FileNotFoundError(f"Noisy directory not found: {noisy_dir}")
    if not denoised_dir.exists():
        raise FileNotFoundError(f"Denoised directory not found: {denoised_dir}")
    
    # Find all BMP files (use ground truth as reference)
    gt_files = sorted(gt_dir.glob('*.bmp'), key=lambda f: get_frame_number(str(f)))
    
    if not gt_files:
        raise ValueError(f"No BMP files found in {gt_dir}")
    
    # Filter by frame range
    if start_frame is not None or end_frame is not None:
        filtered_files = []
        for f in gt_files:
            frame_num = get_frame_number(str(f))
            if start_frame is not None and frame_num < start_frame:
                continue
            if end_frame is not None and frame_num > end_frame:
                continue
            filtered_files.append(f)
        gt_files = filtered_files
    
    if not gt_files:
        raise ValueError(f"No frames found in range {start_frame} to {end_frame}")
    
    print(f"  Loading {len(gt_files)} frame triplets...")
    
    # Load and combine frames
    combined_frames = []
    for gt_file in gt_files:
        frame_num = get_frame_number(str(gt_file))
        noisy_file = noisy_dir / f"frame_{frame_num:05d}.bmp"
        denoised_file = denoised_dir / f"frame_{frame_num:05d}.bmp"
        
        if not noisy_file.exists():
            print(f"  Warning: Noisy frame {frame_num:05d} not found, skipping...")
            continue
        if not denoised_file.exists():
            print(f"  Warning: Denoised frame {frame_num:05d} not found, skipping...")
            continue
        
        # Load all three images
        gt_img = Image.open(gt_file)
        noisy_img = Image.open(noisy_file)
        denoised_img = Image.open(denoised_file)
        
        # Convert to RGB if needed
        if gt_img.mode != 'RGB':
            gt_img = gt_img.convert('RGB')
        if noisy_img.mode != 'RGB':
            noisy_img = noisy_img.convert('RGB')
        if denoised_img.mode != 'RGB':
            denoised_img = denoised_img.convert('RGB')
        
        # Get dimensions (assuming all same size)
        img_w, img_h = gt_img.size
        
        # Create combined image (three images side by side)
        if gap > 0:
            combined_w = img_w * 3 + gap * 2
        else:
            combined_w = img_w * 3
        combined_h = img_h
        
        combined_img = Image.new('RGB', (combined_w, combined_h), color='white')
        
        # Paste images side by side
        combined_img.paste(gt_img, (0, 0))
        combined_img.paste(noisy_img, (img_w + gap, 0))
        combined_img.paste(denoised_img, (img_w * 2 + gap * 2, 0))
        
        # Add text labels (optional - using PIL's ImageDraw)
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(combined_img)
            
            # Try to use a nice font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
                except:
                    font = ImageFont.load_default()
            
            # Add labels at the top
            label_y = 5
            draw.text((10, label_y), "Ground Truth", fill='black', font=font)
            draw.text((img_w + gap + 10, label_y), "Noisy", fill='black', font=font)
            draw.text((img_w * 2 + gap * 2 + 10, label_y), "Denoised", fill='black', font=font)
        except:
            # If ImageDraw fails, continue without labels
            pass
        
        combined_frames.append(combined_img)
    
    if not combined_frames:
        raise ValueError("No valid frame triplets found")
    
    # Save as GIF
    print(f"  Creating side-by-side GIF: {output_path}...")
    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    
    print(f"  ✓ Created side-by-side GIF: {output_path} ({len(combined_frames)} frames, {duration}ms per frame)")
    return len(combined_frames)

def main():
    parser = argparse.ArgumentParser(
        description='Create presentation GIFs from video frames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all GIFs with default settings
  python3 create_presentation_gifs.py
  
  # Create GIFs for frames 1-50
  python3 create_presentation_gifs.py --start 1 --end 50
  
  # Faster animation (50ms per frame)
  python3 create_presentation_gifs.py --duration 50
        """
    )
    
    parser.add_argument('--gt-dir', type=str, default='ground_truth_video',
                       help='Ground truth frames directory (default: ground_truth_video)')
    parser.add_argument('--noisy-dir', type=str, default='noisy_video',
                       help='Noisy frames directory (default: noisy_video)')
    parser.add_argument('--denoised-dir', type=str, default='denoised_video',
                       help='Denoised frames directory (default: denoised_video)')
    parser.add_argument('--output-dir', type=str, default='experiment_result',
                       help='Output directory for GIFs (default: experiment_result)')
    parser.add_argument('--start', type=int, default=None,
                       help='Start frame number (default: all frames)')
    parser.add_argument('--end', type=int, default=None,
                       help='End frame number (default: all frames)')
    parser.add_argument('--duration', type=int, default=100,
                       help='Frame duration in milliseconds (default: 100)')
    parser.add_argument('--loop', type=int, default=0,
                       help='Number of loops (0 = infinite, default: 0)')
    parser.add_argument('--gap', type=int, default=10,
                       help='Gap between images in side-by-side GIF (default: 10 pixels)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print(" Creating Presentation GIFs")
    print("="*70)
    print(f"Output directory: {args.output_dir}/")
    print(f"Frame range: {args.start or 'first'} to {args.end or 'last'}")
    print(f"Frame duration: {args.duration} ms")
    print()
    
    try:
        # Create ground truth GIF
        print("1. Creating ground truth GIF...")
        gt_gif_path = os.path.join(args.output_dir, "ground_truth.gif")
        gt_frame_count = create_gif_from_frames(
            args.gt_dir, gt_gif_path,
            args.start, args.end,
            args.duration, args.loop
        )
        print()
        
        # Create noisy GIF
        print("2. Creating noisy GIF...")
        noisy_gif_path = os.path.join(args.output_dir, "noisy.gif")
        noisy_frame_count = create_gif_from_frames(
            args.noisy_dir, noisy_gif_path,
            args.start, args.end,
            args.duration, args.loop
        )
        print()
        
        # Create denoised GIF
        print("3. Creating denoised GIF...")
        denoised_gif_path = os.path.join(args.output_dir, "denoised.gif")
        denoised_frame_count = create_gif_from_frames(
            args.denoised_dir, denoised_gif_path,
            args.start, args.end,
            args.duration, args.loop
        )
        print()
        
        # Create side-by-side comparison GIF
        print("4. Creating side-by-side comparison GIF...")
        comparison_gif_path = os.path.join(args.output_dir, "comparison.gif")
        comparison_frame_count = create_side_by_side_gif(
            args.gt_dir, args.noisy_dir, args.denoised_dir, comparison_gif_path,
            args.start, args.end,
            args.duration, args.loop, args.gap
        )
        print()
        
        # Summary
        print("="*70)
        print(" Summary")
        print("="*70)
        print(f"Ground truth GIF:  {gt_gif_path} ({gt_frame_count} frames)")
        print(f"Noisy GIF:         {noisy_gif_path} ({noisy_frame_count} frames)")
        print(f"Denoised GIF:      {denoised_gif_path} ({denoised_frame_count} frames)")
        print(f"Comparison GIF:    {comparison_gif_path} ({comparison_frame_count} frames)")
        print("="*70)
        print("✓ All GIFs created successfully!")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

