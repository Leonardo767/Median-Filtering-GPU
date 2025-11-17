#!/usr/bin/env python3
"""
Create a simple test BMP image with salt-and-pepper noise for testing the median filter.
"""

import numpy as np
from PIL import Image
import sys

def create_test_image_with_noise(width=256, height=256, noise_prob=0.1, output_path="inputs/test_image.bmp"):
    """
    Create a test image with salt-and-pepper noise.
    
    Args:
        width: Image width
        height: Image height
        noise_prob: Probability of noise (0.0 to 1.0)
        output_path: Output file path
    """
    # Create a simple pattern (checkerboard or gradient)
    # Let's create a gradient pattern
    img_array = np.zeros((height, width), dtype=np.uint8)
    
    # Create a gradient pattern
    for y in range(height):
        for x in range(width):
            # Create a pattern: diagonal stripes
            value = ((x + y) % 64) * 4
            img_array[y, x] = min(255, value)
    
    # Add salt-and-pepper noise
    noise_mask = np.random.random((height, width)) < noise_prob
    salt_mask = np.random.random((height, width)) < (noise_prob / 2)
    pepper_mask = np.random.random((height, width)) < (noise_prob / 2)
    
    # Apply noise
    img_array[salt_mask] = 255  # Salt (white)
    img_array[pepper_mask] = 0   # Pepper (black)
    
    # Convert to PIL Image and save as BMP
    img = Image.fromarray(img_array, mode='L')  # 'L' mode is 8-bit grayscale
    img.save(output_path)
    
    print(f"Created test image: {output_path}")
    print(f"  Size: {width} x {height}")
    print(f"  Noise level: {noise_prob * 100:.1f}%")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a test BMP image with noise')
    parser.add_argument('--width', type=int, default=256, help='Image width')
    parser.add_argument('--height', type=int, default=256, help='Image height')
    parser.add_argument('--noise', type=float, default=0.1, help='Noise probability (0.0-1.0)')
    parser.add_argument('--output', type=str, default='inputs/test_image.bmp', help='Output file path')
    
    args = parser.parse_args()
    
    create_test_image_with_noise(args.width, args.height, args.noise, args.output)

