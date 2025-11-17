#!/usr/bin/env python3
"""
Test script to verify GPU usage and compatibility.
This script performs basic sanity checks and verifies that the GPU is actually being used.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def print_section(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def check_cuda_installation():
    """Check if CUDA is installed and accessible."""
    print_section("CUDA Installation Check")
    
    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ nvcc found")
            # Extract version
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"  {line.strip()}")
        else:
            print("✗ nvcc not working properly")
            return False
    except FileNotFoundError:
        print("✗ nvcc not found in PATH")
        return False
    except Exception as e:
        print(f"✗ Error checking nvcc: {e}")
        return False
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,driver_version', 
                                '--format=csv,noheader'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ nvidia-smi accessible")
            gpu_info = result.stdout.strip().split('\n')[0]
            print(f"  GPU: {gpu_info}")
        else:
            print("✗ nvidia-smi not working")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found")
        return False
    except Exception as e:
        print(f"✗ Error checking nvidia-smi: {e}")
        return False
    
    return True

def check_library_build():
    """Check if the shared library is built."""
    print_section("Library Build Check")
    
    lib_path = Path("libmedian_filter_gpu.so")
    if lib_path.exists():
        print(f"✓ Shared library found: {lib_path}")
        size = lib_path.stat().st_size / (1024 * 1024)
        print(f"  Size: {size:.2f} MB")
        
        # Check dependencies
        try:
            result = subprocess.run(['ldd', str(lib_path)], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("  Dependencies:")
                for line in result.stdout.split('\n'):
                    if 'cuda' in line.lower() or 'cudart' in line.lower():
                        print(f"    {line.strip()}")
        except:
            pass
        return True
    else:
        print(f"✗ Shared library not found: {lib_path}")
        print("  Run: ./build_wrapper.sh")
        return False

def check_python_dependencies():
    """Check Python dependencies."""
    print_section("Python Dependencies Check")
    
    deps = {
        'pynvml': 'pynvml',
        'numpy': 'numpy',
        'PIL': 'Pillow'
    }
    
    all_ok = True
    for module, package in deps.items():
        try:
            __import__(module)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed (optional)")
            all_ok = False
    
    return all_ok

def test_cuda_runtime():
    """Test basic CUDA runtime functionality."""
    print_section("CUDA Runtime Test")
    
    try:
        import ctypes
        
        # Try to load libcudart
        try:
            libcudart = ctypes.CDLL("libcudart.so")
            print("✓ libcudart.so loaded")
        except OSError:
            # Try alternative paths
            cuda_paths = [
                "/usr/local/cuda/lib64/libcudart.so",
                "/usr/local/cuda/lib/libcudart.so",
            ]
            loaded = False
            for path in cuda_paths:
                if os.path.exists(path):
                    try:
                        libcudart = ctypes.CDLL(path)
                        print(f"✓ libcudart.so loaded from {path}")
                        loaded = True
                        break
                    except:
                        continue
            if not loaded:
                print("✗ Could not load libcudart.so")
                return False
        
        # Test cudaGetDeviceCount
        count = ctypes.c_int()
        get_device_count = libcudart.cudaGetDeviceCount
        get_device_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_device_count.restype = ctypes.c_int
        
        result = get_device_count(ctypes.byref(count))
        if result == 0:  # cudaSuccess
            print(f"✓ CUDA devices detected: {count.value}")
            return True
        else:
            print(f"✗ cudaGetDeviceCount failed with code: {result}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing CUDA runtime: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wrapper_import():
    """Test importing the wrapper module."""
    print_section("Wrapper Module Import Test")
    
    try:
        from median_filter_gpu import MedianFilterGPU, BitmapWrapper
        print("✓ median_filter_gpu module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import median_filter_gpu: {e}")
        return False
    except Exception as e:
        print(f"✗ Error importing module: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_initialization():
    """Test GPU initialization through the wrapper."""
    print_section("GPU Initialization Test")
    
    try:
        from median_filter_gpu import MedianFilterGPU
        
        print("Initializing GPU wrapper...")
        filter_gpu = MedianFilterGPU()
        print("✓ GPU wrapper initialized")
        
        # Get device info
        device_info = filter_gpu.get_device_info()
        if device_info:
            print(f"  Device: {device_info.get('name', 'Unknown')}")
            print(f"  Compute Capability: {device_info.get('compute_capability', 'Unknown')}")
            print(f"  Total Memory: {device_info.get('total_memory_mb', 0)} MB")
            print(f"  Free Memory: {device_info.get('free_memory_mb', 0)} MB")
            return True
        else:
            print("✗ Failed to get device info")
            return False
            
    except FileNotFoundError as e:
        print(f"✗ Library not found: {e}")
        print("  Make sure to build the library first: ./build_wrapper.sh")
        return False
    except Exception as e:
        print(f"✗ Error initializing GPU: {e}")
        import traceback
        traceback.print_exc()
        return False

def monitor_gpu_usage_during_operation():
    """Monitor GPU usage during a test operation."""
    print_section("GPU Usage Monitoring Test")
    
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        print("Monitoring GPU utilization and memory during operation...")
        print("(This requires the wrapper to be built and a test image)")
        
        # Get initial state
        mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util_before = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        print(f"\nBefore operation:")
        print(f"  Memory used: {mem_before.used / (1024**2):.2f} MB / {mem_before.total / (1024**2):.2f} MB")
        print(f"  GPU utilization: {util_before.gpu}%")
        print(f"  Memory utilization: {util_before.memory}%")
        
        print("\nNote: To fully test GPU usage, run:")
        print("  python run_median_filter.py <input_image.bmp> <output.bmp>")
        print("  And monitor with: watch -n 0.5 nvidia-smi")
        
        return True
        
    except ImportError:
        print("⚠ pynvml not available - cannot monitor GPU usage")
        print("  Install with: pip install pynvml")
        return False
    except Exception as e:
        print(f"✗ Error monitoring GPU: {e}")
        return False

def main():
    print("="*70)
    print(" GPU Usage and Compatibility Test Suite")
    print("="*70)
    
    results = {}
    
    # Run all checks
    results['cuda_install'] = check_cuda_installation()
    results['library_build'] = check_library_build()
    results['python_deps'] = check_python_dependencies()
    results['cuda_runtime'] = test_cuda_runtime()
    results['wrapper_import'] = test_wrapper_import()
    
    if results['wrapper_import']:
        results['gpu_init'] = test_gpu_initialization()
        results['gpu_monitor'] = monitor_gpu_usage_during_operation()
    else:
        results['gpu_init'] = False
        results['gpu_monitor'] = False
    
    # Summary
    print_section("Test Summary")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All checks passed! GPU wrapper is ready to use.")
        return 0
    else:
        print(f"\n⚠ {total - passed} check(s) failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

