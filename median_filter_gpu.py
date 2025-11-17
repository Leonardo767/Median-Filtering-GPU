"""
Python Wrapper for CUDA Median Filter
Provides a Python interface to the CUDA median filter implementation
with comprehensive GPU diagnostics.
"""

import ctypes
import os
import sys
from typing import Optional, Tuple, Dict
from pathlib import Path

# Import diagnostics module
try:
    from gpu_diagnostics import GPUDiagnostics, measure_memory_delta
except ImportError:
    print("Warning: gpu_diagnostics module not found. Diagnostics will be limited.", file=sys.stderr)
    GPUDiagnostics = None
    measure_memory_delta = None


# Find and load the shared library
def _find_library():
    """Find the shared library in common locations."""
    script_dir = Path(__file__).parent.absolute()
    possible_paths = [
        script_dir / "libmedian_filter_gpu.so",
        script_dir / ".." / "libmedian_filter_gpu.so",
        Path("/usr/local/lib/libmedian_filter_gpu.so"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # Try loading by name (will use LD_LIBRARY_PATH)
    return "libmedian_filter_gpu.so"


class TimingInfo(ctypes.Structure):
    """Structure to hold CUDA timing information."""
    _fields_ = [
        ("total_time_ms", ctypes.c_float),
        ("h2d_time_ms", ctypes.c_float),
        ("kernel_time_ms", ctypes.c_float),
        ("d2h_time_ms", ctypes.c_float),
        ("success", ctypes.c_int),
    ]


class BitmapWrapper:
    """Python wrapper for the C Bitmap class."""
    
    def __init__(self, lib, handle=None):
        """
        Initialize Bitmap wrapper.
        
        Args:
            lib: Loaded ctypes library
            handle: Optional existing bitmap handle
        """
        self.lib = lib
        self._handle = handle
        self._width = None
        self._height = None
    
    @classmethod
    def create(cls, lib, width: int, height: int):
        """Create a new empty bitmap."""
        handle = lib.create_bitmap(width, height)
        if not handle:
            raise RuntimeError("Failed to create bitmap")
        obj = cls(lib, handle)
        obj._width = width
        obj._height = height
        return obj
    
    @classmethod
    def load(cls, lib, filename: str):
        """Load bitmap from file."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Image file not found: {filename}")
        
        handle = lib.load_bitmap(filename.encode('utf-8'))
        if not handle:
            raise RuntimeError(f"Failed to load bitmap from {filename}")
        
        obj = cls(lib, handle)
        obj._width = lib.get_bitmap_width(handle)
        obj._height = lib.get_bitmap_height(handle)
        return obj
    
    def save(self, filename: str) -> bool:
        """Save bitmap to file."""
        if not self._handle:
            raise RuntimeError("Bitmap not initialized")
        
        result = self.lib.save_bitmap(self._handle, filename.encode('utf-8'))
        return result == 1
    
    @property
    def width(self) -> int:
        """Get bitmap width."""
        if self._width is None and self._handle:
            self._width = self.lib.get_bitmap_width(self._handle)
        return self._width
    
    @property
    def height(self) -> int:
        """Get bitmap height."""
        if self._height is None and self._handle:
            self._height = self.lib.get_bitmap_height(self._handle)
        return self._height
    
    @property
    def handle(self):
        """Get the internal C handle."""
        return self._handle
    
    def get_data(self) -> Optional[ctypes.Array]:
        """Get pointer to image data."""
        if not self._handle:
            return None
        
        data_ptr = self.lib.get_bitmap_data(self._handle)
        if not data_ptr:
            return None
        
        size = self.width * self.height
        # Create a ctypes array view of the data
        # data_ptr is already a pointer to the first element
        return (ctypes.c_ubyte * size).from_address(ctypes.cast(data_ptr, ctypes.c_void_p).value)
    
    def __del__(self):
        """Cleanup bitmap on destruction."""
        if self._handle and self.lib:
            try:
                self.lib.destroy_bitmap(self._handle)
            except:
                pass


class MedianFilterGPU:
    """Main class for GPU median filtering with diagnostics."""
    
    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize the GPU median filter wrapper.
        
        Args:
            library_path: Optional path to shared library. If None, will search.
        """
        self.lib = None
        self.diagnostics = None
        self._load_library(library_path)
        self._init_cuda()
        self._init_diagnostics()
    
    def _load_library(self, library_path: Optional[str] = None):
        """Load the shared library."""
        if library_path is None:
            library_path = _find_library()
        
        if not os.path.exists(library_path) and not library_path.startswith("lib"):
            raise FileNotFoundError(
                f"Shared library not found: {library_path}\n"
                "Please run build_wrapper.sh to build the library first."
            )
        
        try:
            self.lib = ctypes.CDLL(library_path)
        except OSError as e:
            raise RuntimeError(
                f"Failed to load shared library: {e}\n"
                "Make sure CUDA libraries are in LD_LIBRARY_PATH"
            )
        
        # Define function signatures
        self._setup_function_signatures()
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures for all C functions."""
        # Bitmap functions
        self.lib.create_bitmap.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.create_bitmap.restype = ctypes.c_void_p
        
        self.lib.load_bitmap.argtypes = [ctypes.c_char_p]
        self.lib.load_bitmap.restype = ctypes.c_void_p
        
        self.lib.save_bitmap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.save_bitmap.restype = ctypes.c_int
        
        self.lib.get_bitmap_width.argtypes = [ctypes.c_void_p]
        self.lib.get_bitmap_width.restype = ctypes.c_int
        
        self.lib.get_bitmap_height.argtypes = [ctypes.c_void_p]
        self.lib.get_bitmap_height.restype = ctypes.c_int
        
        self.lib.get_bitmap_data.argtypes = [ctypes.c_void_p]
        self.lib.get_bitmap_data.restype = ctypes.POINTER(ctypes.c_ubyte)
        
        self.lib.get_bitmap_size.argtypes = [ctypes.c_void_p]
        self.lib.get_bitmap_size.restype = ctypes.c_int
        
        self.lib.destroy_bitmap.argtypes = [ctypes.c_void_p]
        self.lib.destroy_bitmap.restype = None
        
        # GPU filter function
        self.lib.median_filter_gpu_wrapper.argtypes = [
            ctypes.c_void_p,  # input_handle
            ctypes.c_void_p,  # output_handle
            ctypes.c_int,     # use_shared
            ctypes.POINTER(TimingInfo)  # timing_info
        ]
        self.lib.median_filter_gpu_wrapper.restype = ctypes.c_int
        
        # CUDA utility functions
        self.lib.init_cuda.argtypes = []
        self.lib.init_cuda.restype = ctypes.c_int
        
        self.lib.get_cuda_error_string.argtypes = []
        self.lib.get_cuda_error_string.restype = ctypes.c_char_p
        
        # Logging control functions
        self.lib.enable_wrapper_logging.argtypes = [ctypes.c_int]
        self.lib.enable_wrapper_logging.restype = None
        
        self.lib.set_wrapper_log_file.argtypes = [ctypes.c_char_p]
        self.lib.set_wrapper_log_file.restype = None
        
        # Device info function
        self.lib.get_cuda_device_info.argtypes = [
            ctypes.POINTER(ctypes.c_char),  # name
            ctypes.c_int,                   # name_len
            ctypes.POINTER(ctypes.c_int),  # compute_major
            ctypes.POINTER(ctypes.c_int),  # compute_minor
            ctypes.POINTER(ctypes.c_size_t), # total_memory
            ctypes.POINTER(ctypes.c_size_t)  # free_memory
        ]
        self.lib.get_cuda_device_info.restype = ctypes.c_int
    
    def _init_cuda(self):
        """Initialize CUDA."""
        if not self.lib.init_cuda():
            error_str = self.lib.get_cuda_error_string()
            if error_str:
                error_msg = error_str.decode('utf-8')
            else:
                error_msg = "Unknown CUDA error"
            raise RuntimeError(f"Failed to initialize CUDA: {error_msg}")
    
    def _init_diagnostics(self):
        """Initialize GPU diagnostics."""
        if GPUDiagnostics:
            try:
                self.diagnostics = GPUDiagnostics()
            except Exception as e:
                print(f"Warning: Could not initialize GPU diagnostics: {e}", file=sys.stderr)
                self.diagnostics = None
    
    def enable_logging(self, enable: bool = True, log_file: Optional[str] = None):
        """
        Enable or disable wrapper logging.
        
        Args:
            enable: Whether to enable logging
            log_file: Optional log file path (default: stderr)
        """
        self.lib.enable_wrapper_logging(1 if enable else 0)
        if log_file:
            self.lib.set_wrapper_log_file(log_file.encode('utf-8'))
        else:
            self.lib.set_wrapper_log_file(None)
    
    def get_device_info(self) -> Dict:
        """Get CUDA device information."""
        name_buf = ctypes.create_string_buffer(256)
        compute_major = ctypes.c_int()
        compute_minor = ctypes.c_int()
        total_mem = ctypes.c_size_t()
        free_mem = ctypes.c_size_t()
        
        if self.lib.get_cuda_device_info(name_buf, 256, 
                                         ctypes.byref(compute_major),
                                         ctypes.byref(compute_minor),
                                         ctypes.byref(total_mem),
                                         ctypes.byref(free_mem)):
            return {
                'name': name_buf.value.decode('utf-8'),
                'compute_capability': f"{compute_major.value}.{compute_minor.value}",
                'total_memory_mb': total_mem.value // (1024 * 1024),
                'free_memory_mb': free_mem.value // (1024 * 1024)
            }
        else:
            return {}
    
    def filter(self, input_image: BitmapWrapper, use_shared_memory: bool = False,
               collect_diagnostics: bool = True) -> Tuple[BitmapWrapper, Dict]:
        """
        Apply median filter to an image.
        
        Args:
            input_image: Input bitmap
            use_shared_memory: Whether to use shared memory kernel
            collect_diagnostics: Whether to collect GPU diagnostics
        
        Returns:
            Tuple of (output_bitmap, timing_info_dict)
        """
        # Create output bitmap
        output_image = BitmapWrapper.create(self.lib, input_image.width, input_image.height)
        
        # Collect diagnostics before operation
        stats_before = None
        if collect_diagnostics and self.diagnostics:
            stats_before = self.diagnostics.get_comprehensive_stats()
        
        # Prepare timing info structure
        timing_info = TimingInfo()
        
        # Measure memory delta if diagnostics available
        if collect_diagnostics and self.diagnostics:
            with measure_memory_delta(self.diagnostics, "Median Filter"):
                # Run the filter
                result = self.lib.median_filter_gpu_wrapper(
                    input_image.handle,
                    output_image.handle,
                    1 if use_shared_memory else 0,
                    ctypes.byref(timing_info)
                )
        else:
            # Run the filter without memory tracking
            result = self.lib.median_filter_gpu_wrapper(
                input_image.handle,
                output_image.handle,
                1 if use_shared_memory else 0,
                ctypes.byref(timing_info)
            )
        
        if not result:
            error_str = self.lib.get_cuda_error_string()
            if error_str:
                error_msg = error_str.decode('utf-8')
            else:
                error_msg = "Unknown error"
            raise RuntimeError(f"GPU median filter failed: {error_msg}")
        
        # Collect diagnostics after operation
        stats_after = None
        if collect_diagnostics and self.diagnostics:
            stats_after = self.diagnostics.get_comprehensive_stats()
        
        # Build timing info dictionary
        timing_dict = {
            'total_time_ms': timing_info.total_time_ms,
            'h2d_time_ms': timing_info.h2d_time_ms,
            'kernel_time_ms': timing_info.kernel_time_ms,
            'd2h_time_ms': timing_info.d2h_time_ms,
            'success': bool(timing_info.success)
        }
        
        # Add diagnostics if collected
        if stats_before and stats_after:
            timing_dict['diagnostics'] = {
                'before': stats_before,
                'after': stats_after
            }
        
        return output_image, timing_dict
    
    def print_timing_info(self, timing_dict: Dict):
        """Print timing information in a formatted way."""
        print("\n" + "="*60)
        print("CUDA TIMING INFORMATION")
        print("="*60)
        print(f"Total Time:        {timing_dict['total_time_ms']:.4f} ms")
        print(f"Host-to-Device:    {timing_dict['h2d_time_ms']:.4f} ms")
        print(f"Kernel Execution:  {timing_dict['kernel_time_ms']:.4f} ms")
        print(f"Device-to-Host:    {timing_dict['d2h_time_ms']:.4f} ms")
        print("="*60 + "\n")
        
        if 'diagnostics' in timing_dict:
            print("GPU Diagnostics Before:")
            if self.diagnostics:
                self.diagnostics.print_stats(timing_dict['diagnostics']['before'])
            print("\nGPU Diagnostics After:")
            if self.diagnostics:
                self.diagnostics.print_stats(timing_dict['diagnostics']['after'])


# Convenience function for quick filtering
def filter_image(input_path: str, output_path: str, 
                 use_shared_memory: bool = False,
                 library_path: Optional[str] = None) -> Dict:
    """
    Convenience function to filter an image.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image
        use_shared_memory: Whether to use shared memory kernel
        library_path: Optional path to shared library
    
    Returns:
        Dictionary with timing information
    """
    filter_gpu = MedianFilterGPU(library_path)
    input_img = BitmapWrapper.load(filter_gpu.lib, input_path)
    output_img, timing = filter_gpu.filter(input_img, use_shared_memory)
    output_img.save(output_path)
    return timing

