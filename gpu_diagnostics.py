"""
GPU Diagnostics Module
Provides utilities for monitoring GPU performance, memory, and utilization
using NVIDIA Management Library (NVML) via nvidia-ml-py (provides pynvml module).
"""

import time
from typing import Dict, Optional, Tuple
import sys

try:
    import pynvml  # Provided by nvidia-ml-py package
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: nvidia-ml-py (pynvml) not available. GPU diagnostics will be limited.", file=sys.stderr)


class GPUDiagnostics:
    """Class for collecting GPU diagnostics and performance metrics."""
    
    def __init__(self, device_id: int = 0):
        """
        Initialize GPU diagnostics.
        
        Args:
            device_id: GPU device ID (default: 0)
        """
        self.device_id = device_id
        self.initialized = False
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.initialized = True
            except Exception as e:
                print(f"Warning: Failed to initialize NVML: {e}", file=sys.stderr)
                self.initialized = False
        else:
            self.handle = None
    
    def __del__(self):
        """Cleanup NVML on destruction."""
        if NVML_AVAILABLE and self.initialized:
            try:
                # pynvml doesn't have a cleanup function in all versions
                pass
            except:
                pass
    
    def get_device_info(self) -> Dict:
        """
        Get basic GPU device information.
        
        Returns:
            Dictionary with device information
        """
        info = {
            'name': 'Unknown',
            'compute_capability': 'Unknown',
            'total_memory_mb': 0,
            'driver_version': 'Unknown',
            'cuda_version': 'Unknown'
        }
        
        if not self.initialized:
            return info
        
        try:
            # Get device name
            name = pynvml.nvmlDeviceGetName(self.handle)
            info['name'] = name.decode('utf-8') if isinstance(name, bytes) else name
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            info['total_memory_mb'] = mem_info.total // (1024 * 1024)
            
            # Get driver version
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion()
                info['driver_version'] = driver_version.decode('utf-8') if isinstance(driver_version, bytes) else driver_version
            except:
                pass
            
            # Get compute capability (if available)
            try:
                # This might not be available in all NVML versions
                pass
            except:
                pass
                
        except Exception as e:
            print(f"Error getting device info: {e}", file=sys.stderr)
        
        return info
    
    def get_memory_info(self) -> Tuple[int, int, int]:
        """
        Get current GPU memory usage.
        
        Returns:
            Tuple of (free_memory_bytes, used_memory_bytes, total_memory_bytes)
        """
        if not self.initialized:
            return (0, 0, 0)
        
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return (mem_info.free, mem_info.used, mem_info.total)
        except Exception as e:
            print(f"Error getting memory info: {e}", file=sys.stderr)
            return (0, 0, 0)
    
    def get_utilization(self) -> Dict[str, int]:
        """
        Get GPU utilization percentages.
        
        Returns:
            Dictionary with 'gpu' and 'memory' utilization percentages
        """
        if not self.initialized:
            return {'gpu': 0, 'memory': 0}
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return {
                'gpu': util.gpu,
                'memory': util.memory
            }
        except Exception as e:
            print(f"Error getting utilization: {e}", file=sys.stderr)
            return {'gpu': 0, 'memory': 0}
    
    def get_temperature(self) -> Optional[int]:
        """
        Get GPU temperature in Celsius.
        
        Returns:
            Temperature in Celsius, or None if unavailable
        """
        if not self.initialized:
            return None
        
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            return temp
        except Exception as e:
            print(f"Error getting temperature: {e}", file=sys.stderr)
            return None
    
    def get_power_usage(self) -> Optional[float]:
        """
        Get GPU power usage in watts.
        
        Returns:
            Power usage in watts, or None if unavailable
        """
        if not self.initialized:
            return None
        
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            # Power is returned in milliwatts, convert to watts
            return power / 1000.0
        except Exception as e:
            print(f"Error getting power usage: {e}", file=sys.stderr)
            return None
    
    def get_clock_speeds(self) -> Dict[str, int]:
        """
        Get GPU clock speeds.
        
        Returns:
            Dictionary with 'graphics' and 'memory' clock speeds in MHz
        """
        if not self.initialized:
            return {'graphics': 0, 'memory': 0}
        
        clocks = {'graphics': 0, 'memory': 0}
        try:
            graphics_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
            clocks['graphics'] = graphics_clock
        except:
            pass
        
        try:
            memory_clock = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            # Note: This doesn't actually get memory clock, but keeping structure
            # Memory clock requires different API call
            try:
                memory_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
                clocks['memory'] = memory_clock
            except:
                pass
        except:
            pass
        
        return clocks
    
    def get_comprehensive_stats(self) -> Dict:
        """
        Get comprehensive GPU statistics.
        
        Returns:
            Dictionary with all available GPU metrics
        """
        stats = {
            'device_info': self.get_device_info(),
            'memory': {},
            'utilization': {},
            'temperature': None,
            'power': None,
            'clocks': {}
        }
        
        # Memory info
        free, used, total = self.get_memory_info()
        stats['memory'] = {
            'free_mb': free // (1024 * 1024),
            'used_mb': used // (1024 * 1024),
            'total_mb': total // (1024 * 1024),
            'usage_percent': (used / total * 100) if total > 0 else 0
        }
        
        # Utilization
        stats['utilization'] = self.get_utilization()
        
        # Temperature
        stats['temperature'] = self.get_temperature()
        
        # Power
        stats['power'] = self.get_power_usage()
        
        # Clocks
        stats['clocks'] = self.get_clock_speeds()
        
        return stats
    
    def print_stats(self, stats: Optional[Dict] = None):
        """
        Print GPU statistics in a formatted way.
        
        Args:
            stats: Optional pre-computed stats dictionary. If None, will compute.
        """
        if stats is None:
            stats = self.get_comprehensive_stats()
        
        print("\n" + "="*60)
        print("GPU DIAGNOSTICS")
        print("="*60)
        
        # Device info
        dev_info = stats.get('device_info', {})
        print(f"Device: {dev_info.get('name', 'Unknown')}")
        print(f"Driver Version: {dev_info.get('driver_version', 'Unknown')}")
        
        # Memory
        mem = stats.get('memory', {})
        print(f"\nMemory:")
        print(f"  Total: {mem.get('total_mb', 0)} MB")
        print(f"  Used: {mem.get('used_mb', 0)} MB ({mem.get('usage_percent', 0):.1f}%)")
        print(f"  Free: {mem.get('free_mb', 0)} MB")
        
        # Utilization
        util = stats.get('utilization', {})
        print(f"\nUtilization:")
        print(f"  GPU: {util.get('gpu', 0)}%")
        print(f"  Memory: {util.get('memory', 0)}%")
        
        # Temperature
        temp = stats.get('temperature')
        if temp is not None:
            print(f"\nTemperature: {temp}°C")
        
        # Power
        power = stats.get('power')
        if power is not None:
            print(f"Power Usage: {power:.2f} W")
        
        # Clocks
        clocks = stats.get('clocks', {})
        if clocks.get('graphics', 0) > 0:
            print(f"\nClock Speeds:")
            print(f"  Graphics: {clocks.get('graphics', 0)} MHz")
            if clocks.get('memory', 0) > 0:
                print(f"  Memory: {clocks.get('memory', 0)} MHz")
        
        print("="*60 + "\n")


def measure_memory_delta(diagnostics: GPUDiagnostics, operation_name: str = "Operation"):
    """
    Context manager to measure memory usage before and after an operation.
    
    Usage:
        with measure_memory_delta(diag, "Kernel Launch"):
            # Your GPU operation here
            pass
    """
    class MemoryDelta:
        def __init__(self, diag, name):
            self.diag = diag
            self.name = name
            self.before = None
            self.after = None
        
        def __enter__(self):
            free, used, total = self.diag.get_memory_info()
            self.before = used
            return self
        
        def __exit__(self, *args):
            free, used, total = self.diag.get_memory_info()
            self.after = used
            delta = self.after - self.before
            print(f"{self.name} - Memory Delta: {delta / (1024*1024):.2f} MB")
            return False
    
    return MemoryDelta(diagnostics, operation_name)


def measure_energy_consumption(diagnostics: GPUDiagnostics, operation_name: str = "Operation", 
                                sample_interval: float = 0.01):
    """
    Context manager to measure energy consumption during an operation.
    Samples power consumption at regular intervals and calculates energy.
    
    Usage:
        with measure_energy_consumption(diag, "Kernel Launch") as energy:
            # Your GPU operation here
            pass
        print(f"Energy consumed: {energy.total_energy_joules:.4f} J")
    
    Args:
        diagnostics: GPUDiagnostics instance
        operation_name: Name of the operation (for logging)
        sample_interval: Time between power samples in seconds (default: 0.01 = 10ms)
    
    Returns:
        EnergyMonitor context manager
    """
    class EnergyMonitor:
        def __init__(self, diag, name, interval):
            self.diag = diag
            self.name = name
            self.interval = interval
            self.power_samples = []
            self.start_time = None
            self.end_time = None
            self.total_energy_joules = 0.0
            self.avg_power_watts = 0.0
            self.max_power_watts = 0.0
            self.min_power_watts = float('inf')
            self._monitoring = False
            self._stop_monitoring = False
        
        def __enter__(self):
            import threading
            self.start_time = time.time()
            self._monitoring = True
            self._stop_monitoring = False
            
            # Start power monitoring thread
            def monitor_power():
                while not self._stop_monitoring:
                    if self.diag.initialized:
                        power = self.diag.get_power_usage()
                        if power is not None:
                            self.power_samples.append((time.time(), power))
                            if power > self.max_power_watts:
                                self.max_power_watts = power
                            if power < self.min_power_watts:
                                self.min_power_watts = power
                    time.sleep(self.interval)
            
            self.monitor_thread = threading.Thread(target=monitor_power, daemon=True)
            self.monitor_thread.start()
            return self
        
        def __exit__(self, *args):
            self._stop_monitoring = True
            self.end_time = time.time()
            
            # Wait for monitoring thread to finish
            if hasattr(self, 'monitor_thread'):
                self.monitor_thread.join(timeout=0.1)
            
            # Calculate energy consumption
            if self.power_samples:
                # Calculate average power
                if len(self.power_samples) > 0:
                    total_power = sum(p for _, p in self.power_samples)
                    self.avg_power_watts = total_power / len(self.power_samples)
                
                # Calculate energy: E = P_avg * t
                duration_seconds = self.end_time - self.start_time
                self.total_energy_joules = self.avg_power_watts * duration_seconds
                
                # Alternative: integrate power over time using samples
                if len(self.power_samples) > 1:
                    integrated_energy = 0.0
                    for i in range(1, len(self.power_samples)):
                        t1, p1 = self.power_samples[i-1]
                        t2, p2 = self.power_samples[i]
                        dt = t2 - t1
                        avg_p = (p1 + p2) / 2.0
                        integrated_energy += avg_p * dt
                    # Use integrated energy if available (more accurate)
                    if integrated_energy > 0:
                        self.total_energy_joules = integrated_energy
            else:
                # Fallback: use single power reading
                power = self.diag.get_power_usage()
                if power is not None:
                    duration_seconds = self.end_time - self.start_time
                    self.avg_power_watts = power
                    self.total_energy_joules = power * duration_seconds
            
            if self.min_power_watts == float('inf'):
                self.min_power_watts = self.avg_power_watts
            
            return False
    
    return EnergyMonitor(diagnostics, operation_name, sample_interval)


def get_cuda_device_count() -> int:
    """
    Get the number of CUDA devices available.
    Uses ctypes to query CUDA runtime directly.
    
    Returns:
        Number of CUDA devices
    """
    try:
        import ctypes
        libcudart = ctypes.CDLL("libcudart.so")
        count = ctypes.c_int()
        result = libcudart.cudaGetDeviceCount(ctypes.byref(count))
        if result == 0:  # cudaSuccess
            return count.value
    except:
        pass
    return 0

