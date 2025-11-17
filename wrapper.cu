#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include "Bitmap.h"
#include "MedianFilter.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// Logging utility
static bool g_logging_enabled = true;
static FILE* g_log_file = nullptr;

void enable_logging(bool enable) {
    g_logging_enabled = enable;
}

void set_log_file(const char* filename) {
    if (g_log_file && g_log_file != stdout && g_log_file != stderr) {
        fclose(g_log_file);
    }
    if (filename) {
        g_log_file = fopen(filename, "a");
        if (!g_log_file) {
            g_log_file = stderr;
        }
    } else {
        g_log_file = stderr;
    }
}

#define LOG(...) do { \
    if (g_logging_enabled) { \
        if (!g_log_file) g_log_file = stderr; \
        fprintf(g_log_file, "[GPU_WRAPPER] " __VA_ARGS__); \
        fflush(g_log_file); \
    } \
} while(0)

// Forward declarations of CUDA kernels from medianFilter.cu
extern __global__ void medianFilterKernel(unsigned char *inputImageKernel, unsigned char *outputImagekernel, int imageWidth, int imageHeight);
extern __global__ void medianFilterSharedKernel(unsigned char *inputImageKernel, unsigned char *outputImagekernel, int imageWidth, int imageHeight);

extern "C" {

// Opaque handle type for Bitmap objects
typedef void* BitmapHandle;

// Create a new empty bitmap
BitmapHandle create_bitmap(int width, int height) {
    Bitmap* bmp = new Bitmap(width, height);
    return static_cast<BitmapHandle>(bmp);
}

// Load bitmap from file
BitmapHandle load_bitmap(const char* filename) {
    Bitmap* bmp = new Bitmap();
    if (bmp->Load(filename)) {
        return static_cast<BitmapHandle>(bmp);
    }
    delete bmp;
    return nullptr;
}

// Save bitmap to file
int save_bitmap(BitmapHandle handle, const char* filename) {
    if (handle == nullptr) {
        return 0;
    }
    Bitmap* bmp = static_cast<Bitmap*>(handle);
    return bmp->Save(filename) ? 1 : 0;
}

// Get bitmap width
int get_bitmap_width(BitmapHandle handle) {
    if (handle == nullptr) {
        return -1;
    }
    Bitmap* bmp = static_cast<Bitmap*>(handle);
    return bmp->Width();
}

// Get bitmap height
int get_bitmap_height(BitmapHandle handle) {
    if (handle == nullptr) {
        return -1;
    }
    Bitmap* bmp = static_cast<Bitmap*>(handle);
    return bmp->Height();
}

// Get pointer to bitmap image data (for reading)
unsigned char* get_bitmap_data(BitmapHandle handle) {
    if (handle == nullptr) {
        return nullptr;
    }
    Bitmap* bmp = static_cast<Bitmap*>(handle);
    return reinterpret_cast<unsigned char*>(bmp->image);
}

// Get size of bitmap data in bytes
int get_bitmap_size(BitmapHandle handle) {
    if (handle == nullptr) {
        return -1;
    }
    Bitmap* bmp = static_cast<Bitmap*>(handle);
    return bmp->Width() * bmp->Height();
}

// Destroy bitmap and free memory
void destroy_bitmap(BitmapHandle handle) {
    if (handle != nullptr) {
        Bitmap* bmp = static_cast<Bitmap*>(handle);
        delete bmp;
    }
}

// Structure to hold timing information
struct TimingInfo {
    float total_time_ms;
    float h2d_time_ms;
    float kernel_time_ms;
    float d2h_time_ms;
    int success;
};

// Run GPU median filter with timing information
// Returns 1 on success, 0 on failure
// Timing information is written to timing_info structure
int median_filter_gpu_wrapper(BitmapHandle input_handle, BitmapHandle output_handle, 
                               int use_shared, TimingInfo* timing_info) {
    if (input_handle == nullptr || output_handle == nullptr || timing_info == nullptr) {
        LOG("ERROR: Invalid handles or timing_info pointer\n");
        return 0;
    }
    
    Bitmap* input = static_cast<Bitmap*>(input_handle);
    Bitmap* output = static_cast<Bitmap*>(output_handle);
    
    LOG("Starting GPU median filter (shared_memory=%d)\n", use_shared);
    LOG("Image dimensions: %d x %d pixels\n", input->Width(), input->Height());
    
    // Initialize timing info
    timing_info->total_time_ms = 0.0f;
    timing_info->h2d_time_ms = 0.0f;
    timing_info->kernel_time_ms = 0.0f;
    timing_info->d2h_time_ms = 0.0f;
    timing_info->success = 0;
    
    // Create CUDA events for timing
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;
    
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);
    
    cudaEventRecord(start_total);
    
    cudaError_t status;
    int width = input->Width();
    int height = input->Height();
    int size = width * height * sizeof(char);
    
    LOG("Allocating device memory: %d bytes (%.2f MB)\n", size, size / (1024.0 * 1024.0));
    
    // Allocate device memory for input
    unsigned char *device_input_image;
    cudaMalloc((void**) &device_input_image, size);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        LOG("ERROR: Failed to allocate device input memory. Error: %s\n", cudaGetErrorString(status));
        cudaEventDestroy(start_total);
        cudaEventDestroy(stop_total);
        cudaEventDestroy(start_h2d);
        cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_kernel);
        cudaEventDestroy(stop_kernel);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);
        return 0;
    }
    LOG("Device input memory allocated successfully\n");
    
    // Host to device transfer
    LOG("Copying data from host to device...\n");
    cudaEventRecord(start_h2d);
    cudaMemcpy(device_input_image, input->image, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        LOG("ERROR: Host-to-device transfer failed. Error: %s\n", cudaGetErrorString(status));
        cudaFree(device_input_image);
        cudaEventDestroy(start_total);
        cudaEventDestroy(stop_total);
        cudaEventDestroy(start_h2d);
        cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_kernel);
        cudaEventDestroy(stop_kernel);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);
        return 0;
    }
    LOG("Host-to-device transfer completed\n");
    
    // Allocate device memory for output
    unsigned char *device_output_image;
    cudaMalloc((void**) &device_output_image, size);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        LOG("ERROR: Failed to allocate device output memory. Error: %s\n", cudaGetErrorString(status));
        cudaFree(device_input_image);
        cudaEventDestroy(start_total);
        cudaEventDestroy(stop_total);
        cudaEventDestroy(start_h2d);
        cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_kernel);
        cudaEventDestroy(stop_kernel);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);
        return 0;
    }
    LOG("Device output memory allocated successfully\n");
    
    // Setup grid and block dimensions
    const int TILE_SIZE = 4;
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((int)ceil((float)width / (float)TILE_SIZE),
                 (int)ceil((float)height / (float)TILE_SIZE));
    
    LOG("Launching kernel: Grid(%d, %d), Block(%d, %d), Total threads: %d\n", 
        dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y, dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y);
    
    // Launch kernel
    cudaEventRecord(start_kernel);
    if (!use_shared) {
        LOG("Using regular kernel (no shared memory)\n");
        medianFilterKernel<<<dimGrid, dimBlock>>>(device_input_image, device_output_image, width, height);
    } else {
        LOG("Using shared memory kernel\n");
        medianFilterSharedKernel<<<dimGrid, dimBlock>>>(device_input_image, device_output_image, width, height);
    }
    cudaEventRecord(stop_kernel);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        LOG("ERROR: Kernel launch failed. Error: %s\n", cudaGetErrorString(status));
        cudaFree(device_input_image);
        cudaFree(device_output_image);
        cudaEventDestroy(start_total);
        cudaEventDestroy(stop_total);
        cudaEventDestroy(start_h2d);
        cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_kernel);
        cudaEventDestroy(stop_kernel);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);
        return 0;
    }
    LOG("Kernel launched successfully, waiting for completion...\n");
    
    // Device to host transfer
    LOG("Copying data from device to host...\n");
    cudaEventRecord(start_d2h);
    cudaMemcpy(output->image, device_output_image, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        LOG("ERROR: Device-to-host transfer failed. Error: %s\n", cudaGetErrorString(status));
        cudaFree(device_input_image);
        cudaFree(device_output_image);
        cudaEventDestroy(start_total);
        cudaEventDestroy(stop_total);
        cudaEventDestroy(start_h2d);
        cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_kernel);
        cudaEventDestroy(stop_kernel);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);
        return 0;
    }
    LOG("Device-to-host transfer completed\n");
    
    // Synchronize and get timing
    cudaEventSynchronize(stop_total);
    cudaEventSynchronize(stop_h2d);
    cudaEventSynchronize(stop_kernel);
    cudaEventSynchronize(stop_d2h);
    
    cudaEventElapsedTime(&timing_info->total_time_ms, start_total, stop_total);
    cudaEventElapsedTime(&timing_info->h2d_time_ms, start_h2d, stop_h2d);
    cudaEventElapsedTime(&timing_info->kernel_time_ms, start_kernel, stop_kernel);
    cudaEventElapsedTime(&timing_info->d2h_time_ms, start_d2h, stop_d2h);
    
    LOG("Timing results: Total=%.4f ms, H2D=%.4f ms, Kernel=%.4f ms, D2H=%.4f ms\n",
        timing_info->total_time_ms, timing_info->h2d_time_ms, 
        timing_info->kernel_time_ms, timing_info->d2h_time_ms);
    
    // Cleanup
    LOG("Freeing device memory...\n");
    cudaFree(device_input_image);
    cudaFree(device_output_image);
    
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);
    
    LOG("GPU median filter completed successfully\n");
    timing_info->success = 1;
    return 1;
}

// Get CUDA error string (helper function)
const char* get_cuda_error_string() {
    cudaError_t err = cudaGetLastError();
    return cudaGetErrorString(err);
}

// Logging control functions
void enable_wrapper_logging(int enable) {
    enable_logging(enable != 0);
}

void set_wrapper_log_file(const char* filename) {
    set_log_file(filename);
}

// Initialize CUDA (returns 1 on success, 0 on failure)
int init_cuda() {
    LOG("Initializing CUDA...\n");
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        LOG("ERROR: Failed to get device count or no devices found. Error: %s\n", cudaGetErrorString(err));
        return 0;
    }
    LOG("Found %d CUDA device(s)\n", deviceCount);
    
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        LOG("ERROR: Failed to set device 0. Error: %s\n", cudaGetErrorString(err));
        return 0;
    }
    
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err == cudaSuccess) {
        LOG("Using device: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
        LOG("Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    
    LOG("CUDA initialized successfully\n");
    return 1;
}

// Get CUDA device properties
int get_cuda_device_info(char* name, int name_len, int* compute_major, int* compute_minor, 
                         size_t* total_memory, size_t* free_memory) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return 0;
    }
    
    if (name != nullptr && name_len > 0) {
        strncpy(name, prop.name, name_len - 1);
        name[name_len - 1] = '\0';
    }
    
    if (compute_major != nullptr) {
        *compute_major = prop.major;
    }
    if (compute_minor != nullptr) {
        *compute_minor = prop.minor;
    }
    
    if (total_memory != nullptr) {
        *total_memory = prop.totalGlobalMem;
    }
    
    if (free_memory != nullptr) {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        *free_memory = free;
    }
    
    return 1;
}

} // extern "C"

