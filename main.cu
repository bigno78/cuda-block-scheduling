#ifdef _WIN32
#include "device_launch_parameters.h"
#endif

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>


void cuda_check_result(cudaError_t err, int line)
{
    if (err == cudaSuccess)
        return;

    std::cerr << "[" << line << "] "
              << cudaGetErrorName(err) << ": "
              << cudaGetErrorString(err) << "\n";

    std::exit(1);
}

#define CUDA_CHECK_RESULT(expr) \
    cuda_check_result((expr), __LINE__)


struct DeviceDeleter
{
    void operator()(void* ptr) { cudaFree(ptr); }
};

template<typename T>
using device_ptr = std::unique_ptr<T, DeviceDeleter>;


template<typename T>
device_ptr<T> device_malloc(int count)
{
    int bytes = count*sizeof(T);
    T* raw_ptr;

    CUDA_CHECK_RESULT( cudaMalloc((void**)&raw_ptr, bytes) );

    return device_ptr<T>(raw_ptr);
}

template<typename T>
device_ptr<T> transfer_to_device(const std::vector<T>& vec)
{
    device_ptr<T> ptr = device_malloc<T>(vec.size());

    CUDA_CHECK_RESULT( cudaMemcpy(ptr.get(), (void*)vec.data(),
                                  vec.size()*sizeof(T), cudaMemcpyDefault) );

    return std::move(ptr);
}

template<typename T>
std::vector<T> transfer_to_host(const T* ptr, int count)
{
    int bytes = count*sizeof(T);
    std::vector<T> vec(bytes);

    CUDA_CHECK_RESULT( cudaMemcpy((void*)vec.data(), ptr, bytes,
                                  cudaMemcpyDefault) );

    return vec;
}


__global__ void kernel() {}

int main()
{
    std::vector<int> v1 = { 1, 2, 3, 4 };
    auto p = transfer_to_device(v1);
    auto v2 = transfer_to_host(p.get(), v1.size());
    std::cout << v2[0] << " " << v2[1] << " " << v2[2] << " " << v2[3];
}
