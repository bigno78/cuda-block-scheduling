#ifdef _WIN32
#include "device_launch_parameters.h"
#endif

#include <cstdint>
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


__global__ void kernel(int rows, int cols, int max_nnz_per_row,
                       const float* values, const int* column_indices,
                       const float*x, float*y,
                       uint64_t* start_times, uint64_t* end_times)
{
    if (threadIdx.x == 0)
        start_times[blockIdx.x] = clock64();

    int row = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < rows)
    {
        float sum = 0.0f;
        for (int i = 0; i < max_nnz_per_row; ++i)
        {
            int col = column_indices[i*rows + row];
            if (0 <= col && col < cols)
                sum += values[i*rows + row]*x[col];
        }
        y[row] = sum;
    }

    __syncthreads();
    if (threadIdx.x == 0)
        end_times[blockIdx.x] = clock64();
}


struct TimingData
{
    int grid_size_x;
    int grid_size_y;

    std::vector<uint64_t> start_times;
    std::vector<uint64_t> end_times;
};

struct ELLMatrix
{
    int rows;
    int cols;
    int max_nnz_per_row;

    std::vector<float> values;
    std::vector<int> indices;
};


ELLMatrix initialize_matrix(int rows, int cols, int max_nnz_per_row)
{
    ELLMatrix mat { rows, cols, max_nnz_per_row };

    mat.values.resize(rows*max_nnz_per_row, 1);
    mat.indices.resize(rows*max_nnz_per_row);

    int col = 0;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < max_nnz_per_row; ++j)
        {
            mat.values[j*rows + i] = (i % 3) + 1;
            mat.indices[j*rows + i] = col;
            col = (col + 1) % cols;
        }
    }

    return mat;
}

std::vector<float> initialize_x(int cols)
{
    return std::vector<float>(cols, 1);
}

TimingData call_kernel()
{
    int n = 1024*1024;
    int max_nnz_per_row = 32;

    int block_size = 256;
    int grid_size = (n + block_size - 1)/block_size;

    ELLMatrix mat = initialize_matrix(n, n, max_nnz_per_row);
    auto x = initialize_x(n);

    auto values_dptr = transfer_to_device(mat.values);
    auto indices_dptr = transfer_to_device(mat.indices);
    auto x_dptr = transfer_to_device(x);
    auto y_dptr = device_malloc<float>(n);

    auto start_times_dptr = device_malloc<uint64_t>(grid_size);
    auto end_times_dptr = device_malloc<uint64_t>(grid_size);

    std::cout << "Launching grid of size " << grid_size << "\n";

    kernel<<<grid_size, block_size>>>(n, n, max_nnz_per_row,
                                      values_dptr.get(), indices_dptr.get(),
                                      x_dptr.get(), y_dptr.get(),
                                      start_times_dptr.get(),
                                      end_times_dptr.get());

    auto y = transfer_to_host(y_dptr.get(), n);

    TimingData res { grid_size, 1 };
    res.start_times = transfer_to_host(start_times_dptr.get(), grid_size);
    res.end_times = transfer_to_host(end_times_dptr.get(), grid_size);

    return res;
}


int main()
{
    call_kernel();
}
