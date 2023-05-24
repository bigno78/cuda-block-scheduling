#ifdef _WIN32
#include "device_launch_parameters.h"
#endif

#include "external/gif.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <set>

uint32_t from_rgb(uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t raw_bytes[] = { r, g, b, 0 };
    uint32_t result = 0;

    std::memcpy(&result, raw_bytes, 4);

    return result;
}

namespace colors
{
    constexpr uint32_t black = 0;
    constexpr uint32_t white = 0xffffffff;
    constexpr uint32_t red = 0x000000ff;
    constexpr uint32_t green = 0x0000ff00;

    // The palette is taken from:
    // https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=6
    std::vector<uint32_t> palette =
        [] () { return std::vector<uint32_t> {
                    from_rgb(228, 26, 28),
                    from_rgb(55, 126, 184),
                    from_rgb(77, 175, 74),
                    from_rgb(152, 78, 163),
                    from_rgb(255, 127, 0),
                    from_rgb(255, 255, 51)
                }; }();
};

using clock64_t = long long int;

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
    std::vector<T> vec(count);

    CUDA_CHECK_RESULT( cudaMemcpy((void*)vec.data(), ptr, bytes,
                                  cudaMemcpyDefault) );

    return vec;
}


__device__ int dummy_work(int n)
{
    if (n == 0)
        return 0;
    if (n == 1)
        return 1;
    return dummy_work(n - 1) + dummy_work(n - 2);
}

__global__ void kernel(int rows, int cols, int max_nnz_per_row,
                       const float* values, const int* column_indices,
                       const float*x, float*y, clock64_t* start_times,
                       clock64_t* end_times, int* smids)
{
    int block_id = blockIdx.y*gridDim.x + blockIdx.x;

    if (threadIdx.x == 0)
    {
        int smid;
        asm("mov.u32 %0, %%smid;" : "=r"(smid));

        start_times[block_id] = clock64();
        smids[block_id] = smid;
    }
    __syncthreads();

    int row = block_id*blockDim.x + threadIdx.x;

    if (row < rows)
    {
        float sum = 0.0f;
        for (int i = 0; i < max_nnz_per_row; ++i)
        {
            int col = column_indices[i*rows + row];
            //if (0 <= col && col < cols)
                sum += values[i*rows + row]*x[col];
        }
        y[row] = sum;
        // y[row] = dummy_work(20);
    }

    __syncthreads();
    if (threadIdx.x == 0)
        end_times[block_id] = clock64();
}


struct TimingData
{
    int grid_size_x;
    int grid_size_y;

    std::vector<clock64_t> start_times;
    std::vector<clock64_t> end_times;
    std::vector<int> smids;
};

struct ELLMatrix
{
    int rows;
    int cols;
    int max_nnz_per_row;

    std::vector<float> values;
    std::vector<int> indices;
};

struct Image
{
    int width;
    int height;

    std::vector<uint32_t> data;
};


void write_pixel(Image& image, int x, int y, uint32_t color)
{
    image.data[y*image.width + x] = color;
}

Image initialize_image(int width, int height)
{
    Image img { width, height };
    img.data.resize(width*height);

    for (int i = 0; i < width*height; ++i)
        img.data[i] = colors::green;

    return img;
}

void draw_grid(Image& img, int cell_size)
{
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            if (y % cell_size == 0 || x % cell_size == 0)
                write_pixel(img, x, y, colors::black);
        }
    }
}

void draw_frame(Image& img, int cell_size)
{
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            if (y % cell_size == 0 || x % cell_size == 0)
                write_pixel(img, x, y, colors::black);
        }
    }
}

clock64_t min(const std::vector<clock64_t>& times)
{
    clock64_t res = times[0];
    for (auto t : times)
        if (t < res)
            res = t;
    return res;
}

void animate_grid(TimingData& data, int frames, int seconds)
{
    int centi_seconds = 100*seconds;
    int gif_delay = centi_seconds/(frames - 1);

    int sm_count = 1 + *max_element(data.smids.begin(), data.smids.end());
    int block_count = data.grid_size_x * data.grid_size_y;

    std::vector<clock64_t> sm_start(sm_count,
                                    std::numeric_limits<clock64_t>::max());
    std::vector<clock64_t> sm_end(sm_count, 0);

    for (int i = 0; i < block_count; ++i)
    {
        int smid = data.smids[i];

        if (data.start_times[i] < sm_start[smid])
            sm_start[smid] = data.start_times[i];
        if (data.end_times[i] > sm_end[smid])
            sm_end[smid] = data.end_times[i];
    }

    clock64_t duration = 0;

    for (int i = 0; i < sm_count; ++i)
        if (sm_end[i] - sm_start[i] > duration)
            duration = sm_end[i] - sm_start[i];

    clock64_t frame_time = duration/frames;

    std::vector<clock64_t> durs(block_count);
    std::vector<int> indices(block_count);
    for (int i = 0; i < block_count; ++i)
    {
        indices[i] = i;
        durs[i] = data.end_times[i] - data.start_times[i];
    }
    std::sort(indices.begin(), indices.end(), [&] (int i, int j) { return durs[i] < durs[j]; });

    constexpr int cell_size = 4;

    int dislay_width = data.grid_size_x;
    int dislay_height = data.grid_size_y;
    if (data.grid_size_y == 1)
    {
        dislay_width = std::ceil(std::sqrt(data.grid_size_x));
        dislay_height = dislay_width;
    }

    Image img = initialize_image(dislay_width*cell_size, dislay_height*cell_size);

    GifWriter g;
	GifBegin(&g, "bwgif.gif", img.width, img.height, gif_delay);
    GifWriteFrame(&g, (const uint8_t*)img.data.data(), img.width, img.height, gif_delay);

    for (uint64_t frame = 0; frame < frames; ++frame)
    {
        int count = 0;
        for (int i = 0; i < block_count; ++i)
        {
            int smid = data.smids[i];

            clock64_t from = sm_start[smid] + frame*frame_time;
            clock64_t to   = from + frame_time;

            int cell_x = i % dislay_width;
            int cell_y = i / dislay_width;

            uint32_t color = colors::white;
            if (data.start_times[i] <= to && data.end_times[i] > from)
            {

                color = colors::palette[smid % colors::palette.size()];
                count++;
            }

            for (int x = 0; x < cell_size; ++x)
                for (int y = 0; y < cell_size; ++y)
                    write_pixel(img, cell_x*cell_size + x, cell_y*cell_size + y,
                                color);
        }

        //std::cout << "Frame " << frame << ": " << count << " blocks\n";

        GifWriteFrame(&g, (const uint8_t*)img.data.data(), img.width, img.height, gif_delay);
    }

    GifEnd(&g);
}


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

TimingData call_kernel(int grid_size_x, int grid_size_y)
{
    int blocks = grid_size_x*grid_size_y;
    int block_size = 256;

    int n = blocks*block_size;
    int max_nnz_per_row = 32;

    std::cout << "Creating matrix of size " << n << "\n";

    ELLMatrix mat = initialize_matrix(n, n, max_nnz_per_row);
    auto x = initialize_x(n);

    auto values_dptr = transfer_to_device(mat.values);
    auto indices_dptr = transfer_to_device(mat.indices);
    auto x_dptr = transfer_to_device(x);
    auto y_dptr = device_malloc<float>(n);

    auto start_times_dptr = device_malloc<clock64_t>(blocks);
    auto end_times_dptr = device_malloc<clock64_t>(blocks);
    auto smids_dptr = device_malloc<int>(blocks);

    std::cout << "Launching grid of size " << blocks << "\n";

    CUDA_CHECK_RESULT( cudaDeviceSynchronize() );

    dim3 grid_size(grid_size_x, grid_size_y);
    kernel<<<grid_size, block_size>>>(n, n, max_nnz_per_row,
                                      values_dptr.get(), indices_dptr.get(),
                                      x_dptr.get(), y_dptr.get(),
                                      start_times_dptr.get(),
                                      end_times_dptr.get(),
                                      smids_dptr.get());

    auto y = transfer_to_host(y_dptr.get(), n);

    TimingData res { grid_size_x, grid_size_y };
    res.start_times = transfer_to_host(start_times_dptr.get(), blocks);
    res.end_times = transfer_to_host(end_times_dptr.get(), blocks);
    res.smids = transfer_to_host(smids_dptr.get(), blocks);

    for (int i = 0; i < y.size(); ++i)
    {
        float expected = max_nnz_per_row*((i % 3) + 1);
        if (y[i] != expected)
        {
            std::cout << "Invalid result at index " << i << ": " << y[i] << " vs " << expected << "\n";
        }
    }

    return res;
}


void print_device_info()
{
    cudaDeviceProp prop;

    CUDA_CHECK_RESULT( cudaGetDeviceProperties(&prop, 0) );

    std::cout << "Using device: " << prop.name << "\n";
    std::cout << "Number of SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "L2 size: " << prop.l2CacheSize << "\n";
}


int main()
{
    print_device_info();

    auto data = call_kernel(128, 64);
    animate_grid(data, 150, 10);

	return 0;
}
