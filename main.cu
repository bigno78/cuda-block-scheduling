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
#include <string>
#include <vector>

//############################################################################//
//##                            CONSTANTS                                   ##//
//############################################################################//

namespace config
{
    constexpr int block_size = 256;
    constexpr int cell_size = 4;
    constexpr int max_nnz_per_row = 32;
    constexpr int frames = 200;
    constexpr int gif_length_seconds = 7;
    constexpr int border_width = 2;
}


//############################################################################//
//##                            CUDA STUFF                                  ##//
//############################################################################//

using clock64_t = long long int;

void cuda_check_result(cudaError_t err, int line)
{
    if (err == cudaSuccess)
        return;

    std::cerr << "[" << line << "] "
              << cudaGetErrorName(err) << ": "
              << cudaGetErrorString(err) << "\n"
              << "Aborting...\n";

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


//############################################################################//
//##                            KERNEL                                      ##//
//############################################################################//

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
            if (0 <= col && col < cols)
                sum += values[i*rows + row]*x[col];
        }
        y[row] = sum;
    }

    __syncthreads();
    if (threadIdx.x == 0)
        end_times[block_id] = clock64();
}


//############################################################################//
//##                            DATA DEFINITIONS                            ##//
//############################################################################//

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


//############################################################################//
//##                            COLOR STUFF                                 ##//
//############################################################################//

uint32_t from_rgb(uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t raw_bytes[] = { r, g, b, 0 };
    uint32_t result = 0;

    std::memcpy(&result, raw_bytes, 4);

    return result;
}

namespace colors
{
    constexpr uint32_t white = 0xffffffff;
    constexpr uint32_t black = 0;

    // The color palette is taken from:
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


//############################################################################//
//##                            GIF STUFF                                   ##//
//############################################################################//

void write_pixel(Image& image, int x, int y, uint32_t color)
{
    image.data[y*image.width + x] = color;
}

Image initialize_image(int width, int height)
{
    Image img { width, height };
    img.data.resize(width*height);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (y < config::border_width
                || y > height - config::border_width - 1
                || x < config::border_width
                || x > width - config::border_width - 1)
            {
                write_pixel(img, x, y, colors::black);
            }
            else
            {
                write_pixel(img, x, y, colors::white);
            }
        }
    }

    return img;
}


//############################################################################//
//##                          CORE IMPLEMENTATION                           ##//
//############################################################################//

void animate_grid(TimingData& data, int frames, int seconds,
                  const std::string& filename)
{
    std::cout << "Animating the grid...\n";

    int sm_count = 1 + *max_element(data.smids.begin(), data.smids.end());
    int block_count = data.grid_size_x * data.grid_size_y;

    clock64_t max_time = std::numeric_limits<clock64_t>::max();
    std::vector<clock64_t> sm_start_times(sm_count, max_time);
    std::vector<clock64_t> sm_end_times(sm_count, 0);

    for (int i = 0; i < block_count; ++i)
    {
        int smid = data.smids[i];

        if (data.start_times[i] < sm_start_times[smid])
            sm_start_times[smid] = data.start_times[i];
        if (data.end_times[i] > sm_end_times[smid])
            sm_end_times[smid] = data.end_times[i];
    }

    clock64_t duration = 0;

    for (int i = 0; i < sm_count; ++i)
        if (sm_end_times[i] - sm_start_times[i] > duration)
            duration = sm_end_times[i] - sm_start_times[i];

    clock64_t frame_time = duration/frames;

    int dislay_width = data.grid_size_x;
    int dislay_height = data.grid_size_y;
    if (data.grid_size_y == 1)
    {
        dislay_width = std::ceil(std::sqrt(data.grid_size_x));
        dislay_height = dislay_width;
    }

    int img_width = dislay_width*config::cell_size + 2*config::border_width;
    int img_height = dislay_height*config::cell_size + 2*config::border_width;
    Image img = initialize_image(img_width, img_height);

    int centi_seconds = 100*seconds;
    int gif_delay = centi_seconds/(frames - 1);

    GifWriter g;
	GifBegin(&g, filename.c_str(), img.width, img.height, gif_delay);
    GifWriteFrame(&g, (const uint8_t*)img.data.data(), img.width, img.height,
                  gif_delay);

    for (uint64_t frame = 0; frame < frames; ++frame)
    {
        for (int i = 0; i < block_count; ++i)
        {
            int smid = data.smids[i];

            clock64_t from = sm_start_times[smid] + frame*frame_time;
            clock64_t to   = from + frame_time;

            uint32_t color = colors::white;
            if (data.start_times[i] <= to && data.end_times[i] > from)
                color = colors::palette[smid % colors::palette.size()];

            int cell_x = i % dislay_width;
            int cell_y = i / dislay_width;

            for (int x = 0; x < config::cell_size; ++x)
            {
                for (int y = 0; y < config::cell_size; ++y)
                {
                    int pixel_x = cell_x*config::cell_size + x
                                    + config::border_width;
                    int pixel_y = cell_y*config::cell_size + y
                                    + config::border_width;

                    write_pixel(img, pixel_x, pixel_y, color);
                }
            }
        }

        GifWriteFrame(&g, (const uint8_t*)img.data.data(), img.width,
                      img.height, gif_delay);
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

TimingData measure_block_times(int grid_size_x, int grid_size_y)
{
    int blocks = grid_size_x*grid_size_y;
    int n = blocks*config::block_size;
    int max_nnz_per_row = config::max_nnz_per_row;

    std::cout << "Creating matrix of size "
              << n << " x " << n << " with "
              << max_nnz_per_row*n << " non-zeros...\n";

    auto mat = initialize_matrix(n, n, max_nnz_per_row);
    auto x = std::vector<float>(n, 1);

    auto values_dptr = transfer_to_device(mat.values);
    auto indices_dptr = transfer_to_device(mat.indices);
    auto x_dptr = transfer_to_device(x);
    auto y_dptr = device_malloc<float>(n);

    auto start_times_dptr = device_malloc<clock64_t>(blocks);
    auto end_times_dptr = device_malloc<clock64_t>(blocks);
    auto smids_dptr = device_malloc<int>(blocks);

    std::cout << "Launching grid of size "
              << grid_size_x << " x " << grid_size_y
              << " with a total of " << blocks << " blocks...\n";

    dim3 grid_size(grid_size_x, grid_size_y);
    kernel<<<grid_size, config::block_size>>>(n, n, max_nnz_per_row,
                                      values_dptr.get(), indices_dptr.get(),
                                      x_dptr.get(), y_dptr.get(),
                                      start_times_dptr.get(),
                                      end_times_dptr.get(),
                                      smids_dptr.get());

    auto y = transfer_to_host(y_dptr.get(), n);

    for (int i = 0; i < y.size(); ++i)
    {
        float expected = max_nnz_per_row*((i % 3) + 1);
        if (y[i] != expected)
        {
            std::cout << "Invalid result at index "
                      << i << ": " << y[i] << " vs " << expected << "\n"
                      << "Aborting...\n";
            std::exit(1);
        }
    }

    std::cout << "SpMV results are correct.\n";

    TimingData res { grid_size_x, grid_size_y };
    res.start_times = transfer_to_host(start_times_dptr.get(), blocks);
    res.end_times = transfer_to_host(end_times_dptr.get(), blocks);
    res.smids = transfer_to_host(smids_dptr.get(), blocks);

    return res;
}


void print_device_info()
{
    cudaDeviceProp prop;

    CUDA_CHECK_RESULT( cudaGetDeviceProperties(&prop, 0) );

    std::cout << "Using " << prop.name << ":\n";
    std::cout << "    Number of SMs: " << prop.multiProcessorCount << "\n";
}


int main()
{
    print_device_info();
    std::cout << "\n";

    auto data_1d = measure_block_times(4096, 1);
    animate_grid(data_1d, config::frames, config::gif_length_seconds,
                 "1D_grid.gif");
    std::cout << "\n";

    auto data_2d = measure_block_times(64, 64);
    animate_grid(data_2d, config::frames, config::gif_length_seconds,
                 "2D_grid.gif");
    std::cout << "\n";

	return 0;
}
