#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

struct square_t
{
    int *d_ptr;
    int incr;

    __device__ void operator()(int i)
    {
        d_ptr[i] = (d_ptr[i] + incr) * d_ptr[i];
    }
};

int main(void)
{
    cudaFree(0);

    thrust::device_vector<int> img = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    square_t op{thrust::raw_pointer_cast(img.data()), 1};

    std::uint8_t *d_temp_storage{};
    std::size_t temp_storage_bytes{};

    // Get and allocate temp storage
    cub::DeviceFor::Bulk(d_temp_storage, temp_storage_bytes, img.size(), op);
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // Perform the operation
    cub::DeviceFor::Bulk(d_temp_storage, temp_storage_bytes, img.size(), op);

    thrust::host_vector<int> result(img.size());

    thrust::copy(img.begin(), img.end(), result.begin());

    std::cout << "< ";
    for (int element : result)
    {
        std::cout << element << " ";
    }

    std::cout << ">" << std::endl;

    return EXIT_SUCCESS;
}
