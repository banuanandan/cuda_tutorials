#include "kernel.h"

__global__ void image_filter_kernel(unsigned char* input_image,
                                     unsigned char* output_image,
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // Assuming 3 channels (RGB)
        auto r = input_image[idx];
        auto g = input_image[idx + 1];
        auto b = input_image[idx + 2];

        // Simple grayscale filter
        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

        output_image[idx] = gray;
        output_image[idx + 1] = gray;
        output_image[idx + 2] = gray;
    }
}