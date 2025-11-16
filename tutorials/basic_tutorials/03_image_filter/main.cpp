#include "kernel.h"

#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>" << std::endl;
        return -1;
    }

    std::string input_image_path = argv[1];
    std::string output_image_path = argv[2];
    // Check if input image exists
    if (!std::filesystem::exists(input_image_path)) {
        std::cerr << "Input image file does not exist: " << input_image_path << std::endl;
        return -1;
    }

    // Load input image
    cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Could not open the image: " << input_image_path << std::endl;
        return -1;
    }

    int width = input_image.cols;
    int height = input_image.rows;

    // Allocate device memory
    unsigned char *d_input_image, *d_output_image;
    size_t image_size = width * height * 3 * sizeof(unsigned char); // 3 channels (RGB)

    CUDA_CHECK(cudaMalloc(&d_input_image, image_size));
    CUDA_CHECK(cudaMalloc(&d_output_image, image_size));

    // Copy input image to device
    CUDA_CHECK(cudaMemcpy(d_input_image, input_image.data, image_size, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    image_filter_kernel<<<gridSize, blockSize>>>(d_input_image, d_output_image, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output image back to host
    cv::Mat output_image(height, width, CV_8UC3);
    CUDA_CHECK(cudaMemcpy(output_image.data, d_output_image, image_size, cudaMemcpyDeviceToHost));

    // Save output image
    cv::imwrite(output_image_path, output_image);

    // Free device memory
    CUDA_CHECK(cudaFree(d_input_image));
    CUDA_CHECK(cudaFree(d_output_image));

    return 0;
}