#include <cuda_runtime.h>
#include <cuComplex.h>

// ---------------------------------------------------------------------------
// Fonction CUDA pour calculer la fractale de Mandelbrot
// ---------------------------------------------------------------------------
__global__ void computeMandelbrotCUDA(unsigned char* pixels, unsigned width, unsigned height,
                                      double minX, double maxX, double minY, double maxY, unsigned maxIter) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px < width && py < height) {
        double x0 = minX + (px / (double)width) * (maxX - minX);
        double y0 = minY + (py / (double)height) * (maxY - minY);

        //std::complex<double> c(x0, y0);
        //std::complex<double> z(0, 0);
        cuDoubleComplex c = make_cuDoubleComplex(x0, y0);
        cuDoubleComplex z = make_cuDoubleComplex(0, 0);

        unsigned iteration = 0;
        while (cuCabs(z) < 2.0 && iteration < maxIter) {
            z = cuCadd(cuCmul(z, z), c);
            iteration++;
        }

        double t = (double)iteration / (double)maxIter;
        unsigned char r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
        unsigned char g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
        unsigned char b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);

        int index = 4 * (py * width + px);
        pixels[index + 0] = r;
        pixels[index + 1] = g;
        pixels[index + 2] = b;
        pixels[index + 3] = 255;
    }
}

extern "C" void launchMandelbrotCUDA(unsigned char* hostPixels, unsigned width, unsigned height,
                                     double minX, double maxX, double minY, double maxY, unsigned maxIter) {
    unsigned char* d_pixels;
    size_t dataSize = width * height * 4;

    // Allocate memory on the device
    cudaMalloc(&d_pixels, dataSize);

    // Configure grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Launch the kernel
    computeMandelbrotCUDA<<<grid, block>>>(d_pixels, width, height, minX, maxX, minY, maxY, maxIter);
    cudaDeviceSynchronize();

    // Copy data back to host
    cudaMemcpy(hostPixels, d_pixels, dataSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_pixels);
}
