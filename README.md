# Acceleration of an application

Authors : Michael Strefeler and Cristhian Ronquillo

## Stage 1 - Choosing an application

We chose a C++ program that generates an interactive visualization window (resolution = `HEIGHT` x `WIDTH`) of the Mandelbrot fractal using the *SFML* graphics library. The fractal is calculated by iterating over each point in the complex plane defined by the limits `minX`, `maxX`, `minY`, and `maxY`. Each pixel on the screen corresponds to a complex point, and a color is assigned to it based on the number of iterations needed to determine whether the point diverges. The `MAX_ITER` parameter controls the maximum number of iterations for these calculations: a higher value improves the precision and details of the fractal patterns but also increases computation time.

The user can interact with the visualization by selecting a region with the mouse to zoom in. When a selection rectangle is drawn, the program dynamically adjusts the limits of the complex plane according to the selected area while maintaining the window's aspect ratio. It then recalculates the fractal for the new region, allowing for progressive and detailed exploration. This process is made smoother by `MAX_ITER`, which directly influences the visual quality and the color gradients based on the depth of the calculations for each point.

We found some code on *GitHub* that generated one image of the Mandelbrot set fractal and we used *ChatGPT* to edit the code to make it interactive.

## Stage 2 - Analysing application bottlenecks

### Execution Time

The execution time of the application is influenced primarily by the `computeMandelbrot` function, which calculates pixel data for the fractal. The function's runtime depends on the resolution and the maximum number of iterations.

We ran the code with `WIDTH` = 1000, `HEIGHT` = 800, and `MAX_ITER` = 3000, the application performs approximately $1000 * 800 * 3000 = 2.4 * 10 ^9$ complex operations. It took 4.08 seconds to show the initial fractal and 65.20 seconds for the next zoom. The initial fractal is less detailed so it takes less time to generate.

### Complexity Analysis

For each pixel, up to `MAX_ITER` iterations are performed.
The overall complexity is O(`WIDTH` x `HEIGHT` × `MAX_ITER`), where each iteration involves basic operations on complex numbers.

The application stores RGBA pixel data, requiring 4 × `WIDTH` × `HEIGHT` bytes (~3.2 MB for 1000 × 800 resolution).

### Bottlenecks

- The nested loops in `computeMandelbrot` are computationally expensive due to the iterative calculations for each pixel.
- The application processes the entire fractal sequentially, leaving room for parallelization.
- Updating the texture after every zoom adds overhead, although less significant compared to fractal computation.

### Acceleration Plan

1. Implement multi-threading using OpenMP to divide the computation of pixel rows or columns across multiple CPU cores.
2. Use CUDA to offload computations to the GPU. Each pixel computation can be treated as an independent task, making it highly suitable for GPU parallelism.
3. Minimize memory access latency by optimizing memory transfers (e.g., using pinned memory or memory tiling for CUDA).

### Theoretical Performance Improvement

Using CPU Parallelization: For an 8-core CPU, the theoretical speedup is close to 8x, minus synchronization overhead.
Using GPU Acceleration: A modern GPU with thousands of CUDA cores could provide a speedup of 50–100x due to massive parallelism.
Algorithmic Optimization: Early exits for quickly diverging points and reducing redundant calculations could further improve runtime by 20–30%.

Expected Performance:
By combining GPU acceleration and memory optimizations, execution time could decrease significantly, achieving near-peak performance on compute-bound segments while mitigating memory bottlenecks.
This approach aligns with the principles of Amdahl's Law, which emphasizes improving parallelizable portions of the code to maximize performance scaling. With these improvements, the application could transition from a memory-bound to a compute-bound scenario, optimizing execution on modern architectures.

## Stage 3 - Acceleration

We had to separate our code into two files (because it wouldn't work in one file), `mandelbrot_host.cpp` and `mandelbrot_device.cu`, the first one contains the host code (OpenMP and rest of code) and the second one contains the device/GPU code.

Here's how we compile it :

```bash
nvcc -c mandelbrot_device.cu -o mandelbrot_device.o
g++ -std=c++11 mandelbrot_host.cpp mandelbrot_device.o -o mandelbrot_acceleration -lsfml-graphics -lsfml-window -lsfml-system -fopenmp -lcudart
```

The first line is to compile the device code and the second one is to compile the rest and link everything together. The first 3 flags (-lsfml-graphics -lsfml-window -lsfml-system) are required to use the SFML library, `-fopenmp` is for OpenMP and `-lcudart` links against the NVIDIA CUDA runtime library.

### CPU Acceleration (with OpenMP)

The `computeMandelbrotOpenMP` function calculates the fractal on the CPU. Its nested loops process each pixel independently, making it ideal for parallelization.
We added OpenMP directives to parallelize the outer loop, splitting the workload across multiple CPU threads.

Here's how we accelerated it (in the `computeMandelbrotOpenMP` function):

```cpp
  std::vector<sf::Uint8> computeMandelbrotOpenMP(unsigned width, unsigned height,
                                               double minX, double maxX,
                                               double minY, double maxY,
                                               unsigned maxIter) {
    std::vector<sf::Uint8> pixels(width * height * 4); // RGBA

    #pragma omp parallel for collapse(2)
    for (unsigned py = 0; py < height; ++py) {
        for (unsigned px = 0; px < width; ++px) {
            double x0 = minX + (px / (double)width) * (maxX - minX);
            double y0 = minY + (py / (double)height) * (maxY - minY);

            std::complex<double> c(x0, y0);
            std::complex<double> z(0, 0);

            unsigned iteration = 0;
            while (std::abs(z) < 2.0 && iteration < maxIter) {
                z = z * z + c;
                iteration++;
            }

            double t = (double)iteration / (double)maxIter;
            sf::Uint8 r = static_cast<sf::Uint8>(9 * (1 - t) * t * t * t * 255);
            sf::Uint8 g = static_cast<sf::Uint8>(15 * (1 - t) * (1 - t) * t * t * 255);
            sf::Uint8 b = static_cast<sf::Uint8>(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);

            unsigned index = 4 * (py * width + px);
            pixels[index + 0] = r;
            pixels[index + 1] = g;
            pixels[index + 2] = b;
            pixels[index + 3] = 255;
        }
    }

    return pixels;
}
```

### GPU Acceleration (with CUDA)

The `computeMandelbrotCUDA` function processes each pixel in parallel on the GPU. We used **cuComplex** for complex number calculations, as it is optimized for GPU operations.

Here's the code we used to create the kernel:

```cpp
__global__ void computeMandelbrotCUDA(unsigned char* pixels, unsigned width, unsigned height,
                                      double minX, double maxX, double minY, double maxY, unsigned maxIter) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px < width && py < height) {
        double x0 = minX + (px / (double)width) * (maxX - minX);
        double y0 = minY + (py / (double)height) * (maxY - minY);

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
```

We run the kernel like so:

```cpp
  dim3 block(16, 16);
  dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
  computeMandelbrotCUDA<<<grid, block>>>(devicePixels, WIDTH, HEIGHT, minX, maxX, minY, maxY, MAX_ITER);
```

## Stage 4 - Analysis of results

### Analysis of the results

We were unable to run the program on the card (we couldn't install everything on), but luckily one of our laptops has a Nvidia GPU, so we could run it there.

Here are the results using `WIDTH` = 1000, `HEIGHT` = 800, and `MAX_ITER` = 3000

| Computation Stage       | Original Code (seconds) | OpenMP (seconds) | CUDA (seconds) |
|-----------------------------|----------------------------|-----------------------|--------------------|
| Initial Mandelbrot          | 6.5155                    | 2.08163              | 0.689945           |
| Mandelbrot Zoom 1           | 75.2991                   | 7.74693              | 0.705067           |
| Mandelbrot Zoom 2           | 40.892                    | 8.95785              | 0.764674           |
| Mandelbrot Zoom 3           | 45.0272                   | 4.66551              | 0.642998           |
| Mandelbrot Zoom 4           | 59.3526                   | 8.74846              | 0.842222           |

The differences in execution times are influenced by the varying complexity of the Mandelbrot set's zoom regions, as some require more iterations per pixel near the set's boundary. This variability adds to the performance differences observed across the original code, OpenMP, and CUDA, with CUDA remaining the fastest despite the computational intensity of each zoom.

1. **OpenMP Acceleration**:
   - OpenMP consistently reduced execution time across all stages, achieving a **3x–10x improvement** over the original code.
   - Performance variations between zooms indicate the effect of varying computational loads on the CPU.

2. **CUDA Acceleration**:
   - CUDA provided the fastest execution times for all computations, achieving up to **100x improvement** over the original code for certain zooms.
   - The GPU's ability to handle thousands of threads in parallel ensures near-constant performance even for regions with higher complexity.

3. **Impact of Zoom Complexity**:
   - Zoom regions with more points closer to the Mandelbrot set boundary required more iterations per pixel, leading to longer computation times across all implementations.
   - However, CUDA’s performance was the least affected by this variation, demonstrating its scalability for computationally intensive tasks.

**CUDA** is the clear winner in terms of raw performance, consistently outperforming both the original code and OpenMP. It is the best choice for accelerating highly parallelizable tasks like Mandelbrot fractal computation.
**OpenMP** offers a substantial improvement over the original code and is a suitable alternative for systems without GPUs.

The performance differences between zooms highlight the importance of computational load and how different implementations handle it. CUDA demonstrates the best scalability, making it ideal for scenarios involving variable computational complexity.

**Source of Improvement**:

1. **OpenMP**:
   - Parallelized the computation by splitting pixel rows among CPU threads, reducing the sequential nature of the original code.
   - Speedup is proportional to the number of available CPU cores, minus some overhead for thread synchronization.
2. **CUDA**:
   - Utilized thousands of GPU cores to compute each pixel in parallel.
   - High throughput and optimized memory access patterns (e.g., coalesced memory access) significantly reduced computation times.

### Potential future lines

We managed to accelerate our code quite a bit (especially with CUDA), so we don't see any future lines

New Bottlenecks

1. **CPU with OpenMP**:
   - **Memory bandwidth**: The CPU may struggle with memory access when threads compete for the same shared resources, limiting further scalability.
   - **Thread synchronization overhead**: Even with OpenMP, the synchronization overhead becomes noticeable as the number of threads increases.
2. **GPU with CUDA**:
   - **Memory transfer overhead**: Host-to-device and device-to-host memory transfers are relatively slow, introducing a bottleneck, especially for small data sizes.
   - **Compute-bound regions**: For highly complex zoom regions, the kernel's computational intensity becomes the bottleneck.
   - **Roofline model link**: CUDA achieves high computational throughput but is limited by memory-bound operations, particularly during frequent host-device data transfers.

What else to accelerate?

1. **Memory Transfers**:
   - Reduce host-device memory transfers by keeping data on the GPU as much as possible during iterative zoom computations.
   - Implement **pinned memory** to speed up data transfers.
2. **Dynamic Iteration Limits**:
   - Instead of using a static `MAX_ITER`, dynamically adjust the iteration limit for each pixel based on its proximity to the Mandelbrot set boundary.
   - This would reduce unnecessary computations for pixels that converge quickly.
3. **Shared Memory Optimization**:
   - Use CUDA shared memory to cache frequently accessed values, reducing global memory access overhead.

Could something have been done differently?

1. **Algorithmic Optimization**:
   - Incorporate early bailout strategies for regions of the Mandelbrot set that converge rapidly.
   - Use adaptive mesh refinement to focus computational resources on complex areas.
2. **Work Distribution**:
   - For OpenMP, experiment with **dynamic scheduling** to balance the workload more evenly across threads.
   - For CUDA, fine-tune thread block sizes to achieve optimal occupancy and performance.
3. **Data Structure Changes**:
   - Explore using single-channel grayscale textures instead of RGBA if color interpolation isn't a priority, reducing memory and computation overhead.
4. **Multi-GPU Scalability**:
   - Distribute the workload across multiple GPUs to handle larger resolutions or higher iteration limits more efficiently.

### **Conclusion**

The performance improvements achieved through OpenMP and CUDA demonstrate the potential of parallel computing for intensive tasks like Mandelbrot fractal generation. While CUDA provides exceptional acceleration, bottlenecks like memory transfers and computational intensity in complex regions remain areas for improvement. Future efforts should focus on reducing memory overhead, optimizing workload distribution, and exploring algorithmic enhancements to further accelerate the application.
