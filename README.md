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

TODO

## Stage 4 - Analysis of results

We were unable to run the program on the card (we couldn't install everything on), but luckily one of our laptops has a Nvidia GPU, so we could run it there.

With CUDA it took 0.69 seconds for the initial fractal and 0.58 seconds for the next zoom (we used `WIDTH` = 1000, `HEIGHT` = 800, and `MAX_ITER` = 3000)

With OpenMP it took 2.13 seconds and then 9.07 seconds (with the same parameters).

Both optimizations are way faster than the base code.
