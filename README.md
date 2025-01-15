# SHA-256 Benchmarking Report

In this project, I explored various implementations of the SHA-256 hashing algorithm, focusing on optimizing performance across different hardware architectures. The goal was to understand how different approaches—single-threaded, multi-threaded, SIMD (Single Instruction, Multiple Data), and CUDA impact the speed and efficiency of SHA-256 hashing. The results were benchmarked on my Lenovo Legion 5i gaming laptop, equipped with an Intel i5-10500H CPU and an NVIDIA GTX 1650 GPU with 4GB of VRAM.

### 1. **Single-Threaded Implementation**

- ~1.5 MH/s (Mega Hashes per second)
- This is a straightforward, single-threaded implementation of the SHA-256 algorithm. It serves as the baseline for comparison with more optimized versions. The code processes data in 512-bit blocks, updating the hash state for each block. While simple, this implementation is limited by its inability to leverage modern multi-core CPUs or GPU acceleration.

### 2. **Multi-Threaded Implementation**

- ~5.7 MH/s
- To improve performance, I parallelized the SHA-256 algorithm using C++'s threading capabilities. By dividing the workload across multiple CPU cores, this implementation significantly increased the hashing speed. The `std::thread` library was used to create worker threads, each responsible for processing a single hash. This achieved nearly 4x the performance of the single-threaded version.

### 3. **SIMD (AVX2) Implementation**

- ~15 MH/s
- Leveraging Intel's AVX2 (Advanced Vector Extensions) instructions, this implementation processes multiple data points in parallel within a single CPU core. By using 256-bit wide registers, the SIMD version can perform operations on 8 integers simultaneously, significantly speeding up the hashing process. The code includes custom macros for AVX2 operations, such as `ADD32`, `ROTR32`, and `MAJ_AVX`, which are used to optimize the SHA-256 computation. This implementation achieved a 10x speedup over the single-threaded version.

### 4. **CUDA Implementation**

- ~270 MH/s (initial), ~1000 MH/s (optimized), ~1200 MH/s (further optimized)
- The CUDA implementation offloads the SHA-256 computation to the GPU, which is capable of executing thousands of threads in parallel. The initial version achieved ~270 MH/s, but after several optimizations—such as reducing memory latency and increasing the number of iterations—the performance improved to ~1000 MH/s. Further tweaks, including better utilization of GPU resources, pushed the speed to ~1200 MH/s.

### 5. **Bitcoin Miner Implementation**

- **Reference:** [CUDA Bitcoin Miner](https://github.com/geedo0/cuda_bitcoin_miner)
- ~173 MH/s
- For comparison, I also tested a CUDA-based Bitcoin miner implementation, which is optimized for mining rather than general-purpose SHA-256 hashing. While it achieved a respectable ~173 MH/s.

---

For a theoretical analysis of the performance limits and bottlenecks, refer to the [ANALYSIS.md](ANALYSIS.md) file.
