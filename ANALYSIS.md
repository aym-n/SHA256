## Theoretical Performance Analysis

To understand the theoretical limits of SHA-256 hashing performance, we need to consider both the computational and memory bandwidth constraints of the hardware. Here's a breakdown of the calculations and their implications:

### Computational Capacity

- **Operations per Hash:** A single SHA-256 hash requires approximately **2,200 operations**. This includes the arithmetic and logical steps involved in processing each 512-bit block of data, such as additions, rotations, and bitwise operations.
- **GPU Compute Capability:** My NVIDIA GTX 1650 GPU has a theoretical peak performance of **6.390 TFLOPS** (trillion floating-point operations per second). Assuming each hash operation can be mapped to a floating-point operation, the maximum number of hashes per second the GPU can handle is:
  \[
  \text{Max Hashes/s (Compute)} = \frac{6.390 \times 10^{12} \text{ FLOPS}}{2,200 \text{ ops/hash}} \approx 2.9 \times 10^9 \text{ hashes/s} \ (\text{2.9 GH/s})
  \]
  This suggests that, computationally, the GPU could theoretically handle up to **2.9 billion hashes per second**.

### Memory Bandwidth

- **Memory Transfer per Hash:** Each SHA-256 hash involves transferring **96 bytes** of data. This includes loading the input data, and storing the final hash output.
- **GPU Memory Bandwidth:** The GTX 1650 has a memory bandwidth of **128.1 GB/s**. The maximum number of hashes per second limited by memory bandwidth is:
  \[
  \text{Max Hashes/s (Memory)} = \frac{128.1 \times 10^9 \text{ bytes/s}}{96 \text{ bytes/hash}} \approx 1.33 \times 10^9 \text{ hashes/s} \ (\text{1.33 GH/s})
  \]
  This indicates that the memory subsystem can support up to **1.33 billion hashes per second**.

### Bottleneck Analysis

- While the GPU's compute capability suggests a potential of **2.9 GH/s**, the memory bandwidth limits the performance to **1.33 GH/s**. This means the system is **memory bottlenecked**—the GPU's computational power cannot be fully utilized because the memory subsystem cannot keep up with the data transfer demands. In real-world scenarios, the actual performance will be closer to the memory-bound limit of **1.33 GH/s**. This explains why my optimized CUDA implementation achieved **~1.2 GH/s**, which is near the theoretical memory limit.

### How the Values Were Calculated

- **2,200 Operations per Hash:** This is derived from the SHA-256 algorithm's structure, which involves 64 rounds of processing per 512-bit block. Each round includes multiple arithmetic and logical operations, totaling approximately 2,200 operations per hash.

- **96 Bytes per Hash:** This includes:
  - 64 bytes for the input block.
  - 32 bytes for the final hash output.
  - Additional overhead for intermediate state and control data is minimal as compared to input and output read/write.

### Conclusion

The theoretical analysis shows that SHA-256 hashing is **memory bottlenecked** on my GPU. While the compute capability suggests a higher potential, the memory bandwidth limits the practical performance to around **1.33 GH/s**. This aligns closely with the observed performance of **~1.2 GH/s** in my CUDA implementation
