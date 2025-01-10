#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;
__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

__constant__ uint8_t constant_input[64];

__device__ uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ uint32_t sigma0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
__device__ uint32_t sigma1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
__device__ uint32_t gamma0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
__device__ uint32_t gamma1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

__global__ void sha256_kernel(uint32_t *output, size_t num_hashes)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_hashes)
    return;

  uint32_t state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint32_t w[64];

#pragma unroll
  for (int i = 0; i < 16; ++i)
  {
    w[i] =
        (constant_input[i * 4] << 24) | (constant_input[i * 4 + 1] << 16) | (constant_input[i * 4 + 2] << 8) | (constant_input[i * 4 + 3]);
  }

#pragma unroll
  for (int i = 16; i < 64; ++i) w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];

  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

#pragma unroll
  for (int i = 0; i < 64; ++i)
  {
    uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K[i] + w[i];
    uint32_t t2 = sigma0(a) + maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;

  for (int i = 0; i < 8; ++i) output[idx * 8 + i] = state[i];
}

void benchmark(const string &input, int num_hashes)
{
  cout << "[Starting Benchmark]" << endl;
  size_t len = input.size();

  uint8_t h_input[64] = {0};
  memcpy(h_input, input.data(), len);
  h_input[len] = 0x80;  // Padding
                        //
  uint64_t bit_len = len * 8;
  for (int i = 0; i < 8; ++i) h_input[56 + i] = (bit_len >> (56 - 8 * i)) & 0xff;

  cudaMemcpyToSymbol(constant_input, h_input, 64);
  vector<uint32_t> h_output(num_hashes * 8);
  uint32_t *d_output;
  cudaMalloc(&d_output, num_hashes * 8 * sizeof(uint32_t));

  int threadsPerBlock = 256;
  int blocksPerGrid = (num_hashes + threadsPerBlock - 1) / threadsPerBlock;

  cout << "[Start Calculating]" << endl;
  auto start = chrono::high_resolution_clock::now();
  sha256_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, num_hashes);
  cudaDeviceSynchronize();
  auto end = chrono::high_resolution_clock::now();

  cudaMemcpy(h_output.data(), d_output, num_hashes * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaFree(d_output);

  auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
  cout << "Time taken: " << duration << " ms\n";
  cout << "Speed: " << (float)num_hashes / (duration / 1000.0) / 1e6 << " MH/s\n";

  cout << "\nSHA256:\n";
  for (int i = 0; i < 8; ++i) cout << hex << setw(8) << setfill('0') << h_output[i] << " ";
  cout << dec << endl;
}

int main()
{
  string input = "Hello Vicharak";
  benchmark(input, 10000000);
  return 0;
}
