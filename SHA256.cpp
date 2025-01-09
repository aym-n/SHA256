#include <immintrin.h>  // AVX intrinsics

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

class SHA256
{
public:
  SHA256() { reset(); }

  void update(const char *data, size_t len)
  {
    for (size_t i = 0; i < len; ++i)
    {
      buffer[bufferLength++] = data[i];
      if (bufferLength == 64)
      {
        transform(buffer.data());
        bufferLength = 0;
        bitLength += 512;
      }
    }
  }

  vector<uint8_t> finalize()
  {
    vector<uint8_t> hash(32, 0);
    pad();
    transform(buffer.data());
    for (size_t i = 0; i < 8; ++i)
    {
      hash[i * 4 + 0] = (state[i] >> 24) & 0xff;
      hash[i * 4 + 1] = (state[i] >> 16) & 0xff;
      hash[i * 4 + 2] = (state[i] >> 8) & 0xff;
      hash[i * 4 + 3] = state[i] & 0xff;
    }
    reset();
    return hash;
  }

private:
  void reset()
  {
    state[0] = 0x6a09e667;
    state[1] = 0xbb67ae85;
    state[2] = 0x3c6ef372;
    state[3] = 0xa54ff53a;
    state[4] = 0x510e527f;
    state[5] = 0x9b05688c;
    state[6] = 0x1f83d9ab;
    state[7] = 0x5be0cd19;
    bitLength = 0;
    bufferLength = 0;
    buffer.assign(64, 0);
  }

  void transform(const uint8_t data[])
  {
    __m256i a, b, c, d, e, f, g, h;
    __m256i W[64];
    __m256i T1, T2;

    // Load initial state into AVX registers
    a = _mm256_set1_epi32(state[0]);
    b = _mm256_set1_epi32(state[1]);
    c = _mm256_set1_epi32(state[2]);
    d = _mm256_set1_epi32(state[3]);
    e = _mm256_set1_epi32(state[4]);
    f = _mm256_set1_epi32(state[5]);
    g = _mm256_set1_epi32(state[6]);
    h = _mm256_set1_epi32(state[7]);

    // Prepare message schedule
    for (int i = 0; i < 16; ++i)
      W[i] = _mm256_set1_epi32((data[i * 4] << 24) | (data[i * 4 + 1] << 16) | (data[i * 4 + 2] << 8) | (data[i * 4 + 3]));
    for (int i = 16; i < 64; ++i)
      W[i] = _mm256_add_epi32(_mm256_add_epi32(SIG1(W[i - 2]), W[i - 7]), _mm256_add_epi32(SIG0(W[i - 15]), W[i - 16]));

    // SHA-256 compression function
    for (int i = 0; i < 64; ++i)
    {
      T1 = _mm256_add_epi32(_mm256_add_epi32(h, EP1(e)), _mm256_add_epi32(CH(e, f, g), _mm256_add_epi32(_mm256_set1_epi32(K[i]), W[i])));
      T2 = _mm256_add_epi32(EP0(a), MAJ(a, b, c));
      h = g;
      g = f;
      f = e;
      e = _mm256_add_epi32(d, T1);
      d = c;
      c = b;
      b = a;
      a = _mm256_add_epi32(T1, T2);
    }

    // Update state
    state[0] += _mm256_extract_epi32(a, 0);
    state[1] += _mm256_extract_epi32(b, 0);
    state[2] += _mm256_extract_epi32(c, 0);
    state[3] += _mm256_extract_epi32(d, 0);
    state[4] += _mm256_extract_epi32(e, 0);
    state[5] += _mm256_extract_epi32(f, 0);
    state[6] += _mm256_extract_epi32(g, 0);
    state[7] += _mm256_extract_epi32(h, 0);
  }

  void pad()
  {
    size_t orig_len = bufferLength;
    size_t pad_len;

    buffer[orig_len] = 0x80;
    pad_len = (orig_len < 56) ? (56 - orig_len) : (120 - orig_len);
    memset(buffer.data() + orig_len + 1, 0, pad_len - 1);

    uint64_t bit_len = bitLength + bufferLength * 8;
    for (int i = 0; i < 8; ++i) buffer[bufferLength + pad_len + i] = (bit_len >> (56 - 8 * i)) & 0xff;
    bufferLength += pad_len + 8;
  }

  static __m256i rightRotate(__m256i x, int n) { return _mm256_or_si256(_mm256_srli_epi32(x, n), _mm256_slli_epi32(x, 32 - n)); }

  static __m256i SIG0(__m256i x)
  {
    return _mm256_xor_si256(_mm256_xor_si256(rightRotate(x, 7), rightRotate(x, 18)), _mm256_srli_epi32(x, 3));
  }

  static __m256i SIG1(__m256i x)
  {
    return _mm256_xor_si256(_mm256_xor_si256(rightRotate(x, 17), rightRotate(x, 19)), _mm256_srli_epi32(x, 10));
  }

  static __m256i EP0(__m256i x) { return _mm256_xor_si256(_mm256_xor_si256(rightRotate(x, 2), rightRotate(x, 13)), rightRotate(x, 22)); }

  static __m256i EP1(__m256i x) { return _mm256_xor_si256(_mm256_xor_si256(rightRotate(x, 6), rightRotate(x, 11)), rightRotate(x, 25)); }

  static __m256i CH(__m256i x, __m256i y, __m256i z)
  {
    return _mm256_xor_si256(_mm256_and_si256(x, y), _mm256_and_si256(_mm256_andnot_si256(x, _mm256_set1_epi32(-1)), z));
  }

  static __m256i MAJ(__m256i x, __m256i y, __m256i z)
  {
    return _mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(x, y), _mm256_and_si256(y, z)), _mm256_and_si256(x, z));
  }

  static const uint32_t K[64];

  uint32_t state[8];
  uint64_t bitLength;
  size_t bufferLength;
  vector<uint8_t> buffer;
};

const uint32_t SHA256::K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

void benchmark(const string &input, int duration_seconds)
{
  SHA256 hasher;
  int iterations = 0;
  auto start = chrono::high_resolution_clock::now();

  while (true)
  {
    auto now = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::seconds>(now - start).count();

    if (elapsed >= duration_seconds)
      break;

    hasher.update(reinterpret_cast<const char *>(input.data()), input.size());
    iterations++;
  }

  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::seconds>(end - start).count();

  cout << "Time taken: " << duration << " Seconds\n";
  cout << "Number of hashes performed: " << iterations << "\n";
  cout << "Speed: " << (float)iterations / (duration * 1000000) << " MH/s\n";
}

int main()
{
  string input = "Hello Vicharak";
  benchmark(input, 5);
}
