#include <immintrin.h>

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
        transform_avx(buffer.data());
        bufferLength = 0;
        bitLength += 512;
      }
    }
  }

  vector<uint8_t> finalize()
  {
    vector<uint8_t> hash(32, 0);
    pad();
    transform_avx(buffer.data());  // Use AVX transform
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

  void transform_avx(const uint8_t data[64 * 8])
  {
    __m256i s[8], w[64], T0, T1;

    // Load words and transform data correctly
    for (int i = 0; i < 8; i++)
    {
      w[i] = _mm256_loadu_si256((__m256i *)(data + 64 * i));
      w[i + 8] = _mm256_loadu_si256((__m256i *)(data + 32 + 64 * i));
    }

    transpose(w);
    transpose(w + 8);

    // Initial State
    s[0] = _mm256_set1_epi32(0x6a09e667);
    s[1] = _mm256_set1_epi32(0xbb67ae85);
    s[2] = _mm256_set1_epi32(0x3c6ef372);
    s[3] = _mm256_set1_epi32(0xa54ff53a);
    s[4] = _mm256_set1_epi32(0x510e527f);
    s[5] = _mm256_set1_epi32(0x9b05688c);
    s[6] = _mm256_set1_epi32(0x1f83d9ab);
    s[7] = _mm256_set1_epi32(0x5be0cd19);

    // SHA-256 rounds
    for (int i = 0; i < 64; i++)
    {
      T0 = _mm256_add_epi32(
          _mm256_add_epi32(_mm256_add_epi32(s[7], _mm256_set1_epi32(K[i])),
                           _mm256_add_epi32(_mm256_xor_si256(_mm256_xor_si256(_mm256_srli_epi32(s[4], 6), _mm256_srli_epi32(s[4], 11)),
                                                             _mm256_srli_epi32(s[4], 25)),
                                            _mm256_xor_si256(_mm256_and_si256(s[4], s[5]), _mm256_andnot_si256(s[4], s[6])))),
          w[i]);
      T1 = _mm256_add_epi32(
          _mm256_xor_si256(_mm256_xor_si256(_mm256_srli_epi32(s[0], 2), _mm256_srli_epi32(s[0], 13)), _mm256_srli_epi32(s[0], 22)),
          _mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(s[0], s[1]), _mm256_and_si256(s[0], s[2])), _mm256_and_si256(s[1], s[2])));
      s[7] = s[6];
      s[6] = s[5];
      s[5] = s[4];
      s[4] = _mm256_add_epi32(s[3], T0);
      s[3] = s[2];
      s[2] = s[1];
      s[1] = s[0];
      s[0] = _mm256_add_epi32(T0, T1);
    }

    // Feed Forward
    s[0] = _mm256_add_epi32(s[0], _mm256_set1_epi32(0x6a09e667));
    s[1] = _mm256_add_epi32(s[1], _mm256_set1_epi32(0xbb67ae85));
    s[2] = _mm256_add_epi32(s[2], _mm256_set1_epi32(0x3c6ef372));
    s[3] = _mm256_add_epi32(s[3], _mm256_set1_epi32(0xa54ff53a));
    s[4] = _mm256_add_epi32(s[4], _mm256_set1_epi32(0x510e527f));
    s[5] = _mm256_add_epi32(s[5], _mm256_set1_epi32(0x9b05688c));
    s[6] = _mm256_add_epi32(s[6], _mm256_set1_epi32(0x1f83d9ab));
    s[7] = _mm256_add_epi32(s[7], _mm256_set1_epi32(0x5be0cd19));

    // Transpose data again to get correct output
    transpose(s);

    // Store Hash value
    for (int i = 0; i < 8; i++) _mm256_storeu_si256((__m256i *)(state + i * 8), s[i]);
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
    uint32_t a, b, c, d, e, f, g, h, i, j, T1, T2, W[64];

    for (i = 0, j = 0; i < 16; ++i, j += 4) W[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
    for (; i < 64; ++i) W[i] = SIG1(W[i - 2]) + W[i - 7] + SIG0(W[i - 15]) + W[i - 16];

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    for (i = 0; i < 64; ++i)
    {
      T1 = h + EP1(e) + CH(e, f, g) + K[i] + W[i];
      T2 = EP0(a) + MAJ(a, b, c);
      h = g;
      g = f;
      f = e;
      e = d + T1;
      d = c;
      c = b;
      b = a;
      a = T1 + T2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
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

  static uint32_t rightRotate(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  static uint32_t SIG0(uint32_t x) { return rightRotate(x, 7) ^ rightRotate(x, 18) ^ (x >> 3); }
  static uint32_t SIG1(uint32_t x) { return rightRotate(x, 17) ^ rightRotate(x, 19) ^ (x >> 10); }
  static uint32_t EP0(uint32_t x) { return rightRotate(x, 2) ^ rightRotate(x, 13) ^ rightRotate(x, 22); }
  static uint32_t EP1(uint32_t x) { return rightRotate(x, 6) ^ rightRotate(x, 11) ^ rightRotate(x, 25); }
  static uint32_t CH(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
  static uint32_t MAJ(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (y & z) ^ (x & z); }

  static const uint32_t K[64];

  uint32_t state[8];
  uint64_t bitLength;
  size_t bufferLength;
  vector<uint8_t> buffer;

  // AVX-specific helper functions
  void transpose(__m256i s[8])
  {
    __m256i tmp0[8], tmp1[8];
    tmp0[0] = _mm256_unpacklo_epi32(s[0], s[1]);
    tmp0[1] = _mm256_unpackhi_epi32(s[0], s[1]);
    tmp0[2] = _mm256_unpacklo_epi32(s[2], s[3]);
    tmp0[3] = _mm256_unpackhi_epi32(s[2], s[3]);
    tmp0[4] = _mm256_unpacklo_epi32(s[4], s[5]);
    tmp0[5] = _mm256_unpackhi_epi32(s[4], s[5]);
    tmp0[6] = _mm256_unpacklo_epi32(s[6], s[7]);
    tmp0[7] = _mm256_unpackhi_epi32(s[6], s[7]);

    tmp1[0] = _mm256_unpacklo_epi64(tmp0[0], tmp0[2]);
    tmp1[1] = _mm256_unpackhi_epi64(tmp0[0], tmp0[2]);
    tmp1[2] = _mm256_unpacklo_epi64(tmp0[1], tmp0[3]);
    tmp1[3] = _mm256_unpackhi_epi64(tmp0[1], tmp0[3]);
    tmp1[4] = _mm256_unpacklo_epi64(tmp0[4], tmp0[6]);
    tmp1[5] = _mm256_unpackhi_epi64(tmp0[4], tmp0[6]);
    tmp1[6] = _mm256_unpacklo_epi64(tmp0[5], tmp0[7]);
    tmp1[7] = _mm256_unpackhi_epi64(tmp0[5], tmp0[7]);

    s[0] = _mm256_permute2x128_si256(tmp1[0], tmp1[4], 0x20);
    s[1] = _mm256_permute2x128_si256(tmp1[1], tmp1[5], 0x20);
    s[2] = _mm256_permute2x128_si256(tmp1[2], tmp1[6], 0x20);
    s[3] = _mm256_permute2x128_si256(tmp1[3], tmp1[7], 0x20);
    s[4] = _mm256_permute2x128_si256(tmp1[0], tmp1[4], 0x31);
    s[5] = _mm256_permute2x128_si256(tmp1[1], tmp1[5], 0x31);
    s[6] = _mm256_permute2x128_si256(tmp1[2], tmp1[6], 0x31);
    s[7] = _mm256_permute2x128_si256(tmp1[3], tmp1[7], 0x31);
  }
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
