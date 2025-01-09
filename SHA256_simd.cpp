#include <immintrin.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static const uint32_t RC[] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define LOAD(src) _mm256_loadu_si256((__m256i *)(src))

#define XOR _mm256_xor_si256
#define OR _mm256_or_si256
#define AND _mm256_and_si256
#define ADD32 _mm256_add_epi32
#define NOT(x) _mm256_xor_si256(x, _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1))

#define LOAD(src) _mm256_loadu_si256((__m256i *)(src))
#define STORE(dest, src) _mm256_storeu_si256((__m256i *)(dest), src)

#define SHIFTR32(x, y) _mm256_srli_epi32(x, y)
#define SHIFTL32(x, y) _mm256_slli_epi32(x, y)

#define ROTR32(x, y) OR(SHIFTR32(x, y), SHIFTL32(x, 32 - y))
#define ROTL32(x, y) OR(SHIFTL32(x, y), SHIFTR32(x, 32 - y))

#define XOR3(a, b, c) XOR(XOR(a, b), c)

#define ADD3_32(a, b, c) ADD32(ADD32(a, b), c)
#define ADD4_32(a, b, c, d) ADD32(ADD32(ADD32(a, b), c), d)
#define ADD5_32(a, b, c, d, e) ADD32(ADD32(ADD32(ADD32(a, b), c), d), e)

#define MAJ_AVX(a, b, c) XOR3(AND(a, b), AND(a, c), AND(b, c))
#define CH_AVX(a, b, c) XOR(AND(a, b), AND(NOT(a), c))

#define SIGMA1_AVX(x) XOR3(ROTR32(x, 6), ROTR32(x, 11), ROTR32(x, 25))
#define SIGMA0_AVX(x) XOR3(ROTR32(x, 2), ROTR32(x, 13), ROTR32(x, 22))

#define WSIGMA1_AVX(x) XOR3(ROTR32(x, 17), ROTR32(x, 19), SHIFTR32(x, 10))
#define WSIGMA0_AVX(x) XOR3(ROTR32(x, 7), ROTR32(x, 18), SHIFTR32(x, 3))

#define SHA256ROUND_AVX(a, b, c, d, e, f, g, h, rc, w)                           \
  T0 = ADD5_32(h, SIGMA1_AVX(e), CH_AVX(e, f, g), _mm256_set1_epi32(RC[rc]), w); \
  d = ADD32(d, T0);                                                              \
  T1 = ADD32(SIGMA0_AVX(a), MAJ_AVX(a, b, c));                                   \
  h = ADD32(T0, T1)

void transpose(__m256i s[8])
{
  __m256i tmp0[8];
  __m256i tmp1[8];
  /*

s[0] = [A0 A1 A2 A3 A4 A5 A6 A7]
s[1] = [B0 B1 B2 B3 B4 B5 B6 B7]
s[2] = [C0 C1 C2 C3 C4 C5 C6 C7]
s[3] = [D0 D1 D2 D3 D4 D5 D6 D7]
s[4] = [E0 E1 E2 E3 E4 E5 E6 E7]
s[5] = [F0 F1 F2 F3 F4 F5 F6 F7]
s[6] = [G0 G1 G2 G3 G4 G5 G6 G7]
s[7] = [H0 H1 H2 H3 H4 H5 H6 H7]

*/
  tmp0[0] = _mm256_unpacklo_epi32(s[0], s[1]);  // Takes lower 4 elements: [A0 B0 A1 B1 A4 B4 A5 B5]
  tmp0[1] = _mm256_unpackhi_epi32(s[0], s[1]);  // Takes higher 4 elements: [A2 B2 A3 B3 A6 B6 A7 B7]
  tmp0[2] = _mm256_unpacklo_epi32(s[2], s[3]);  // [C0 D0 C1 D1 C4 D4 C5 D5]
  tmp0[3] = _mm256_unpackhi_epi32(s[2], s[3]);  // [C2 D2 C3 D3 C6 D6 C7 D7]
  tmp0[4] = _mm256_unpacklo_epi32(s[4], s[5]);  // [E0 F0 E1 F1 E4 F4 E5 F5]
  tmp0[5] = _mm256_unpackhi_epi32(s[4], s[5]);  // [E2 F2 E3 F3 E6 F6 E7 F7]
  tmp0[6] = _mm256_unpacklo_epi32(s[6], s[7]);  // [G0 H0 G1 H1 G4 H4 G5 H5]
  tmp0[7] = _mm256_unpackhi_epi32(s[6], s[7]);  // [G2 H2 G3 H3 G6 H6 G7 H7]

  tmp1[0] = _mm256_unpacklo_epi64(tmp0[0], tmp0[2]);  // [A0 B0 C0 D0 A4 B4 C4 D4]
  tmp1[1] = _mm256_unpackhi_epi64(tmp0[0], tmp0[2]);  // [A1 B1 C1 D1 A5 B5 C5 D5]
  tmp1[2] = _mm256_unpacklo_epi64(tmp0[1], tmp0[3]);  // [A2 B2 C2 D2 A6 B6 C6 D6]
  tmp1[3] = _mm256_unpackhi_epi64(tmp0[1], tmp0[3]);  // [A3 B3 C3 D3 A7 B7 C7 D7]
  tmp1[4] = _mm256_unpacklo_epi64(tmp0[4], tmp0[6]);  // [E0 F0 G0 H0 E4 F4 G4 H4]
  tmp1[5] = _mm256_unpackhi_epi64(tmp0[4], tmp0[6]);  // [E1 F1 G1 H1 E5 F5 G5 H5]
  tmp1[6] = _mm256_unpacklo_epi64(tmp0[5], tmp0[7]);  // [E2 F2 G2 H2 E6 F6 G6 H6]
  tmp1[7] = _mm256_unpackhi_epi64(tmp0[5], tmp0[7]);  // [E3 F3 G3 H3 E7 F7 G7 H7]

  // 0x20 = 00100000 binary - selects low 128 bits from first vector, low 128 from second
  s[0] = _mm256_permute2x128_si256(tmp1[0], tmp1[4], 0x20);
  // [A0 B0 C0 D0 E0 F0 G0 H0] - First row of transposed matrix
  s[1] = _mm256_permute2x128_si256(tmp1[1], tmp1[5], 0x20);
  s[2] = _mm256_permute2x128_si256(tmp1[2], tmp1[6], 0x20);
  s[3] = _mm256_permute2x128_si256(tmp1[3], tmp1[7], 0x20);

  // 0x31 = 00110001 binary - selects high 128 bits from first vector, high 128 from second
  s[4] = _mm256_permute2x128_si256(tmp1[0], tmp1[4], 0x31);
  // [A4 B4 C4 D4 E4 F4 G4 H4] - Fifth row of transposed matrix  s[5] = _mm256_permute2x128_si256(tmp1[1], tmp1[5], 0x31);
  s[6] = _mm256_permute2x128_si256(tmp1[2], tmp1[6], 0x31);
  s[7] = _mm256_permute2x128_si256(tmp1[3], tmp1[7], 0x31);
}

void hash(unsigned char *in, unsigned char *out)
{
  __m256i s[8], w[64], T0, T1;  // s -> State(a, b, c, d .. h) , W -> message schedule

  for (int i = 0; i < 8; i++)
  {
    w[i] = LOAD(in + 64 * i);
    w[i + 8] = LOAD(in + 32 + 64 * i);
  }

  // transpose message schedule
  transpose(w);
  transpose(w + 8);

  // Set the initial State
  s[0] = _mm256_set_epi32(0x6a09e667, 0x6a09e667, 0x6a09e667, 0x6a09e667, 0x6a09e667, 0x6a09e667, 0x6a09e667, 0x6a09e667);
  s[1] = _mm256_set_epi32(0xbb67ae85, 0xbb67ae85, 0xbb67ae85, 0xbb67ae85, 0xbb67ae85, 0xbb67ae85, 0xbb67ae85, 0xbb67ae85);
  s[2] = _mm256_set_epi32(0x3c6ef372, 0x3c6ef372, 0x3c6ef372, 0x3c6ef372, 0x3c6ef372, 0x3c6ef372, 0x3c6ef372, 0x3c6ef372);
  s[3] = _mm256_set_epi32(0xa54ff53a, 0xa54ff53a, 0xa54ff53a, 0xa54ff53a, 0xa54ff53a, 0xa54ff53a, 0xa54ff53a, 0xa54ff53a);
  s[4] = _mm256_set_epi32(0x510e527f, 0x510e527f, 0x510e527f, 0x510e527f, 0x510e527f, 0x510e527f, 0x510e527f, 0x510e527f);
  s[5] = _mm256_set_epi32(0x9b05688c, 0x9b05688c, 0x9b05688c, 0x9b05688c, 0x9b05688c, 0x9b05688c, 0x9b05688c, 0x9b05688c);
  s[6] = _mm256_set_epi32(0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab);
  s[7] = _mm256_set_epi32(0x5be0cd19, 0x5be0cd19, 0x5be0cd19, 0x5be0cd19, 0x5be0cd19, 0x5be0cd19, 0x5be0cd19, 0x5be0cd19);

  SHA256ROUND_AVX(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 0, w[0]);
  SHA256ROUND_AVX(s[7], s[0], s[1], s[2], s[3], s[4], s[5], s[6], 1, w[1]);
  SHA256ROUND_AVX(s[6], s[7], s[0], s[1], s[2], s[3], s[4], s[5], 2, w[2]);
  SHA256ROUND_AVX(s[5], s[6], s[7], s[0], s[1], s[2], s[3], s[4], 3, w[3]);
  SHA256ROUND_AVX(s[4], s[5], s[6], s[7], s[0], s[1], s[2], s[3], 4, w[4]);
  SHA256ROUND_AVX(s[3], s[4], s[5], s[6], s[7], s[0], s[1], s[2], 5, w[5]);
  SHA256ROUND_AVX(s[2], s[3], s[4], s[5], s[6], s[7], s[0], s[1], 6, w[6]);
  SHA256ROUND_AVX(s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[0], 7, w[7]);
  SHA256ROUND_AVX(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 8, w[8]);
  SHA256ROUND_AVX(s[7], s[0], s[1], s[2], s[3], s[4], s[5], s[6], 9, w[9]);
  SHA256ROUND_AVX(s[6], s[7], s[0], s[1], s[2], s[3], s[4], s[5], 10, w[10]);
  SHA256ROUND_AVX(s[5], s[6], s[7], s[0], s[1], s[2], s[3], s[4], 11, w[11]);
  SHA256ROUND_AVX(s[4], s[5], s[6], s[7], s[0], s[1], s[2], s[3], 12, w[12]);
  SHA256ROUND_AVX(s[3], s[4], s[5], s[6], s[7], s[0], s[1], s[2], 13, w[13]);
  SHA256ROUND_AVX(s[2], s[3], s[4], s[5], s[6], s[7], s[0], s[1], 14, w[14]);
  SHA256ROUND_AVX(s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[0], 15, w[15]);

  w[16] = ADD4_32(WSIGMA1_AVX(w[14]), w[0], w[9], WSIGMA0_AVX(w[1]));
  SHA256ROUND_AVX(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 16, w[16]);

  w[17] = ADD4_32(WSIGMA1_AVX(w[15]), w[1], w[10], WSIGMA0_AVX(w[2]));
  SHA256ROUND_AVX(s[7], s[0], s[1], s[2], s[3], s[4], s[5], s[6], 17, w[17]);

  w[18] = ADD4_32(WSIGMA1_AVX(w[16]), w[2], w[11], WSIGMA0_AVX(w[3]));
  SHA256ROUND_AVX(s[6], s[7], s[0], s[1], s[2], s[3], s[4], s[5], 18, w[18]);

  w[19] = ADD4_32(WSIGMA1_AVX(w[17]), w[3], w[12], WSIGMA0_AVX(w[4]));
  SHA256ROUND_AVX(s[5], s[6], s[7], s[0], s[1], s[2], s[3], s[4], 19, w[19]);

  w[20] = ADD4_32(WSIGMA1_AVX(w[18]), w[4], w[13], WSIGMA0_AVX(w[5]));
  SHA256ROUND_AVX(s[4], s[5], s[6], s[7], s[0], s[1], s[2], s[3], 20, w[20]);

  w[21] = ADD4_32(WSIGMA1_AVX(w[19]), w[5], w[14], WSIGMA0_AVX(w[6]));
  SHA256ROUND_AVX(s[3], s[4], s[5], s[6], s[7], s[0], s[1], s[2], 21, w[21]);

  w[22] = ADD4_32(WSIGMA1_AVX(w[20]), w[6], w[15], WSIGMA0_AVX(w[7]));
  SHA256ROUND_AVX(s[2], s[3], s[4], s[5], s[6], s[7], s[0], s[1], 22, w[22]);

  w[23] = ADD4_32(WSIGMA1_AVX(w[21]), w[7], w[16], WSIGMA0_AVX(w[8]));
  SHA256ROUND_AVX(s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[0], 23, w[23]);

  w[24] = ADD4_32(WSIGMA1_AVX(w[22]), w[8], w[17], WSIGMA0_AVX(w[9]));
  SHA256ROUND_AVX(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 24, w[24]);

  w[25] = ADD4_32(WSIGMA1_AVX(w[23]), w[9], w[18], WSIGMA0_AVX(w[10]));
  SHA256ROUND_AVX(s[7], s[0], s[1], s[2], s[3], s[4], s[5], s[6], 25, w[25]);

  w[26] = ADD4_32(WSIGMA1_AVX(w[24]), w[10], w[19], WSIGMA0_AVX(w[11]));
  SHA256ROUND_AVX(s[6], s[7], s[0], s[1], s[2], s[3], s[4], s[5], 26, w[26]);

  w[27] = ADD4_32(WSIGMA1_AVX(w[25]), w[11], w[20], WSIGMA0_AVX(w[12]));
  SHA256ROUND_AVX(s[5], s[6], s[7], s[0], s[1], s[2], s[3], s[4], 27, w[27]);

  w[28] = ADD4_32(WSIGMA1_AVX(w[26]), w[12], w[21], WSIGMA0_AVX(w[13]));
  SHA256ROUND_AVX(s[4], s[5], s[6], s[7], s[0], s[1], s[2], s[3], 28, w[28]);

  w[29] = ADD4_32(WSIGMA1_AVX(w[27]), w[13], w[22], WSIGMA0_AVX(w[14]));
  SHA256ROUND_AVX(s[3], s[4], s[5], s[6], s[7], s[0], s[1], s[2], 29, w[29]);

  w[30] = ADD4_32(WSIGMA1_AVX(w[28]), w[14], w[23], WSIGMA0_AVX(w[15]));
  SHA256ROUND_AVX(s[2], s[3], s[4], s[5], s[6], s[7], s[0], s[1], 30, w[30]);

  w[31] = ADD4_32(WSIGMA1_AVX(w[29]), w[15], w[24], WSIGMA0_AVX(w[16]));
  SHA256ROUND_AVX(s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[0], 31, w[31]);

  w[32] = ADD4_32(WSIGMA1_AVX(w[30]), w[16], w[25], WSIGMA0_AVX(w[17]));
  SHA256ROUND_AVX(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 32, w[32]);

  w[33] = ADD4_32(WSIGMA1_AVX(w[31]), w[17], w[26], WSIGMA0_AVX(w[18]));
  SHA256ROUND_AVX(s[7], s[0], s[1], s[2], s[3], s[4], s[5], s[6], 33, w[33]);

  w[34] = ADD4_32(WSIGMA1_AVX(w[32]), w[18], w[27], WSIGMA0_AVX(w[19]));
  SHA256ROUND_AVX(s[6], s[7], s[0], s[1], s[2], s[3], s[4], s[5], 34, w[34]);

  w[35] = ADD4_32(WSIGMA1_AVX(w[33]), w[19], w[28], WSIGMA0_AVX(w[20]));
  SHA256ROUND_AVX(s[5], s[6], s[7], s[0], s[1], s[2], s[3], s[4], 35, w[35]);

  w[36] = ADD4_32(WSIGMA1_AVX(w[34]), w[20], w[29], WSIGMA0_AVX(w[21]));
  SHA256ROUND_AVX(s[4], s[5], s[6], s[7], s[0], s[1], s[2], s[3], 36, w[36]);

  w[37] = ADD4_32(WSIGMA1_AVX(w[35]), w[21], w[30], WSIGMA0_AVX(w[22]));
  SHA256ROUND_AVX(s[3], s[4], s[5], s[6], s[7], s[0], s[1], s[2], 37, w[37]);

  w[38] = ADD4_32(WSIGMA1_AVX(w[36]), w[22], w[31], WSIGMA0_AVX(w[23]));
  SHA256ROUND_AVX(s[2], s[3], s[4], s[5], s[6], s[7], s[0], s[1], 38, w[38]);

  w[39] = ADD4_32(WSIGMA1_AVX(w[37]), w[23], w[32], WSIGMA0_AVX(w[24]));
  SHA256ROUND_AVX(s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[0], 39, w[39]);

  w[40] = ADD4_32(WSIGMA1_AVX(w[38]), w[24], w[33], WSIGMA0_AVX(w[25]));
  SHA256ROUND_AVX(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 40, w[40]);

  w[41] = ADD4_32(WSIGMA1_AVX(w[39]), w[25], w[34], WSIGMA0_AVX(w[26]));
  SHA256ROUND_AVX(s[7], s[0], s[1], s[2], s[3], s[4], s[5], s[6], 41, w[41]);

  w[42] = ADD4_32(WSIGMA1_AVX(w[40]), w[26], w[35], WSIGMA0_AVX(w[27]));
  SHA256ROUND_AVX(s[6], s[7], s[0], s[1], s[2], s[3], s[4], s[5], 42, w[42]);

  w[43] = ADD4_32(WSIGMA1_AVX(w[41]), w[27], w[36], WSIGMA0_AVX(w[28]));
  SHA256ROUND_AVX(s[5], s[6], s[7], s[0], s[1], s[2], s[3], s[4], 43, w[43]);

  w[44] = ADD4_32(WSIGMA1_AVX(w[42]), w[28], w[37], WSIGMA0_AVX(w[29]));
  SHA256ROUND_AVX(s[4], s[5], s[6], s[7], s[0], s[1], s[2], s[3], 44, w[44]);

  w[45] = ADD4_32(WSIGMA1_AVX(w[43]), w[29], w[38], WSIGMA0_AVX(w[30]));
  SHA256ROUND_AVX(s[3], s[4], s[5], s[6], s[7], s[0], s[1], s[2], 45, w[45]);

  w[46] = ADD4_32(WSIGMA1_AVX(w[44]), w[30], w[39], WSIGMA0_AVX(w[31]));
  SHA256ROUND_AVX(s[2], s[3], s[4], s[5], s[6], s[7], s[0], s[1], 46, w[46]);

  w[47] = ADD4_32(WSIGMA1_AVX(w[45]), w[31], w[40], WSIGMA0_AVX(w[32]));
  SHA256ROUND_AVX(s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[0], 47, w[47]);

  w[48] = ADD4_32(WSIGMA1_AVX(w[46]), w[32], w[41], WSIGMA0_AVX(w[33]));
  SHA256ROUND_AVX(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 48, w[48]);

  w[49] = ADD4_32(WSIGMA1_AVX(w[47]), w[33], w[42], WSIGMA0_AVX(w[34]));
  SHA256ROUND_AVX(s[7], s[0], s[1], s[2], s[3], s[4], s[5], s[6], 49, w[49]);

  w[50] = ADD4_32(WSIGMA1_AVX(w[48]), w[34], w[43], WSIGMA0_AVX(w[35]));
  SHA256ROUND_AVX(s[6], s[7], s[0], s[1], s[2], s[3], s[4], s[5], 50, w[50]);

  w[51] = ADD4_32(WSIGMA1_AVX(w[49]), w[35], w[44], WSIGMA0_AVX(w[36]));
  SHA256ROUND_AVX(s[5], s[6], s[7], s[0], s[1], s[2], s[3], s[4], 51, w[51]);

  w[52] = ADD4_32(WSIGMA1_AVX(w[50]), w[36], w[45], WSIGMA0_AVX(w[37]));
  SHA256ROUND_AVX(s[4], s[5], s[6], s[7], s[0], s[1], s[2], s[3], 52, w[52]);

  w[53] = ADD4_32(WSIGMA1_AVX(w[51]), w[37], w[46], WSIGMA0_AVX(w[38]));
  SHA256ROUND_AVX(s[3], s[4], s[5], s[6], s[7], s[0], s[1], s[2], 53, w[53]);

  w[54] = ADD4_32(WSIGMA1_AVX(w[52]), w[38], w[47], WSIGMA0_AVX(w[39]));
  SHA256ROUND_AVX(s[2], s[3], s[4], s[5], s[6], s[7], s[0], s[1], 54, w[54]);

  w[55] = ADD4_32(WSIGMA1_AVX(w[53]), w[39], w[48], WSIGMA0_AVX(w[40]));
  SHA256ROUND_AVX(s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[0], 55, w[55]);

  w[56] = ADD4_32(WSIGMA1_AVX(w[54]), w[40], w[49], WSIGMA0_AVX(w[41]));
  SHA256ROUND_AVX(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 56, w[56]);

  w[57] = ADD4_32(WSIGMA1_AVX(w[55]), w[41], w[50], WSIGMA0_AVX(w[42]));
  SHA256ROUND_AVX(s[7], s[0], s[1], s[2], s[3], s[4], s[5], s[6], 57, w[57]);

  w[58] = ADD4_32(WSIGMA1_AVX(w[56]), w[42], w[51], WSIGMA0_AVX(w[43]));
  SHA256ROUND_AVX(s[6], s[7], s[0], s[1], s[2], s[3], s[4], s[5], 58, w[58]);

  w[59] = ADD4_32(WSIGMA1_AVX(w[57]), w[43], w[52], WSIGMA0_AVX(w[44]));
  SHA256ROUND_AVX(s[5], s[6], s[7], s[0], s[1], s[2], s[3], s[4], 59, w[59]);

  w[60] = ADD4_32(WSIGMA1_AVX(w[58]), w[44], w[53], WSIGMA0_AVX(w[45]));
  SHA256ROUND_AVX(s[4], s[5], s[6], s[7], s[0], s[1], s[2], s[3], 60, w[60]);

  w[61] = ADD4_32(WSIGMA1_AVX(w[59]), w[45], w[54], WSIGMA0_AVX(w[46]));
  SHA256ROUND_AVX(s[3], s[4], s[5], s[6], s[7], s[0], s[1], s[2], 61, w[61]);

  w[62] = ADD4_32(WSIGMA1_AVX(w[60]), w[46], w[55], WSIGMA0_AVX(w[47]));
  SHA256ROUND_AVX(s[2], s[3], s[4], s[5], s[6], s[7], s[0], s[1], 62, w[62]);

  w[63] = ADD4_32(WSIGMA1_AVX(w[61]), w[47], w[56], WSIGMA0_AVX(w[48]));
  SHA256ROUND_AVX(s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[0], 63, w[63]);

  // Feed Forward
  s[0] = ADD32(s[0], _mm256_set_epi32(0x6a09e667, 0x6a09e667, 0x6a09e667, 0x6a09e667, 0x6a09e667, 0x6a09e667, 0x6a09e667, 0x6a09e667));
  s[1] = ADD32(s[1], _mm256_set_epi32(0xbb67ae85, 0xbb67ae85, 0xbb67ae85, 0xbb67ae85, 0xbb67ae85, 0xbb67ae85, 0xbb67ae85, 0xbb67ae85));
  s[2] = ADD32(s[2], _mm256_set_epi32(0x3c6ef372, 0x3c6ef372, 0x3c6ef372, 0x3c6ef372, 0x3c6ef372, 0x3c6ef372, 0x3c6ef372, 0x3c6ef372));
  s[3] = ADD32(s[3], _mm256_set_epi32(0xa54ff53a, 0xa54ff53a, 0xa54ff53a, 0xa54ff53a, 0xa54ff53a, 0xa54ff53a, 0xa54ff53a, 0xa54ff53a));
  s[4] = ADD32(s[4], _mm256_set_epi32(0x510e527f, 0x510e527f, 0x510e527f, 0x510e527f, 0x510e527f, 0x510e527f, 0x510e527f, 0x510e527f));
  s[5] = ADD32(s[5], _mm256_set_epi32(0x9b05688c, 0x9b05688c, 0x9b05688c, 0x9b05688c, 0x9b05688c, 0x9b05688c, 0x9b05688c, 0x9b05688c));
  s[6] = ADD32(s[6], _mm256_set_epi32(0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab, 0x1f83d9ab));
  s[7] = ADD32(s[7], _mm256_set_epi32(0x5be0cd19, 0x5be0cd19, 0x5be0cd19, 0x5be0cd19, 0x5be0cd19, 0x5be0cd19, 0x5be0cd19, 0x5be0cd19));

  // Transpose data again to get correct output
  transpose(s);

  // Store Hash value
  for (int i = 0; i < 8; i++) STORE(out + 32 * i, s[i]);
  return;
}
std::string hash(const std::string &input)
{
  unsigned char in[64 * 8] = {0};
  unsigned char out[32 * 8] = {0};

  std::memcpy(in, input.data(), input.size());
  hash(in, out);

  std::stringstream ss;
  for (int i = 0; i < 32; i++) ss << std::hex << std::setw(2) << std::setfill('0') << (int)out[i];

  return ss.str();
}

void benchmark(const std::string &input, int duration_seconds, int num_threads)
{
  std::atomic<int> iterations{0};
  std::vector<std::thread> threads;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_threads; ++i)
  {
    threads.emplace_back(
        [&iterations, &input, start, duration_seconds]()
        {
          while (true)
          {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
            if (elapsed >= duration_seconds)
              break;
            hash(input);
            iterations += 8;
          }
        });
  }

  for (auto &t : threads) t.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << "Time taken: " << duration << " Seconds\n";
  std::cout << "Number of hashes performed: " << iterations << "\n";
  std::cout << "Speed: " << (float)iterations / (duration * 1000000) << " MH/s\n";
}

int main()
{
  std::string input = "Hello Vicharak!";
  int duration_seconds = 5;
  int num_threads = std::thread::hardware_concurrency();
  benchmark(input, duration_seconds, num_threads);

  return 0;
}
