// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/digest.h"

#include <string.h>

typedef struct loom_serve_media_sha256_t {
  uint32_t state[8];
  uint64_t data_length;
  uint8_t block[64];
  uint8_t block_length;
} loom_serve_media_sha256_t;

static const uint32_t loom_serve_media_sha256_round_constants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static inline uint32_t loom_serve_media_rotate_right(uint32_t value,
                                                     uint32_t count) {
  return (value >> count) | (value << (32u - count));
}

static inline uint32_t loom_serve_media_load_be_u32(const uint8_t* data) {
  return ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) |
         ((uint32_t)data[2] << 8) | (uint32_t)data[3];
}

static inline void loom_serve_media_store_be_u32(uint32_t value,
                                                 uint8_t* data) {
  data[0] = (uint8_t)(value >> 24);
  data[1] = (uint8_t)(value >> 16);
  data[2] = (uint8_t)(value >> 8);
  data[3] = (uint8_t)value;
}

static void loom_serve_media_sha256_transform(
    loom_serve_media_sha256_t* context, const uint8_t block[64]) {
  uint32_t schedule[64];
  for (uint32_t i = 0; i < 16; ++i) {
    schedule[i] = loom_serve_media_load_be_u32(block + i * 4);
  }
  for (uint32_t i = 16; i < 64; ++i) {
    const uint32_t x = schedule[i - 15];
    const uint32_t y = schedule[i - 2];
    const uint32_t sigma0 = loom_serve_media_rotate_right(x, 7) ^
                            loom_serve_media_rotate_right(x, 18) ^ (x >> 3);
    const uint32_t sigma1 = loom_serve_media_rotate_right(y, 17) ^
                            loom_serve_media_rotate_right(y, 19) ^ (y >> 10);
    schedule[i] = schedule[i - 16] + sigma0 + schedule[i - 7] + sigma1;
  }

  uint32_t a = context->state[0];
  uint32_t b = context->state[1];
  uint32_t c = context->state[2];
  uint32_t d = context->state[3];
  uint32_t e = context->state[4];
  uint32_t f = context->state[5];
  uint32_t g = context->state[6];
  uint32_t h = context->state[7];
  for (uint32_t i = 0; i < 64; ++i) {
    const uint32_t sum1 = loom_serve_media_rotate_right(e, 6) ^
                          loom_serve_media_rotate_right(e, 11) ^
                          loom_serve_media_rotate_right(e, 25);
    const uint32_t choice = (e & f) ^ (~e & g);
    const uint32_t temporary1 = h + sum1 + choice +
                                loom_serve_media_sha256_round_constants[i] +
                                schedule[i];
    const uint32_t sum0 = loom_serve_media_rotate_right(a, 2) ^
                          loom_serve_media_rotate_right(a, 13) ^
                          loom_serve_media_rotate_right(a, 22);
    const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temporary2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }
  context->state[0] += a;
  context->state[1] += b;
  context->state[2] += c;
  context->state[3] += d;
  context->state[4] += e;
  context->state[5] += f;
  context->state[6] += g;
  context->state[7] += h;
}

static void loom_serve_media_sha256_update(loom_serve_media_sha256_t* context,
                                           iree_const_byte_span_t data) {
  context->data_length += data.data_length;
  while (data.data_length > 0) {
    const iree_host_size_t available =
        sizeof(context->block) - (iree_host_size_t)context->block_length;
    const iree_host_size_t length = iree_min(available, data.data_length);
    memcpy(context->block + context->block_length, data.data, length);
    context->block_length += (uint8_t)length;
    data.data += length;
    data.data_length -= length;
    if (context->block_length == sizeof(context->block)) {
      loom_serve_media_sha256_transform(context, context->block);
      context->block_length = 0;
    }
  }
}

static void loom_serve_media_sha256_finish(
    loom_serve_media_sha256_t* context, loom_serve_media_digest_t* out_digest) {
  const uint64_t bit_length = context->data_length * 8u;
  context->block[context->block_length++] = 0x80;
  if (context->block_length > 56) {
    memset(context->block + context->block_length, 0,
           sizeof(context->block) - context->block_length);
    loom_serve_media_sha256_transform(context, context->block);
    context->block_length = 0;
  }
  memset(context->block + context->block_length, 0, 56 - context->block_length);
  for (uint32_t i = 0; i < 8; ++i) {
    context->block[63 - i] = (uint8_t)(bit_length >> (i * 8));
  }
  loom_serve_media_sha256_transform(context, context->block);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(context->state); ++i) {
    loom_serve_media_store_be_u32(context->state[i], out_digest->bytes + i * 4);
  }
}

void loom_serve_media_digest_compute(iree_const_byte_span_t data,
                                     loom_serve_media_digest_t* out_digest) {
  loom_serve_media_sha256_t context = {
      .state = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu,
                0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u},
  };
  loom_serve_media_sha256_update(&context, data);
  loom_serve_media_sha256_finish(&context, out_digest);
}

bool loom_serve_media_digest_equal(const loom_serve_media_digest_t* lhs,
                                   const loom_serve_media_digest_t* rhs) {
  uint8_t difference = 0;
  for (iree_host_size_t i = 0; i < sizeof(lhs->bytes); ++i) {
    difference |= lhs->bytes[i] ^ rhs->bytes[i];
  }
  return difference == 0;
}
