// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>

__forceinline__ unsigned increment_byte(unsigned input) {
  unsigned char value = (unsigned char)input;
  unsigned previous = (value)++;
  unsigned next = ++(value);
  unsigned retreat = value--;
  unsigned final = --value;
  return previous | (next << 8) | (retreat << 16) | (final << 24);
}

__forceinline__ int decrement_short(int input) {
  short value = (short)input;
  int previous = value--;
  int next = --value;
  return previous * 65536 + next + 32768;
}

__forceinline__ unsigned long long increment_wide(unsigned long long value) {
  unsigned long long previous = value++;
  unsigned long long next = ++value;
  return previous ^ (next * 17ull) ^ (value * 3ull);
}

// Failed bounds skip every subsequent update, including across nested regions.
__forceinline__ unsigned increment_chain(unsigned value) {
  unsigned visits = 0;
  bool accepted = value < 256u && visits++ < 1u && value < 128u &&
                  visits++ < 2u && value < 64u && visits++ < 3u &&
                  value < 32u && visits++ < 4u && value < 16u &&
                  visits++ < 5u && value < 8u && visits++ < 6u && value < 4u;
  return visits + (accepted ? 256u : 0u);
}

__forceinline__ unsigned increment_or(unsigned value, unsigned choose) {
  unsigned visits = value;
  bool accepted = choose || visits++ < 128u;
  return visits * 2u + (unsigned)accepted;
}

__forceinline__ unsigned increment_select(unsigned input, unsigned choose) {
  unsigned char first = (unsigned char)input;
  unsigned char second = (unsigned char)(input * 3u + 1u);
  unsigned value = (choose & 1u) ? first++ : --second;
  unsigned nested =
      (choose & 2u) ? ((choose & 4u) ? ++first : second--) : first--;
  return value | (nested << 8) | ((unsigned)first << 16) |
         ((unsigned)second << 24);
}

__forceinline__ unsigned increment_condition(unsigned value) {
  unsigned selected = (value++ & 1u) ? ++value : value--;
  return selected ^ (value * 17u);
}

__forceinline__ unsigned increment_while(unsigned count) {
  unsigned index = 0;
  unsigned total = 0;
  while (index++ < count) {
    total += index;
  }
  return total * 257u + index;
}

__forceinline__ unsigned increment_do(unsigned count) {
  unsigned index = 0;
  unsigned total = 0;
  do {
    total += index;
  } while (++index < count);
  return total * 257u + index;
}

__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void increment_values(unsigned* output, unsigned input) {
  unsigned lane = threadIdx.x;
  unsigned value = input + lane;
  unsigned long long wide =
      increment_wide((unsigned long long)input * 4294967297ull + lane);
  output[lane * 10u] = increment_byte(value);
  output[lane * 10u + 1u] = (unsigned)decrement_short((int)lane - 31);
  output[lane * 10u + 2u] = (unsigned)wide;
  output[lane * 10u + 3u] = (unsigned)(wide >> 32);
  output[lane * 10u + 4u] = increment_chain(value);
  output[lane * 10u + 5u] = increment_or(value, lane & 1u);
  output[lane * 10u + 6u] = increment_select(value, lane & 7u);
  output[lane * 10u + 7u] = increment_condition(value);
  output[lane * 10u + 8u] = increment_while(lane & 7u);
  output[lane * 10u + 9u] = increment_do(lane & 7u);
}

__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void increment_pointers(const int* input, int* output, unsigned length,
                        unsigned choose) {
  unsigned lane = threadIdx.x;
  if (lane < length) {
    const int* cursor = input + lane * 8u + 1u;
    int* result = output + lane * 12u;
    *result++ = *cursor++;
    *result++ = *++cursor;
    *result++ = *cursor--;
    *result++ = *--cursor;
    const int* selected = (choose & 1u) ? cursor++ : ++cursor;
    *result++ = *selected;
    *result++ = *cursor;
    bool accepted = (choose & 2u) || *cursor++ != 0;
    *result++ = (int)accepted;
    *result++ = *cursor;
    bool guarded = (choose & 4u) && *cursor++ != 0;
    *result++ = (int)guarded;
    *result++ = *cursor;
    unsigned index = 0;
    int total = 0;
    while (index++ < 2u) {
      total += *cursor++;
    }
    *result++ = total;
    *result++ = *cursor;
  }
}
