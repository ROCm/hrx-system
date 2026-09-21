// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>

template <class Word>
static unsigned bitmap_update(Word* words, unsigned index, unsigned mask) {
  return words[index] |= mask;
}

template <class Word>
static unsigned sequenced_update(Word* words) {
  unsigned index = 0;
  // The RHS writes element one before the LHS selects and updates it once.
  unsigned result = words[index++] |= (index = 1, words[1] = 32, 5);
  return result + 100 * index;
}

template <class Word>
static unsigned increment_storage(Word* words) {
  unsigned previous = (*words++)++;
  unsigned next = ++words[0];
  return previous + 100 * next;
}

template <class Word>
static unsigned update_words(Word* words) {
  words[1] = 7;
  words[2] = 11;
  unsigned bitmap = bitmap_update(words, 1, 8);
  unsigned sequenced = sequenced_update(words);
  unsigned increments = increment_storage(words + 1);
  return bitmap + sequenced + increments;
}

unsigned storage_words(unsigned* words) { return update_words(words); }

unsigned storage_volatile(volatile unsigned* words) {
  return update_words(words);
}

unsigned storage_bytes(unsigned char* bytes) {
  bytes[1] = 255;
  bytes[2] = 1;
  unsigned previous = bytes[1]++;
  unsigned assigned = bytes[2] += 258u;
  return previous + 100 * assigned;
}

int storage_signed(signed char* bytes) {
  bytes[1] = -127;
  bytes[2] = 125;
  int first = bytes[1] -= 2;
  int second = bytes[2] += -250;
  bytes[1] >>= 1ull;
  --bytes[2];
  return first + 100 * second;
}

int storage_divide(signed char* bytes) {
  bytes[1] = -125;
  return bytes[1] /= 3;
}

float storage_floats(float* values) {
  values[1] = 1.5f;
  values[2] = -3.0f;
  float first = values[1] *= 2;
  float second = values[2] /= 2;
  values[1] += 0.25f;
  values[2] -= 0.5f;
  return first + second;
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void storage_words_kernel(unsigned* words, unsigned* output) {
  output[1] = storage_words(words);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void storage_volatile_kernel(volatile unsigned* words, unsigned* output) {
  output[1] = storage_volatile(words);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void storage_bytes_kernel(unsigned char* bytes, unsigned* output) {
  output[1] = storage_bytes(bytes);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void storage_signed_kernel(signed char* bytes, int* output) {
  output[1] = storage_signed(bytes);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void storage_floats_kernel(float* values, float* output) {
  output[1] = storage_floats(values);
}
