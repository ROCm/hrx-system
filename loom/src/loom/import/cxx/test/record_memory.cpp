// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>

struct Sample {
  // Byte surrounded by natural record padding.
  unsigned char tag;
  // Floating-point payload with a distinct typed memory projection.
  float position;
  // Unsigned payload whose high bit must survive arithmetic conversion.
  unsigned flags;
  // Narrow field updated through promoted integer arithmetic.
  unsigned short count;
  // Signed field followed by one byte of tail padding.
  signed char delta;
};
static_assert(sizeof(Sample) == 16 && alignof(Sample) == 4);
static_assert(__builtin_offsetof(Sample, position) == 4);
static_assert(__builtin_offsetof(Sample, flags) == 8);
static_assert(__builtin_offsetof(Sample, count) == 12);
static_assert(__builtin_offsetof(Sample, delta) == 14);

template <class Record>
static unsigned update_sample(Record* records, const Sample* input,
                              unsigned index) {
  auto* sample = records + index;
  sample->position = input->position * -2.0f;
  sample->flags = input->flags / 2u;
  sample->flags |= 1u;
  sample->count = 65535;
  sample->count += 2u;
  sample->delta = -127;
  sample->delta -= 2;
  sample->tag = 255;
  unsigned previous = records[index].tag++;
  auto* position = &sample->position;
  *position += 0.5f;
  return unsigned(sample->position) + previous + sample->count + sample->delta;
}

unsigned record_update(Sample* records, const Sample* input, unsigned index) {
  return update_sample(records, input, index);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void record_update_kernel(Sample* records, const Sample* input,
                          unsigned* output) {
  output[1] = record_update(records, input, 1);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void record_volatile_kernel(volatile Sample* records, const Sample* input,
                            unsigned* output) {
  output[1] = update_sample(records, input, 1);
}

typedef float Float4 __attribute__((vector_size(16)));

struct alignas(32) VectorRecord {
  // Leading word before vector alignment padding.
  unsigned tag;
  // Four lanes loaded and stored as one source vector.
  Float4 value;
};

struct Envelope {
  // Leading word before the nested record's alignment padding.
  unsigned prefix;
  // Overaligned nested object with its own byte origin.
  VectorRecord record;
  // Trailing word before the envelope's tail padding.
  unsigned suffix;
};
static_assert(sizeof(VectorRecord) == 32 && alignof(VectorRecord) == 32);
static_assert(__builtin_offsetof(VectorRecord, value) == 16);
static_assert(sizeof(Envelope) == 96 && alignof(Envelope) == 32);
static_assert(__builtin_offsetof(Envelope, record) == 32);

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void record_vector_kernel(Envelope* envelopes, const VectorRecord* input) {
  const Float4 delta = {2.0f, 3.0f, 4.0f, 5.0f};
  auto* vector = &envelopes[1].record.value;
  *vector = input->value + delta;
}
