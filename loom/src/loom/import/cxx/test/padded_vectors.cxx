// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/buffer.h>
#include <loomcxx/check.h>
#include <loomcxx/view.h>

// clang-format off
using Half3 = _Float16 __attribute__((ext_vector_type(3)));
using BFloat3 = __bf16 __attribute__((ext_vector_type(3)));
using Float3 = float __attribute__((ext_vector_type(3)));

static_assert(sizeof(Half3) == 8);
static_assert(sizeof(BFloat3) == 8);

struct Observation {
  unsigned values;
  unsigned padding;
  unsigned neighbors;
};

static Half3 adjust_half(Half3 value) {
  Float3 wide = __builtin_convertvector(value, Float3);
  return __builtin_convertvector(wide + Float3{1.0f, 2.0f, 3.0f}, Half3);
}

static BFloat3 adjust_bfloat(BFloat3 value) {
  Float3 wide = __builtin_convertvector(value, Float3);
  return __builtin_convertvector(wide + Float3{1.0f, 2.0f, 3.0f}, BFloat3);
}

static unsigned pack_values(Half3 first, Half3 second) {
  return static_cast<unsigned>(first[0]) +
         10u * static_cast<unsigned>(first[1]) +
         100u * static_cast<unsigned>(first[2]) +
         1000u * static_cast<unsigned>(second[0]) +
         10000u * static_cast<unsigned>(second[1]) +
         100000u * static_cast<unsigned>(second[2]);
}

static unsigned pack_values(BFloat3 first, BFloat3 second) {
  return static_cast<unsigned>(first[0]) +
         10u * static_cast<unsigned>(first[1]) +
         100u * static_cast<unsigned>(first[2]) +
         1000u * static_cast<unsigned>(second[0]) +
         10000u * static_cast<unsigned>(second[1]) +
         100000u * static_cast<unsigned>(second[2]);
}

static unsigned pack_padding(const unsigned char* first, const unsigned char* second) {
  return static_cast<unsigned>(first[6]) |
         (static_cast<unsigned>(first[7]) << 8u) |
         (static_cast<unsigned>(second[6]) << 16u) |
         (static_cast<unsigned>(second[7]) << 24u);
}

static Observation half_objects(Half3* values, unsigned index) {
  values[0] = Half3{1.0f, 2.0f, 3.0f};
  values[1] = Half3{4.0f, 5.0f, 6.0f};
  auto* first_bytes = reinterpret_cast<unsigned char*>(&values[0]);
  auto* second_bytes = reinterpret_cast<unsigned char*>(&values[1]);
  first_bytes[6] = 0xa5;
  first_bytes[7] = 0x5a;
  second_bytes[6] = 0x3c;
  second_bytes[7] = 0xc3;
  values[1u - index] = adjust_half(values[index]);
  return {pack_values(values[0], values[1]),
          pack_padding(first_bytes, second_bytes), 0};
}

static Observation half_array(unsigned index) {
  Half3 values[2];
  return half_objects(values, index);
}

struct BFloatRecord {
  unsigned before;
  BFloat3 value;
  unsigned after;
};
static_assert(sizeof(BFloatRecord) == 24);
static_assert(alignof(BFloatRecord) == 8);
static_assert(__builtin_offsetof(BFloatRecord, value) == 8);
static_assert(__builtin_offsetof(BFloatRecord, after) == 16);

static Observation bfloat_objects(BFloatRecord* records) {
  records[0].before = 11;
  records[0].value = BFloat3{1.0f, 2.0f, 3.0f};
  records[0].after = 13;
  records[1].before = 17;
  records[1].value = BFloat3{4.0f, 5.0f, 6.0f};
  records[1].after = 19;
  auto* first_bytes = reinterpret_cast<unsigned char*>(&records[0].value);
  auto* second_bytes = reinterpret_cast<unsigned char*>(&records[1].value);
  first_bytes[6] = 0xa5;
  first_bytes[7] = 0x5a;
  second_bytes[6] = 0x3c;
  second_bytes[7] = 0xc3;
  records[0].value = adjust_bfloat(records[1].value);
  return {pack_values(records[0].value, records[1].value),
          pack_padding(first_bytes, second_bytes),
          records[0].before + 100u * records[0].after +
              10000u * records[1].before + 1000000u * records[1].after};
}

static Observation bfloat_records() {
  BFloatRecord records[2];
  return bfloat_objects(records);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void padded_vectors_kernel(unsigned long long* storage, unsigned* output) {
  auto* bytes = reinterpret_cast<unsigned char*>(storage);
  const auto first = half_objects(reinterpret_cast<Half3*>(bytes), 0);
  const auto second = half_objects(reinterpret_cast<Half3*>(bytes + 16), 1);
  const auto records = bfloat_objects(reinterpret_cast<BFloatRecord*>(bytes + 32));
  output[0] = first.values;
  output[1] = second.values;
  output[2] = first.padding;
  output[3] = second.padding;
  output[4] = records.values;
  output[5] = records.padding;
  output[6] = records.neighbors;
}

static void padded_vectors_oracle(loom::type::buffer<unsigned long long> storage, loom::type::buffer<unsigned> output) {
  auto values = loom::buffer::view<7>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(642321u, values, 0);
  loom::view::store(654975u, values, 1);
  loom::view::store(0xc33c5aa5u, values, 2);
  loom::view::store(0xc33c5aa5u, values, 3);
  loom::view::store(654975u, values, 4);
  loom::view::store(0xc33c5aa5u, values, 5);
  loom::view::store(19171311u, values, 6);
}

LOOM_CHECK_CASE(padded_vectors_functions) {
  const auto first = half_array(0);
  const auto second = half_array(1);
  const auto records = bfloat_records();
  loom::check::expect_equal(first.values, 642321u);
  loom::check::expect_equal(second.values, 654975u);
  loom::check::expect_equal(first.padding, 0xc33c5aa5u);
  loom::check::expect_equal(second.padding, 0xc33c5aa5u);
  loom::check::expect_equal(records.values, 654975u);
  loom::check::expect_equal(records.padding, 0xc33c5aa5u);
  loom::check::expect_equal(records.neighbors, 19171311u);
}

LOOM_CHECK_SCENARIO(padded_vectors_values) {
  loom::check::trial<1>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto storage = loom::check::fill<10, unsigned long long>(0ull);
        const auto output = loom::check::fill<7, unsigned>(0u);
        loom::check::compare<padded_vectors_kernel, padded_vectors_oracle>(
            storage, output,
            [&] { loom::check::expect_bitwise(output, output); });
      });
}
// clang-format on
