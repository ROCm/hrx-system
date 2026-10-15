// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"

using I32x8 = int32_t __attribute__((vector_size(32)));
using F16 = _Float16;

// These definitions are linked directly from the object emitted by Loom in a
// separate build action. The compiler process exits before the link begins.
extern "C" uint64_t reuse_mutated_argument(uint64_t value, uint64_t delta);
extern "C" uint64_t reuse_aliased_arguments(uint64_t value);
extern "C" uint64_t mix(uint64_t input, uint64_t delta);
extern "C" void independent_integer_updates(uint32_t left32, uint32_t right32,
                                            uint64_t left64, uint64_t right64,
                                            uint64_t* output);
extern "C" uint32_t immediate_mix32(uint32_t input);
extern "C" uint64_t immediate_mix64(uint64_t input);
extern "C" uint64_t immediate_scale(uint64_t coordinate);
extern "C" uint32_t read_memory_mix32(const uint32_t* input, uint64_t position,
                                      uint32_t seed);
extern "C" uint64_t read_memory_mix64(const uint64_t* input, uint64_t position,
                                      uint64_t seed);
extern "C" uint64_t choose_mix(uint64_t input, uint64_t delta, uint64_t limit);
extern "C" uint32_t immediate_branch_family(int8_t value8, int16_t value16,
                                            int32_t value32, int64_t value64);
extern "C" uint32_t register_branch_false_fallthrough(int32_t lhs, int32_t rhs,
                                                      uint32_t true_value,
                                                      uint32_t false_value);
extern "C" uint64_t compare_select_mix(uint32_t lhs32, uint32_t rhs32,
                                       int64_t lhs64);
extern "C" uint32_t narrow_compare_mask(int8_t lhs8, int8_t rhs8, int16_t lhs16,
                                        int16_t rhs16);
extern "C" uint64_t narrow_compare_select_mix(int8_t lhs8, int8_t rhs8,
                                              int16_t lhs16);
extern "C" uint64_t preserve(uint64_t a, uint64_t b, uint64_t c, uint64_t d,
                             uint64_t e, uint64_t f);
extern "C" uint64_t shift_mix(uint64_t unused_first, uint64_t count,
                              uint64_t unused_third, uint64_t input);
extern "C" uint64_t load_word(uint64_t unused, const uint64_t* input,
                              uint64_t index);
extern "C" uint64_t add_word(uint32_t word, uint64_t bias);
extern "C" uint32_t divide_mix(uint64_t unused_first, uint64_t unused_second,
                               uint32_t word);
extern "C" uint64_t stored_pair(uint64_t first, uint64_t second);
extern "C" uint32_t stored_eleven32(uint32_t v0, uint32_t v1, uint32_t v2,
                                    uint32_t v3, uint32_t v4, uint32_t v5,
                                    uint32_t v6, uint32_t v7, uint32_t v8,
                                    uint32_t v9, uint32_t v10);
extern "C" uint32_t replace_narrow(uint8_t* bytes, uint16_t* words,
                                   uint64_t index, uint32_t replacement);
extern "C" void signed_unary_i8(uint8_t input, uint8_t* output);
extern "C" void signed_unary_i16(uint16_t input, uint16_t* output);
extern "C" void signed_unary_i32(uint32_t input, uint32_t* output);
extern "C" void signed_unary_i64(uint64_t input, uint64_t* output);
extern "C" bool truncate_wide_to_bit(uint64_t input);
extern "C" uint8_t truncate_wide_to_byte(uint64_t input);
extern "C" uint16_t truncate_wide_to_halfword(uint64_t input);
extern "C" uint32_t truncate_wide_to_word(uint64_t input);
extern "C" int16_t extend_signed_byte_to_halfword(int8_t input);
extern "C" uint16_t extend_unsigned_byte_to_halfword(uint8_t input);
extern "C" int64_t extend_signed_byte_to_wide(int8_t input);
extern "C" uint64_t extend_unsigned_byte_to_wide(uint8_t input);
extern "C" int64_t extend_signed_halfword_to_wide(int16_t input);
extern "C" uint64_t extend_unsigned_halfword_to_wide(uint16_t input);
extern "C" int64_t extend_signed_predicate_to_wide(bool input);
extern "C" uint64_t extend_unsigned_predicate_to_wide(bool input);
extern "C" uint64_t high_mix(uint64_t unused, uint64_t factor, uint64_t word);
extern "C" uint64_t recurrence(uint64_t first, uint64_t second,
                               uint64_t iterations);
extern "C" const uint32_t* immutable_data();
extern "C" const void* empty_data();
extern "C" const uint8_t* external_data();
extern "C" uint32_t immutable_lookup(uint64_t position);
extern "C" const uint8_t native_host_table[] = {7, 19, 41};

extern "C" uint64_t pressure64(const uint64_t* values);
extern "C" uint32_t pressure32(const uint32_t* values);
extern "C" uint64_t storage_spaces(uint64_t input, uint32_t word);
extern "C" uint64_t local_pair(uint64_t first, uint64_t second, uint64_t index);
extern "C" uint32_t sum_previous_instances(uint64_t count);
extern "C" void add_i32x4(const int32_t* lhs, const int32_t* rhs,
                          int32_t* output);
extern "C" void write_exact_avx2_lanes(uint8_t* byte_output,
                                       uint16_t* word_output,
                                       float* dword_output,
                                       int64_t* qword_output);
extern "C" void add_constant_i32x4(const int32_t* input, int32_t* output);
extern "C" void reverse_i8x32_lanes(const uint8_t* input, uint8_t* output);
extern "C" void spill_i8x32(const uint8_t* input, uint8_t* output);
extern "C" void write_mixed_results(const int32_t* input,
                                    int32_t* vector_output,
                                    uint64_t* scalar_output);
extern "C" void avx512_select_i32x16(const int32_t* scalar, const int32_t* lhs,
                                     const int32_t* rhs,
                                     const int32_t* fallback, int32_t* output);
extern "C" void avx512_reverse_i32x16(const int32_t* input, int32_t* output);
extern "C" void avx512_reduce_f32x16(const float* values, const float* bias,
                                     const float* initial, float* output);
extern "C" void write_exact_avx512_lanes(double* output);
extern "C" void avx512_scalar_load_pressure(const float* input, float* output);
extern "C" void avx512_bf16_convert_indexed_masked(
    const float* low_values, uint64_t position, const float* high_values,
    const int8_t* mask_lhs, const int8_t* mask_rhs, const uint16_t* passthrough,
    uint16_t* output);
extern "C" void call_vector_mix(uint64_t* output, I32x8 v0, uint64_t g0,
                                I32x8 v1, uint64_t g1, I32x8 v2, uint64_t g2,
                                I32x8 v3, uint64_t g3, I32x8 v4, uint64_t g4,
                                I32x8 v5, uint64_t g5, I32x8 v6, uint64_t g6,
                                I32x8 v7, I32x8 v8);
extern "C" void native_vector_mix(uint64_t* output, I32x8 v0, uint64_t g0,
                                  I32x8 v1, uint64_t g1, I32x8 v2, uint64_t g2,
                                  I32x8 v3, uint64_t g3, I32x8 v4, uint64_t g4,
                                  I32x8 v5, uint64_t g5, I32x8 v6, uint64_t g6,
                                  I32x8 v7, I32x8 v8) {
  const I32x8 vectors[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8};
  size_t output_index = 0;
  for (const I32x8& vector : vectors) {
    for (size_t lane = 0; lane < 8; ++lane) {
      output[output_index++] = static_cast<uint32_t>(vector[lane]);
    }
  }
  const uint64_t scalars[] = {g0, g1, g2, g3, g4, g5, g6};
  for (uint64_t scalar : scalars) {
    output[output_index++] = scalar;
  }
}
extern "C" void call_scalar_stack(double* output, float f0, double d0, float f1,
                                  double d1, float f2, double d2, float f3,
                                  double d3, float f4, double d4);
extern "C" void native_scalar_stack(double* output, float f0, double d0,
                                    float f1, double d1, float f2, double d2,
                                    float f3, double d3, float f4, double d4) {
  output[0] = f0;
  output[1] = d0;
  output[2] = f1;
  output[3] = d1;
  output[4] = f2;
  output[5] = d2;
  output[6] = f3;
  output[7] = d3;
  output[8] = f4;
  output[9] = d4;
}
extern "C" float identity_f32(float value);
extern "C" double identity_f64(double value);
extern "C" float copy_scalar_f32(const uint32_t* input, uint32_t* output,
                                 uint64_t base, uint64_t row, uint64_t column);
extern "C" double copy_scalar_f64(const uint64_t* input, uint64_t* output,
                                  uint64_t base, uint64_t row, uint64_t column);
extern "C" uint32_t widen_i1(bool value);
extern "C" uint32_t widen_i8(uint8_t value);
extern "C" uint32_t widen_i16(uint16_t value);
extern "C" uint32_t retain_first_narrow(uint32_t value, bool unused_i1,
                                        uint8_t unused_i8,
                                        uint8_t unused_f8e4m3,
                                        uint8_t unused_f8e5m2,
                                        uint16_t unused_i16);
extern "C" uint32_t call_narrow_i1(uint32_t value);
extern "C" uint32_t call_narrow_i8(uint32_t value);
extern "C" uint32_t call_narrow_i16(uint32_t value);
extern "C" uint32_t ignore_narrow_call_result(uint32_t value);
extern "C" bool native_narrow_i1(uint32_t value) { return value != 0; }
extern "C" uint8_t native_narrow_i8(uint32_t value) {
  return static_cast<uint8_t>(value);
}
extern "C" uint16_t native_narrow_i16(uint32_t value) {
  return static_cast<uint16_t>(value);
}
static uint32_t ignored_i8_result_call_count = 0;
extern "C" uint8_t native_ignored_i8_result() {
  ++ignored_i8_result_call_count;
  return 0xa5;
}
extern "C" F16 identity_f16(F16 value);
extern "C" I32x8 identity_i32x8(I32x8 value);
extern "C" I32x8 identity_packed_i32x8(I32x8 value);
extern "C" I32x8 call_identity_packed_i32x8(I32x8 value);
extern "C" void call_f16_stack(uint16_t* output, F16 v0, F16 v1, F16 v2, F16 v3,
                               F16 v4, F16 v5, F16 v6, F16 v7, F16 v8);
extern "C" void native_f16_stack(uint16_t* output, F16 v0, F16 v1, F16 v2,
                                 F16 v3, F16 v4, F16 v5, F16 v6, F16 v7,
                                 F16 v8) {
  const F16 values[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8};
  std::memcpy(output, values, sizeof(values));
}

extern "C" uint64_t call_pair(uint64_t, uint64_t);
extern "C" void write_narrow_triplet(uint32_t*, uint32_t);
extern "C" uint64_t incoming_eight(uint64_t, uint64_t, uint64_t, uint64_t,
                                   uint64_t, uint64_t, uint64_t, uint64_t);
extern "C" uint64_t recursive_sum(uint64_t);

extern "C" uint64_t call_host(uint64_t x, uint64_t y);
extern "C" uint64_t native_host_mix(uint64_t x, uint64_t y) {
  return x * 37 + y * 19;
}
extern "C" uint64_t call_store(uint64_t* output, uint32_t word, uint64_t wide);
extern "C" void native_host_store(uint64_t* output, uint32_t a, uint64_t b,
                                  uint32_t c, uint64_t d, uint32_t e,
                                  uint64_t f, uint32_t g) {
  *output = uint64_t{a} + b + c + d + e + f + g;
}

extern "C" uint64_t reverse_eight(uint64_t, uint64_t, uint64_t, uint64_t,
                                  uint64_t, uint64_t, uint64_t, uint64_t);
extern "C" uint64_t incoming_aligned(uint64_t, uint64_t, uint64_t, uint64_t,
                                     uint64_t, uint64_t, uint64_t, uint64_t);
extern "C" uint64_t native_host_alignment(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) & 63;
}

extern "C" uint64_t call_fifteen(uint64_t seed);
extern "C" uint64_t native_host_fifteen(uint64_t a, uint64_t b, uint64_t c,
                                        uint64_t d, uint64_t e, uint64_t f,
                                        uint64_t g, uint64_t h, uint64_t i,
                                        uint64_t j, uint64_t k, uint64_t l,
                                        uint64_t m, uint64_t n, uint64_t o) {
  return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f + 7 * g + 8 * h + 9 * i +
         10 * j + 11 * k + 12 * l + 13 * m + 14 * n + 15 * o;
}
extern "C" uint64_t call_loop(uint64_t seed, uint64_t count);
extern "C" uint64_t saturated_permutation(uint64_t, uint64_t, uint64_t,
                                          uint64_t, uint64_t, uint64_t,
                                          uint64_t, uint64_t, uint64_t,
                                          uint64_t, uint64_t, uint64_t,
                                          uint64_t, uint64_t, uint32_t);

namespace {

template <typename T>
T ExpectedReadMemoryMix(const std::array<T, 12>& values, size_t position,
                        T seed) {
  T value = values[position] + seed;
  value += values[1];
  value -= values[2];
  value = values[3] * value;
  value *= values[4];
  value = values[5] & value;
  value &= values[6];
  value = values[7] | value;
  value |= values[8];
  value = values[9] ^ value;
  return value ^ values[10];
}

template <typename T>
struct SignedUnaryCase {
  // Authored input bit pattern.
  T input;
  // Expected wrapping negation bit pattern.
  T negative;
  // Expected wrapping absolute-value bit pattern.
  T absolute;
};

template <typename T, size_t N>
void CheckSignedUnary(void (*function)(T, T*),
                      const std::array<SignedUnaryCase<T>, N>& test_cases) {
  for (const SignedUnaryCase<T>& test_case : test_cases) {
    std::array<T, 2> output = {};
    function(test_case.input, output.data());
    EXPECT_EQ(output[0], test_case.negative);
    EXPECT_EQ(output[1], test_case.absolute);
  }
}

float FloatFromBits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint16_t ExpectedFiniteF32ToBf16Daz(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & UINT32_C(0x7f800000)) == 0) {
    return static_cast<uint16_t>((bits >> 16) & UINT32_C(0x8000));
  }
  const uint32_t retained_lsb = (bits >> 16) & 1;
  return static_cast<uint16_t>((bits + UINT32_C(0x7fff) + retained_lsb) >> 16);
}

TEST(NativeCallableTest, InlinedLowArgumentsRetainIndependentContents) {
  for (uint64_t value :
       {0ull, 1ull, 0x123456789abcdef0ull, 0xffffffffffffffffull}) {
    for (uint64_t delta : {0ull, 7ull, 0x8000000000000000ull}) {
      EXPECT_EQ(reuse_mutated_argument(value, delta),
                value ^ (value + value + delta));
    }
    EXPECT_EQ(reuse_aliased_arguments(value), value ^ (value + value + 1));
  }
}

TEST(NativeCallableTest, ReadonlyDataOutlivesTheCompilerAndCrossesCalls) {
  const std::array<uint32_t, 4> expected = {0x44332211, 0x7e00ff80, 0x89abcdef,
                                            0x01234567};
  const uint32_t* data = immutable_data();
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(data) % 64, 0u);
  EXPECT_EQ(immutable_data(), data);
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(data[i], expected[i]);
    EXPECT_EQ(immutable_lookup(i), expected[i] ^ expected[0]);
  }
  ASSERT_NE(empty_data(), nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(empty_data()) % 32, 0u);
  EXPECT_EQ(external_data(), native_host_table);
}

TEST(NativeCallableTest, SaturatedLoopPermutation) {
  const std::array<uint64_t, 14> words = {UINT64_C(0x0123456789abcdef),
                                          UINT64_C(0xfedcba9876543210),
                                          3,
                                          5,
                                          8,
                                          13,
                                          21,
                                          34,
                                          55,
                                          89,
                                          144,
                                          233,
                                          377,
                                          610};
  uint64_t difference = 0;
  for (size_t i = 0; i < words.size(); i += 2) {
    difference += words[i] - words[i + 1];
  }
  for (uint32_t count = 0; count < 20; ++count) {
    EXPECT_EQ(saturated_permutation(words[0], words[1], words[2], words[3],
                                    words[4], words[5], words[6], words[7],
                                    words[8], words[9], words[10], words[11],
                                    words[12], words[13], count),
              count % 2 ? uint64_t{0} - difference : difference);
  }
}

TEST(NativeCallableTest, OrdinaryCLinkage) {
  uint64_t state = UINT64_C(0x243f6a8885a308d3);
  for (unsigned i = 0; i < 1000; ++i) {
    uint64_t words[6];
    for (uint64_t& word : words) {
      state = state * UINT64_C(6364136223846793005) + 1;
      word = state;
    }
    const uint64_t a = words[0], b = words[1], c = words[2];
    const uint64_t d = words[3], e = words[4], f = words[5];
    ASSERT_EQ(mix(a, b), (a + b) ^ a);
    ASSERT_EQ(choose_mix(a, b, c), a < c ? ((a + b) ^ a) : ((a * b) ^ c));
    const int32_t signed_a = static_cast<int32_t>(a);
    const int32_t signed_b = static_cast<int32_t>(b);
    ASSERT_EQ(register_branch_false_fallthrough(signed_a, signed_b, 19, 23),
              signed_a < signed_b ? 19u : 23u);
    const uint64_t selected64 =
        static_cast<uint32_t>(a) < static_cast<uint32_t>(b)
            ? UINT64_C(4294967297)
            : UINT64_C(8589934594);
    const int64_t signed_value = static_cast<int64_t>(i) - 500;
    const uint64_t selected32 = signed_value >= -129 ? 17 : 23;
    ASSERT_EQ(compare_select_mix(static_cast<uint32_t>(a),
                                 static_cast<uint32_t>(b), signed_value),
              selected64 ^ selected32);
    const int8_t lhs8 = static_cast<int8_t>(a);
    const int8_t rhs8 = static_cast<int8_t>(b);
    const int16_t lhs16 = static_cast<int16_t>(c);
    const int16_t rhs16 = static_cast<int16_t>(d);
    const uint32_t narrow_mask =
        (lhs8 < rhs8 ? 1u : 0u) | (lhs8 == -17 ? 2u : 0u) |
        (static_cast<uint16_t>(lhs16) > static_cast<uint16_t>(rhs16) ? 4u
                                                                     : 0u) |
        (static_cast<uint16_t>(lhs16) <= UINT16_C(1234) ? 8u : 0u);
    ASSERT_EQ(narrow_compare_mask(lhs8, rhs8, lhs16, rhs16), narrow_mask);
    const uint64_t narrow_selected16 = lhs8 < rhs8 ? 17 : 300;
    const uint64_t narrow_selected64 =
        lhs16 >= -129 ? UINT64_C(4294967297) : UINT64_C(8589934594);
    ASSERT_EQ(narrow_compare_select_mix(lhs8, rhs8, lhs16),
              narrow_selected16 ^ narrow_selected64);
    ASSERT_EQ(preserve(a, b, c, d, e, f),
              ((a + b) * (c + d) + (e + f) * a) ^ (b ^ c));
    ASSERT_EQ(shift_mix(a, b, c, d), (d >> (b & 63)) ^ d);
    ASSERT_EQ(load_word(a, words, i % 6), words[i % 6]);
    ASSERT_EQ(add_word(static_cast<uint32_t>(a), b),
              static_cast<uint64_t>(static_cast<uint32_t>(a)) + b);
    const uint32_t word = static_cast<uint32_t>(a);
    ASSERT_EQ(divide_mix(b, c, word), ((word / 7) + (word % 7)) ^ word);
    const auto product = static_cast<unsigned __int128>(b) * c;
    ASSERT_EQ(high_mix(a, b, c), static_cast<uint64_t>(product >> 64) ^ c);
    uint64_t first = a, second = b;
    for (unsigned step = 0; step < i % 23; ++step) {
      const uint64_t sum = first + second;
      first = second;
      second = sum;
    }
    ASSERT_EQ(recurrence(a, b, i % 23), first);
  }
}

TEST(NativeCallableTest, CompareBranchesUseOrdinaryCLinkage) {
  EXPECT_EQ(immediate_branch_family(-17, 0, 129, -129), 0u);
  EXPECT_EQ(immediate_branch_family(0, 0, 129, -129), 1u);
  EXPECT_EQ(immediate_branch_family(-17, -129, 129, -129), 2u);
  EXPECT_EQ(immediate_branch_family(-17, 0, 128, -129), 3u);
  EXPECT_EQ(immediate_branch_family(-17, 0, 129, -130), 4u);
  EXPECT_EQ(register_branch_false_fallthrough(-4, 7, 19, 23), 19u);
  EXPECT_EQ(register_branch_false_fallthrough(7, -4, 19, 23), 23u);
}

TEST(NativeCallableTest, IndependentResultsPreservePhysicallyTiedInputs) {
  for (uint64_t left :
       {UINT64_C(0), UINT64_C(7), UINT64_C(0x123456789abcdef0), UINT64_MAX}) {
    for (uint64_t right : {UINT64_C(1), UINT64_C(19), UINT64_MAX}) {
      std::array<uint64_t, 3> output = {};
      independent_integer_updates(static_cast<uint32_t>(left),
                                  static_cast<uint32_t>(right), left, right,
                                  output.data());
      const uint32_t sum = static_cast<uint32_t>(left + right);
      EXPECT_EQ(output[0], (uint64_t{sum} << 32) | static_cast<uint32_t>(left));
      EXPECT_EQ(output[1], left);
      EXPECT_EQ(output[2], left - right);
    }
  }
}

TEST(NativeCallableTest, IntegerImmediateArithmetic) {
  const std::array<uint32_t, 6> words32 = {
      0,          1, UINT32_C(127), UINT32_C(0x7fffffff), UINT32_C(0x80000000),
      UINT32_MAX,
  };
  for (uint32_t input : words32) {
    EXPECT_EQ(immediate_mix32(input),
              (input + uint32_t{23}) * uint32_t{129} - uint32_t{17});
  }

  const std::array<uint64_t, 6> words64 = {
      0,
      1,
      UINT64_C(127),
      UINT64_C(0x7fffffffffffffff),
      UINT64_C(0x8000000000000000),
      UINT64_MAX,
  };
  for (uint64_t input : words64) {
    EXPECT_EQ(immediate_mix64(input),
              (input - UINT64_C(2147483648)) * UINT64_C(2147483647));
    EXPECT_EQ(immediate_scale(input), input * UINT64_C(65));
  }
}

TEST(NativeCallableTest, ScalarReadMemoryArithmetic) {
  const std::array<uint32_t, 12> values32 = {
      UINT32_C(0x00000000), UINT32_C(0x00000001), UINT32_C(0x7fffffff),
      UINT32_C(0x80000001), UINT32_C(0xfffffffb), UINT32_C(0x5a5aa5a5),
      UINT32_C(0xf0ff0ff0), UINT32_C(0x01020408), UINT32_C(0x80402010),
      UINT32_C(0xaaaaaaaa), UINT32_C(0x55555555), UINT32_C(0xffffffff),
  };
  const std::array<uint32_t, 4> seeds32 = {0, 1, UINT32_C(0x80000000),
                                           UINT32_MAX};
  for (size_t position : {size_t{0}, size_t{5}, size_t{11}}) {
    for (uint32_t seed : seeds32) {
      EXPECT_EQ(read_memory_mix32(values32.data(), position, seed),
                ExpectedReadMemoryMix(values32, position, seed));
    }
  }

  const std::array<uint64_t, 12> values64 = {
      UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000001),
      UINT64_C(0x7fffffffffffffff), UINT64_C(0x8000000000000001),
      UINT64_C(0xfffffffffffffffb), UINT64_C(0x5a5aa5a55a5aa5a5),
      UINT64_C(0xf0ff0ff00ff0ff0f), UINT64_C(0x0102040810204080),
      UINT64_C(0x8040201008040201), UINT64_C(0xaaaaaaaaaaaaaaaa),
      UINT64_C(0x5555555555555555), UINT64_C(0xffffffffffffffff),
  };
  const std::array<uint64_t, 4> seeds64 = {0, 1, UINT64_C(0x8000000000000000),
                                           UINT64_MAX};
  for (size_t position : {size_t{0}, size_t{5}, size_t{11}}) {
    for (uint64_t seed : seeds64) {
      EXPECT_EQ(read_memory_mix64(values64.data(), position, seed),
                ExpectedReadMemoryMix(values64, position, seed));
    }
  }
}

TEST(NativeCallableTest, Avx2VectorFunctionUsesOrdinaryCLinkage) {
  const std::array<int32_t, 4> lhs = {INT32_MIN, -2, 3, INT32_MAX};
  const std::array<int32_t, 4> rhs = {-1, 5, 7, 1};
  std::array<int32_t, 4> output = {};
  add_i32x4(lhs.data(), rhs.data(), output.data());
  EXPECT_EQ(output, (std::array<int32_t, 4>{INT32_MAX, 3, 10, INT32_MIN}));

  const std::array<int32_t, 4> input = {4, 7, -100, -1};
  add_constant_i32x4(input.data(), output.data());
  EXPECT_EQ(output, (std::array<int32_t, 4>{5, 2, 0, INT32_MAX}));

  std::array<uint8_t, 32> bytes = {};
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<uint8_t>(i);
  }
  std::array<uint8_t, 32> reversed = {};
  reverse_i8x32_lanes(bytes.data(), reversed.data());
  for (size_t i = 0; i < reversed.size(); ++i) {
    EXPECT_EQ(reversed[i], 15 - i % 16 + i / 16 * 16);
  }

  std::array<uint8_t, 32> spilled = {};
  spill_i8x32(bytes.data(), spilled.data());
  EXPECT_EQ(spilled, bytes);

  std::array<uint8_t, 16> exact_bytes = {};
  std::array<uint16_t, 8> exact_words = {};
  std::array<float, 8> exact_dwords = {};
  std::array<int64_t, 4> exact_qwords = {};
  write_exact_avx2_lanes(exact_bytes.data(), exact_words.data(),
                         exact_dwords.data(), exact_qwords.data());
  for (size_t i = 0; i < exact_words.size(); ++i) {
    EXPECT_EQ(exact_bytes[2 * i], 1);
    EXPECT_EQ(exact_bytes[2 * i + 1], 0xfe);
    EXPECT_EQ(exact_words[i], i % 2 == 0 ? 0x3fc0 : 0xc000);
    EXPECT_EQ(exact_dwords[i], i % 2 == 0 ? 1.5f : -2.25f);
  }
  EXPECT_EQ(exact_qwords,
            (std::array<int64_t, 4>{INT64_C(0x0123456789abcdef), -2,
                                    INT64_C(0x0123456789abcdef), -2}));
}

template <typename Bits, typename Copy>
void CheckScalarMemory(const std::array<Bits, 8>& patterns, Copy copy) {
  constexpr Bits kGuard = static_cast<Bits>(0xDEADBEEFCAFE1234ull);
  std::array<Bits, 64> input;
  for (size_t rotation = 0; rotation < patterns.size(); ++rotation) {
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = patterns[(i + rotation) % patterns.size()];
    }
    const auto original = input;
    for (size_t base = 0; base < 4; ++base) {
      for (size_t row = 0; row < 2; ++row) {
        for (size_t column = 0; column < 5; ++column) {
          std::array<Bits, 64> output;
          output.fill(kGuard);
          auto expected = output;
          const size_t position = base + (row + 1) * 13 + column + 1;
          expected[base] = input[base];
          expected[position] = input[position];
          auto returned = copy(input.data(), output.data(), base * sizeof(Bits),
                               row, column);
          Bits returned_bits;
          static_assert(sizeof(returned_bits) == sizeof(returned));
          std::memcpy(&returned_bits, &returned, sizeof(returned_bits));
          EXPECT_EQ(returned_bits, input[position]);
          EXPECT_EQ(output, expected);
          EXPECT_EQ(input, original);
        }
      }
    }
  }
}

TEST(NativeCallableTest, ScalarFloatingMemoryPreservesBitsAndNeighbors) {
  // Signed zero, infinities, quiet/signaling NaNs, subnormals, and arbitrary
  // finite payloads are compared as bits, including the native scalar return.
  CheckScalarMemory<uint32_t>({0x00000000, 0x80000000, 0x7F800000, 0xFF800000,
                               0x7FC12345, 0x7F812345, 0x00000001, 0x12345678},
                              copy_scalar_f32);
  CheckScalarMemory<uint64_t>(
      {0x0000000000000000ull, 0x8000000000000000ull, 0x7FF0000000000000ull,
       0xFFF0000000000000ull, 0x7FF8123456789ABCull, 0x7FF0123456789ABCull,
       0x0000000000000001ull, 0x123456789ABCDEFull},
      copy_scalar_f64);
}

TEST(NativeCallableTest, MixedPrivateResultsOverflowIndependentBanks) {
  std::array<int32_t, 24> input;
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<int32_t>(1000 + i);
  }
  std::array<int32_t, 24> vectors = {};
  std::array<uint64_t, 3> scalars = {};
  write_mixed_results(input.data(), vectors.data(), scalars.data());
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(vectors[i], input[16 + i]);
    EXPECT_EQ(vectors[8 + i], input[i]);
    EXPECT_EQ(vectors[16 + i], input[8 + i]);
  }
  EXPECT_EQ(scalars, (std::array<uint64_t, 3>{43, 17, 29}));
}

TEST(NativeCallableTest, Avx512CoreUsesOrdinaryCLinkage) {
  if (!__builtin_cpu_supports("avx512f") ||
      !__builtin_cpu_supports("avx512bw") ||
      !__builtin_cpu_supports("avx512dq") ||
      !__builtin_cpu_supports("avx512vl")) {
    GTEST_SKIP() << "AVX-512F/BW/DQ/VL are unavailable on this test host";
  }

  const int32_t scalar = 3;
  std::array<int32_t, 16> lhs;
  std::array<int32_t, 16> rhs;
  std::array<int32_t, 16> fallback;
  for (size_t i = 0; i < lhs.size(); ++i) {
    lhs[i] = static_cast<int32_t>(i) - 8;
    rhs[i] = static_cast<int32_t>(i / 2);
    fallback[i] = i % 3 == 0 ? rhs[i] : 1000 + static_cast<int32_t>(i);
  }
  std::array<int32_t, 16> selected = {};
  avx512_select_i32x16(&scalar, lhs.data(), rhs.data(), fallback.data(),
                       selected.data());
  for (size_t i = 0; i < selected.size(); ++i) {
    const int32_t sum = lhs[i] + scalar;
    const bool condition = sum > rhs[i] || rhs[i] == fallback[i];
    EXPECT_EQ(selected[i], condition ? sum : fallback[i]) << "lane " << i;
  }

  std::array<int32_t, 16> reversed = {};
  avx512_reverse_i32x16(lhs.data(), reversed.data());
  for (size_t i = 0; i < reversed.size(); ++i) {
    EXPECT_EQ(reversed[i], lhs[lhs.size() - i - 1]) << "lane " << i;
  }

  std::array<float, 16> values;
  std::array<float, 16> bias;
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i);
    bias[i] = 1.0f;
  }
  const std::array<float, 4> initial = {8.0f, 0.0f, 0.0f, 0.0f};
  std::array<float, 4> reduced = {};
  avx512_reduce_f32x16(values.data(), bias.data(), initial.data(),
                       reduced.data());
  EXPECT_EQ(reduced[0], 144.0f);

  std::array<double, 8> exact_lanes = {};
  write_exact_avx512_lanes(exact_lanes.data());
  for (size_t i = 0; i < exact_lanes.size(); ++i) {
    EXPECT_EQ(exact_lanes[i], i % 2 == 0 ? 1.25 : -2.5);
  }
}

TEST(NativeCallableTest, Avx512ScalarLoadsRespectVexRegisterDomain) {
  if (!__builtin_cpu_supports("avx512f")) {
    GTEST_SKIP() << "AVX-512F is unavailable on this test host";
  }
  std::array<float, 17> input;
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i) - 8.5f;
  }
  std::array<float, 17 * 16> output = {};
  avx512_scalar_load_pressure(input.data(), output.data());
  for (size_t i = 0; i < input.size(); ++i) {
    for (size_t lane = 0; lane < 16; ++lane) {
      EXPECT_EQ(output[i * 16 + lane], input[i])
          << "scalar " << i << ", lane " << lane;
    }
  }
}

TEST(NativeCallableTest,
     Avx512Bf16IndexedMaskedConversionUsesOrdinaryCLinkage) {
  if (!__builtin_cpu_supports("avx512f") ||
      !__builtin_cpu_supports("avx512bw") ||
      !__builtin_cpu_supports("avx512dq") ||
      !__builtin_cpu_supports("avx512vl") ||
      !__builtin_cpu_supports("avx512bf16")) {
    GTEST_SKIP() << "AVX-512F/BW/DQ/VL/BF16 are unavailable on this test host";
  }

  const std::array<float, 16> conversion_cases = {
      0.0f,
      -0.0f,
      1.0f,
      -1.0f,
      1.00390625f,
      1.01171875f,
      -1.00390625f,
      -1.01171875f,
      3.1415927f,
      -2.7182818f,
      FloatFromBits(UINT32_C(0x7f7fffff)),
      FloatFromBits(UINT32_C(0xff7fffff)),
      FloatFromBits(UINT32_C(0x00800000)),
      FloatFromBits(UINT32_C(0x80800000)),
      FloatFromBits(UINT32_C(0x00000001)),
      FloatFromBits(UINT32_C(0x807fffff)),
  };
  constexpr uint64_t kPosition = 3;
  std::array<float, 32> low_values = {};
  std::copy(conversion_cases.begin(), conversion_cases.end(),
            low_values.begin() + kPosition);
  std::array<float, 16> high_values;
  for (size_t i = 0; i < high_values.size(); ++i) {
    high_values[i] = conversion_cases[high_values.size() - i - 1];
  }

  std::array<int8_t, 32> mask_lhs;
  std::array<int8_t, 32> mask_rhs;
  std::array<uint16_t, 32> passthrough;
  for (size_t i = 0; i < passthrough.size(); ++i) {
    mask_lhs[i] = 0;
    mask_rhs[i] = i % 2 == 0 ? 1 : -1;
    passthrough[i] = static_cast<uint16_t>(0x5100 + i);
  }

  std::array<uint16_t, 32> output = {};
  avx512_bf16_convert_indexed_masked(
      low_values.data(), kPosition, high_values.data(), mask_lhs.data(),
      mask_rhs.data(), passthrough.data(), output.data());
  for (size_t i = 0; i < output.size(); ++i) {
    const float converted_value =
        i < 16 ? low_values[kPosition + i] : high_values[i - 16];
    const uint16_t expected = mask_lhs[i] < mask_rhs[i]
                                  ? ExpectedFiniteF32ToBf16Daz(converted_value)
                                  : passthrough[i];
    EXPECT_EQ(output[i], expected) << "lane " << i;
  }
}

TEST(NativeCallableTest, NarrowMemoryPreservesNeighbors) {
  const std::array<uint8_t, 6> initial_bytes = {0, 127, 128, 255, 17, 201};
  const std::array<uint16_t, 6> initial_words = {0,     32767, 32768,
                                                 65535, 513,   54321};
  for (size_t index = 0; index < initial_bytes.size(); ++index) {
    auto bytes = initial_bytes;
    auto words = initial_words;
    const uint32_t previous = bytes[index] | (uint32_t{words[index]} << 8);
    const uint32_t replacement = 0xabcdef42u + static_cast<uint32_t>(index);
    auto expected_bytes = initial_bytes;
    auto expected_words = initial_words;
    expected_bytes[index] = static_cast<uint8_t>(replacement);
    expected_words[index] = static_cast<uint16_t>(replacement);
    ASSERT_EQ(replace_narrow(bytes.data(), words.data(), index, replacement),
              previous);
    EXPECT_EQ(bytes, expected_bytes);
    EXPECT_EQ(words, expected_words);
  }
}

TEST(NativeCallableTest, StackStorageAndAllocationSpills) {
  uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
  for (unsigned repetition = 0; repetition < 100; ++repetition) {
    std::array<uint64_t, 20> wide;
    std::array<uint32_t, 20> narrow;
    uint64_t wide_sum = 0;
    uint32_t narrow_sum = 0;
    for (size_t i = 0; i < wide.size(); ++i) {
      state = state * UINT64_C(6364136223846793005) + 1;
      wide[i] = state;
      narrow[i] = static_cast<uint32_t>(state >> 32);
      wide_sum += wide[i];
      narrow_sum += narrow[i];
    }
    ASSERT_EQ(pressure64(wide.data()), wide_sum);
    ASSERT_EQ(pressure32(narrow.data()), narrow_sum);
    ASSERT_EQ(storage_spaces(wide[0], narrow[0]),
              wide[0] ^ (wide[0] + 258) ^ narrow[0]);
    ASSERT_EQ(stored_pair(wide[0], wide[1]), wide[0] - wide[1]);
    ASSERT_EQ(stored_eleven32(narrow[0], narrow[1], narrow[2], narrow[3],
                              narrow[4], narrow[5], narrow[6], narrow[7],
                              narrow[8], narrow[9], narrow[10]),
              narrow[0] ^ narrow[10]);
    ASSERT_EQ(local_pair(wide[0], wide[1], 0), wide[0]);
    ASSERT_EQ(local_pair(wide[0], wide[1], 1), wide[1]);
  }
}

TEST(NativeCallableTest, ReusesOnlyDeadAllocationInstances) {
  for (uint32_t count = 0; count < 20; ++count) {
    const uint32_t expected = count == 0 ? 0 : count * (count - 1) / 2;
    EXPECT_EQ(sum_previous_instances(count), expected);
  }
}

TEST(NativeCallableTest, CompleteCallOwnsItsOverflowAndAlignedLocal) {
  for (uint64_t x : {uint64_t{0}, uint64_t{11}, uint64_t{0x123456789abcdef0}}) {
    for (uint64_t y :
         {uint64_t{0}, uint64_t{23}, uint64_t{0xfedcba9876543210}}) {
      // The seven scalar fields have different weights; the local contributes
      // y to each call, independently of its first scalar argument.
      const uint64_t first = y + 23 * x + 33 * y + 19 * 5;
      const uint64_t second = y + 23 * y + 33 * x + 19 * 9;
      EXPECT_EQ(call_pair(x, y), first + second + (x ^ y));
    }
  }
}

TEST(NativeCallableTest, NestedPrivateCallsReturnRegisterAndOverflowValues) {
  for (uint32_t value : {0u, 1u, 0x12345678u, 0xffffffffu}) {
    std::array<uint32_t, 3> output = {};
    write_narrow_triplet(output.data(), value);
    EXPECT_EQ(output[0], value != 0 ? 1u : 0u);
    EXPECT_EQ(output[1], value != 0 ? value & UINT8_MAX : 0x5au);
    EXPECT_EQ(output[2], value != 0 ? value & UINT16_MAX : 0xa55au);
  }
}

TEST(NativeCallableTest, IncomingStackArgumentsFromIndependentCxxCaller) {
  EXPECT_EQ(incoming_eight(1, 2, 7, 9, 3, 5, 11, 13),
            ((uint64_t{1} + 2) + (7 ^ 9)) ^ (3 * 5 + 11 * 13));
  EXPECT_EQ(
      incoming_eight(9, 4, 13, 7, 3, 17, 0x100000001, 19),
      ((uint64_t{9} + 4) + (13 ^ 7)) ^ (3 * 17 + uint64_t{0x100000001} * 19));
}

TEST(NativeCallableTest, RecursiveFramesPreserveCallerValues) {
  for (uint64_t n : {uint64_t{0}, uint64_t{1}, uint64_t{17}, uint64_t{61}}) {
    EXPECT_EQ(recursive_sum(n), n * (n + 1) / 2);
  }
}

TEST(NativeCallableTest, ExternalCLinkage) {
  EXPECT_EQ(call_host(12345, 67890), (67890 * 37 + 12345 * 19) ^ 12345);
}

TEST(NativeCallableTest, MixedWidthVoidCall) {
  uint64_t output = 0;
  const uint32_t word = 0xfedcba98u;
  const uint64_t wide = UINT64_C(0x123456789abcdef0);
  const uint64_t expected = 4 * uint64_t{word} + 3 * wide;
  EXPECT_EQ(call_store(&output, word, wide), expected);
  EXPECT_EQ(output, expected);
}

TEST(NativeCallableTest, PermutesRegisterAndStackArguments) {
  const uint64_t a = 7, b = 23, c = 31, d = 43, e = 59, f = 67;
  const uint64_t g = UINT64_C(0x123456789abcdef0);
  const uint64_t h = UINT64_C(0xfedcba9876543210);
  EXPECT_EQ(reverse_eight(a, b, c, d, e, f, g, h),
            ((h + g) + (f ^ e)) ^ (d * c + b * a));
  EXPECT_EQ(incoming_aligned(a, b, c, d, e, f, g, h),
            h + 3 * a + 5 * b + 7 * c + 11 * d + 13 * e + 17 * f + 19 * g);
}

TEST(NativeCallableTest, ReusesConsumedOverflowForRegisterPermutation) {
  for (uint64_t seed : {uint64_t{0}, uint64_t{13}, UINT64_MAX}) {
    EXPECT_EQ(call_fifteen(seed), 1360 * seed);
  }
}

TEST(NativeCallableTest, CallsPreserveLoopState) {
  for (uint64_t count : {uint64_t{0}, uint64_t{1}, uint64_t{29}}) {
    const uint64_t seed = UINT64_C(0xfedcba9876543210);
    uint64_t expected = 0;
    for (uint64_t i = 0; i < count; ++i) {
      expected += 37 * (seed + i) + 19 * expected;
    }
    EXPECT_EQ(call_loop(seed, count), expected);
  }
}

TEST(NativeCallableTest, IndependentIntegerAndSseArgumentBanks) {
  I32x8 vectors[9];
  for (size_t vector_index = 0; vector_index < 9; ++vector_index) {
    for (size_t lane = 0; lane < 8; ++lane) {
      vectors[vector_index][lane] =
          static_cast<int32_t>(vector_index * 100 + lane);
    }
  }
  const uint64_t scalars[] = {
      UINT64_C(0x1111222233334444), UINT64_C(0x5555666677778888),
      UINT64_C(0x9999aaaabbbbcccc), UINT64_C(0xddddeeeeffff0000),
      UINT64_C(0x0123456789abcdef), UINT64_C(0xfedcba9876543210),
      UINT64_C(0x0f1e2d3c4b5a6978),
  };
  std::array<uint64_t, 79> output = {};
  call_vector_mix(output.data(), vectors[0], scalars[0], vectors[1], scalars[1],
                  vectors[2], scalars[2], vectors[3], scalars[3], vectors[4],
                  scalars[4], vectors[5], scalars[5], vectors[6], scalars[6],
                  vectors[7], vectors[8]);
  size_t output_index = 0;
  for (const I32x8& vector : vectors) {
    for (size_t lane = 0; lane < 8; ++lane) {
      EXPECT_EQ(output[output_index++], static_cast<uint32_t>(vector[lane]));
    }
  }
  for (uint64_t scalar : scalars) {
    EXPECT_EQ(output[output_index++], scalar);
  }
}

TEST(NativeCallableTest, ScalarSseStackArgumentsAndReturns) {
  const float floats[] = {1.25f, -2.5f, 3.75f, -4.125f, 5.5f};
  const double doubles[] = {11.125, -22.25, 33.5, -44.75, 55.875};
  double output[10] = {};
  call_scalar_stack(output, floats[0], doubles[0], floats[1], doubles[1],
                    floats[2], doubles[2], floats[3], doubles[3], floats[4],
                    doubles[4]);
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(output[2 * i], floats[i]);
    EXPECT_EQ(output[2 * i + 1], doubles[i]);
  }
  EXPECT_FLOAT_EQ(identity_f32(-123.75f), -123.75f);
  EXPECT_DOUBLE_EQ(identity_f64(0x1.23456789abcdep+27), 0x1.23456789abcdep+27);
}

TEST(NativeCallableTest, NarrowIntegerBoundariesNormalizeCarriers) {
  EXPECT_EQ(widen_i1(false), 0u);
  EXPECT_EQ(widen_i1(true), 1u);
  EXPECT_EQ(widen_i8(0xa5), 0xa5u);
  EXPECT_EQ(widen_i16(0xa55a), 0xa55au);

  for (uint32_t value : {0u, 1u, 0x12345678u, 0xffffffffu}) {
    EXPECT_EQ(call_narrow_i1(value), value != 0 ? 1u : 0u);
    EXPECT_EQ(call_narrow_i8(value), value & UINT8_MAX);
    EXPECT_EQ(call_narrow_i16(value), value & UINT16_MAX);
  }
}

TEST(NativeCallableTest, SignedUnaryWrapsAtEveryStorageWidth) {
  CheckSignedUnary<uint8_t, 5>(signed_unary_i8, {{{0x00, 0x00, 0x00},
                                                  {0x01, 0xff, 0x01},
                                                  {0x7f, 0x81, 0x7f},
                                                  {0x80, 0x80, 0x80},
                                                  {0xff, 0x01, 0x01}}});
  CheckSignedUnary<uint16_t, 5>(signed_unary_i16, {{{0x0000, 0x0000, 0x0000},
                                                    {0x0001, 0xffff, 0x0001},
                                                    {0x7fff, 0x8001, 0x7fff},
                                                    {0x8000, 0x8000, 0x8000},
                                                    {0xffff, 0x0001, 0x0001}}});
  CheckSignedUnary<uint32_t, 5>(
      signed_unary_i32,
      {{{UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000)},
        {UINT32_C(0x00000001), UINT32_C(0xffffffff), UINT32_C(0x00000001)},
        {UINT32_C(0x7fffffff), UINT32_C(0x80000001), UINT32_C(0x7fffffff)},
        {UINT32_C(0x80000000), UINT32_C(0x80000000), UINT32_C(0x80000000)},
        {UINT32_C(0xffffffff), UINT32_C(0x00000001), UINT32_C(0x00000001)}}});
  CheckSignedUnary<uint64_t, 5>(
      signed_unary_i64,
      {{{UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000),
         UINT64_C(0x0000000000000000)},
        {UINT64_C(0x0000000000000001), UINT64_C(0xffffffffffffffff),
         UINT64_C(0x0000000000000001)},
        {UINT64_C(0x7fffffffffffffff), UINT64_C(0x8000000000000001),
         UINT64_C(0x7fffffffffffffff)},
        {UINT64_C(0x8000000000000000), UINT64_C(0x8000000000000000),
         UINT64_C(0x8000000000000000)},
        {UINT64_C(0xffffffffffffffff), UINT64_C(0x0000000000000001),
         UINT64_C(0x0000000000000001)}}});
}

TEST(NativeCallableTest, IntegerConversionsPreserveLogicalWidthSemantics) {
  constexpr uint64_t kPayload = UINT64_C(0xfedcba98765489ab);
  EXPECT_TRUE(truncate_wide_to_bit(kPayload));
  EXPECT_FALSE(truncate_wide_to_bit(kPayload - 1));
  EXPECT_EQ(truncate_wide_to_byte(kPayload), UINT8_C(0xab));
  EXPECT_EQ(truncate_wide_to_halfword(kPayload), UINT16_C(0x89ab));
  EXPECT_EQ(truncate_wide_to_word(kPayload), UINT32_C(0x765489ab));

  EXPECT_EQ(extend_signed_byte_to_halfword(INT8_MIN), INT16_C(-128));
  EXPECT_EQ(extend_unsigned_byte_to_halfword(UINT8_MAX), UINT16_C(255));
  EXPECT_EQ(extend_signed_byte_to_wide(INT8_MIN), INT64_C(-128));
  EXPECT_EQ(extend_unsigned_byte_to_wide(UINT8_MAX), UINT64_C(255));
  EXPECT_EQ(extend_signed_halfword_to_wide(INT16_MIN), INT64_C(-32768));
  EXPECT_EQ(extend_unsigned_halfword_to_wide(UINT16_MAX), UINT64_C(65535));
  EXPECT_EQ(extend_signed_predicate_to_wide(false), INT64_C(0));
  EXPECT_EQ(extend_signed_predicate_to_wide(true), INT64_C(-1));
  EXPECT_EQ(extend_unsigned_predicate_to_wide(false), UINT64_C(0));
  EXPECT_EQ(extend_unsigned_predicate_to_wide(true), UINT64_C(1));
}

TEST(NativeCallableTest, UnusedNarrowBoundariesRetainObservableBehavior) {
  EXPECT_EQ(retain_first_narrow(0x12345678u, true, 0xa5, 0x7f, 0x80, 0xa55a),
            0x12345678u);
  const uint32_t call_count = ignored_i8_result_call_count;
  EXPECT_EQ(ignore_narrow_call_result(0x89abcdefu), 0x89abcdefu);
  EXPECT_EQ(ignored_i8_result_call_count, call_count + 1);
}

TEST(NativeCallableTest, HalfPrecisionCrossClassRegistersAndStack) {
  static_assert(sizeof(F16) == sizeof(uint16_t));
  const std::array<uint16_t, 9> input_bits = {
      0x0000, 0x8000, 0x3c00, 0xc100, 0x3555, 0x7bff, 0x0400, 0x7c01, 0xfc01,
  };
  std::array<F16, 9> values;
  std::memcpy(values.data(), input_bits.data(), sizeof(input_bits));
  std::array<uint16_t, 9> output_bits = {};
  call_f16_stack(output_bits.data(), values[0], values[1], values[2], values[3],
                 values[4], values[5], values[6], values[7], values[8]);
  EXPECT_EQ(output_bits, input_bits);
  for (size_t i = 0; i < values.size(); ++i) {
    const F16 returned = identity_f16(values[i]);
    uint16_t returned_bits = 0;
    std::memcpy(&returned_bits, &returned, sizeof(returned_bits));
    EXPECT_EQ(returned_bits, input_bits[i]);
  }
}

TEST(NativeCallableTest, VectorReturnUsesSseBank) {
  I32x8 input;
  for (size_t i = 0; i < 8; ++i) {
    input[i] = static_cast<int32_t>(i * 101 - 303);
  }
  const I32x8 output = identity_i32x8(input);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(output[i], input[i]);
  }
  const I32x8 packed_output = identity_packed_i32x8(input);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(packed_output[i], input[i]);
  }
  const I32x8 packed_call_output = call_identity_packed_i32x8(input);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(packed_call_output[i], input[i]);
  }
}

}  // namespace

static uint64_t Weighted(const std::array<uint64_t, 32>& values) {
  uint64_t result = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    result += values[i] * (i + 1);
  }
  return result;
}

extern "C" uint64_t incoming_many(
    uint32_t value0, uint64_t value1, uint32_t value2, uint64_t value3,
    uint32_t value4, uint64_t value5, uint32_t value6, uint64_t value7,
    uint32_t value8, uint64_t value9, uint32_t value10, uint64_t value11,
    uint32_t value12, uint64_t value13, uint32_t value14, uint64_t value15,
    uint32_t value16, uint64_t value17, uint32_t value18, uint64_t value19,
    uint32_t value20, uint64_t value21, uint32_t value22, uint64_t value23,
    uint32_t value24, uint64_t value25, uint32_t value26, uint64_t value27,
    uint32_t value28, uint64_t value29, uint32_t value30, uint64_t value31);
extern "C" uint64_t outgoing_many(const uint64_t* input);
extern "C" uint64_t host_weighted(
    uint32_t value0, uint64_t value1, uint32_t value2, uint64_t value3,
    uint32_t value4, uint64_t value5, uint32_t value6, uint64_t value7,
    uint32_t value8, uint64_t value9, uint32_t value10, uint64_t value11,
    uint32_t value12, uint64_t value13, uint32_t value14, uint64_t value15,
    uint32_t value16, uint64_t value17, uint32_t value18, uint64_t value19,
    uint32_t value20, uint64_t value21, uint32_t value22, uint64_t value23,
    uint32_t value24, uint64_t value25, uint32_t value26, uint64_t value27,
    uint32_t value28, uint64_t value29, uint32_t value30, uint64_t value31) {
  return Weighted(
      {value0,  value1,  value2,  value3,  value4,  value5,  value6,  value7,
       value8,  value9,  value10, value11, value12, value13, value14, value15,
       value16, value17, value18, value19, value20, value21, value22, value23,
       value24, value25, value26, value27, value28, value29, value30, value31});
}

TEST(NativeCallableTest, StoredIncomingAndOutgoingArguments) {
  for (uint64_t seed : {0ull, 1ull, 0x123456789abcdef0ull}) {
    std::array<uint64_t, 32> values;
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = (seed + i * 0x9e3779b97f4a7c15ull) ^ (seed >> (i % 17));
    }
    for (size_t i = 0; i < values.size(); i += 2) {
      values[i] = static_cast<uint32_t>(values[i]);
    }
    EXPECT_EQ(incoming_many(values[0], values[1], values[2], values[3],
                            values[4], values[5], values[6], values[7],
                            values[8], values[9], values[10], values[11],
                            values[12], values[13], values[14], values[15],
                            values[16], values[17], values[18], values[19],
                            values[20], values[21], values[22], values[23],
                            values[24], values[25], values[26], values[27],
                            values[28], values[29], values[30], values[31]),
              Weighted(values));
    auto forward = values;
    auto reverse = values;
    forward[31] = 5;
    for (size_t i = 0; i < 31; ++i) {
      reverse[i] = values[30 - i];
    }
    reverse[31] = 9;
    EXPECT_EQ(outgoing_many(values.data()),
              Weighted(forward) ^ Weighted(reverse));
  }
}
