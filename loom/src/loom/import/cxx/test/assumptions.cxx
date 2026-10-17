// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>
#include <loomcxx/buffer.h>
#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/view.h>

[[loom::force_inline]] unsigned bound_pair(unsigned first, unsigned second) {
  __builtin_assume(first < 256u && second < 256u);
  return first * 257u + second;
}

[[loom::force_inline]] unsigned bound_seven(unsigned a, unsigned b, unsigned c,
                                            unsigned d, unsigned e, unsigned f,
                                            unsigned g) {
  loom::assume(a < 256u && b < 256u && c < 256u && d < 256u && e < 256u &&
               f < 256u && g < 256u);
  return a + 2u * b + 3u * c + 5u * d + 7u * e + 11u * f + 13u * g;
}

template <unsigned Limit>
[[loom::force_inline]] unsigned refine(unsigned value) {
  loom::assume(((value < 256u) && ((value) < (1u << Limit))) && value < 64u);
  return value * 17u;
}

[[loom::force_inline]] unsigned bound_repeated(unsigned value) {
  return refine<5>(value);
}

[[loom::force_inline]] unsigned bound_capacity(unsigned value) {
  constexpr unsigned capacity = 28672;
  constexpr unsigned stride = 16;
  loom::assume(value <
               ((capacity / sizeof(unsigned) - 16u - 320u) / stride + 1u));
  return value * 16u + 336u;
}

[[loom::force_inline]] unsigned bound_cast(unsigned value) {
  loom::assume(value < static_cast<unsigned char>(272u));
  return value + 5u;
}

[[loom::force_inline]] unsigned bound_byte(unsigned char value) {
  loom::assume(value < 256u);
  return value + (value >= 128u ? 1024u : 0u);
}

[[loom::force_inline]] unsigned bound_wide(unsigned long long value) {
  loom::assume(value < (0xffffffffu + 257u));
  return (unsigned)(value * 3u);
}

[[loom::force_inline]] unsigned bound_size(unsigned value) {
  loom::assume(value < sizeof(++value) * 4u);
  return value;
}

[[loom::force_inline]] unsigned bound_scoped(unsigned value) {
  if (value < 256u) {
    loom::assume(value < 256u);
    return value + 1u;
  }
  return value + 3u;
}

[[loom::force_inline]] unsigned bound_inclusive(unsigned tokens) {
  constexpr unsigned capacity = 427u;
  loom::assume(tokens <= capacity);
  return tokens * 19u + 3u;
}

[[loom::force_inline]] int bound_signed(int hidden, int capacity) {
  loom::assume(hidden > 0 && hidden <= capacity);
  return hidden * 23 + capacity;
}

[[loom::force_inline]] unsigned bound_unsigned(unsigned tokens,
                                               unsigned capacity) {
  loom::assume(tokens <= capacity);
  return tokens ^ capacity;
}

[[loom::kernel, loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void assumption_kernel(unsigned* output, unsigned input) {
  unsigned lane = threadIdx.x;
  unsigned value = input + lane;
  unsigned output_offset = lane * 12u;
  output[output_offset] = bound_pair(value & 255u, (value >> 8u) & 255u);
  output[output_offset + 1u] =
      bound_seven(value & 255u, 1u, 2u, 3u, 5u, 7u, 11u);
  output[output_offset + 2u] = bound_repeated(value & 31u);
  output[output_offset + 3u] = bound_capacity(value % 428u);
  output[output_offset + 4u] = bound_cast(value & 15u);
  output[output_offset + 5u] = bound_byte((unsigned char)value);
  output[output_offset + 6u] = bound_wide(value & 255u);
  output[output_offset + 7u] = bound_size(value & 15u);
  output[output_offset + 8u] = bound_scoped(value);
  output[output_offset + 9u] = bound_inclusive(value % 428u);
  int hidden = (int)(value % 63u) + 1;
  int capacity = hidden + (int)(value % 5u);
  output[output_offset + 10u] = (unsigned)bound_signed(hidden, capacity);
  unsigned tokens = value & 0x7fffffffu;
  unsigned token_capacity = tokens | 0x80000000u;
  output[output_offset + 11u] = bound_unsigned(tokens, token_capacity);
}

unsigned assumption_value_at(unsigned position) {
  if (position == 0u) {
    return 0u;
  }
  if (position == 1u) {
    return 1u;
  }
  if (position == 2u) {
    return 127u;
  }
  if (position == 3u) {
    return 128u;
  }
  if (position == 4u) {
    return 254u;
  }
  return 255u;
}

unsigned assumption_value_input(loom::check::ordinal trial) {
  return assumption_value_at((unsigned)trial);
}

unsigned assumption_pair_input(loom::check::ordinal trial, unsigned position) {
  unsigned trial_value = trial;
  return assumption_value_at(position ? trial_value % 6u : trial_value / 6u);
}

unsigned assumption_seven_input(loom::check::ordinal trial, unsigned position) {
  unsigned trial_value = trial;
  if (trial_value == 0u) {
    return 0u;
  }
  if (trial_value == 1u) {
    return 255u;
  }
  return trial_value == position + 2u ? 255u : 0u;
}

unsigned assumption_repeated_input(loom::check::ordinal trial) {
  unsigned trial_value = trial;
  if (trial_value == 0u) {
    return 0u;
  }
  if (trial_value == 1u) {
    return 1u;
  }
  if (trial_value == 2u) {
    return 15u;
  }
  if (trial_value == 3u) {
    return 16u;
  }
  return 31u;
}

unsigned assumption_capacity_input(loom::check::ordinal trial) {
  unsigned trial_value = trial;
  if (trial_value == 0u) {
    return 0u;
  }
  if (trial_value == 1u) {
    return 1u;
  }
  if (trial_value == 2u) {
    return 255u;
  }
  if (trial_value == 3u) {
    return 256u;
  }
  if (trial_value == 4u) {
    return 426u;
  }
  return 427u;
}

unsigned assumption_cast_size_input(loom::check::ordinal trial) {
  unsigned trial_value = trial;
  if (trial_value == 0u) {
    return 0u;
  }
  if (trial_value == 1u) {
    return 1u;
  }
  if (trial_value == 2u) {
    return 7u;
  }
  return 15u;
}

unsigned char assumption_byte_input(loom::check::ordinal trial) {
  return (unsigned char)(unsigned)trial;
}

unsigned long long assumption_wide_input(loom::check::ordinal trial) {
  return assumption_value_input(trial);
}

unsigned assumption_scoped_input(loom::check::ordinal trial) {
  unsigned trial_value = trial;
  if (trial_value == 0u) {
    return 0u;
  }
  if (trial_value == 1u) {
    return 1u;
  }
  if (trial_value == 2u) {
    return 127u;
  }
  if (trial_value == 3u) {
    return 128u;
  }
  if (trial_value == 4u) {
    return 255u;
  }
  if (trial_value == 5u) {
    return 256u;
  }
  if (trial_value == 6u) {
    return 427u;
  }
  if (trial_value == 7u) {
    return 0x7fffffffu;
  }
  return 0xffffffffu;
}

unsigned assumption_inclusive_input(loom::check::ordinal trial) {
  unsigned trial_value = trial;
  if (trial_value == 0u) {
    return 0u;
  }
  if (trial_value == 1u) {
    return 1u;
  }
  if (trial_value == 2u) {
    return 127u;
  }
  if (trial_value == 3u) {
    return 426u;
  }
  return 427u;
}

int assumption_signed_input(loom::check::ordinal trial, unsigned position) {
  unsigned trial_value = trial;
  if (position == 0u) {
    if (trial_value < 2u) {
      return 1;
    }
    if (trial_value < 4u) {
      return 31;
    }
    return 127;
  }
  if (trial_value == 0u) {
    return 1;
  }
  if (trial_value == 1u) {
    return 7;
  }
  if (trial_value == 2u) {
    return 31;
  }
  if (trial_value == 3u) {
    return 63;
  }
  return 255;
}

unsigned assumption_unsigned_input(loom::check::ordinal trial,
                                   unsigned position) {
  unsigned trial_value = trial;
  if (trial_value == 0u) {
    return 0u;
  }
  if (trial_value == 1u) {
    return position ? 0x80000000u : 0x7fffffffu;
  }
  if (trial_value == 2u) {
    return position ? 0xffffffffu : 0x80000000u;
  }
  if (trial_value == 3u) {
    return position ? 0xffffffffu : 0xfffffffeu;
  }
  return 0xffffffffu;
}

unsigned assumption_kernel_input(loom::check::ordinal trial) {
  unsigned trial_value = trial;
  if (trial_value == 0u) {
    return 0u;
  }
  if (trial_value == 1u) {
    return 1u;
  }
  if (trial_value == 2u) {
    return 127u;
  }
  if (trial_value == 3u) {
    return 128u;
  }
  if (trial_value == 4u) {
    return 255u;
  }
  if (trial_value == 5u) {
    return 256u;
  }
  if (trial_value == 6u) {
    return 427u;
  }
  if (trial_value == 7u) {
    return 0xffffffc0u;
  }
  return 0xffffffffu;
}

unsigned bound_pair_oracle(unsigned first, unsigned second) {
  return first * 257u + second;
}

unsigned bound_seven_oracle(unsigned a, unsigned b, unsigned c, unsigned d,
                            unsigned e, unsigned f, unsigned g) {
  return a + 2u * b + 3u * c + 5u * d + 7u * e + 11u * f + 13u * g;
}

unsigned bound_repeated_oracle(unsigned value) { return value * 17u; }

unsigned bound_capacity_oracle(unsigned value) { return value * 16u + 336u; }

unsigned bound_cast_oracle(unsigned value) { return value + 5u; }

unsigned bound_byte_oracle(unsigned char value) {
  return (unsigned)value + (value >= 128u ? 1024u : 0u);
}

unsigned bound_wide_oracle(unsigned long long value) {
  return (unsigned)(value * 3ull);
}

unsigned bound_size_oracle(unsigned value) { return value; }

unsigned bound_scoped_oracle(unsigned value) {
  return value + (value < 256u ? 1u : 3u);
}

unsigned bound_inclusive_oracle(unsigned tokens) { return tokens * 19u + 3u; }

int bound_signed_oracle(int hidden, int capacity) {
  return hidden * 23 + capacity;
}

unsigned bound_unsigned_oracle(unsigned tokens, unsigned capacity) {
  return tokens ^ capacity;
}

void bound_pair_observation(unsigned first, unsigned second,
                            loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_pair(first, second), values, 0);
}

void bound_pair_reference(unsigned first, unsigned second,
                          loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_pair_oracle(first, second), values, 0);
}

void bound_seven_observation(unsigned a, unsigned b, unsigned c, unsigned d,
                             unsigned e, unsigned f, unsigned g,
                             loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_seven(a, b, c, d, e, f, g), values, 0);
}

void bound_seven_reference(unsigned a, unsigned b, unsigned c, unsigned d,
                           unsigned e, unsigned f, unsigned g,
                           loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_seven_oracle(a, b, c, d, e, f, g), values, 0);
}

void bound_repeated_observation(unsigned value,
                                loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_repeated(value), values, 0);
}

void bound_repeated_reference(unsigned value,
                              loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_repeated_oracle(value), values, 0);
}

void bound_capacity_observation(unsigned value,
                                loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_capacity(value), values, 0);
}

void bound_capacity_reference(unsigned value,
                              loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_capacity_oracle(value), values, 0);
}

void bound_cast_size_observations(unsigned value,
                                  loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<2>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_cast(value), values, 0);
  loom::view::store(bound_size(value), values, 1);
}

void bound_cast_size_reference(unsigned value,
                               loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<2>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_cast_oracle(value), values, 0);
  loom::view::store(bound_size_oracle(value), values, 1);
}

void bound_byte_observation(unsigned char value,
                            loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_byte(value), values, 0);
}

void bound_byte_reference(unsigned char value,
                          loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_byte_oracle(value), values, 0);
}

void bound_wide_observation(unsigned long long value,
                            loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_wide(value), values, 0);
}

void bound_wide_reference(unsigned long long value,
                          loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_wide_oracle(value), values, 0);
}

void bound_scoped_observation(unsigned value,
                              loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_scoped(value), values, 0);
}

void bound_scoped_reference(unsigned value,
                            loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_scoped_oracle(value), values, 0);
}

void bound_inclusive_observation(unsigned value,
                                 loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_inclusive(value), values, 0);
}

void bound_inclusive_reference(unsigned value,
                               loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_inclusive_oracle(value), values, 0);
}

void bound_signed_observation(int hidden, int capacity,
                              loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store((unsigned)bound_signed(hidden, capacity), values, 0);
}

void bound_signed_reference(int hidden, int capacity,
                            loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store((unsigned)bound_signed_oracle(hidden, capacity), values, 0);
}

void bound_unsigned_observation(unsigned tokens, unsigned capacity,
                                loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_unsigned(tokens, capacity), values, 0);
}

void bound_unsigned_reference(unsigned tokens, unsigned capacity,
                              loom::type::buffer<unsigned> output) {
  auto values =
      loom::buffer::view<1>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(bound_unsigned_oracle(tokens, capacity), values, 0);
}

void assumption_kernel_oracle(loom::type::buffer<unsigned> output,
                              unsigned input) {
  auto values =
      loom::buffer::view<768>(output, {}, loom::encoding::layout::dense<1>());
  for (unsigned lane = 0; lane < 64u; ++lane) {
    unsigned value = input + lane;
    unsigned byte = value & 255u;
    unsigned position = lane * 12u;
    loom::view::store(bound_pair_oracle(byte, (value >> 8u) & 255u), values,
                      position);
    loom::view::store(bound_seven_oracle(byte, 1u, 2u, 3u, 5u, 7u, 11u), values,
                      position + 1u);
    loom::view::store(bound_repeated_oracle(value & 31u), values,
                      position + 2u);
    loom::view::store(bound_capacity_oracle(value % 428u), values,
                      position + 3u);
    loom::view::store(bound_cast_oracle(value & 15u), values, position + 4u);
    loom::view::store(bound_byte_oracle((unsigned char)byte), values,
                      position + 5u);
    loom::view::store(bound_wide_oracle(byte), values, position + 6u);
    loom::view::store(bound_size_oracle(value & 15u), values, position + 7u);
    loom::view::store(bound_scoped_oracle(value), values, position + 8u);
    loom::view::store(bound_inclusive_oracle(value % 428u), values,
                      position + 9u);
    int hidden = (int)(value % 63u) + 1;
    int capacity = hidden + (int)(value % 5u);
    loom::view::store((unsigned)bound_signed_oracle(hidden, capacity), values,
                      position + 10u);
    unsigned tokens = value & 0x7fffffffu;
    loom::view::store(bound_unsigned_oracle(tokens, tokens | 0x80000000u),
                      values, position + 11u);
  }
}

LOOM_CHECK_SCENARIO(assumptions_functions) {
  loom::check::trial<36>([](loom::check::ordinal trial,
                            loom::check::entropy entropy) {
    const auto first = loom::check::generate<assumption_pair_input>(trial, 0u);
    const auto second = loom::check::generate<assumption_pair_input>(trial, 1u);
    const auto output = loom::check::fill<1, unsigned>(-123u);
    loom::check::compare<bound_pair_observation, bound_pair_reference>(
        first, second, output,
        [&] { loom::check::expect_bitwise(output, output); });
  });
  loom::check::trial<9>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto a = loom::check::generate<assumption_seven_input>(trial, 0u);
        const auto b = loom::check::generate<assumption_seven_input>(trial, 1u);
        const auto c = loom::check::generate<assumption_seven_input>(trial, 2u);
        const auto d = loom::check::generate<assumption_seven_input>(trial, 3u);
        const auto e = loom::check::generate<assumption_seven_input>(trial, 4u);
        const auto f = loom::check::generate<assumption_seven_input>(trial, 5u);
        const auto g = loom::check::generate<assumption_seven_input>(trial, 6u);
        const auto output = loom::check::fill<1, unsigned>(-123u);
        loom::check::compare<bound_seven_observation, bound_seven_reference>(
            a, b, c, d, e, f, g, output,
            [&] { loom::check::expect_bitwise(output, output); });
      });
  loom::check::trial<5>([](loom::check::ordinal trial,
                           loom::check::entropy entropy) {
    const auto value = loom::check::generate<assumption_repeated_input>(trial);
    const auto output = loom::check::fill<1, unsigned>(-123u);
    loom::check::compare<bound_repeated_observation, bound_repeated_reference>(
        value, output, [&] { loom::check::expect_bitwise(output, output); });
  });
  loom::check::trial<6>([](loom::check::ordinal trial,
                           loom::check::entropy entropy) {
    const auto value = loom::check::generate<assumption_capacity_input>(trial);
    const auto output = loom::check::fill<1, unsigned>(-123u);
    loom::check::compare<bound_capacity_observation, bound_capacity_reference>(
        value, output, [&] { loom::check::expect_bitwise(output, output); });
  });
  loom::check::trial<4>([](loom::check::ordinal trial,
                           loom::check::entropy entropy) {
    const auto value = loom::check::generate<assumption_cast_size_input>(trial);
    const auto output = loom::check::fill<2, unsigned>(-123u);
    loom::check::compare<bound_cast_size_observations,
                         bound_cast_size_reference>(
        value, output, [&] { loom::check::expect_bitwise(output, output); });
  });
  loom::check::trial<256>([](loom::check::ordinal trial,
                             loom::check::entropy entropy) {
    const auto value = loom::check::generate<assumption_byte_input>(trial);
    const auto output = loom::check::fill<1, unsigned>(-123u);
    loom::check::compare<bound_byte_observation, bound_byte_reference>(
        value, output, [&] { loom::check::expect_bitwise(output, output); });
  });
  loom::check::trial<6>([](loom::check::ordinal trial,
                           loom::check::entropy entropy) {
    const auto value = loom::check::generate<assumption_wide_input>(trial);
    const auto output = loom::check::fill<1, unsigned>(-123u);
    loom::check::compare<bound_wide_observation, bound_wide_reference>(
        value, output, [&] { loom::check::expect_bitwise(output, output); });
  });
  loom::check::trial<9>([](loom::check::ordinal trial,
                           loom::check::entropy entropy) {
    const auto value = loom::check::generate<assumption_scoped_input>(trial);
    const auto output = loom::check::fill<1, unsigned>(-123u);
    loom::check::compare<bound_scoped_observation, bound_scoped_reference>(
        value, output, [&] { loom::check::expect_bitwise(output, output); });
  });
  loom::check::trial<5>([](loom::check::ordinal trial,
                           loom::check::entropy entropy) {
    const auto value = loom::check::generate<assumption_inclusive_input>(trial);
    const auto output = loom::check::fill<1, unsigned>(-123u);
    loom::check::compare<bound_inclusive_observation,
                         bound_inclusive_reference>(
        value, output, [&] { loom::check::expect_bitwise(output, output); });
  });
  loom::check::trial<5>(
      [](loom::check::ordinal trial, loom::check::entropy entropy) {
        const auto hidden =
            loom::check::generate<assumption_signed_input>(trial, 0u);
        const auto capacity =
            loom::check::generate<assumption_signed_input>(trial, 1u);
        const auto output = loom::check::fill<1, unsigned>(-123u);
        loom::check::compare<bound_signed_observation, bound_signed_reference>(
            hidden, capacity, output,
            [&] { loom::check::expect_bitwise(output, output); });
      });
  loom::check::trial<5>([](loom::check::ordinal trial,
                           loom::check::entropy entropy) {
    const auto tokens =
        loom::check::generate<assumption_unsigned_input>(trial, 0u);
    const auto capacity =
        loom::check::generate<assumption_unsigned_input>(trial, 1u);
    const auto output = loom::check::fill<1, unsigned>(-123u);
    loom::check::compare<bound_unsigned_observation, bound_unsigned_reference>(
        tokens, capacity, output,
        [&] { loom::check::expect_bitwise(output, output); });
  });
}

LOOM_CHECK_SCENARIO(assumption_kernel_values) {
  loom::check::trial<9>([](loom::check::ordinal trial,
                           loom::check::entropy entropy) {
    const auto input = loom::check::generate<assumption_kernel_input>(trial);
    const auto storage = loom::check::fill<800, unsigned>(-123u);
    const auto output = loom::check::slice<768>(storage, 16);
    loom::check::compare<assumption_kernel, assumption_kernel_oracle>(
        output, input, [&] { loom::check::expect_bitwise(storage, storage); });
  });
}

template <class Index>
static LOOM_FORCE_INLINE void assumed_table_indices(const Index* input,
                                                    unsigned* output) {
  unsigned lane = loom::kernel::workitem::id.x;
  auto* table =
      loom::buffer::alloca<unsigned, loom::memory_space::workgroup, 16>(6);
  if (lane < 6u) {
    table[lane] = lane + 10u;
  }
  loom::kernel::workgroup::barrier();

  // Each relation constrains a separate unbounded input. The dynamic-operand
  // assumption must prove both nonnegativity and the table's upper bound.
  Index first = input[4u * lane];
  Index second = input[4u * lane + 1u];
  Index third = input[4u * lane + 2u];
  Index fourth = input[4u * lane + 3u];
  Index count = 6;
  Index last = 5;
  loom::assume(first < count);
  loom::assume(second <= last);
  loom::assume(count > third);
  loom::assume(last >= fourth);
  output[4u * lane] = table[first];
  output[4u * lane + 1u] = table[second];
  output[4u * lane + 2u] = table[third];
  output[4u * lane + 3u] = table[fourth];
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void assumed_table_u32(const unsigned* input, unsigned* output) {
  assumed_table_indices(input, output);
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void assumed_table_u64(const unsigned long long* input, unsigned* output) {
  assumed_table_indices(input, output);
}

LOOM_CHECK_CASE(unsigned_table_u32) {
  const auto input = loom::check::iota<128, unsigned>(0u, 1u, 6);
  const auto storage = loom::check::fill<132, unsigned>(0xA5A5A5A5u);
  const auto output = loom::check::slice<128>(storage, 2);
  loom::check::launch<assumed_table_u32>(input, output);
  loom::check::expect_bitwise(output,
                              loom::check::iota<128, unsigned>(10u, 1u, 6));
  loom::check::expect_bitwise(loom::check::slice<2>(storage, 0),
                              loom::check::fill<2, unsigned>(0xA5A5A5A5u));
  loom::check::expect_bitwise(loom::check::slice<2>(storage, 130),
                              loom::check::fill<2, unsigned>(0xA5A5A5A5u));
}

LOOM_CHECK_CASE(unsigned_table_u64) {
  const auto input = loom::check::iota<128, unsigned long long>(0ull, 1ull, 6);
  const auto storage = loom::check::fill<132, unsigned>(0xA5A5A5A5u);
  const auto output = loom::check::slice<128>(storage, 2);
  loom::check::launch<assumed_table_u64>(input, output);
  loom::check::expect_bitwise(output,
                              loom::check::iota<128, unsigned>(10u, 1u, 6));
  loom::check::expect_bitwise(loom::check::slice<2>(storage, 0),
                              loom::check::fill<2, unsigned>(0xA5A5A5A5u));
  loom::check::expect_bitwise(loom::check::slice<2>(storage, 130),
                              loom::check::fill<2, unsigned>(0xA5A5A5A5u));
}
