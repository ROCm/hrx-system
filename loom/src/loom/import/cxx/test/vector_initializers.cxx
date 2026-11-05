// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/view.h>

typedef unsigned U4 __attribute__((vector_size(16)));
typedef int I4 __attribute__((vector_size(16)));
typedef float F4 __attribute__((vector_size(16)));
typedef unsigned char B16 __attribute__((vector_size(16)));
typedef unsigned E4 __attribute__((ext_vector_type(4)));

[[loom::inline]] static I4 offsets(int input) {
  return I4{input, input + 1, input + 2, input + 3};
}

[[loom::inline]] static U4 add(U4 first, U4 second) { return first + second; }

template <typename Vector>
[[loom::inline]] static Vector first(unsigned input) {
  return Vector{input};
}

int constructor_return(int input, unsigned lane) {
  return offsets(input)[lane];
}

unsigned constructor_argument(unsigned input, unsigned lane) {
  return add(U4{input, 7u}, U4{})[lane];
}

unsigned constructor_single(unsigned input, unsigned lane) {
  return first<E4>(input)[lane];
}

unsigned constructor_narrow(unsigned input, unsigned lane) {
  return B16{(unsigned char)input, 255u, 128u}[lane];
}

unsigned constructor_float(int input, unsigned lane) {
  return __builtin_bit_cast(U4, F4{float(input), -0.0f, 1.5f})[lane];
}

unsigned constructor_nested(unsigned input, unsigned lane) {
  return (U4{input, 1u, 2u, 3u} + U4{9u, 8u, 7u, 6u})[lane];
}

[[loom::inline]] static unsigned record(unsigned* trace, unsigned digit,
                                        unsigned value) {
  trace[0] = trace[0] * 10u + digit;
  return value;
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void vector_initializers(U4* output, unsigned input) {
  output[0] = (U4)offsets((int)input);
  output[1] = add(U4{input, 7u}, U4{});
  output[2] = (U4)first<E4>(input);
  output[3] = __builtin_bit_cast(U4, B16{(unsigned char)input, 255u, 128u});
  output[4] = __builtin_bit_cast(U4, F4{float((int)input), -0.0f, 1.5f});
  output[5] = U4{input, 1u, 2u, 3u} + U4{9u, 8u, 7u, 6u};
  output[7] = U4{};
  unsigned* trace = reinterpret_cast<unsigned*>(output + 7);
  output[6] = U4{record(trace, 1, input), record(trace, 2, input + 1u),
                 record(trace, 3, input + 2u), record(trace, 4, input + 3u)};
}

static unsigned vector_initializer_input(loom::check::ordinal trial) {
  unsigned index = trial;
  if (index == 0u) {
    return 0u;
  }
  if (index == 1u) {
    return 1u;
  }
  if (index == 2u) {
    return 0x7fu;
  }
  if (index == 3u) {
    return 0xffu;
  }
  if (index == 4u) {
    return 0x100u;
  }
  if (index == 5u) {
    return 0xffffff80u;
  }
  if (index == 6u) {
    return 0x7ffffffcu;
  }
  return 0xffffffffu;
}

static void vector_initializer_observations(
    unsigned input, loom::type::buffer<unsigned> output) {
  auto results =
      loom::buffer::view<36>(output, {}, loom::encoding::layout::dense<1>());
  for (unsigned lane = 0; lane < 4; ++lane) {
    loom::view::store((unsigned)constructor_return((int)input, lane), results,
                      lane);
    loom::view::store(constructor_argument(input, lane), results, 4u + lane);
    loom::view::store(constructor_single(input, lane), results, 8u + lane);
    loom::view::store(constructor_float((int)input, lane), results, 28u + lane);
    loom::view::store(constructor_nested(input, lane), results, 32u + lane);
  }
  for (unsigned lane = 0; lane < 16; ++lane) {
    loom::view::store(constructor_narrow(input, lane), results, 12u + lane);
  }
}

static void vector_initializer_reference(unsigned input,
                                         loom::type::buffer<unsigned> output) {
  auto results =
      loom::buffer::view<36>(output, {}, loom::encoding::layout::dense<1>());
  const unsigned input_float_bits =
      __builtin_bit_cast(unsigned, (float)(int)input);
  for (unsigned lane = 0; lane < 4; ++lane) {
    const unsigned return_value = (unsigned)((int)input + (int)lane);
    const unsigned argument_value = lane == 0 ? input : lane == 1 ? 7u : 0u;
    const unsigned single_value = lane == 0 ? input : 0u;
    const unsigned float_value = lane == 0   ? input_float_bits
                                 : lane == 1 ? 0x80000000u
                                 : lane == 2 ? 0x3fc00000u
                                             : 0u;
    const unsigned nested_value = lane == 0 ? input + 9u : 9u;
    loom::view::store(return_value, results, lane);
    loom::view::store(argument_value, results, 4u + lane);
    loom::view::store(single_value, results, 8u + lane);
    loom::view::store(float_value, results, 28u + lane);
    loom::view::store(nested_value, results, 32u + lane);
  }
  for (unsigned lane = 0; lane < 16; ++lane) {
    const unsigned narrow_value = lane == 0   ? (unsigned char)input
                                  : lane == 1 ? 255u
                                  : lane == 2 ? 128u
                                              : 0u;
    loom::view::store(narrow_value, results, 12u + lane);
  }
}

static void vector_initializers_oracle(loom::type::buffer<unsigned> output,
                                       unsigned input) {
  auto results =
      loom::buffer::view<32>(output, {}, loom::encoding::layout::dense<1>());
  const unsigned input_float_bits =
      __builtin_bit_cast(unsigned, (float)(int)input);
  const unsigned narrow_word = (input & 0xffu) | 0x0080ff00u;
  for (unsigned lane = 0; lane < 4; ++lane) {
    const unsigned return_value = (unsigned)((int)input + (int)lane);
    const unsigned argument_value = lane == 0 ? input : lane == 1 ? 7u : 0u;
    const unsigned single_value = lane == 0 ? input : 0u;
    const unsigned float_value = lane == 0   ? input_float_bits
                                 : lane == 1 ? 0x80000000u
                                 : lane == 2 ? 0x3fc00000u
                                             : 0u;
    const unsigned nested_value = lane == 0 ? input + 9u : 9u;
    loom::view::store(return_value, results, lane);
    loom::view::store(argument_value, results, 4u + lane);
    loom::view::store(single_value, results, 8u + lane);
    loom::view::store(lane == 0 ? narrow_word : 0u, results, 12u + lane);
    loom::view::store(float_value, results, 16u + lane);
    loom::view::store(nested_value, results, 20u + lane);
    loom::view::store(return_value, results, 24u + lane);
  }
  loom::view::store(1234u, results, 28u);
  loom::view::store(0u, results, 29u);
  loom::view::store(0u, results, 30u);
  loom::view::store(0u, results, 31u);
}

LOOM_CHECK_SCENARIO(vector_initializers_functions) {
  loom::check::trial<8>([](loom::check::ordinal trial,
                           loom::check::entropy entropy) {
    const auto input = loom::check::generate<vector_initializer_input>(trial);
    const auto output = loom::check::fill<36, unsigned>(-123u);
    loom::check::compare<vector_initializer_observations,
                         vector_initializer_reference>(
        input, output, [&] { loom::check::expect_bitwise(output, output); });
  });
}

LOOM_CHECK_SCENARIO(vector_initializer_values) {
  loom::check::trial<8>([](loom::check::ordinal trial,
                           loom::check::entropy entropy) {
    const auto input = loom::check::generate<vector_initializer_input>(trial);
    const auto storage = loom::check::fill<16, U4>(-123u);
    const auto output = loom::check::slice<8>(storage, 4);
    loom::check::compare<vector_initializers, vector_initializers_oracle>(
        output, input, [&] { loom::check::expect_bitwise(storage, storage); });
  });
}
