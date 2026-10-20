// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/vector.h>

namespace loomt = loom::type;

template <class T>
static LOOM_FORCE_INLINE void slice_payloads(const T* input, T* output) {
  using Source = T __attribute__((ext_vector_type(16)));
  using Four = T __attribute__((ext_vector_type(4)));
  using Eight = T __attribute__((ext_vector_type(8)));
  using Two = T __attribute__((ext_vector_type(2)));
  auto source =
      loom::buffer::view<16>(input, {}, loom::encoding::layout::dense<1>());
  auto destination =
      loom::buffer::view<32>(output, {}, loom::encoding::layout::dense<1>());
  const auto values = loom::vector::load<Source>(source, {0});
  loom::vector::store(loom::vector::slice<Four>(values, 0u), destination, {0});
  loom::vector::store(loom::vector::slice<Four>(values, 1u), destination, {4});
  loom::vector::store(loom::vector::slice<Four>(values, 2u), destination, {8});
  loom::vector::store(loom::vector::slice<Four>(values, 3u), destination, {12});
  loom::vector::store(loom::vector::slice<Eight>(values, 3u), destination,
                      {16});
  loom::vector::store(loom::vector::slice<Two>(values, 13u), destination, {24});
  loom::vector::store(loom::vector::slice<Two>(values, 14u), destination, {26});
}

// Scalar source coordinates establish the expected byte-preserving result,
// independently of register slicing and packing.
template <class T>
void slice_oracle(loomt::buffer<T> input, loomt::buffer<T> output) {
  auto source =
      loom::buffer::view<16>(input, {}, loom::encoding::layout::dense<1>());
  auto destination =
      loom::buffer::view<32>(output, {}, loom::encoding::layout::dense<1>());
  for (unsigned group = 0; group < 4; ++group) {
    for (unsigned index = 0; index < 4; ++index) {
      loom::view::store(loom::view::load(source, group + index), destination,
                        group * 4 + index);
    }
  }
  for (unsigned index = 0; index < 8; ++index) {
    loom::view::store(loom::view::load(source, 3 + index), destination,
                      16 + index);
  }
  for (unsigned index = 0; index < 2; ++index) {
    loom::view::store(loom::view::load(source, 13 + index), destination,
                      24 + index);
    loom::view::store(loom::view::load(source, 14 + index), destination,
                      26 + index);
  }
}

// Rotating through raw patterns includes zero, all ones, high bits, and NaNs.
// Floating values participate only in bitcasts and memory transport.
template <class T>
void slice_inputs(loomt::buffer<T> input, unsigned trial) {
  auto values =
      loom::buffer::view<16>(input, {}, loom::encoding::layout::dense<1>());
  for (unsigned index = 0; index < 16; ++index) {
    unsigned ordinal = (index + trial) % 16;
    unsigned bits = ordinal == 0   ? 0u
                    : ordinal == 1 ? 0xffffu
                                   : 0x8080u + ordinal * 7919u;
    T value = {};
    if constexpr (sizeof(T) == 1) {
      value = __builtin_bit_cast(T, static_cast<unsigned char>(bits));
    } else {
      value = __builtin_bit_cast(T, static_cast<unsigned short>(bits));
    }
    loom::view::store(value, values, index);
  }
}

unsigned slice_trial(loom::check::ordinal trial) {
  return static_cast<unsigned>(trial);
}

#define SLICE_SCENARIO(T, Name)                                             \
  [[loom::kernel, loom::workgroup_size(1, 1, 1),                            \
    loom::workgroup_count(1, 1, 1)]]                                        \
  void Name(const T* input, T* output) {                                    \
    slice_payloads(input, output);                                          \
  }                                                                         \
  LOOM_CHECK_SCENARIO(Name##_values) {                                      \
    loom::check::trial<16>(                                                 \
        [](loom::check::ordinal trial, loom::check::entropy entropy) {      \
          const auto selected = loom::check::generate<slice_trial>(trial);  \
          const auto input = loom::check::fill<16, T>(static_cast<T>(0));   \
          loom::check::generate<slice_inputs<T>>(input, selected);          \
          const auto storage = loom::check::fill<48, T>(static_cast<T>(1)); \
          const auto output = loom::check::slice<32>(storage, 8);           \
          loom::check::compare<Name, slice_oracle<T>>(input, output, [&] {  \
            loom::check::expect_bitwise(input, input);                      \
            loom::check::expect_bitwise(storage, storage);                  \
          });                                                               \
        });                                                                 \
  }

SLICE_SCENARIO(unsigned char, slice_u8)
SLICE_SCENARIO(unsigned short, slice_u16)
SLICE_SCENARIO(_Float16, slice_f16)
SLICE_SCENARIO(__bf16, slice_bf16)
SLICE_SCENARIO(__float8_e4m3fn, slice_f8e4m3)
SLICE_SCENARIO(__float8_e5m2, slice_f8e5m2)
