// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/predicate.h>
#include <loomcxx/vector.h>

namespace loomt = loom::type;

[[loom::config("test.vector.slice.width"),
  loom::where(loom::predicate::range(4u, 16u))]]
extern const unsigned slice_width;

template <class T>
static LOOM_FORCE_INLINE void slice_values(const T* input, T* output,
                                           unsigned trial) {
  using Dynamic = loomt::vector<loomt::shape<loomt::dynamic>, T>;
  using Pair = T __attribute__((ext_vector_type(2)));
  using SemanticPair = loomt::vector<loomt::shape<2>, T>;
  using Tile = loomt::vector<loomt::shape<4, 8>, T>;
  using Part = loomt::vector<loomt::shape<2, loomt::dynamic>, T>;
  auto source =
      loom::buffer::view<32>(input, {}, loom::encoding::layout::dense<1>());
  auto destination =
      loom::buffer::view<48>(output, {}, loom::encoding::layout::dense<1>());
  Dynamic values = loom::vector::load<Dynamic>(
      source, {0}, {loom::index::cast(slice_width)});
  Dynamic middle = loom::vector::slice<Dynamic>(
      values, 1u, {loom::index::cast(slice_width / 2)});
  loom::vector::store(middle, destination, {0});
  Pair tail = loom::vector::slice<Pair>(values, slice_width - 2);
  loom::vector::store(tail, destination, {16});
  SemanticPair copy = loom::vector::slice<SemanticPair>(tail, {0});
  loom::vector::store(copy, destination, {20});

  auto source_tile =
      loom::buffer::view<4, 8>(input, {}, loom::encoding::layout::dense<2>());
  auto destination_tile = loom::buffer::view<2, 8>(
      output + 24, {}, loom::encoding::layout::dense<2>());
  Tile tile = loom::vector::load<Tile>(source_tile, {0, 0});
  Part part = loom::vector::slice<Part>(tile, {trial % 3, trial % 3},
                                        {loom::index::cast(slice_width / 4)});
  loom::vector::store(part, destination_tile, {0, 0});
}

// The reference copies individual logical elements, without register vectors,
// and leaves every location outside the selected rectangles untouched.
template <class T>
void slice_oracle(loomt::buffer<T> input, loomt::buffer<T> output,
                  unsigned trial) {
  auto source =
      loom::buffer::view<32>(input, {}, loom::encoding::layout::dense<1>());
  auto destination =
      loom::buffer::view<48>(output, {}, loom::encoding::layout::dense<1>());
  for (unsigned index = 0; index < slice_width / 2; ++index) {
    loom::view::store(loom::view::load(source, index + 1), destination, index);
  }
  for (unsigned index = 0; index < 2; ++index) {
    T value = loom::view::load(source, slice_width - 2 + index);
    loom::view::store(value, destination, 16 + index);
    loom::view::store(value, destination, 20 + index);
  }
  for (unsigned row = 0; row < 2; ++row) {
    for (unsigned column = 0; column < slice_width / 4; ++column) {
      T value =
          loom::view::load(source, (trial % 3 + row) * 8 + trial % 3 + column);
      loom::view::store(value, destination, 24 + row * 8 + column);
    }
  }
}

unsigned slice_trial(loom::check::ordinal trial) {
  return static_cast<unsigned>(trial);
}

template <class T>
void slice_inputs(loomt::buffer<T> input) {
  auto values =
      loom::buffer::view<32>(input, {}, loom::encoding::layout::dense<1>());
  for (unsigned index = 0; index < 32; ++index) {
    unsigned bits = index == 0   ? 0u
                    : index == 1 ? 0xffffffffu
                                 : 0x80000000u + index * 7919u;
    T value = {};
    if constexpr (sizeof(T) == 1) {
      value = __builtin_bit_cast(T, static_cast<unsigned char>(bits));
    } else if constexpr (sizeof(T) == 2) {
      value = __builtin_bit_cast(T, static_cast<unsigned short>(bits));
    } else if constexpr (sizeof(T) == 4) {
      value = __builtin_bit_cast(T, bits);
    } else {
      unsigned long long wide =
          (static_cast<unsigned long long>(bits) << 32) | (bits ^ 0x13579bdfu);
      value = __builtin_bit_cast(T, wide);
    }
    loom::view::store(value, values, index);
  }
}

#define SLICE_SCENARIO(T, Name)                                              \
  [[loom::kernel, loom::workgroup_size(1, 1, 1),                             \
    loom::workgroup_count(1, 1, 1)]]                                         \
  void Name(const T* input, T* output, unsigned trial) {                     \
    slice_values(input, output, trial);                                      \
  }                                                                          \
  LOOM_CHECK_SCENARIO(Name##_values) {                                       \
    loom::check::trial<3>(                                                   \
        [](loom::check::ordinal trial, loom::check::entropy entropy) {       \
          const auto input = loom::check::fill<32, T>(static_cast<T>(0));    \
          loom::check::generate<slice_inputs<T>>(input);                     \
          const auto storage = loom::check::fill<64, T>(static_cast<T>(99)); \
          const auto output = loom::check::slice<48>(storage, 8);            \
          const auto selected = loom::check::generate<slice_trial>(trial);   \
          loom::check::compare<Name, slice_oracle<T>>(                       \
              input, output, selected, [&] {                                 \
                loom::check::expect_bitwise(input, input);                   \
                loom::check::expect_bitwise(storage, storage);               \
              });                                                            \
        });                                                                  \
  }

SLICE_SCENARIO(unsigned char, slice_u8)
SLICE_SCENARIO(unsigned short, slice_u16)
SLICE_SCENARIO(unsigned, slice_u32)
SLICE_SCENARIO(unsigned long long, slice_u64)
SLICE_SCENARIO(_Float16, slice_f16)
SLICE_SCENARIO(float, slice_f32)
