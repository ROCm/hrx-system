// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>
#include <loomcxx/view.h>

template <class T>
struct Source {
  // Borrowed storage; the pointee qualifier determines each observation.
  T* data;
};

template <class T>
static unsigned read_twice(Source<T> source) {
  source.data[0];
  static_cast<void>(*source.data);
  unsigned first = source.data[0];
  unsigned second = source.data[0];
  return first + second;
}

template <class Input, class Output>
static unsigned observe_one(Input* input, Output* output) {
  unsigned value = read_twice(Source<Input>{input});
  output[0] = 0;
  output[0] = value;
  return output[0];
}

unsigned volatile_word(const volatile unsigned* input,
                       volatile unsigned* output) {
  return observe_one(input, output);
}

unsigned volatile_parameter(volatile unsigned value) {
  static_cast<void>(value);
  return value;
}

template <class Input, class Output>
static void copy_observations(Input* input, Output* output, unsigned count,
                              unsigned lane) {
  loom::assume(count < 65);
  if (lane < count) {
    unsigned observed = observe_one(input + lane, output + lane);
    output[64 + lane] = observed ^ 0xa5a55a5au;
  }
}

[[loom::kernel, loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void volatile_memory(const volatile unsigned* input, volatile unsigned* output,
                     unsigned count) {
  copy_observations(input, output, count, loom::workitem_id.x);
}

[[loom::kernel, loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void ordinary_memory(const unsigned* input, unsigned* output, unsigned count) {
  copy_observations(input, output, count, loom::workitem_id.x);
}

using Words = unsigned __attribute__((vector_size(16)));

static Words volatile_vector_parameter(volatile Words value) {
  static_cast<void>(value);
  return value;
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void volatile_scalar_parameter(const unsigned* input, unsigned* output) {
  output[0] = volatile_parameter(input[0]);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void volatile_vector_parameter_kernel(const Words* input, Words* output) {
  output[0] = volatile_vector_parameter(input[0]);
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void volatile_vectors(const volatile Words* input, volatile Words* output) {
  static_cast<void>(*input);
  Words value = input[0];
  output[0] = Words{0, 0, 0, 0};
  output[0] = value;
}

[[loom::kernel, loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void volatile_shared(const unsigned* input, unsigned* output) {
  [[loom::workgroup]] volatile unsigned table[64];
  unsigned lane = loom::workitem_id.x;
  // Each invocation owns its element, so no cross-invocation ordering is
  // needed.
  table[lane] = input[lane];
  table[lane];
  unsigned first = table[lane];
  table[lane] = first + 1u;
  output[lane] = table[lane];
}

template <class View>
static unsigned read_view_twice(View source) {
  loom::view::load(source, 0, 1);
  unsigned first = loom::view::load(source, 0, 1);
  unsigned second = loom::view::load(source, 0, 1);
  return first + second;
}

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void volatile_views(const volatile unsigned* input, volatile unsigned* output,
                    unsigned origin, unsigned stride) {
  loom::assume(origin < 16);
  loom::assume(stride < 16);
  auto layout = loom::encoding::layout::strided(stride, 1u);
  auto source = loom::buffer::view<2, 4>(input + origin, {}, layout);
  auto row = loom::view::subview<1, 4>(source, {1, 0}, {});
  auto destination =
      loom::buffer::view<1, 4>(output, {}, loom::encoding::layout::dense<2>());
  unsigned value = read_view_twice(row);
  loom::view::store(0u, destination, 0, 1);
  loom::view::store(value, destination, 0, 1);
}
