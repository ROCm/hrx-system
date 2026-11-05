// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>
#include <loomcxx/numeric.h>
#include <loomcxx/pipeline.h>
#include <loomcxx/predicate.h>
#include <loomcxx/target/aie2p.h>
#include <loomcxx/vector.h>

#include <stdfloat>

namespace loomt = loom::type;

using BFloat16x16 = std::bfloat16_t __attribute__((ext_vector_type(16)));
using BFloat16x32 = std::bfloat16_t __attribute__((ext_vector_type(32)));
using Float32x16 = float __attribute__((ext_vector_type(16)));

using Bytes = loomt::buffer<unsigned char>;
using InputTile = loomt::tile<loomt::shape<loomt::dynamic>, std::bfloat16_t>;
using WeightTile =
    loomt::tile<loomt::shape<2, loomt::dynamic>, std::bfloat16_t>;
using OutputTile = loomt::tile<loomt::shape<1>, float>;
using InputChannel = loomt::channel<InputTile>;
using WeightChannel = loomt::channel<WeightTile>;
using OutputChannel = loomt::channel<OutputTile>;
using InputView =
    loomt::view<loomt::shape<loomt::dynamic>, const std::bfloat16_t>;
using WeightView =
    loomt::view<loomt::shape<2, loomt::dynamic>, const std::bfloat16_t>;
using OutputView = loomt::view<loomt::shape<1>, float>;

constexpr loom::aie2p::target npu{.kind = "array"};

[[loom::config("ffn_gate_up.input_size"),
  loom::where(loom::predicate::range(512, 4096) &&
              loom::predicate::multiple_of(32))]]
extern const loomt::index input_size;

static LOOM_INLINE BFloat16x16 bf16_product(BFloat16x16 lhs, BFloat16x16 rhs) {
  BFloat16x16 zeros{};
  Float32x16 initial{};
  auto left = loom::vector::interleave<0, BFloat16x32>(lhs, zeros);
  auto right = loom::vector::interleave<0, BFloat16x32>(rhs, zeros);
  return loom::convert<BFloat16x16>(loom::vector::dot2f(left, right, initial));
}

static LOOM_INLINE void gate_up_tile(InputView input, WeightView weights,
                                     OutputView output) {
  auto* scratch =
      loom::buffer::alloca<float, loom::memory_space::private_, 64>(16);
  auto scratch_view = loom::buffer::view<16>(scratch, {});
  float gate_sum = 0.0f;
  float up_sum = 0.0f;
  for (loomt::index row{0}; row < loomt::index{2};
       row = row + loomt::index{1}) {
    Float32x16 parts{};
    for (loomt::index position{0}; position < input_size;
         position = position + loomt::index{32}) {
      auto input_values =
          loom::vector::load<BFloat16x32>(input, loomt::indices{position});
      auto weight_values = loom::vector::load<BFloat16x32>(
          weights, loomt::indices{row, position});
      parts = loom::vector::dot2f(input_values, weight_values, parts);
    }
    loom::vector::store(parts, scratch_view, loomt::indices{loomt::index{0}});
    float sum = 0.0f;
    for (loomt::index lane{0}; lane < loomt::index{16};
         lane = lane + loomt::index{1}) {
      sum += loom::view::load(scratch_view, lane);
    }
    if (row == loomt::index{0}) {
      gate_sum = sum;
    } else {
      up_sum = sum;
    }
  }

  Float32x16 gate_vector = gate_sum;
  Float32x16 up_vector = up_sum;
  BFloat16x16 gate = loom::convert<BFloat16x16>(gate_vector);
  BFloat16x16 up = loom::convert<BFloat16x16>(up_vector);
  BFloat16x16 quarters = std::bfloat16_t(0.25f);
  Float32x16 halves = 0.5f;
  auto linear = bf16_product(gate, quarters);
  auto scale =
      loom::convert<BFloat16x16>(halves + loom::convert<Float32x16>(linear));
  auto activated = bf16_product(gate, scale);
  auto result = bf16_product(activated, up);
  loom::view::store(static_cast<float>(result[0]), output, loomt::index{0});
}

// --8<-- [start:fill-input]
static LOOM_INLINE void fill_input(loomt::index record, Bytes input,
                                   InputChannel inputs) {
  auto byte_offset = loom::offset::cast(record * input_size * loomt::index{2});
  auto source = loom::buffer::view<loomt::dynamic>(
      loom::buffer::cast<std::bfloat16_t>(input), byte_offset,
      loomt::indices{input_size});
  auto reservation = loom::channel::reserve(inputs);
  auto copy = loom::kernel::async::copy<
      loom::kernel::async::direction::global_to_workgroup,
      loom::cache::scope::workgroup, loom::cache::temporal::regular>(
      source, reservation.payload);
  loom::kernel::async::wait<0>(loom::kernel::async::group(copy));
  loom::channel::publish(reservation.write);
}
// --8<-- [end:fill-input]

// --8<-- [start:drain-output]
static LOOM_INLINE void drain_output(loomt::index record, OutputChannel outputs,
                                     Bytes output) {
  auto byte_offset = loom::offset::cast(record * loomt::index{4});
  auto destination =
      loom::buffer::view<1>(loom::buffer::cast<float>(output), byte_offset, {});
  auto acquisition = loom::channel::acquire(outputs);
  auto copy = loom::kernel::async::copy<
      loom::kernel::async::direction::workgroup_to_global,
      loom::cache::scope::workgroup, loom::cache::temporal::regular>(
      acquisition.payload, destination);
  loom::kernel::async::wait<0>(loom::kernel::async::group(copy));
  loom::channel::release(acquisition.read);
}
// --8<-- [end:drain-output]

[[loom::pipeline("kernel"), loom::target(npu)]]
void ffn_gate_up_quadratic_bf16([[loom::launch]] Bytes input, Bytes weights,
                                Bytes output) {
  auto input_aligned = loom::buffer::assume::alignment<64>(input);
  auto weights_aligned = loom::buffer::assume::alignment<64>(weights);
  auto output_aligned = loom::buffer::assume::alignment<64>(output);
  auto tile_memory = loom::pipeline::memory<loom::memory_space::workgroup>(
      loomt::indices{loomt::index{0}, loomt::index{3}});
  auto* input_storage =
      loom::buffer::alloca<std::bfloat16_t, loom::memory_space::workgroup, 64>(
          loomt::index{2} * input_size, tile_memory);
  auto input_slots = loom::buffer::view<2, loomt::dynamic>(
      input_storage, loomt::indices{input_size});
  auto inputs = loom::channel::bind<loom::channel::discipline::fifo>(
      input_slots, loomt::index{2});

  auto* weight_storage =
      loom::buffer::alloca<std::bfloat16_t, loom::memory_space::workgroup, 64>(
          loomt::index{2} * input_size, tile_memory);
  auto weight_slots = loom::buffer::view<1, 2, loomt::dynamic>(
      weight_storage, loomt::indices{input_size});
  auto resident_weights = loom::channel::bind<loom::channel::discipline::fifo>(
      weight_slots, loomt::index{1});

  auto* output_storage =
      loom::buffer::alloca<float, loom::memory_space::workgroup, 64>(
          loomt::index{2}, tile_memory);
  auto output_slots = loom::buffer::view<2, 1>(output_storage, {});
  auto outputs = loom::channel::bind<loom::channel::discipline::fifo>(
      output_slots, loomt::index{2});

  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{2}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    auto source_weights = loom::buffer::view<2, loomt::dynamic>(
        loom::buffer::cast<std::bfloat16_t>(weights_aligned),
        loomt::indices{input_size});
    auto reservation = loom::channel::reserve(resident_weights);
    auto copy = loom::kernel::async::copy<
        loom::kernel::async::direction::global_to_workgroup,
        loom::cache::scope::workgroup, loom::cache::temporal::regular>(
        source_weights, reservation.payload);
    loom::kernel::async::wait<0>(loom::kernel::async::group(copy));
    loom::channel::publish(reservation.write);

    auto prefetch_count = loom::index::min(loomt::index{2}, loomt::index{1});
    for (loomt::index record{0}; record < prefetch_count;
         record = record + loomt::index{1}) {
      fill_input(record, input_aligned, inputs);
    }
    for (loomt::index record{0}; record < loomt::index{1};
         record = record + loomt::index{1}) {
      drain_output(record, outputs, output_aligned);
      auto next_record = record + prefetch_count;
      if (next_record < loomt::index{1}) {
        fill_input(next_record, input_aligned, inputs);
      }
    }
  }

  [[loom::strand(loomt::indices{loomt::index{0}, loomt::index{3}},
                 loomt::indices{loomt::index{1}, loomt::index{1}},
                 loomt::indices{loomt::index{1}, loomt::index{1}})]] {
    auto weights_record = loom::channel::acquire(resident_weights);
    for (loomt::index record{0}; record < loomt::index{1};
         record = record + loomt::index{1}) {
      auto input_record = loom::channel::acquire(inputs);
      auto output_record = loom::channel::reserve(outputs);
      gate_up_tile(input_record.payload, weights_record.payload,
                   output_record.payload);
      loom::channel::publish(output_record.write);
      loom::channel::release(input_record.read);
    }
    loom::channel::release(weights_record.read);
  }
}
