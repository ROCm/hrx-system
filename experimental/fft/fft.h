// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_FFT_FFT_H_
#define IREE_EXPERIMENTAL_FFT_FFT_H_
#include "frequency_packet.h"
#include "real_packet.h"
namespace fft_experiment {
template <bool Inverse, unsigned Length, unsigned Width, class InputScalar,
          class OutputScalar, class Input, class Output, class PacketTable,
          class ScalarTable>
[[loom::force_inline]] static void packet_tail(
    Input input, Output output, PacketTable packet_table,
    ScalarTable scalar_table, loom::type::buffer<float> workspace_buffer,
    loom::type::index frame, loom::type::index frame_count) {
  if constexpr (Width == 1) {
    if (frame < frame_count) {
      const auto scalar = make_workspace<Length / 2, 1>(workspace_buffer);
      if constexpr (Inverse) {
        inverse_packet<Length, float, 1, OutputScalar>(
            input, output, scalar_table, scalar, frame);
      } else {
        forward_packet<Length, float, 1, SpectrumOutput::Complex, InputScalar,
                       OutputScalar>(input, output, scalar_table, scalar,
                                     frame);
      }
    }
  } else {
    if (frame + loom::type::index{Width} <= frame_count) {
      using Frames = float __attribute__((vector_size(Width * sizeof(float))));
      const auto remainder = real_workspace<Length, Width>(workspace_buffer);
      real_packet<Inverse, Length, Frames, Width, InputScalar, OutputScalar>(
          input, output, packet_table, remainder, frame);
      frame = frame + loom::type::index{Width};
    }
    packet_tail<Inverse, Length, Width / 2, InputScalar, OutputScalar>(
        input, output, packet_table, scalar_table, workspace_buffer, frame,
        frame_count);
  }
}

// Transforms runtime-counted contiguous real frames. Forward output stores
// Length/2+1 complex bins as interleaved real/imaginary scalars; inverse scales
// by 1/Length and ignores the DC/Nyquist imaginary components. All arithmetic
// and scratch use FP32, with conversion at the input and output boundaries.
//
// Input, output, table, and workspace regions are borrowed and disjoint. Tables
// contain table_elements<Length>() floats from tables.transform_table;
// workspace contains workspace_elements<Length, Width>() floats and is
// exclusive to this invocation. Tables may be shared by concurrent invocations;
// scratch may be reused after completion. Zero frames performs no loads or
// stores.
template <bool Inverse, unsigned Length, unsigned Width, class InputScalar,
          class OutputScalar>
  requires(supported_real_length(Length) && Width > 0 &&
           (Width & (Width - 1)) == 0)
[[loom::force_inline]] static void cascade_transform(
    loom::type::buffer<InputScalar> input_buffer,
    loom::type::buffer<OutputScalar> output_buffer,
    loom::type::buffer<float> table_buffer,
    loom::type::buffer<float> workspace_buffer, unsigned frames) {
  static_assert(supported_real_length(Length),
                "real FFT length must be positive, even, and have only factors "
                "two and five");
  static_assert(Width > 0 && (Width & (Width - 1)) == 0,
                "FFT frame packet width must be a power of two");
  static_assert(__is_same(InputScalar, float) ||
                    __is_same(InputScalar, std::float16_t) ||
                    __is_same(InputScalar, std::bfloat16_t),
                "FFT input storage must be f32, f16, or bf16");
  static_assert(__is_same(OutputScalar, float) ||
                    __is_same(OutputScalar, std::float16_t) ||
                    __is_same(OutputScalar, std::bfloat16_t),
                "FFT output storage must be f32, f16, or bf16");
  static_assert(Length <= ~0u / 2 / Width,
                "FFT workspace extent exceeds the index representation");
  const auto frame_count = loom::index::cast(frames);
  const auto packet_count = frame_count / loom::type::index{Width};
  auto input =
      loom::buffer::view<loom::type::dynamic, Length + (Inverse ? 2 : 0)>(
          input_buffer, {frame_count}, loom::encoding::layout::dense<2>());
  auto output =
      loom::buffer::view<loom::type::dynamic, Length + (Inverse ? 0 : 2)>(
          output_buffer, {frame_count}, loom::encoding::layout::dense<2>());
  constexpr unsigned PacketTableElements = real_table_elements(Length);
  auto tables = loom::buffer::view<table_elements<Length>()>(
      table_buffer, {}, loom::encoding::layout::dense<1>());
  auto packet_table = loom::view::subview<PacketTableElements>(tables, {0}, {});
  auto scalar_table =
      loom::view::subview<2 * Length>(tables, {PacketTableElements}, {});
  for (loom::type::index packet{0}; packet < packet_count;
       packet = packet + loom::type::index{1}) {
    if constexpr (Width == 1) {
      packet_tail<Inverse, Length, 1, InputScalar, OutputScalar>(
          input, output, packet_table, scalar_table, workspace_buffer, packet,
          frame_count);
    } else {
      using Frames = float __attribute__((vector_size(Width * sizeof(float))));
      const auto packets = real_workspace<Length, Width>(workspace_buffer);
      real_packet<Inverse, Length, Frames, Width, InputScalar, OutputScalar>(
          input, output, packet_table, packets,
          packet * loom::type::index{Width});
    }
  }
  if constexpr (Width > 1) {
    packet_tail<Inverse, Length, Width / 2, InputScalar, OutputScalar>(
        input, output, packet_table, scalar_table, workspace_buffer,
        packet_count * loom::type::index{Width}, frame_count);
  }
}
}  // namespace fft_experiment
#endif  // IREE_EXPERIMENTAL_FFT_FFT_H_
