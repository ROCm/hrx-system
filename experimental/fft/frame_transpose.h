// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_FFT_FRAME_TRANSPOSE_H_
#define IREE_EXPERIMENTAL_FFT_FRAME_TRANSPOSE_H_
#include <loomcxx/kernel.h>

#include "frame_io.h"
namespace fft_experiment {
struct FrameOrigin {
  // Runtime caller frame.
  loom::type::index frame;
  // Contiguous position within that frame.
  unsigned position;
};
struct Block {
  // First transposed row.
  Packet first;
  // Second transposed row.
  Packet second;
  // Third transposed row.
  Packet third;
  // Fourth transposed row.
  Packet fourth;
};
LOOM_TEMPLATE_DECL("fft.transpose4")
Block transpose_block(Packet first, Packet second, Packet third, Packet fourth);

struct TwoRows {
  // First row, or first two interleaved positions.
  Packet first;
  // Second row, or last two interleaved positions.
  Packet second;
};
LOOM_TEMPLATE_DECL("fft.interleave2x4")
TwoRows interleave_pair(Packet first, Packet second);
LOOM_TEMPLATE_DECL("fft.deinterleave2x4")
TwoRows deinterleave_pair(Packet first, Packet second);
template <class InputScalar, unsigned Width, unsigned Lane = 0, class Input,
          class Storage>
[[loom::force_inline]] static void transpose_frame_input(
    Input input, loom::type::index frame, unsigned source, Storage destination,
    unsigned position) {
  using InputPacket =
      InputScalar __attribute__((vector_size(4 * sizeof(InputScalar))));
  if constexpr (Width == 2) {
    const auto block = interleave_pair(
        __builtin_convertvector(
            loom::vector::load<InputPacket>(input, FrameOrigin{frame, source}),
            Packet),
        __builtin_convertvector(
            loom::vector::load<InputPacket>(
                input, FrameOrigin{frame + loom::type::index{1}, source}),
            Packet));
    loom::vector::store(block.first, destination.values, {position * Width});
    loom::vector::store(block.second, destination.values,
                        {(position + 2) * Width});
  } else {
    const auto block = transpose_block(
        __builtin_convertvector(
            loom::vector::load<InputPacket>(
                input, FrameOrigin{frame + loom::type::index{Lane}, source}),
            Packet),
        __builtin_convertvector(
            loom::vector::load<InputPacket>(
                input,
                FrameOrigin{frame + loom::type::index{Lane + 1}, source}),
            Packet),
        __builtin_convertvector(
            loom::vector::load<InputPacket>(
                input,
                FrameOrigin{frame + loom::type::index{Lane + 2}, source}),
            Packet),
        __builtin_convertvector(
            loom::vector::load<InputPacket>(
                input,
                FrameOrigin{frame + loom::type::index{Lane + 3}, source}),
            Packet));
    loom::vector::store(block.first, destination.values,
                        {position * Width + Lane});
    loom::vector::store(block.second, destination.values,
                        {(position + 1) * Width + Lane});
    loom::vector::store(block.third, destination.values,
                        {(position + 2) * Width + Lane});
    loom::vector::store(block.fourth, destination.values,
                        {(position + 3) * Width + Lane});
    if constexpr (Lane + 4 < Width) {
      transpose_frame_input<InputScalar, Width, Lane + 4>(
          input, frame, source, destination, position);
    }
  }
}
template <class OutputScalar, unsigned Length, bool Normalize, unsigned Width,
          unsigned Lane = 0, class Output, class Storage>
[[loom::force_inline]] static void transpose_frame_output(
    Storage source, unsigned position, Output output, loom::type::index frame,
    unsigned destination) {
  if constexpr (Width == 2) {
    auto first = loom::vector::load<Packet>(source.values, {position * Width});
    auto second =
        loom::vector::load<Packet>(source.values, {(position + 2) * Width});
    if constexpr (Normalize) {
      first *= 1.0f / Length;
      second *= 1.0f / Length;
    }
    using OutputPacket =
        OutputScalar __attribute__((vector_size(4 * sizeof(OutputScalar))));
    const auto block = deinterleave_pair(first, second);
    loom::vector::store(__builtin_convertvector(block.first, OutputPacket),
                        output, FrameOrigin{frame, destination});
    loom::vector::store(__builtin_convertvector(block.second, OutputPacket),
                        output,
                        FrameOrigin{frame + loom::type::index{1}, destination});
  } else {
    auto first =
        loom::vector::load<Packet>(source.values, {position * Width + Lane});
    auto second = loom::vector::load<Packet>(source.values,
                                             {(position + 1) * Width + Lane});
    auto third = loom::vector::load<Packet>(source.values,
                                            {(position + 2) * Width + Lane});
    auto fourth = loom::vector::load<Packet>(source.values,
                                             {(position + 3) * Width + Lane});
    if constexpr (Normalize) {
      first *= 1.0f / Length;
      second *= 1.0f / Length;
      third *= 1.0f / Length;
      fourth *= 1.0f / Length;
    }
    using OutputPacket =
        OutputScalar __attribute__((vector_size(4 * sizeof(OutputScalar))));
    const auto block = transpose_block(first, second, third, fourth);
    loom::vector::store(
        __builtin_convertvector(block.first, OutputPacket), output,
        FrameOrigin{frame + loom::type::index{Lane}, destination});
    loom::vector::store(
        __builtin_convertvector(block.second, OutputPacket), output,
        FrameOrigin{frame + loom::type::index{Lane + 1}, destination});
    loom::vector::store(
        __builtin_convertvector(block.third, OutputPacket), output,
        FrameOrigin{frame + loom::type::index{Lane + 2}, destination});
    loom::vector::store(
        __builtin_convertvector(block.fourth, OutputPacket), output,
        FrameOrigin{frame + loom::type::index{Lane + 3}, destination});
    if constexpr (Lane + 4 < Width) {
      transpose_frame_output<OutputScalar, Length, Normalize, Width, Lane + 4>(
          source, position, output, frame, destination);
    }
  }
}

}  // namespace fft_experiment
#endif
