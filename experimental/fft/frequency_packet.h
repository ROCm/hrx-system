// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_FFT_FREQUENCY_PACKET_H_
#define IREE_EXPERIMENTAL_FFT_FREQUENCY_PACKET_H_

#include <stdfloat>

#include "deinterleave.h"
#include "frame_io.h"
#include "geometry.h"

namespace fft_experiment {

template <unsigned Length, class T, unsigned Width, class Storage, class Table>
[[loom::force_inline]] static Spectrum<Storage> transform_packet(
    Workspace<Storage> workspace, Table table) {
  static_assert(Width == 1);
  if constexpr (Length == 1) {
    return {workspace.first_real, workspace.first_imag};
  } else {
    return stages<Length, 1>(workspace.first_real, workspace.first_imag,
                             workspace.second_real, workspace.second_imag,
                             table);
  }
}

template <unsigned Length, class T, unsigned Width, SpectrumOutput Mode,
          class InputScalar, class OutputScalar, class Input, class Output,
          class Table, class Storage>
[[loom::force_inline, loom::fastmath("contract")]] static void forward_packet(
    Input input, Output output, Table table, Workspace<Storage> workspace,
    loom::type::index frame) {
  constexpr unsigned Half = Length / 2;
  static_assert(Width == 1);
  using InputPacket =
      InputScalar __attribute__((vector_size(4 * sizeof(InputScalar))));
  for (unsigned position = 0; position < Half / 4 * 4; position += 4) {
    const auto values =
        deinterleave(__builtin_convertvector(
                         loom::vector::load<InputPacket>(
                             input, ScalarFrameOrigin{frame, 2 * position}),
                         Packet),
                     __builtin_convertvector(
                         loom::vector::load<InputPacket>(
                             input, ScalarFrameOrigin{frame, 2 * position + 4}),
                         Packet));
    store_plane(workspace.first_real, position, values.real);
    store_plane(workspace.first_imag, position, values.imag);
  }
  for (unsigned position = Half / 4 * 4; position < Half; ++position) {
    store_plane(workspace.first_real, position,
                read_frames<T, Width>(input, frame, 2 * position));
    store_plane(workspace.first_imag, position,
                read_frames<T, Width>(input, frame, 2 * position + 1));
  }
  const auto spectrum = transform_packet<Half, T, Width>(workspace, table);
  const T first_real = load_plane<T>(spectrum.real, 0);
  const T first_imag = load_plane<T>(spectrum.imag, 0);
  consume_spectrum<Mode, OutputScalar, Width>(
      output, frame, 0, Complex<T>{first_real + first_imag, T{}});
  consume_spectrum<Mode, OutputScalar, Width>(
      output, frame, Half, Complex<T>{first_real - first_imag, T{}});
  for (unsigned frequency = 1; frequency < Half; ++frequency) {
    const Complex<T> first{load_plane<T>(spectrum.real, frequency),
                           load_plane<T>(spectrum.imag, frequency)};
    const Complex<T> second{load_plane<T>(spectrum.real, Half - frequency),
                            -load_plane<T>(spectrum.imag, Half - frequency)};
    const auto even = scale(add(first, second), 0.5f);
    const auto odd = scale(minus_i(sub(first, second)), 0.5f);
    const Complex<float> twiddle{
        loom::view::load(table, 2 * (Half - 1) + frequency),
        loom::view::load(table, 2 * (Half - 1) + Half + 1 + frequency)};
    consume_spectrum<Mode, OutputScalar, Width>(
        output, frame, frequency, add(even, multiply(odd, twiddle)));
  }
}

template <unsigned Length, class T, unsigned Width, class OutputScalar,
          class Input, class Output, class Table, class Storage>
[[loom::force_inline, loom::fastmath("contract")]] static void inverse_packet(
    Input input, Output output, Table table, Workspace<Storage> workspace,
    loom::type::index frame) {
  constexpr unsigned Half = Length / 2;
  const T dc = read_frames<T, Width>(input, frame, 0);
  const T nyquist = read_frames<T, Width>(input, frame, Length);
  store_plane(workspace.first_real, 0, (dc + nyquist) * 0.5f);
  store_plane(workspace.first_imag, 0, (nyquist - dc) * 0.5f);
  for (unsigned frequency = 1; frequency < (Half + 1) / 2; ++frequency) {
    const Complex<T> first{
        read_frames<T, Width>(input, frame, 2 * frequency),
        read_frames<T, Width>(input, frame, 2 * frequency + 1)};
    const Complex<T> second{
        read_frames<T, Width>(input, frame, 2 * (Half - frequency)),
        -read_frames<T, Width>(input, frame, 2 * (Half - frequency) + 1)};
    const auto even = scale(add(first, second), 0.5f);
    const Complex<float> inverse_twiddle{
        loom::view::load(table, 2 * (Half - 1) + frequency),
        -loom::view::load(table, 2 * (Half - 1) + Half + 1 + frequency)};
    const auto odd = multiply(scale(sub(first, second), 0.5f), inverse_twiddle);
    // Conjugate the packed spectrum so the shared forward stages implement
    // IFFT.
    store_plane(workspace.first_real, frequency, even.real - odd.imag);
    store_plane(workspace.first_imag, frequency, -even.imag - odd.real);
    store_plane(workspace.first_real, Half - frequency, even.real + odd.imag);
    store_plane(workspace.first_imag, Half - frequency, even.imag - odd.real);
  }
  if constexpr (Half % 2 == 0) {
    // The conjugated packed self-pair equals this input bin directly.
    store_plane(workspace.first_real, Half / 2,
                read_frames<T, Width>(input, frame, Half));
    store_plane(workspace.first_imag, Half / 2,
                read_frames<T, Width>(input, frame, Half + 1));
  }
  const auto samples = transform_packet<Half, T, Width>(workspace, table);
  for (unsigned position = 0; position < Half; ++position) {
    write_frames<OutputScalar, Width>(
        output, frame, 2 * position,
        load_plane<T>(samples.real, position) * (1.0f / Half));
    write_frames<OutputScalar, Width>(
        output, frame, 2 * position + 1,
        load_plane<T>(samples.imag, position) * (-1.0f / Half));
  }
}

template <unsigned Length, class T, unsigned Width, SpectrumOutput Mode,
          class InputScalar, class OutputScalar>
  requires(supported_real_length(Length) && Width == 1 && __is_same(T, float))
[[loom::force_inline]] static void forward(
    loom::type::buffer<InputScalar> input_buffer,
    loom::type::buffer<OutputScalar> output_buffer,
    loom::type::buffer<float> table_buffer,
    loom::type::buffer<float> workspace_buffer, unsigned frames) {
  constexpr unsigned Half = Length / 2;
  constexpr unsigned Outputs =
      Mode == SpectrumOutput::Power ? Half + 1 : Length + 2;
  const auto frame_count = loom::index::cast(frames);
  const auto packet_count = frame_count / loom::type::index{Width};
  auto input = loom::buffer::view<loom::type::dynamic, Length>(
      input_buffer, {frame_count}, loom::encoding::layout::dense<2>());
  auto output = loom::buffer::view<loom::type::dynamic, Outputs>(
      output_buffer, {frame_count}, loom::encoding::layout::dense<2>());
  auto table = loom::buffer::view<2 * Length>(
      table_buffer, {}, loom::encoding::layout::dense<1>());
  const auto packets = make_workspace<Half, Width>(workspace_buffer);
  for (loom::type::index packet{0}; packet < packet_count;
       packet = packet + loom::type::index{1}) {
    forward_packet<Length, T, Width, Mode, InputScalar, OutputScalar>(
        input, output, table, packets, packet * loom::type::index{Width});
  }
  const auto scalar = make_workspace<Half, 1>(workspace_buffer);
  for (auto frame = packet_count * loom::type::index{Width};
       frame < frame_count; frame = frame + loom::type::index{1}) {
    forward_packet<Length, float, 1, Mode, InputScalar, OutputScalar>(
        input, output, table, scalar, frame);
  }
}

template <unsigned Length, class T, unsigned Width, class InputScalar,
          class OutputScalar>
  requires(supported_real_length(Length) && Width == 1 && __is_same(T, float))
[[loom::force_inline]] static void inverse(
    loom::type::buffer<InputScalar> input_buffer,
    loom::type::buffer<OutputScalar> output_buffer,
    loom::type::buffer<float> table_buffer,
    loom::type::buffer<float> workspace_buffer, unsigned frames) {
  constexpr unsigned Half = Length / 2;
  const auto frame_count = loom::index::cast(frames);
  const auto packet_count = frame_count / loom::type::index{Width};
  auto input = loom::buffer::view<loom::type::dynamic, Length + 2>(
      input_buffer, {frame_count}, loom::encoding::layout::dense<2>());
  auto output = loom::buffer::view<loom::type::dynamic, Length>(
      output_buffer, {frame_count}, loom::encoding::layout::dense<2>());
  auto table = loom::buffer::view<2 * Length>(
      table_buffer, {}, loom::encoding::layout::dense<1>());
  const auto packets = make_workspace<Half, Width>(workspace_buffer);
  for (loom::type::index packet{0}; packet < packet_count;
       packet = packet + loom::type::index{1}) {
    inverse_packet<Length, T, Width, OutputScalar>(
        input, output, table, packets, packet * loom::type::index{Width});
  }
  const auto scalar = make_workspace<Half, 1>(workspace_buffer);
  for (auto frame = packet_count * loom::type::index{Width};
       frame < frame_count; frame = frame + loom::type::index{1}) {
    inverse_packet<Length, float, 1, OutputScalar>(input, output, table, scalar,
                                                   frame);
  }
}

}  // namespace fft_experiment
#endif
