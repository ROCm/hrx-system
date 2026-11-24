// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_FFT_REAL_PACKET_H_
#define IREE_EXPERIMENTAL_FFT_REAL_PACKET_H_
#include "frame_transpose.h"
#include "real_radix.h"
namespace fft_experiment {
// A real stage stores DC at zero, Nyquist at Length-1, and interior bin k at
// positions 2*k-1 (real) and 2*k (imaginary). Each coordinate contains Width
// independent frame lanes. The returned plane borrows one of the two banks.
template <unsigned Length, unsigned Ido, class T, class Storage, class Table>
[[loom::force_inline]] static Storage real_forward_stages(Storage input,
                                                          Storage output,
                                                          Table table) {
  constexpr unsigned Radix = forward_radix(Length / Ido);
  constexpr unsigned Groups = Length / (Ido * Radix);
  if constexpr (Radix == 2) {
    radf2<Length, Ido, Groups, T>(input, output, table);
  } else if constexpr (Radix == 4) {
    radf4<Length, Ido, Groups, T>(input, output, table);
  } else {
    radf5<Length, Ido, Groups, T>(input, output, table);
  }
  if constexpr (Ido * Radix == Length) {
    return output;
  } else {
    return real_forward_stages<Length, Ido * Radix, T>(output, input, table);
  }
}

template <unsigned Length, unsigned Groups, class T, class Storage, class Table>
[[loom::force_inline]] static Storage real_inverse_stages(Storage input,
                                                          Storage output,
                                                          Table table) {
  constexpr unsigned Radix = inverse_radix(Length / Groups);
  constexpr unsigned Ido = Length / (Groups * Radix);
  if constexpr (Radix == 2) {
    radb2<Length, Ido, Groups, T>(input, output, table);
  } else if constexpr (Radix == 4) {
    radb4<Length, Ido, Groups, T>(input, output, table);
  } else {
    radb5<Length, Ido, Groups, T>(input, output, table);
  }
  if constexpr (Groups * Radix == Length) {
    return output;
  } else {
    return real_inverse_stages<Length, Groups * Radix, T>(output, input, table);
  }
}

template <class Storage>
struct RealWorkspace {
  // First caller-owned stage bank, populated from the input frames.
  Storage first;
  // Second caller-owned stage bank, alternating with first between stages.
  Storage second;
};

template <unsigned Length, unsigned Width>
[[loom::force_inline]] static auto real_workspace(
    loom::type::buffer<float> buffer) {
  constexpr unsigned Elements = Length * Width;
  auto view = loom::buffer::view<2 * Elements>(
      buffer, {}, loom::encoding::layout::dense<1>());
  using Storage = Plane<Elements, Width>;
  return RealWorkspace<Storage>{
      {loom::view::subview<Elements>(view, {0}, {})},
      {loom::view::subview<Elements>(view, {Elements}, {})}};
}

template <bool Inverse, unsigned Length, class T, unsigned Width,
          class InputScalar, class OutputScalar, class Input, class Output,
          class Table, class Storage>
[[loom::force_inline, loom::fastmath("contract")]] static void real_packet(
    Input input, Output output, Table table, RealWorkspace<Storage> workspace,
    loom::type::index frame) {
  if constexpr (Inverse) {
    store_plane(workspace.first, 0, read_frames<T, Width>(input, frame, 0));
    store_plane(workspace.first, Length - 1,
                read_frames<T, Width>(input, frame, Length));
    if constexpr (Width > 1) {
      constexpr unsigned InteriorPrefix = (Length - 2) / 4 * 4;
      for (unsigned position = 0; position < InteriorPrefix; position += 4) {
        transpose_frame_input<InputScalar, Width>(
            input, frame, position + 2, workspace.first, position + 1);
      }
      if constexpr (InteriorPrefix != Length - 2) {
        store_plane(workspace.first, Length - 3,
                    read_frames<T, Width>(input, frame, Length - 2));
        store_plane(workspace.first, Length - 2,
                    read_frames<T, Width>(input, frame, Length - 1));
      }
    } else {
      for (unsigned frequency = 1; frequency < Length / 2; ++frequency) {
        store_plane(workspace.first, 2 * frequency - 1,
                    read_frames<T, Width>(input, frame, 2 * frequency));
        store_plane(workspace.first, 2 * frequency,
                    read_frames<T, Width>(input, frame, 2 * frequency + 1));
      }
    }
    const auto samples = real_inverse_stages<Length, 1, T>(
        workspace.first, workspace.second, table);
    if constexpr (Width > 1) {
      for (unsigned position = 0; position < Length / 4 * 4; position += 4) {
        transpose_frame_output<OutputScalar, Length, true, Width>(
            samples, position, output, frame, position);
      }
      for (unsigned position = Length / 4 * 4; position < Length; ++position) {
        write_frames<OutputScalar, Width>(
            output, frame, position,
            load_plane<T>(samples, position) * (1.0f / Length));
      }
    } else {
      for (unsigned position = 0; position < Length; ++position) {
        write_frames<OutputScalar, Width>(
            output, frame, position,
            load_plane<T>(samples, position) * (1.0f / Length));
      }
    }
  } else {
    if constexpr (Width > 1) {
      for (unsigned position = 0; position < Length / 4 * 4; position += 4) {
        transpose_frame_input<InputScalar, Width>(input, frame, position,
                                                  workspace.first, position);
      }
      for (unsigned position = Length / 4 * 4; position < Length; ++position) {
        store_plane(workspace.first, position,
                    read_frames<T, Width>(input, frame, position));
      }
    } else {
      for (unsigned position = 0; position < Length; ++position) {
        store_plane(workspace.first, position,
                    read_frames<T, Width>(input, frame, position));
      }
    }
    const auto spectrum = real_forward_stages<Length, 1, T>(
        workspace.first, workspace.second, table);
    consume_spectrum<SpectrumOutput::Complex, OutputScalar, Width>(
        output, frame, 0, Complex<T>{load_plane<T>(spectrum, 0), T{}});
    consume_spectrum<SpectrumOutput::Complex, OutputScalar, Width>(
        output, frame, Length / 2,
        Complex<T>{load_plane<T>(spectrum, Length - 1), T{}});
    if constexpr (Width > 1) {
      constexpr unsigned InteriorPrefix = (Length - 2) / 4 * 4;
      for (unsigned position = 0; position < InteriorPrefix; position += 4) {
        transpose_frame_output<OutputScalar, Length, false, Width>(
            spectrum, position + 1, output, frame, position + 2);
      }
      if constexpr (InteriorPrefix != Length - 2) {
        consume_spectrum<SpectrumOutput::Complex, OutputScalar, Width>(
            output, frame, Length / 2 - 1,
            Complex<T>{load_plane<T>(spectrum, Length - 3),
                       load_plane<T>(spectrum, Length - 2)});
      }
    } else {
      for (unsigned frequency = 1; frequency < Length / 2; ++frequency) {
        consume_spectrum<SpectrumOutput::Complex, OutputScalar, Width>(
            output, frame, frequency,
            Complex<T>{load_plane<T>(spectrum, 2 * frequency - 1),
                       load_plane<T>(spectrum, 2 * frequency)});
      }
    }
  }
}

}  // namespace fft_experiment
#endif
