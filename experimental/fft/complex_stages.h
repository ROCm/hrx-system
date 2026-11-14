// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_FFT_COMPLEX_STAGES_H_
#define IREE_EXPERIMENTAL_FFT_COMPLEX_STAGES_H_

#include <loomcxx/buffer.h>
#include <loomcxx/vector.h>

namespace fft_experiment {

using Packet = float __attribute__((vector_size(16)));

template <class T>
struct Complex {
  // Real component, or independent real lanes.
  T real;
  // Imaginary component, or matching independent imaginary lanes.
  T imag;
};

template <class T>
[[loom::force_inline, loom::fastmath("contract")]] static Complex<T> add(
    Complex<T> a, Complex<T> b) {
  return {a.real + b.real, a.imag + b.imag};
}
template <class T>
[[loom::force_inline, loom::fastmath("contract")]] static Complex<T> sub(
    Complex<T> a, Complex<T> b) {
  return {a.real - b.real, a.imag - b.imag};
}
template <class T, class U>
[[loom::force_inline, loom::fastmath("contract")]] static Complex<T> multiply(
    Complex<T> a, Complex<U> b) {
  return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}
template <class T>
[[loom::force_inline, loom::fastmath("contract")]] static Complex<T> scale(
    Complex<T> a, float factor) {
  return {a.real * factor, a.imag * factor};
}
template <class T>
[[loom::force_inline, loom::fastmath("contract")]] static Complex<T> minus_i(
    Complex<T> a) {
  return {a.imag, -a.real};
}

template <class T>
struct Pair {
  // Frequency zero.
  Complex<T> frequency0;
  // Frequency one.
  Complex<T> frequency1;
};
template <class T>
struct Quad {
  // Frequency zero.
  Complex<T> frequency0;
  // Frequency one.
  Complex<T> frequency1;
  // Frequency two.
  Complex<T> frequency2;
  // Frequency three.
  Complex<T> frequency3;
};
template <class T>
struct Quint {
  // Frequency zero.
  Complex<T> frequency0;
  // Frequency one.
  Complex<T> frequency1;
  // Frequency two.
  Complex<T> frequency2;
  // Frequency three.
  Complex<T> frequency3;
  // Frequency four.
  Complex<T> frequency4;
};

template <class T>
[[loom::force_inline, loom::fastmath("contract")]] static Pair<T> butterfly(
    Complex<T> first, Complex<T> second) {
  return {add(first, second), sub(first, second)};
}

template <class T>
[[loom::force_inline, loom::fastmath("contract")]] static Quad<T> butterfly(
    Complex<T> first, Complex<T> second, Complex<T> third, Complex<T> fourth) {
  const auto even = add(first, third);
  const auto odd = add(second, fourth);
  const auto even_difference = sub(first, third);
  const auto odd_difference = minus_i(sub(second, fourth));
  return {add(even, odd), add(even_difference, odd_difference), sub(even, odd),
          sub(even_difference, odd_difference)};
}

template <class T>
[[loom::force_inline, loom::fastmath("contract")]] static Quint<T> butterfly(
    Complex<T> first, Complex<T> second, Complex<T> third, Complex<T> fourth,
    Complex<T> fifth) {
  const auto first_sum = add(second, fifth);
  const auto second_sum = add(third, fourth);
  const auto first_difference = sub(second, fifth);
  const auto second_difference = sub(third, fourth);
  const auto first_base =
      add(first, add(scale(first_sum, 0.30901699437494745f),
                     scale(second_sum, -0.80901699437494745f)));
  const auto second_base =
      add(first, add(scale(first_sum, -0.80901699437494745f),
                     scale(second_sum, 0.30901699437494745f)));
  const auto first_rotation =
      minus_i(add(scale(first_difference, 0.95105651629515353f),
                  scale(second_difference, 0.58778525229247314f)));
  const auto second_rotation =
      minus_i(sub(scale(first_difference, 0.58778525229247314f),
                  scale(second_difference, 0.95105651629515353f)));
  return {add(first, add(first_sum, second_sum)),
          add(first_base, first_rotation), add(second_base, second_rotation),
          sub(second_base, second_rotation), sub(first_base, first_rotation)};
}

template <class T, unsigned Width, class Table>
[[loom::force_inline, loom::fastmath("contract")]] static auto load_twiddle(
    Table table, unsigned real_offset, unsigned imag_offset) {
  if constexpr (Width == 1) {
    return Complex<float>{loom::view::load(table, real_offset),
                          loom::view::load(table, imag_offset)};
  } else {
    return Complex<T>{loom::vector::load<T>(table, {real_offset}),
                      loom::vector::load<T>(table, {imag_offset})};
  }
}

// A borrowed scalar plane. Stride is the number of independent frame lanes
// stored at each FFT coordinate; frequency packets use stride one.
template <unsigned Elements, unsigned Stride>
struct Plane {
  // Caller-owned contiguous FP32 storage for this plane.
  loom::type::view<loom::type::shape<Elements>, float> values;
};

template <class T, unsigned Elements, unsigned Stride>
[[loom::force_inline]] static T load_plane(Plane<Elements, Stride> plane,
                                           unsigned position) {
  if constexpr (sizeof(T) == sizeof(float)) {
    return loom::view::load(plane.values, position * Stride);
  } else {
    return loom::vector::load<T>(plane.values, {position * Stride});
  }
}

template <class T, unsigned Elements, unsigned Stride>
[[loom::force_inline]] static void store_plane(Plane<Elements, Stride> plane,
                                               unsigned position, T value) {
  if constexpr (sizeof(T) == sizeof(float)) {
    loom::view::store(value, plane.values, position * Stride);
  } else {
    loom::vector::store(value, plane.values, {position * Stride});
  }
}

template <class Storage>
struct Workspace {
  // First real plane, borrowed for this invocation.
  Storage first_real;
  // First imaginary plane in the same caller-owned buffer.
  Storage first_imag;
  // Second real plane, used as the other stage destination.
  Storage second_real;
  // Second imaginary plane, paired with second_real.
  Storage second_imag;
};

template <unsigned Length, unsigned Width>
[[loom::force_inline]] static auto make_workspace(
    loom::type::buffer<float> workspace_buffer) {
  constexpr unsigned PlaneElements = Length * Width;
  auto storage = loom::buffer::view<4 * PlaneElements>(
      workspace_buffer, {}, loom::encoding::layout::dense<1>());
  using Storage = Plane<PlaneElements, Width>;
  return Workspace<Storage>{
      {loom::view::subview<PlaneElements>(storage, {0}, {})},
      {loom::view::subview<PlaneElements>(storage, {PlaneElements}, {})},
      {loom::view::subview<PlaneElements>(storage, {2 * PlaneElements}, {})},
      {loom::view::subview<PlaneElements>(storage, {3 * PlaneElements}, {})}};
}

template <unsigned Length, unsigned Previous, unsigned Radix,
          unsigned Component, class T, unsigned Width, class Table,
          class Storage>
[[loom::force_inline, loom::fastmath("contract")]] static Complex<T>
load_component(Storage real, Storage imag, Table table, unsigned group,
               unsigned position) {
  const unsigned source =
      group * Previous + position + Component * (Length / Radix);
  const Complex<T> value{load_plane<T>(real, source),
                         load_plane<T>(imag, source)};
  if constexpr (Component != 0 && Previous != 1) {
    const unsigned offset =
        2 * (Previous - 1) + (Component - 1) * 2 * Previous + position;
    return multiply(value,
                    load_twiddle<T, Width>(table, offset, offset + Previous));
  } else {
    return value;
  }
}

template <unsigned Previous, unsigned Radix, unsigned Component, class T,
          class Storage>
[[loom::force_inline, loom::fastmath("contract")]] static void store_component(
    Storage real, Storage imag, unsigned group, unsigned position,
    Complex<T> value) {
  const unsigned destination =
      group * Previous * Radix + position + Component * Previous;
  store_plane(real, destination, value.real);
  store_plane(imag, destination, value.imag);
}

template <unsigned Previous, unsigned Radix, class Result, class Storage>
[[loom::force_inline, loom::fastmath("contract")]] static void store_result(
    Storage real, Storage imag, unsigned group, unsigned position,
    Result result) {
  store_component<Previous, Radix, 0>(real, imag, group, position,
                                      result.frequency0);
  store_component<Previous, Radix, 1>(real, imag, group, position,
                                      result.frequency1);
  if constexpr (Radix >= 4) {
    store_component<Previous, Radix, 2>(real, imag, group, position,
                                        result.frequency2);
    store_component<Previous, Radix, 3>(real, imag, group, position,
                                        result.frequency3);
  }
  if constexpr (Radix == 5) {
    store_component<Previous, Radix, 4>(real, imag, group, position,
                                        result.frequency4);
  }
}

template <unsigned Length, unsigned Previous, unsigned Radix, class T,
          unsigned Width, class Table, class Storage>
[[loom::force_inline, loom::fastmath("contract")]] static void stage_packet(
    Storage input_real, Storage input_imag, Storage output_real,
    Storage output_imag, Table table, unsigned group, unsigned position) {
  const auto first = load_component<Length, Previous, Radix, 0, T, Width>(
      input_real, input_imag, table, group, position);
  const auto second = load_component<Length, Previous, Radix, 1, T, Width>(
      input_real, input_imag, table, group, position);
  if constexpr (Radix == 2) {
    store_result<Previous, Radix>(output_real, output_imag, group, position,
                                  butterfly(first, second));
  } else {
    const auto third = load_component<Length, Previous, Radix, 2, T, Width>(
        input_real, input_imag, table, group, position);
    const auto fourth = load_component<Length, Previous, Radix, 3, T, Width>(
        input_real, input_imag, table, group, position);
    if constexpr (Radix == 4) {
      store_result<Previous, Radix>(output_real, output_imag, group, position,
                                    butterfly(first, second, third, fourth));
    } else {
      static_assert(Radix == 5);
      const auto fifth = load_component<Length, Previous, Radix, 4, T, Width>(
          input_real, input_imag, table, group, position);
      store_result<Previous, Radix>(
          output_real, output_imag, group, position,
          butterfly(first, second, third, fourth, fifth));
    }
  }
}

template <class Storage>
struct Spectrum {
  // Borrowed final real plane in the caller-owned workspace.
  Storage real;
  // Borrowed final imaginary plane in the same scratch bank.
  Storage imag;
};

template <unsigned Length, unsigned Previous, class Table, class Storage>
[[loom::force_inline, loom::fastmath("contract")]] static Spectrum<Storage>
stages(Storage input_real, Storage input_imag, Storage output_real,
       Storage output_imag, Table table) {
  constexpr unsigned Remaining = Length / Previous;
  constexpr unsigned Radix = Remaining % 4 == 0   ? 4
                             : Remaining % 5 == 0 ? 5
                                                  : 2;
  static_assert(Remaining % Radix == 0);
  for (unsigned group = 0; group < Length / (Radix * Previous); ++group) {
    if constexpr (Previous % 4 == 0) {
      for (unsigned position = 0; position < Previous; position += 4) {
        stage_packet<Length, Previous, Radix, Packet, 4>(
            input_real, input_imag, output_real, output_imag, table, group,
            position);
      }
    } else {
      for (unsigned position = 0; position < Previous; ++position) {
        stage_packet<Length, Previous, Radix, float, 1>(
            input_real, input_imag, output_real, output_imag, table, group,
            position);
      }
    }
  }
  if constexpr (Previous * Radix == Length) {
    return {output_real, output_imag};
  } else {
    return stages<Length, Previous * Radix>(output_real, output_imag,
                                            input_real, input_imag, table);
  }
}

}  // namespace fft_experiment
#endif
