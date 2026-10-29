// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_FFT_WORKGROUP_H_
#define IREE_EXPERIMENTAL_FFT_WORKGROUP_H_

#include <loomcxx/kernel.h>
#include <loomcxx/scalar.h>

#include "complex_stages.h"
#include "workgroup_geometry.h"

namespace fft_experiment::workgroup {
using fft_experiment::add;
using fft_experiment::butterfly;
using fft_experiment::Complex;
using fft_experiment::minus_i;
using fft_experiment::multiply;
using fft_experiment::scale;
using fft_experiment::sub;

struct Octet {
  // Frequency zero.
  Complex<float> frequency0;
  // Frequency one.
  Complex<float> frequency1;
  // Frequency two.
  Complex<float> frequency2;
  // Frequency three.
  Complex<float> frequency3;
  // Frequency four.
  Complex<float> frequency4;
  // Frequency five.
  Complex<float> frequency5;
  // Frequency six.
  Complex<float> frequency6;
  // Frequency seven.
  Complex<float> frequency7;
};

[[loom::force_inline, loom::fastmath("contract")]] static Octet butterfly8(
    Complex<float> a, Complex<float> b, Complex<float> c, Complex<float> d,
    Complex<float> e, Complex<float> f, Complex<float> g, Complex<float> h) {
  const auto even = butterfly(a, c, e, g);
  const auto odd = butterfly(b, d, f, h);
  const auto first =
      multiply(odd.frequency1,
               Complex<float>{0.7071067811865475f, -0.7071067811865475f});
  const auto second = minus_i(odd.frequency2);
  const auto third =
      multiply(odd.frequency3,
               Complex<float>{-0.7071067811865475f, -0.7071067811865475f});
  return {add(even.frequency0, odd.frequency0), add(even.frequency1, first),
          add(even.frequency2, second),         add(even.frequency3, third),
          sub(even.frequency0, odd.frequency0), sub(even.frequency1, first),
          sub(even.frequency2, second),         sub(even.frequency3, third)};
}

typedef float Float2 __attribute__((ext_vector_type(2)));
[[loom::force_inline]] static Complex<float> load(const float* buffer,
                                                  unsigned position) {
  const Float2 value = reinterpret_cast<const Float2*>(buffer)[position];
  return {value[0], value[1]};
}
[[loom::force_inline]] static void store(float* buffer, unsigned position,
                                         Complex<float> value) {
  reinterpret_cast<Float2*>(buffer)[position] = Float2{value.real, value.imag};
}

[[loom::force_inline]] static unsigned local_position(unsigned position) {
  // One complex padding slot per eight values separates LDS bank accesses.
  return position + position / 8;
}
[[loom::force_inline]] static Complex<float> load_local(const float* buffer,
                                                        unsigned position) {
  return load(buffer, local_position(position));
}
[[loom::force_inline]] static void store_local(float* buffer, unsigned position,
                                               Complex<float> value) {
  store(buffer, local_position(position), value);
}

// Cardinal twiddle zero signs do not affect the FFT's numerical contract.
[[loom::force_inline, loom::fastmath("nsz")]] static Complex<float>
twiddle_turns(float turns) {
  return {loom::scalar::approximate::costurnsf(turns),
          loom::scalar::approximate::sinturnsf(turns)};
}

template <unsigned Previous, class Input>
[[loom::force_inline]] static Complex<float> read_stage(Input input,
                                                        unsigned index) {
  if constexpr (Previous == 1) {
    return load(input, index);
  } else {
    return load_local(input, index);
  }
}

template <unsigned Length, unsigned Previous, unsigned Radix,
          unsigned Component, class Input>
[[loom::force_inline, loom::fastmath("contract")]] static Complex<float>
component(Input input, unsigned group, unsigned position) {
  const unsigned index =
      group * Previous + position + Component * (Length / Radix);
  const auto value = read_stage<Previous>(input, index);
  if constexpr (Previous == 1 || Component == 0) {
    return value;
  } else {
    return multiply(value,
                    twiddle_turns((-1.0f * Component / (Previous * Radix)) *
                                  static_cast<float>(position)));
  }
}

template <unsigned Previous, unsigned Radix, unsigned Component>
[[loom::force_inline]] static void output_component(float* output,
                                                    unsigned group,
                                                    unsigned position,
                                                    Complex<float> value) {
  store_local(output,
              group * Previous * Radix + position + Component * Previous,
              value);
}

template <unsigned Previous, unsigned Radix, class Result>
[[loom::force_inline]] static void output_result(float* output, unsigned group,
                                                 unsigned position,
                                                 Result value) {
  output_component<Previous, Radix, 0>(output, group, position,
                                       value.frequency0);
  output_component<Previous, Radix, 1>(output, group, position,
                                       value.frequency1);
  if constexpr (Radix >= 4) {
    output_component<Previous, Radix, 2>(output, group, position,
                                         value.frequency2);
    output_component<Previous, Radix, 3>(output, group, position,
                                         value.frequency3);
  }
  if constexpr (Radix >= 5) {
    output_component<Previous, Radix, 4>(output, group, position,
                                         value.frequency4);
  }
  if constexpr (Radix == 8) {
    output_component<Previous, Radix, 5>(output, group, position,
                                         value.frequency5);
    output_component<Previous, Radix, 6>(output, group, position,
                                         value.frequency6);
    output_component<Previous, Radix, 7>(output, group, position,
                                         value.frequency7);
  }
}

// Three native sin/cos pairs supply W, W^2, and W^4 to the radix-2 factors.
[[loom::force_inline, loom::fastmath("contract")]] static Octet twiddled8(
    Complex<float> x0, Complex<float> x1, Complex<float> x2, Complex<float> x3,
    Complex<float> x4, Complex<float> x5, Complex<float> x6, Complex<float> x7,
    float turns) {
  const auto w4 = twiddle_turns(4.0f * turns);
  const auto w2 = twiddle_turns(2.0f * turns);
  const auto w1 = twiddle_turns(turns);
  const auto a0 = butterfly(x0, multiply(x4, w4));
  const auto a1 = butterfly(x1, multiply(x5, w4));
  const auto a2 = butterfly(x2, multiply(x6, w4));
  const auto a3 = butterfly(x3, multiply(x7, w4));
  const auto b0 = butterfly(a0.frequency0, multiply(a2.frequency0, w2));
  const auto b1 = butterfly(a1.frequency0, multiply(a3.frequency0, w2));
  const auto b2 =
      butterfly(a0.frequency1, multiply(a2.frequency1, minus_i(w2)));
  const auto b3 =
      butterfly(a1.frequency1, multiply(a3.frequency1, minus_i(w2)));
  const auto w1_rotated =
      multiply(w1, Complex<float>{0.7071067811865475f, -0.7071067811865475f});
  const auto c0 = butterfly(b0.frequency0, multiply(b1.frequency0, w1));
  const auto c2 =
      butterfly(b0.frequency1, multiply(b1.frequency1, minus_i(w1)));
  const auto c1 = butterfly(b2.frequency0, multiply(b3.frequency0, w1_rotated));
  const auto c3 =
      butterfly(b2.frequency1, multiply(b3.frequency1, minus_i(w1_rotated)));
  return {c0.frequency0, c1.frequency0, c2.frequency0, c3.frequency0,
          c0.frequency1, c1.frequency1, c2.frequency1, c3.frequency1};
}

template <unsigned Length, unsigned Previous, unsigned Radix, class Input>
[[loom::force_inline, loom::fastmath("contract")]] static auto compute(
    Input input, unsigned job) {
  const unsigned group = job / Previous;
  const unsigned position = job % Previous;
  if constexpr (Radix == 8 && Previous > 1) {
    const unsigned base = group * Previous + position;
    return twiddled8(
        load_local(input, base), load_local(input, base + Length / 8),
        load_local(input, base + 2 * Length / 8),
        load_local(input, base + 3 * Length / 8),
        load_local(input, base + 4 * Length / 8),
        load_local(input, base + 5 * Length / 8),
        load_local(input, base + 6 * Length / 8),
        load_local(input, base + 7 * Length / 8),
        (-1.0f / (Previous * Radix)) * static_cast<float>(position));
  } else {
    const auto a =
        component<Length, Previous, Radix, 0>(input, group, position);
    const auto b =
        component<Length, Previous, Radix, 1>(input, group, position);
    if constexpr (Radix == 2) {
      return butterfly(a, b);
    } else {
      const auto c =
          component<Length, Previous, Radix, 2>(input, group, position);
      const auto d =
          component<Length, Previous, Radix, 3>(input, group, position);
      if constexpr (Radix == 4) {
        return butterfly(a, b, c, d);
      } else {
        const auto e =
            component<Length, Previous, Radix, 4>(input, group, position);
        if constexpr (Radix == 5) {
          return butterfly(a, b, c, d, e);
        } else {
          const auto f =
              component<Length, Previous, Radix, 5>(input, group, position);
          const auto g =
              component<Length, Previous, Radix, 6>(input, group, position);
          const auto h =
              component<Length, Previous, Radix, 7>(input, group, position);
          return butterfly8(a, b, c, d, e, f, g, h);
        }
      }
    }
  }
}

template <unsigned Length, unsigned Previous, class Input>
[[loom::force_inline, loom::fastmath("contract")]] static float* stages(
    Input input, float* workspace) {
  constexpr unsigned remaining = Length / Previous;
  constexpr unsigned radix = stage_radix(remaining);
  const unsigned lane = loom::kernel::workitem::id.x;
  using Result = decltype(compute<Length, Previous, radix>(input, 0));
  Result values{};
  if (lane < Length / radix) {
    values = compute<Length, Previous, radix>(input, lane);
  }
  if constexpr (Previous > 1) {
    // Every stage reads its operands before overwriting the shared bank.
    loom::kernel::barrier<loom::memory_space::workgroup,
                          loom::atomic::scope::workgroup,
                          loom::atomic::ordering::acq_rel>();
  }
  if (lane < Length / radix) {
    output_result<Previous, radix>(workspace, lane / Previous, lane % Previous,
                                   values);
  }
  loom::kernel::barrier<loom::memory_space::workgroup,
                        loom::atomic::scope::workgroup,
                        loom::atomic::ordering::acq_rel>();
  if constexpr (Previous * radix == Length) {
    return workspace;
  } else {
    return stages<Length, Previous * radix>(workspace, workspace);
  }
}

template <unsigned Length, class Output>
[[loom::force_inline, loom::fastmath("contract")]] static void reconstruct_pair(
    Output output, unsigned index, Complex<float> a, Complex<float> b) {
  b.imag = -b.imag;
  const auto even = scale(add(a, b), 0.5f);
  const auto odd = minus_i(scale(sub(a, b), 0.5f));
  const auto rotated = multiply(
      odd, twiddle_turns((-0.5f / Length) * static_cast<float>(index)));
  store(output, index, add(even, rotated));
  auto mirror = sub(even, rotated);
  mirror.imag = -mirror.imag;
  store(output, Length - index, mirror);
}

// Computes one real frame cooperatively. All x workitems participate, with
// y=z=1 and width at least minimum_workgroup_width<Length>(). The caller owns
// workspace_elements<Length>() workgroup floats and disjoint input/output
// spans. Unqualified load/store calls compose the producer and consumer.
// Scratch remains live until every workitem finishes reconstruction; callers
// synchronize workgroup memory before reusing it in another collective.
template <unsigned Length, class Input, class Output>
[[loom::force_inline, loom::fastmath("contract")]] static void forward(
    Input input, Output output, float* workspace) {
  static_assert(supported_real_length(Length),
                "workgroup real FFT length must be positive, even, and have "
                "only factors two and five");
  constexpr unsigned half = Length / 2;
  if constexpr (half == 1) {
    if (loom::kernel::workitem::id.x == 0) {
      const auto packed = load(input, 0);
      store(output, 0, {packed.real + packed.imag, 0.0f});
      store(output, 1, {packed.real - packed.imag, 0.0f});
    }
  } else {
    const auto spectrum = stages<half, 1>(input, workspace);
    const unsigned workgroup_width = loom::kernel::workgroup::size.x;
    constexpr unsigned pair_limit = (half + 1) / 2;
    const unsigned tile_count =
        (pair_limit - 1 + workgroup_width - 1) / workgroup_width;
    [[loom::unroll]]
    for (unsigned tile = 0; tile < tile_count; ++tile) {
      const unsigned frequency =
          1 + tile * workgroup_width + loom::kernel::workitem::id.x;
      if (frequency < pair_limit) {
        reconstruct_pair<half>(output, frequency,
                               load_local(spectrum, frequency),
                               load_local(spectrum, half - frequency));
      }
    }
    if (loom::kernel::workitem::id.x == 0) {
      const auto packed = load_local(spectrum, 0);
      store(output, 0, {packed.real + packed.imag, 0.0f});
      store(output, half, {packed.real - packed.imag, 0.0f});
      if constexpr (half % 2 == 0) {
        auto middle = load_local(spectrum, half / 2);
        middle.imag = -middle.imag;
        store(output, half / 2, middle);
      }
    }
  }
}

}  // namespace fft_experiment::workgroup

#endif  // IREE_EXPERIMENTAL_FFT_WORKGROUP_H_
