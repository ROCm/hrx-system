/*
This file is part of pocketfft.

Copyright (C) 2010-2024 Max-Planck-Society
Copyright (C) 2019-2020 Peter Bell

For the odd-sized DCT-IV transforms:
  Copyright (C) 2003, 2007-14 Matteo Frigo
  Copyright (C) 2003, 2007-14 Massachusetts Institute of Technology

For the prev_good_size search:
  Copyright (C) 2024 Tan Ping Liang, Peter Bell

For the safeguards against integer overflow in good_size search:
  Copyright (C) 2024 Cris Luengo

Authors: Martin Reinecke, Peter Bell

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// SPDX-License-Identifier: BSD-3-Clause
// Bounded adaptation of PocketFFT's licensed real-radix formulas for Loom.
#ifndef IREE_EXPERIMENTAL_FFT_REAL_RADIX_H_
#define IREE_EXPERIMENTAL_FFT_REAL_RADIX_H_
#include "complex_stages.h"
#include "geometry.h"
namespace fft_experiment {
template <unsigned Length, unsigned Radix, unsigned Groups, unsigned Component,
          bool Imaginary, class Table>
[[loom::force_inline]] static float real_twiddle(Table table,
                                                 unsigned frequency) {
  constexpr unsigned Ido = Length / (Radix * Groups);
  constexpr unsigned Offset =
      real_table_offset(Length, Ido) + (Component - 1) * (Ido - 1);
  return loom::view::load(table,
                          Offset + 2 * frequency - 2 + (Imaginary ? 1 : 0));
}
template <unsigned Length, unsigned ido, unsigned l1, class T, class Storage,
          class Table>
[[loom::force_inline, loom::fastmath("contract")]] static void radf2(
    Storage input, Storage output, Table table) {
  for (unsigned k = 0; k < l1; k++) {
    const T sample0 = load_plane<T>(input, (0) + ido * ((k) + l1 * (0)));
    const T sample1 = load_plane<T>(input, (0) + ido * ((k) + l1 * (1)));
    {
      const auto first_result = (sample0) + (sample1);
      const auto second_result = (sample0) - (sample1);
      store_plane(output, (0) + ido * ((0) + 2 * (k)), first_result);
      store_plane(output, (ido - 1) + ido * ((1) + 2 * (k)), second_result);
    };
  }
  if ((ido & 1) == 0) {
    for (unsigned k = 0; k < l1; k++) {
      const T sample0 =
          load_plane<T>(input, (ido - 1) + ido * ((k) + l1 * (1)));
      const T sample1 =
          load_plane<T>(input, (ido - 1) + ido * ((k) + l1 * (0)));

      store_plane(output, (0) + ido * ((1) + 2 * (k)), -sample0);
      store_plane(output, (ido - 1) + ido * ((0) + 2 * (k)), sample1);
    }
  }
  if (ido <= 2) {
    return;
  }
  for (unsigned k = 0; k < l1; k++) {
    for (unsigned i = 2; i < ido; i += 2) {
      unsigned ic = ido - i;
      const T sample0 = load_plane<T>(input, (i - 1) + ido * ((k) + l1 * (1)));
      const T sample1 = load_plane<T>(input, (i) + ido * ((k) + l1 * (1)));
      const T sample2 = load_plane<T>(input, (i - 1) + ido * ((k) + l1 * (0)));
      const T sample3 = load_plane<T>(input, (i) + ido * ((k) + l1 * (0)));

      T tr2{}, ti2{};
      {
        const auto first_result =
            (real_twiddle<Length, 2, l1, 1, false>(table, i / 2)) * (sample0) +
            (real_twiddle<Length, 2, l1, 1, true>(table, i / 2)) * (sample1);
        const auto second_result =
            (real_twiddle<Length, 2, l1, 1, false>(table, i / 2)) * (sample1) -
            (real_twiddle<Length, 2, l1, 1, true>(table, i / 2)) * (sample0);
        tr2 = first_result;
        ti2 = second_result;
      };
      {
        const auto first_result = (sample2) + (tr2);
        const auto second_result = (sample2) - (tr2);
        store_plane(output, (i - 1) + ido * ((0) + 2 * (k)), first_result);
        store_plane(output, (ic - 1) + ido * ((1) + 2 * (k)), second_result);
      };
      {
        const auto first_result = (ti2) + (sample3);
        const auto second_result = (ti2) - (sample3);
        store_plane(output, (i) + ido * ((0) + 2 * (k)), first_result);
        store_plane(output, (ic) + ido * ((1) + 2 * (k)), second_result);
      };
    }
  }
}

template <unsigned Length, unsigned ido, unsigned l1, class T, class Storage,
          class Table>
[[loom::force_inline, loom::fastmath("contract")]] static void radf4(
    Storage input, Storage output, Table table) {
  constexpr float hsqt2 = float(0.707106781186547524400844362104849L);

  for (unsigned k = 0; k < l1; k++) {
    const T sample0 = load_plane<T>(input, (0) + ido * ((k) + l1 * (3)));
    const T sample1 = load_plane<T>(input, (0) + ido * ((k) + l1 * (1)));
    const T sample2 = load_plane<T>(input, (0) + ido * ((k) + l1 * (0)));
    const T sample3 = load_plane<T>(input, (0) + ido * ((k) + l1 * (2)));

    T tr1{}, tr2{};
    {
      const auto first_result = (sample0) + (sample1);
      const auto second_result = (sample0) - (sample1);
      tr1 = first_result;
      store_plane(output, (0) + ido * ((2) + 4 * (k)), second_result);
    };
    {
      const auto first_result = (sample2) + (sample3);
      const auto second_result = (sample2) - (sample3);
      tr2 = first_result;
      store_plane(output, (ido - 1) + ido * ((1) + 4 * (k)), second_result);
    };
    {
      const auto first_result = (tr2) + (tr1);
      const auto second_result = (tr2) - (tr1);
      store_plane(output, (0) + ido * ((0) + 4 * (k)), first_result);
      store_plane(output, (ido - 1) + ido * ((3) + 4 * (k)), second_result);
    };
  }
  if ((ido & 1) == 0) {
    for (unsigned k = 0; k < l1; k++) {
      const T sample0 =
          load_plane<T>(input, (ido - 1) + ido * ((k) + l1 * (1)));
      const T sample1 =
          load_plane<T>(input, (ido - 1) + ido * ((k) + l1 * (3)));
      const T sample2 =
          load_plane<T>(input, (ido - 1) + ido * ((k) + l1 * (0)));
      const T sample3 =
          load_plane<T>(input, (ido - 1) + ido * ((k) + l1 * (2)));

      T ti1 = -hsqt2 * (sample0 + sample1);
      T tr1 = hsqt2 * (sample0 - sample1);
      {
        const auto first_result = (sample2) + (tr1);
        const auto second_result = (sample2) - (tr1);
        store_plane(output, (ido - 1) + ido * ((0) + 4 * (k)), first_result);
        store_plane(output, (ido - 1) + ido * ((2) + 4 * (k)), second_result);
      };
      {
        const auto first_result = (ti1) + (sample3);
        const auto second_result = (ti1) - (sample3);
        store_plane(output, (0) + ido * ((3) + 4 * (k)), first_result);
        store_plane(output, (0) + ido * ((1) + 4 * (k)), second_result);
      };
    }
  }
  if (ido <= 2) {
    return;
  }
  for (unsigned k = 0; k < l1; k++) {
    for (unsigned i = 2; i < ido; i += 2) {
      unsigned ic = ido - i;
      const T sample0 = load_plane<T>(input, (i - 1) + ido * ((k) + l1 * (1)));
      const T sample1 = load_plane<T>(input, (i) + ido * ((k) + l1 * (1)));
      const T sample2 = load_plane<T>(input, (i - 1) + ido * ((k) + l1 * (2)));
      const T sample3 = load_plane<T>(input, (i) + ido * ((k) + l1 * (2)));
      const T sample4 = load_plane<T>(input, (i - 1) + ido * ((k) + l1 * (3)));
      const T sample5 = load_plane<T>(input, (i) + ido * ((k) + l1 * (3)));
      const T sample6 = load_plane<T>(input, (i - 1) + ido * ((k) + l1 * (0)));
      const T sample7 = load_plane<T>(input, (i) + ido * ((k) + l1 * (0)));

      T ci2{}, ci3{}, ci4{}, cr2{}, cr3{}, cr4{}, ti1{}, ti2{}, ti3{}, ti4{},
          tr1{}, tr2{}, tr3{}, tr4{};
      {
        const auto first_result =
            (real_twiddle<Length, 4, l1, 1, false>(table, i / 2)) * (sample0) +
            (real_twiddle<Length, 4, l1, 1, true>(table, i / 2)) * (sample1);
        const auto second_result =
            (real_twiddle<Length, 4, l1, 1, false>(table, i / 2)) * (sample1) -
            (real_twiddle<Length, 4, l1, 1, true>(table, i / 2)) * (sample0);
        cr2 = first_result;
        ci2 = second_result;
      };
      {
        const auto first_result =
            (real_twiddle<Length, 4, l1, 2, false>(table, i / 2)) * (sample2) +
            (real_twiddle<Length, 4, l1, 2, true>(table, i / 2)) * (sample3);
        const auto second_result =
            (real_twiddle<Length, 4, l1, 2, false>(table, i / 2)) * (sample3) -
            (real_twiddle<Length, 4, l1, 2, true>(table, i / 2)) * (sample2);
        cr3 = first_result;
        ci3 = second_result;
      };
      {
        const auto first_result =
            (real_twiddle<Length, 4, l1, 3, false>(table, i / 2)) * (sample4) +
            (real_twiddle<Length, 4, l1, 3, true>(table, i / 2)) * (sample5);
        const auto second_result =
            (real_twiddle<Length, 4, l1, 3, false>(table, i / 2)) * (sample5) -
            (real_twiddle<Length, 4, l1, 3, true>(table, i / 2)) * (sample4);
        cr4 = first_result;
        ci4 = second_result;
      };
      {
        const auto first_result = (cr4) + (cr2);
        const auto second_result = (cr4) - (cr2);
        tr1 = first_result;
        tr4 = second_result;
      };
      {
        const auto first_result = (ci2) + (ci4);
        const auto second_result = (ci2) - (ci4);
        ti1 = first_result;
        ti4 = second_result;
      };
      {
        const auto first_result = (sample6) + (cr3);
        const auto second_result = (sample6) - (cr3);
        tr2 = first_result;
        tr3 = second_result;
      };
      {
        const auto first_result = (sample7) + (ci3);
        const auto second_result = (sample7) - (ci3);
        ti2 = first_result;
        ti3 = second_result;
      };
      {
        const auto first_result = (tr2) + (tr1);
        const auto second_result = (tr2) - (tr1);
        store_plane(output, (i - 1) + ido * ((0) + 4 * (k)), first_result);
        store_plane(output, (ic - 1) + ido * ((3) + 4 * (k)), second_result);
      };
      {
        const auto first_result = (ti1) + (ti2);
        const auto second_result = (ti1) - (ti2);
        store_plane(output, (i) + ido * ((0) + 4 * (k)), first_result);
        store_plane(output, (ic) + ido * ((3) + 4 * (k)), second_result);
      };
      {
        const auto first_result = (tr3) + (ti4);
        const auto second_result = (tr3) - (ti4);
        store_plane(output, (i - 1) + ido * ((2) + 4 * (k)), first_result);
        store_plane(output, (ic - 1) + ido * ((1) + 4 * (k)), second_result);
      };
      {
        const auto first_result = (tr4) + (ti3);
        const auto second_result = (tr4) - (ti3);
        store_plane(output, (i) + ido * ((2) + 4 * (k)), first_result);
        store_plane(output, (ic) + ido * ((1) + 4 * (k)), second_result);
      };
    }
  }
}

template <unsigned Length, unsigned ido, unsigned l1, class T, class Storage,
          class Table>
[[loom::force_inline, loom::fastmath("contract")]] static void radf5(
    Storage input, Storage output, Table table) {
  constexpr float tr11 = float(0.3090169943749474241022934171828191L),
                  ti11 = float(0.9510565162951535721164393333793821L),
                  tr12 = float(-0.8090169943749474241022934171828191L),
                  ti12 = float(0.5877852522924731291687059546390728L);

  for (unsigned k = 0; k < l1; k++) {
    const T sample0 = load_plane<T>(input, (0) + ido * ((k) + l1 * (4)));
    const T sample1 = load_plane<T>(input, (0) + ido * ((k) + l1 * (1)));
    const T sample2 = load_plane<T>(input, (0) + ido * ((k) + l1 * (3)));
    const T sample3 = load_plane<T>(input, (0) + ido * ((k) + l1 * (2)));
    const T sample4 = load_plane<T>(input, (0) + ido * ((k) + l1 * (0)));

    T cr2{}, cr3{}, ci4{}, ci5{};
    {
      const auto first_result = (sample0) + (sample1);
      const auto second_result = (sample0) - (sample1);
      cr2 = first_result;
      ci5 = second_result;
    };
    {
      const auto first_result = (sample2) + (sample3);
      const auto second_result = (sample2) - (sample3);
      cr3 = first_result;
      ci4 = second_result;
    };
    store_plane(output, (0) + ido * ((0) + 5 * (k)), sample4 + cr2 + cr3);
    store_plane(output, (ido - 1) + ido * ((1) + 5 * (k)),
                sample4 + tr11 * cr2 + tr12 * cr3);
    store_plane(output, (0) + ido * ((2) + 5 * (k)), ti11 * ci5 + ti12 * ci4);
    store_plane(output, (ido - 1) + ido * ((3) + 5 * (k)),
                sample4 + tr12 * cr2 + tr11 * cr3);
    store_plane(output, (0) + ido * ((4) + 5 * (k)), ti12 * ci5 - ti11 * ci4);
  }
  if (ido == 1) {
    return;
  }
  for (unsigned k = 0; k < l1; ++k) {
    for (unsigned i = 2, ic = ido - 2; i < ido; i += 2, ic -= 2) {
      const T sample0 = load_plane<T>(input, (i - 1) + ido * ((k) + l1 * (1)));
      const T sample1 = load_plane<T>(input, (i) + ido * ((k) + l1 * (1)));
      const T sample2 = load_plane<T>(input, (i - 1) + ido * ((k) + l1 * (2)));
      const T sample3 = load_plane<T>(input, (i) + ido * ((k) + l1 * (2)));
      const T sample4 = load_plane<T>(input, (i - 1) + ido * ((k) + l1 * (3)));
      const T sample5 = load_plane<T>(input, (i) + ido * ((k) + l1 * (3)));
      const T sample6 = load_plane<T>(input, (i - 1) + ido * ((k) + l1 * (4)));
      const T sample7 = load_plane<T>(input, (i) + ido * ((k) + l1 * (4)));
      const T sample8 = load_plane<T>(input, (i - 1) + ido * ((k) + l1 * (0)));
      const T sample9 = load_plane<T>(input, (i) + ido * ((k) + l1 * (0)));

      T di2{}, di3{}, di4{}, di5{}, dr2{}, dr3{}, dr4{}, dr5{};
      {
        const auto first_result =
            (real_twiddle<Length, 5, l1, 1, false>(table, i / 2)) * (sample0) +
            (real_twiddle<Length, 5, l1, 1, true>(table, i / 2)) * (sample1);
        const auto second_result =
            (real_twiddle<Length, 5, l1, 1, false>(table, i / 2)) * (sample1) -
            (real_twiddle<Length, 5, l1, 1, true>(table, i / 2)) * (sample0);
        dr2 = first_result;
        di2 = second_result;
      };
      {
        const auto first_result =
            (real_twiddle<Length, 5, l1, 2, false>(table, i / 2)) * (sample2) +
            (real_twiddle<Length, 5, l1, 2, true>(table, i / 2)) * (sample3);
        const auto second_result =
            (real_twiddle<Length, 5, l1, 2, false>(table, i / 2)) * (sample3) -
            (real_twiddle<Length, 5, l1, 2, true>(table, i / 2)) * (sample2);
        dr3 = first_result;
        di3 = second_result;
      };
      {
        const auto first_result =
            (real_twiddle<Length, 5, l1, 3, false>(table, i / 2)) * (sample4) +
            (real_twiddle<Length, 5, l1, 3, true>(table, i / 2)) * (sample5);
        const auto second_result =
            (real_twiddle<Length, 5, l1, 3, false>(table, i / 2)) * (sample5) -
            (real_twiddle<Length, 5, l1, 3, true>(table, i / 2)) * (sample4);
        dr4 = first_result;
        di4 = second_result;
      };
      {
        const auto first_result =
            (real_twiddle<Length, 5, l1, 4, false>(table, i / 2)) * (sample6) +
            (real_twiddle<Length, 5, l1, 4, true>(table, i / 2)) * (sample7);
        const auto second_result =
            (real_twiddle<Length, 5, l1, 4, false>(table, i / 2)) * (sample7) -
            (real_twiddle<Length, 5, l1, 4, true>(table, i / 2)) * (sample6);
        dr5 = first_result;
        di5 = second_result;
      };
      {
        const auto t1 = dr2 + dr5, t2 = dr5 - dr2, t3 = di2 + di5,
                   t4 = di2 - di5;
        dr2 = t1;
        di2 = t3;
        dr5 = t4;
        di5 = t2;
      };
      {
        const auto t1 = dr3 + dr4, t2 = dr4 - dr3, t3 = di3 + di4,
                   t4 = di3 - di4;
        dr3 = t1;
        di3 = t3;
        dr4 = t4;
        di4 = t2;
      };
      store_plane(output, (i - 1) + ido * ((0) + 5 * (k)), sample8 + dr2 + dr3);
      store_plane(output, (i) + ido * ((0) + 5 * (k)), sample9 + di2 + di3);
      T tr2 = sample8 + tr11 * dr2 + tr12 * dr3;
      T ti2 = sample9 + tr11 * di2 + tr12 * di3;
      T tr3 = sample8 + tr12 * dr2 + tr11 * dr3;
      T ti3 = sample9 + tr12 * di2 + tr11 * di3;
      T tr5 = ti11 * dr5 + ti12 * dr4;
      T ti5 = ti11 * di5 + ti12 * di4;
      T tr4 = ti12 * dr5 - ti11 * dr4;
      T ti4 = ti12 * di5 - ti11 * di4;
      {
        const auto first_result = (tr2) + (tr5);
        const auto second_result = (tr2) - (tr5);
        store_plane(output, (i - 1) + ido * ((2) + 5 * (k)), first_result);
        store_plane(output, (ic - 1) + ido * ((1) + 5 * (k)), second_result);
      };
      {
        const auto first_result = (ti5) + (ti2);
        const auto second_result = (ti5) - (ti2);
        store_plane(output, (i) + ido * ((2) + 5 * (k)), first_result);
        store_plane(output, (ic) + ido * ((1) + 5 * (k)), second_result);
      };
      {
        const auto first_result = (tr3) + (tr4);
        const auto second_result = (tr3) - (tr4);
        store_plane(output, (i - 1) + ido * ((4) + 5 * (k)), first_result);
        store_plane(output, (ic - 1) + ido * ((3) + 5 * (k)), second_result);
      };
      {
        const auto first_result = (ti4) + (ti3);
        const auto second_result = (ti4) - (ti3);
        store_plane(output, (i) + ido * ((4) + 5 * (k)), first_result);
        store_plane(output, (ic) + ido * ((3) + 5 * (k)), second_result);
      };
    }
  }
}

template <unsigned Length, unsigned ido, unsigned l1, class T, class Storage,
          class Table>
[[loom::force_inline, loom::fastmath("contract")]] static void radb2(
    Storage input, Storage output, Table table) {
  for (unsigned k = 0; k < l1; k++) {
    const T sample0 = load_plane<T>(input, (0) + ido * ((0) + 2 * (k)));
    const T sample1 = load_plane<T>(input, (ido - 1) + ido * ((1) + 2 * (k)));
    {
      const auto first_result = (sample0) + (sample1);
      const auto second_result = (sample0) - (sample1);
      store_plane(output, (0) + ido * ((k) + l1 * (0)), first_result);
      store_plane(output, (0) + ido * ((k) + l1 * (1)), second_result);
    };
  }
  if ((ido & 1) == 0) {
    for (unsigned k = 0; k < l1; k++) {
      const T sample0 = load_plane<T>(input, (ido - 1) + ido * ((0) + 2 * (k)));
      const T sample1 = load_plane<T>(input, (0) + ido * ((1) + 2 * (k)));

      store_plane(output, (ido - 1) + ido * ((k) + l1 * (0)), 2.0f * sample0);
      store_plane(output, (ido - 1) + ido * ((k) + l1 * (1)), -2.0f * sample1);
    }
  }
  if (ido <= 2) {
    return;
  }
  for (unsigned k = 0; k < l1; ++k) {
    for (unsigned i = 2; i < ido; i += 2) {
      unsigned ic = ido - i;
      const T sample0 = load_plane<T>(input, (i - 1) + ido * ((0) + 2 * (k)));
      const T sample1 = load_plane<T>(input, (ic - 1) + ido * ((1) + 2 * (k)));
      const T sample2 = load_plane<T>(input, (i) + ido * ((0) + 2 * (k)));
      const T sample3 = load_plane<T>(input, (ic) + ido * ((1) + 2 * (k)));

      T ti2{}, tr2{};
      {
        const auto first_result = (sample0) + (sample1);
        const auto second_result = (sample0) - (sample1);
        store_plane(output, (i - 1) + ido * ((k) + l1 * (0)), first_result);
        tr2 = second_result;
      };
      {
        const auto first_result = (sample2) + (sample3);
        const auto second_result = (sample2) - (sample3);
        ti2 = first_result;
        store_plane(output, (i) + ido * ((k) + l1 * (0)), second_result);
      };
      {
        const auto first_result =
            (real_twiddle<Length, 2, l1, 1, false>(table, i / 2)) * (ti2) +
            (real_twiddle<Length, 2, l1, 1, true>(table, i / 2)) * (tr2);
        const auto second_result =
            (real_twiddle<Length, 2, l1, 1, false>(table, i / 2)) * (tr2) -
            (real_twiddle<Length, 2, l1, 1, true>(table, i / 2)) * (ti2);
        store_plane(output, (i) + ido * ((k) + l1 * (1)), first_result);
        store_plane(output, (i - 1) + ido * ((k) + l1 * (1)), second_result);
      };
    }
  }
}

template <unsigned Length, unsigned ido, unsigned l1, class T, class Storage,
          class Table>
[[loom::force_inline, loom::fastmath("contract")]] static void radb4(
    Storage input, Storage output, Table table) {
  constexpr float sqrt2 = float(1.414213562373095048801688724209698L);

  for (unsigned k = 0; k < l1; k++) {
    const T sample0 = load_plane<T>(input, (0) + ido * ((0) + 4 * (k)));
    const T sample1 = load_plane<T>(input, (ido - 1) + ido * ((3) + 4 * (k)));
    const T sample2 = load_plane<T>(input, (ido - 1) + ido * ((1) + 4 * (k)));
    const T sample3 = load_plane<T>(input, (0) + ido * ((2) + 4 * (k)));

    T tr1{}, tr2{};
    {
      const auto first_result = (sample0) + (sample1);
      const auto second_result = (sample0) - (sample1);
      tr2 = first_result;
      tr1 = second_result;
    };
    T tr3 = 2.0f * sample2;
    T tr4 = 2.0f * sample3;
    {
      const auto first_result = (tr2) + (tr3);
      const auto second_result = (tr2) - (tr3);
      store_plane(output, (0) + ido * ((k) + l1 * (0)), first_result);
      store_plane(output, (0) + ido * ((k) + l1 * (2)), second_result);
    };
    {
      const auto first_result = (tr1) + (tr4);
      const auto second_result = (tr1) - (tr4);
      store_plane(output, (0) + ido * ((k) + l1 * (3)), first_result);
      store_plane(output, (0) + ido * ((k) + l1 * (1)), second_result);
    };
  }
  if ((ido & 1) == 0) {
    for (unsigned k = 0; k < l1; k++) {
      const T sample0 = load_plane<T>(input, (0) + ido * ((3) + 4 * (k)));
      const T sample1 = load_plane<T>(input, (0) + ido * ((1) + 4 * (k)));
      const T sample2 = load_plane<T>(input, (ido - 1) + ido * ((0) + 4 * (k)));
      const T sample3 = load_plane<T>(input, (ido - 1) + ido * ((2) + 4 * (k)));

      T tr1{}, tr2{}, ti1{}, ti2{};
      {
        const auto first_result = (sample0) + (sample1);
        const auto second_result = (sample0) - (sample1);
        ti1 = first_result;
        ti2 = second_result;
      };
      {
        const auto first_result = (sample2) + (sample3);
        const auto second_result = (sample2) - (sample3);
        tr2 = first_result;
        tr1 = second_result;
      };
      store_plane(output, (ido - 1) + ido * ((k) + l1 * (0)), tr2 + tr2);
      store_plane(output, (ido - 1) + ido * ((k) + l1 * (1)),
                  sqrt2 * (tr1 - ti1));
      store_plane(output, (ido - 1) + ido * ((k) + l1 * (2)), ti2 + ti2);
      store_plane(output, (ido - 1) + ido * ((k) + l1 * (3)),
                  -sqrt2 * (tr1 + ti1));
    }
  }
  if (ido <= 2) {
    return;
  }
  for (unsigned k = 0; k < l1; ++k) {
    for (unsigned i = 2; i < ido; i += 2) {
      T ci2{}, ci3{}, ci4{}, cr2{}, cr3{}, cr4{}, ti1{}, ti2{}, ti3{}, ti4{},
          tr1{}, tr2{}, tr3{}, tr4{};
      unsigned ic = ido - i;
      const T sample0 = load_plane<T>(input, (i - 1) + ido * ((0) + 4 * (k)));
      const T sample1 = load_plane<T>(input, (ic - 1) + ido * ((3) + 4 * (k)));
      const T sample2 = load_plane<T>(input, (i) + ido * ((0) + 4 * (k)));
      const T sample3 = load_plane<T>(input, (ic) + ido * ((3) + 4 * (k)));
      const T sample4 = load_plane<T>(input, (i) + ido * ((2) + 4 * (k)));
      const T sample5 = load_plane<T>(input, (ic) + ido * ((1) + 4 * (k)));
      const T sample6 = load_plane<T>(input, (i - 1) + ido * ((2) + 4 * (k)));
      const T sample7 = load_plane<T>(input, (ic - 1) + ido * ((1) + 4 * (k)));

      {
        const auto first_result = (sample0) + (sample1);
        const auto second_result = (sample0) - (sample1);
        tr2 = first_result;
        tr1 = second_result;
      };
      {
        const auto first_result = (sample2) + (sample3);
        const auto second_result = (sample2) - (sample3);
        ti1 = first_result;
        ti2 = second_result;
      };
      {
        const auto first_result = (sample4) + (sample5);
        const auto second_result = (sample4) - (sample5);
        tr4 = first_result;
        ti3 = second_result;
      };
      {
        const auto first_result = (sample6) + (sample7);
        const auto second_result = (sample6) - (sample7);
        tr3 = first_result;
        ti4 = second_result;
      };
      {
        const auto first_result = (tr2) + (tr3);
        const auto second_result = (tr2) - (tr3);
        store_plane(output, (i - 1) + ido * ((k) + l1 * (0)), first_result);
        cr3 = second_result;
      };
      {
        const auto first_result = (ti2) + (ti3);
        const auto second_result = (ti2) - (ti3);
        store_plane(output, (i) + ido * ((k) + l1 * (0)), first_result);
        ci3 = second_result;
      };
      {
        const auto first_result = (tr1) + (tr4);
        const auto second_result = (tr1) - (tr4);
        cr4 = first_result;
        cr2 = second_result;
      };
      {
        const auto first_result = (ti1) + (ti4);
        const auto second_result = (ti1) - (ti4);
        ci2 = first_result;
        ci4 = second_result;
      };
      {
        const auto first_result =
            (real_twiddle<Length, 4, l1, 1, false>(table, i / 2)) * (ci2) +
            (real_twiddle<Length, 4, l1, 1, true>(table, i / 2)) * (cr2);
        const auto second_result =
            (real_twiddle<Length, 4, l1, 1, false>(table, i / 2)) * (cr2) -
            (real_twiddle<Length, 4, l1, 1, true>(table, i / 2)) * (ci2);
        store_plane(output, (i) + ido * ((k) + l1 * (1)), first_result);
        store_plane(output, (i - 1) + ido * ((k) + l1 * (1)), second_result);
      };
      {
        const auto first_result =
            (real_twiddle<Length, 4, l1, 2, false>(table, i / 2)) * (ci3) +
            (real_twiddle<Length, 4, l1, 2, true>(table, i / 2)) * (cr3);
        const auto second_result =
            (real_twiddle<Length, 4, l1, 2, false>(table, i / 2)) * (cr3) -
            (real_twiddle<Length, 4, l1, 2, true>(table, i / 2)) * (ci3);
        store_plane(output, (i) + ido * ((k) + l1 * (2)), first_result);
        store_plane(output, (i - 1) + ido * ((k) + l1 * (2)), second_result);
      };
      {
        const auto first_result =
            (real_twiddle<Length, 4, l1, 3, false>(table, i / 2)) * (ci4) +
            (real_twiddle<Length, 4, l1, 3, true>(table, i / 2)) * (cr4);
        const auto second_result =
            (real_twiddle<Length, 4, l1, 3, false>(table, i / 2)) * (cr4) -
            (real_twiddle<Length, 4, l1, 3, true>(table, i / 2)) * (ci4);
        store_plane(output, (i) + ido * ((k) + l1 * (3)), first_result);
        store_plane(output, (i - 1) + ido * ((k) + l1 * (3)), second_result);
      };
    }
  }
}

template <unsigned Length, unsigned ido, unsigned l1, class T, class Storage,
          class Table>
[[loom::force_inline, loom::fastmath("contract")]] static void radb5(
    Storage input, Storage output, Table table) {
  constexpr float tr11 = float(0.3090169943749474241022934171828191L),
                  ti11 = float(0.9510565162951535721164393333793821L),
                  tr12 = float(-0.8090169943749474241022934171828191L),
                  ti12 = float(0.5877852522924731291687059546390728L);

  for (unsigned k = 0; k < l1; k++) {
    const T sample0 = load_plane<T>(input, (0) + ido * ((2) + 5 * (k)));
    const T sample1 = load_plane<T>(input, (0) + ido * ((4) + 5 * (k)));
    const T sample2 = load_plane<T>(input, (ido - 1) + ido * ((1) + 5 * (k)));
    const T sample3 = load_plane<T>(input, (ido - 1) + ido * ((3) + 5 * (k)));
    const T sample4 = load_plane<T>(input, (0) + ido * ((0) + 5 * (k)));

    T ti5 = sample0 + sample0;
    T ti4 = sample1 + sample1;
    T tr2 = sample2 + sample2;
    T tr3 = sample3 + sample3;
    store_plane(output, (0) + ido * ((k) + l1 * (0)), sample4 + tr2 + tr3);
    T cr2 = sample4 + tr11 * tr2 + tr12 * tr3;
    T cr3 = sample4 + tr12 * tr2 + tr11 * tr3;
    T ci4{}, ci5{};
    {
      const auto first_result = (ti5) * (ti11) + (ti4) * (ti12);
      const auto second_result = (ti5) * (ti12) - (ti4) * (ti11);
      ci5 = first_result;
      ci4 = second_result;
    };
    {
      const auto first_result = (cr2) + (ci5);
      const auto second_result = (cr2) - (ci5);
      store_plane(output, (0) + ido * ((k) + l1 * (4)), first_result);
      store_plane(output, (0) + ido * ((k) + l1 * (1)), second_result);
    };
    {
      const auto first_result = (cr3) + (ci4);
      const auto second_result = (cr3) - (ci4);
      store_plane(output, (0) + ido * ((k) + l1 * (3)), first_result);
      store_plane(output, (0) + ido * ((k) + l1 * (2)), second_result);
    };
  }
  if (ido == 1) {
    return;
  }
  for (unsigned k = 0; k < l1; ++k) {
    for (unsigned i = 2, ic = ido - 2; i < ido; i += 2, ic -= 2) {
      const T sample0 = load_plane<T>(input, (i - 1) + ido * ((2) + 5 * (k)));
      const T sample1 = load_plane<T>(input, (ic - 1) + ido * ((1) + 5 * (k)));
      const T sample2 = load_plane<T>(input, (i) + ido * ((2) + 5 * (k)));
      const T sample3 = load_plane<T>(input, (ic) + ido * ((1) + 5 * (k)));
      const T sample4 = load_plane<T>(input, (i - 1) + ido * ((4) + 5 * (k)));
      const T sample5 = load_plane<T>(input, (ic - 1) + ido * ((3) + 5 * (k)));
      const T sample6 = load_plane<T>(input, (i) + ido * ((4) + 5 * (k)));
      const T sample7 = load_plane<T>(input, (ic) + ido * ((3) + 5 * (k)));
      const T sample8 = load_plane<T>(input, (i - 1) + ido * ((0) + 5 * (k)));
      const T sample9 = load_plane<T>(input, (i) + ido * ((0) + 5 * (k)));

      T tr2{}, tr3{}, tr4{}, tr5{}, ti2{}, ti3{}, ti4{}, ti5{};
      {
        const auto first_result = (sample0) + (sample1);
        const auto second_result = (sample0) - (sample1);
        tr2 = first_result;
        tr5 = second_result;
      };
      {
        const auto first_result = (sample2) + (sample3);
        const auto second_result = (sample2) - (sample3);
        ti5 = first_result;
        ti2 = second_result;
      };
      {
        const auto first_result = (sample4) + (sample5);
        const auto second_result = (sample4) - (sample5);
        tr3 = first_result;
        tr4 = second_result;
      };
      {
        const auto first_result = (sample6) + (sample7);
        const auto second_result = (sample6) - (sample7);
        ti4 = first_result;
        ti3 = second_result;
      };
      store_plane(output, (i - 1) + ido * ((k) + l1 * (0)),
                  sample8 + tr2 + tr3);
      store_plane(output, (i) + ido * ((k) + l1 * (0)), sample9 + ti2 + ti3);
      T cr2 = sample8 + tr11 * tr2 + tr12 * tr3;
      T ci2 = sample9 + tr11 * ti2 + tr12 * ti3;
      T cr3 = sample8 + tr12 * tr2 + tr11 * tr3;
      T ci3 = sample9 + tr12 * ti2 + tr11 * ti3;
      T ci4{}, ci5{}, cr5{}, cr4{};
      {
        const auto first_result = (tr5) * (ti11) + (tr4) * (ti12);
        const auto second_result = (tr5) * (ti12) - (tr4) * (ti11);
        cr5 = first_result;
        cr4 = second_result;
      };
      {
        const auto first_result = (ti5) * (ti11) + (ti4) * (ti12);
        const auto second_result = (ti5) * (ti12) - (ti4) * (ti11);
        ci5 = first_result;
        ci4 = second_result;
      };
      T dr2{}, dr3{}, dr4{}, dr5{}, di2{}, di3{}, di4{}, di5{};
      {
        const auto first_result = (cr3) + (ci4);
        const auto second_result = (cr3) - (ci4);
        dr4 = first_result;
        dr3 = second_result;
      };
      {
        const auto first_result = (ci3) + (cr4);
        const auto second_result = (ci3) - (cr4);
        di3 = first_result;
        di4 = second_result;
      };
      {
        const auto first_result = (cr2) + (ci5);
        const auto second_result = (cr2) - (ci5);
        dr5 = first_result;
        dr2 = second_result;
      };
      {
        const auto first_result = (ci2) + (cr5);
        const auto second_result = (ci2) - (cr5);
        di2 = first_result;
        di5 = second_result;
      };
      {
        const auto first_result =
            (real_twiddle<Length, 5, l1, 1, false>(table, i / 2)) * (di2) +
            (real_twiddle<Length, 5, l1, 1, true>(table, i / 2)) * (dr2);
        const auto second_result =
            (real_twiddle<Length, 5, l1, 1, false>(table, i / 2)) * (dr2) -
            (real_twiddle<Length, 5, l1, 1, true>(table, i / 2)) * (di2);
        store_plane(output, (i) + ido * ((k) + l1 * (1)), first_result);
        store_plane(output, (i - 1) + ido * ((k) + l1 * (1)), second_result);
      };
      {
        const auto first_result =
            (real_twiddle<Length, 5, l1, 2, false>(table, i / 2)) * (di3) +
            (real_twiddle<Length, 5, l1, 2, true>(table, i / 2)) * (dr3);
        const auto second_result =
            (real_twiddle<Length, 5, l1, 2, false>(table, i / 2)) * (dr3) -
            (real_twiddle<Length, 5, l1, 2, true>(table, i / 2)) * (di3);
        store_plane(output, (i) + ido * ((k) + l1 * (2)), first_result);
        store_plane(output, (i - 1) + ido * ((k) + l1 * (2)), second_result);
      };
      {
        const auto first_result =
            (real_twiddle<Length, 5, l1, 3, false>(table, i / 2)) * (di4) +
            (real_twiddle<Length, 5, l1, 3, true>(table, i / 2)) * (dr4);
        const auto second_result =
            (real_twiddle<Length, 5, l1, 3, false>(table, i / 2)) * (dr4) -
            (real_twiddle<Length, 5, l1, 3, true>(table, i / 2)) * (di4);
        store_plane(output, (i) + ido * ((k) + l1 * (3)), first_result);
        store_plane(output, (i - 1) + ido * ((k) + l1 * (3)), second_result);
      };
      {
        const auto first_result =
            (real_twiddle<Length, 5, l1, 4, false>(table, i / 2)) * (di5) +
            (real_twiddle<Length, 5, l1, 4, true>(table, i / 2)) * (dr5);
        const auto second_result =
            (real_twiddle<Length, 5, l1, 4, false>(table, i / 2)) * (dr5) -
            (real_twiddle<Length, 5, l1, 4, true>(table, i / 2)) * (di5);
        store_plane(output, (i) + ido * ((k) + l1 * (4)), first_result);
        store_plane(output, (i - 1) + ido * ((k) + l1 * (4)), second_result);
      };
    }
  }
}

}  // namespace fft_experiment
#endif
