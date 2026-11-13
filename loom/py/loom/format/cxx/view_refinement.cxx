// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/view.h>

namespace t = loom::type;
using Layout = t::encoding<loom::encoding::role::layout, 2>;
using Rows = t::view<t::shape<t::dynamic, 4>, const unsigned, Layout>;
using Fixed = t::view<t::shape<3, 4>, const unsigned, Layout>;

[[loom::inline]] static unsigned read_last(Rows source) {
  auto fixed = loom::view::refine<Fixed>(source);
  return loom::view::load(fixed, 2u, 3u);
}

unsigned retained_layout(unsigned stride, unsigned iterations) {
  const unsigned first[] = {10, 11, 12, 13, 14, 15, 16, 17,
                            18, 19, 20, 21, 22, 23, 24, 25};
  const unsigned second[] = {100, 101, 102, 103, 104, 105, 106, 107,
                             108, 109, 110, 111, 112, 113, 114, 115};
  auto rows = loom::buffer::view<t::dynamic, 4>(
      first, {t::index{3}}, loom::encoding::layout::strided(stride, 1u));
  auto next = loom::buffer::view<t::dynamic, 4>(
      second, {t::index{3}}, loom::encoding::layout::strided(4u, 1u));
  for (unsigned i = 0; i < iterations; ++i) {
    auto previous = rows;
    rows = next;
    next = previous;
  }
  return read_last(rows);
}

unsigned retained_storage(unsigned value) {
  unsigned data[8] = {};
  auto dynamic = loom::buffer::view<t::dynamic, 4>(data, {t::index{2}});
  using Writable = t::view<t::shape<2, 4>, unsigned>;
  auto fixed = loom::view::refine<Writable>(dynamic);
  loom::view::store(value, fixed, 1u, 2u);
  return data[6];
}

unsigned explicit_encoding(unsigned value) {
  const unsigned data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, value};
  auto plain = loom::buffer::view<3, 4>(data, {});
  auto rows = loom::view::refine<Rows>(plain, {t::index{3}},
                                       loom::encoding::layout::dense<2>());
  using Plain = t::view<t::shape<3, 4>, const unsigned>;
  auto restored = loom::view::refine<Plain>(rows);
  return loom::view::load(restored, 2u, 3u);
}

unsigned rank_three(unsigned value) {
  const unsigned data[8] = {0, 1, 2, 3, 4, 5, 6, value};
  auto dynamic = loom::buffer::view<t::dynamic, 2, t::dynamic>(
      data, {t::index{2}, t::index{2}}, loom::encoding::layout::dense<3>());
  using Volume = t::view<t::shape<2, 2, 2>, const unsigned,
                         t::encoding<loom::encoding::role::layout, 3>>;
  auto fixed = loom::view::refine<Volume>(dynamic);
  return loom::view::load(fixed, 1u, 1u, 1u);
}
