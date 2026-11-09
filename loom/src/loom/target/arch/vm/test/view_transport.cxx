// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Subjects for view_transport_checks.cxx. The same assertions execute these
// functions before and after public C++ export and reimport.
#include <loomcxx/view.h>

template <class View>
[[loom::inline]] View select(View first, View second, bool condition) {
  View selected;
  if (condition) {
    selected = first;
  } else {
    selected = second;
  }
  return selected;
}

template <class View>
[[loom::inline]] View counted(View first, View second, bool condition,
                              unsigned count) {
  View selected = first;
  for (unsigned i = 0; i < count; ++i) {
    if (condition) {
      selected = first;
    } else {
      selected = second;
    }
  }
  return selected;
}

template <class View>
[[loom::inline]] View exchanged(View first, View second, bool condition) {
  View left = first;
  View right = second;
  unsigned i = 0;
  while (i < 3) {
    View saved = left;
    left = right;
    right = saved;
    ++i;
  }
  if (condition) {
    return right;
  }
  return left;
}

template <class View>
[[loom::inline]] View constructed(View first, View second, bool condition) {
  View selected;
  unsigned i = 0;
  do {
    if (condition) {
      selected = first;
    } else {
      selected = second;
    }
    ++i;
  } while (i < 3);
  return selected;
}

// The second-row loads distinguish each view's payload and column count.
// Reusing the other view's width reads 13 or 35 instead of 12 or 34.
static auto dense_first() {
  return loom::buffer::view<2, loom::type::dynamic>(
      loom::buffer::constant("\x0b\x0c\x0c\x0d"), {2u});
}

static auto dense_second() {
  return loom::buffer::view<2, loom::type::dynamic>(
      loom::buffer::constant("\x21\x22\x23\x22\x24\x25"), {3u});
}

// Distinct row strides make a stale layout read padding (99) or a neighbor.
static auto strided_first() {
  return loom::buffer::view<2, loom::type::dynamic>(
      loom::buffer::constant("\x0b\x0c\x63\x0c\x0d\x63"), {2u},
      loom::encoding::layout::strided(3u, 1u));
}

static auto strided_second() {
  return loom::buffer::view<2, loom::type::dynamic>(
      loom::buffer::constant("\x21\x22\x23\x63\x22\x24\x25\x63"), {3u},
      loom::encoding::layout::strided(4u, 1u));
}

unsigned dense_branch(bool condition) {
  return loom::view::load(select(dense_first(), dense_second(), condition), 1u,
                          0u);
}

unsigned dense_counted(bool condition, unsigned count) {
  return loom::view::load(
      counted(dense_first(), dense_second(), condition, count), 1u, 0u);
}

unsigned dense_swap(bool condition) {
  return loom::view::load(exchanged(dense_first(), dense_second(), condition),
                          1u, 0u);
}

unsigned dense_do_while(bool condition) {
  return loom::view::load(constructed(dense_first(), dense_second(), condition),
                          1u, 0u);
}

unsigned strided_branch(bool condition) {
  return loom::view::load(select(strided_first(), strided_second(), condition),
                          1u, 0u);
}

unsigned strided_counted(bool condition, unsigned count) {
  return loom::view::load(
      counted(strided_first(), strided_second(), condition, count), 1u, 0u);
}

unsigned strided_swap(bool condition) {
  return loom::view::load(
      exchanged(strided_first(), strided_second(), condition), 1u, 0u);
}

unsigned strided_do_while(bool condition) {
  return loom::view::load(
      constructed(strided_first(), strided_second(), condition), 1u, 0u);
}

using Index = loom::type::index;

// A dimension has independent scalar uses as well as constructing a view.
unsigned independent_dimension(bool condition, bool observe_length) {
  Index length{3u};
  if (condition) {
    length = Index{2u};
  }
  auto selected = loom::buffer::view<2, loom::type::dynamic>(
      loom::buffer::constant("\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14"),
      {length});
  for (unsigned i = 0; i < 2; ++i) {
    length = length + Index{1u};
    selected = loom::buffer::view<2, loom::type::dynamic>(
        loom::buffer::constant("\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14"),
        {length});
  }
  if (observe_length) {
    return loom::index::cast<unsigned>(length);
  }
  return loom::view::load(selected, 1u, 0u);
}

using View = loom::type::view<loom::type::shape<loom::type::dynamic>,
                              const unsigned char>;

[[loom::noinline]] static View observed(View input,
                                        loom::type::buffer<unsigned> output,
                                        unsigned iteration) {
  auto writes = loom::buffer::view<3>(output, {});
  loom::view::store(iteration + 1u, writes, iteration);
  return input;
}

// The result carrier is unobserved; the calls producing it still write output.
unsigned unused_carrier(unsigned position) {
  unsigned output[3]{};
  auto input = loom::buffer::view<loom::type::dynamic>(
      loom::buffer::constant("\x0b\x0c"), {2u});
  View unused = input;
  for (unsigned i = 0; i < 3; ++i) {
    unused = observed(input, loom::buffer::root(output), i);
  }
  return output[position];
}
