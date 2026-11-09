// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/view.h>

unsigned choose_case(unsigned selector) {
  switch (selector) {
    case 0u:
      return 11u;
    case 2u:
      return 22u;
    default:
      return 33u;
  }
}

unsigned signed_case(long long selector) {
  switch (selector) {
    case (-9223372036854775807ll - 1ll):
      return 1u;
    case -3ll:
      return 2u;
    case 4294967296ll:
      return 3u;
    case 9223372036854775807ll:
      return 4u;
    default:
      return 5u;
  }
}

unsigned default_case(unsigned selector) {
  switch (selector) {
    default:
      return selector + 1u;
  }
}

unsigned rotate(unsigned count, unsigned selector) {
  unsigned first = 3u;
  unsigned second = 5u;
  for (unsigned iteration = 0u; iteration < count; ++iteration) {
    switch (selector) {
      case 0u: {
        const auto previous = first;
        first = second;
        second = previous;
        break;
      }
      case 1u:
        if (iteration == 0u) {
          first += second;
        } else {
          second += first;
        }
        break;
      default:
        first += second;
        break;
    }
  }
  return first * 100u + second;
}

unsigned observe_effect(unsigned* counter) {
  unsigned result = 0u;
  switch ((*counter)++) {
    case 0u:
      result = 17u;
      break;
    case 1u:
      result = 19u;
      break;
    default:
      result = 23u;
      break;
  }
  return result;
}

unsigned defined_return(bool defined) {
  if (defined) {
    return 7u;
  }
  __builtin_unreachable();
}

unsigned wait_or_return(bool ready) {
  if (ready) {
    return 9u;
  } else {
    for (;;) {
    }
  }
}

unsigned read_view(unsigned selector, const unsigned* first,
                   const unsigned* second, unsigned first_width,
                   unsigned second_width) {
  auto left = loom::buffer::view<2, loom::type::dynamic>(
      first, {first_width}, loom::encoding::layout::strided(first_width, 1u));
  auto right = loom::buffer::view<2, loom::type::dynamic>(
      second, {second_width},
      loom::encoding::layout::strided(second_width, 1u));
  auto selected = left;
  switch (selector) {
    case 1u:
      selected = right;
      break;
    default:
      selected = left;
      break;
  }
  return loom::view::load(selected, 1u, 0u);
}

const unsigned* select_pointer(unsigned selector, const unsigned* first,
                               const unsigned* second) {
  const unsigned* selected = first;
  switch (selector) {
    case 1u:
      selected = second + 1;
      break;
    default:
      selected = first + 2;
      break;
  }
  return selected;
}

unsigned observe_selector_effects(unsigned observation) {
  unsigned counter = 0u;
  const auto first = observe_effect(&counter);
  const auto second = observe_effect(&counter);
  const auto third = observe_effect(&counter);
  switch (observation) {
    case 0u:
      return first;
    case 1u:
      return second;
    case 2u:
      return third;
    default:
      return counter;
  }
}

unsigned observe_views(unsigned observation) {
  const unsigned first[] = {10u, 20u, 30u, 40u};
  const unsigned second[] = {100u, 200u, 300u, 400u, 500u, 600u};
  switch (observation) {
    case 0u:
      return read_view(0u, first, second, 2u, 3u);
    case 1u:
      return read_view(1u, first, second, 2u, 3u);
    case 2u:
      return *select_pointer(0u, first, second);
    default:
      return *select_pointer(1u, first, second);
  }
}
