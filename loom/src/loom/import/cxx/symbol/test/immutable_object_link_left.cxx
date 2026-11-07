// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/check.h>
#include <loomcxx/view.h>

#include "immutable_object_link_tables.h"

static constexpr unsigned local_table[2] = {1, 2};

unsigned immutable_left_lookup(unsigned index) {
  const unsigned lane = index & 1u;
  return local_table[lane] + immutable_link::bias[lane] +
         immutable_link::weights[lane];
}
unsigned immutable_right_lookup(unsigned index);
const unsigned* immutable_right_inline(bool first);
const unsigned* immutable_right_template(unsigned variant);
const unsigned* immutable_right_private();
unsigned immutable_right_address(unsigned index);

static unsigned immutable_combined_lookup(unsigned index) {
  return immutable_left_lookup(index) + immutable_right_lookup(index);
}

static void immutable_local_values(unsigned index,
                                   loom::type::buffer<unsigned> output) {
  index &= 1u;
  auto values =
      loom::buffer::view<12>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(immutable_link::inline_table(true)[index], values, 0);
  loom::view::store(immutable_link::inline_table(false)[index], values, 1);
  loom::view::store(immutable_right_inline(true)[index], values, 2);
  loom::view::store(immutable_right_inline(false)[index], values, 3);
  loom::view::store(immutable_link::template_table<3>()[index], values, 4);
  loom::view::store(immutable_right_template(3)[index], values, 5);
  loom::view::store(immutable_link::template_table<7>()[index], values, 6);
  loom::view::store(immutable_right_template(7)[index], values, 7);
  loom::view::store(immutable_link::private_table()[index], values, 8);
  loom::view::store(immutable_right_private()[index], values, 9);
  loom::view::store(immutable_link::address_table<local_table>()[index], values,
                    10);
  loom::view::store(immutable_right_address(index), values, 11);
}

static void immutable_local_oracle(unsigned index,
                                   loom::type::buffer<unsigned> output) {
  index &= 1u;
  auto values =
      loom::buffer::view<12>(output, {}, loom::encoding::layout::dense<1>());
  loom::view::store(7u + 4u * index, values, 0);
  loom::view::store(13u + 4u * index, values, 1);
  loom::view::store(7u + 4u * index, values, 2);
  loom::view::store(13u + 4u * index, values, 3);
  loom::view::store(3u + index, values, 4);
  loom::view::store(3u + index, values, 5);
  loom::view::store(7u + index, values, 6);
  loom::view::store(7u + index, values, 7);
  loom::view::store(19u + 4u * index, values, 8);
  loom::view::store(19u + 4u * index, values, 9);
  loom::view::store(1u + index, values, 10);
  loom::view::store(100u + 100u * index, values, 11);
}

static unsigned immutable_local_index(loom::check::ordinal trial) {
  return unsigned(trial);
}
