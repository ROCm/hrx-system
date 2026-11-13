// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "immutable_object_link_tables.h"

namespace immutable_link {
extern const unsigned weights[2] = {10, 20};
}  // namespace immutable_link

static constexpr unsigned local_table[2] = {100, 200};

unsigned immutable_right_lookup(unsigned index) {
  const unsigned lane = index & 1u;
  return local_table[lane] + immutable_link::bias[lane];
}

const unsigned* immutable_right_inline(bool first) {
  return immutable_link::inline_table(first);
}

const unsigned* immutable_right_template(unsigned variant) {
  return variant == 3 ? immutable_link::template_table<3>()
                      : immutable_link::template_table<7>();
}

const unsigned* immutable_right_private() {
  return immutable_link::private_table();
}

unsigned immutable_right_address(unsigned index) {
  return immutable_link::address_table<local_table>()[index & 1u];
}
