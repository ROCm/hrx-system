// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_TEST_IMMUTABLE_OBJECT_LINK_TABLES_H_
#define LOOM_IMPORT_CXX_TEST_IMMUTABLE_OBJECT_LINK_TABLES_H_

namespace immutable_link {

inline constexpr unsigned bias[2] = {3, 5};
extern const unsigned weights[2];

inline const unsigned* inline_table(bool first) {
  if (first) {
    static constexpr unsigned table[2] = {7, 11};
    return table;
  } else {
    static constexpr unsigned table[2] = {13, 17};
    return table;
  }
}

template <unsigned Bias>
const unsigned* template_table() {
  static constexpr unsigned table[2] = {Bias, Bias + 1};
  return table;
}

static inline const unsigned* private_table() {
  static constexpr unsigned table[2] = {19, 23};
  return table;
}

template <const unsigned* Values>
const unsigned* address_table() {
  static constexpr unsigned table[2] = {Values[0], Values[1]};
  return table;
}

}  // namespace immutable_link

#endif  // LOOM_IMPORT_CXX_TEST_IMMUTABLE_OBJECT_LINK_TABLES_H_
