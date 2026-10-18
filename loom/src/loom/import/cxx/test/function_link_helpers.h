// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_TEST_FUNCTION_LINK_HELPERS_H_
#define LOOM_IMPORT_CXX_TEST_FUNCTION_LINK_HELPERS_H_

inline unsigned shared_inline(unsigned value) { return value * 3u + 1u; }

template <unsigned Bias>
constexpr unsigned shared_template(unsigned value) {
  return value * 2u + Bias;
}

#endif  // LOOM_IMPORT_CXX_TEST_FUNCTION_LINK_HELPERS_H_
