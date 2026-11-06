// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_TEST_CHECK_HEADER_CASES_H_
#define LOOM_IMPORT_CXX_BINDING_TEST_CHECK_HEADER_CASES_H_

#include <loomcxx/check.h>

#ifndef CHECK_HEADER_EXPECTED
#define CHECK_HEADER_EXPECTED 5u
#endif

unsigned next_value(unsigned value) { return value + 1u; }

LOOM_CHECK_CASE(header_assertion) {
  const auto actual = next_value(4u);
  loom::check::expect_equal(actual, CHECK_HEADER_EXPECTED);
}

[[loom::op("scalar.sinf")]] float exact_sine_intrinsic(float);
float exact_sine(float value) { return exact_sine_intrinsic(value); }

#endif  // LOOM_IMPORT_CXX_BINDING_TEST_CHECK_HEADER_CASES_H_
