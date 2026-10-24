// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_tools/bazel/test/cc_shared_fixture.h"

int main(void) {
  int value =
      shared_middle_value() + shared_component_value() + shared_header_value();
  return value == 84 ? 0 : 1;
}
