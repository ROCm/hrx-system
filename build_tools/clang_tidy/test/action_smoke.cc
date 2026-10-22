// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_tools/clang_tidy/test/action_smoke.h"

#include <vector>

// Parsing the destination standard library requires the toolchain's SDK inputs.
bool iree_clang_tidy_action_smoke_cpp(const std::vector<int>& values) {
  return values.empty();
}
