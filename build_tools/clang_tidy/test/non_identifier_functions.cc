// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct iree_clang_tidy_non_identifier_functions_t {
  iree_clang_tidy_non_identifier_functions_t() {}

  ~iree_clang_tidy_non_identifier_functions_t() {}

  iree_clang_tidy_non_identifier_functions_t& operator=(
      const iree_clang_tidy_non_identifier_functions_t&) {
    return *this;
  }

  explicit operator bool() const { return true; }
};
