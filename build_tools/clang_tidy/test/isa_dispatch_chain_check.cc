// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

namespace iree_clang_tidy_dispatch {

struct First {};
struct Second {};
struct Third {};
struct Fourth {};

template <typename T>
bool isa(const void* value) {
  return value;
}

bool isa(const void* value) { return value; }

template <typename T>
bool cast(const void* value) {
  return value;
}

}  // namespace iree_clang_tidy_dispatch

void iree_clang_tidy_dispatch_cpp_chain(const void* value) {
  using namespace iree_clang_tidy_dispatch;
  if (isa<First>(value)) {
  } else if (isa<Second>(value) || isa<Third>(value)) {
  } else if (isa<Fourth>(value)) {
  }
}

void iree_clang_tidy_dispatch_cpp_qualified(const void* value) {
  if (iree_clang_tidy_dispatch::isa<iree_clang_tidy_dispatch::First>(value) ||
      iree_clang_tidy_dispatch::isa<iree_clang_tidy_dispatch::Second>(value) ||
      iree_clang_tidy_dispatch::isa<iree_clang_tidy_dispatch::Third>(value) ||
      iree_clang_tidy_dispatch::isa<iree_clang_tidy_dispatch::Fourth>(value)) {
  }
}

void iree_clang_tidy_dispatch_cpp_non_template(const void* value) {
  using namespace iree_clang_tidy_dispatch;
  if (isa(value) || isa(value) || isa(value) || isa(value)) {
  }
  if (cast<First>(value) || cast<Second>(value) || cast<Third>(value) ||
      cast<Fourth>(value)) {
  }
}
