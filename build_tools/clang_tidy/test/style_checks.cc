// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct iree_clang_tidy_style_cpp_config_t {
  int ordinal;
  const char* name;
};

struct iree_clang_tidy_style_cpp_sparse_config_t {
  int traits;
  int effective_traits;
  const char* name;
};

#define IREE_CLANG_TIDY_STYLE_CPP_ACCEPT_CONFIG(config) (void)sizeof(config)

iree_clang_tidy_style_cpp_config_t iree_clang_tidy_style_cpp_designated_init = {
    .ordinal = 1,
    .name = "device",
};

iree_clang_tidy_style_cpp_config_t
    iree_clang_tidy_style_cpp_skipped_designated_init = {
        .name = "skipped",
};

iree_clang_tidy_style_cpp_sparse_config_t
    iree_clang_tidy_style_cpp_sparse_designated_init = {
        .traits = 1,
        .name = "sparse",
};

iree_clang_tidy_style_cpp_config_t iree_clang_tidy_style_cpp_comment_labels = {
    /*.ordinal=*/2,
    /*.name=*/"host",
};

void iree_clang_tidy_style_cpp_macro_argument() {
  IREE_CLANG_TIDY_STYLE_CPP_ACCEPT_CONFIG((iree_clang_tidy_style_cpp_config_t{
      .ordinal = 3,
      .name = "macro-arg",
  }));
}
