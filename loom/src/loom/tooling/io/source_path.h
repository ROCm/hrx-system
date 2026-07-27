// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source path normalization helpers shared by Loom command-line tools.

#ifndef LOOM_TOOLING_IO_SOURCE_PATH_H_
#define LOOM_TOOLING_IO_SOURCE_PATH_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_tooling_source_path_options_t {
  // Raw prefix remapping entries in `<old>=<new>` form. Entries are evaluated
  // in reverse order so the last matching map wins, matching Clang's
  // `-ffile-prefix-map=` behavior. Either side may be empty; an empty old
  // prefix matches every path.
  iree_string_view_list_t prefix_maps;
} loom_tooling_source_path_options_t;

// Initializes source path options with no remapping.
void loom_tooling_source_path_options_initialize(
    loom_tooling_source_path_options_t* out_options);

// Applies |options| to |source_path| and returns the logical source filename.
//
// The returned |out_source_path| either aliases |source_path| or points into
// |*out_source_path_storage|. Callers must keep |*out_source_path_storage| live
// for as long as the remapped path may be referenced, then free it with
// |allocator|. A NULL storage pointer means no allocation was needed.
iree_status_t loom_tooling_source_path_remap(
    iree_string_view_t source_path,
    const loom_tooling_source_path_options_t* options,
    iree_allocator_t allocator, iree_string_view_t* out_source_path,
    char** out_source_path_storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_IO_SOURCE_PATH_H_
