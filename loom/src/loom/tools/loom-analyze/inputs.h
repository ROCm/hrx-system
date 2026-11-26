// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Input path expansion for loom-analyze.
//
// Command-line paths name files or directories. Files are analyzed as given,
// whatever their extension. Directories are searched recursively for `.loom`
// files; `.loom-test` containers and other files are skipped. Exclusion
// patterns use '*' and '?' wildcards and match the whole path with '/'
// separators, so `*/testdata/*` excludes every testdata subtree. The expanded
// list is sorted and de-duplicated so output is independent of directory
// enumeration order.

#ifndef LOOM_TOOLS_LOOM_ANALYZE_INPUTS_H_
#define LOOM_TOOLS_LOOM_ANALYZE_INPUTS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_analyze_input_list_t {
  // NUL-terminated paths owned by the list.
  char** paths;
  // Number of entries in paths.
  iree_host_size_t count;
  // Allocated capacity of paths.
  iree_host_size_t capacity;
  // Allocator owning paths and every entry.
  iree_allocator_t allocator;
} loom_analyze_input_list_t;

void loom_analyze_input_list_initialize(iree_allocator_t allocator,
                                        loom_analyze_input_list_t* out_list);

void loom_analyze_input_list_deinitialize(loom_analyze_input_list_t* list);

// Expands |path| into |list|, skipping paths matching any of |excludes|.
// Fails with NOT_FOUND when |path| does not exist.
iree_status_t loom_analyze_input_list_add_path(
    loom_analyze_input_list_t* list, iree_string_view_t path,
    const iree_string_view_t* excludes, iree_host_size_t exclude_count);

// Adds every regular file under |directory|, whatever its extension.
iree_status_t loom_analyze_input_list_add_tree(loom_analyze_input_list_t* list,
                                               iree_string_view_t directory);

// Sorts paths and removes duplicates.
void loom_analyze_input_list_finalize(loom_analyze_input_list_t* list);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_ANALYZE_INPUTS_H_
