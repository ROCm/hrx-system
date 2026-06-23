// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_TOOLING_FILESYSTEM_H_
#define EXPERIMENTAL_ID4_TOOLING_FILESYSTEM_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Ensures that |directory| exists as a filesystem directory.
iree_status_t id4_tooling_ensure_directory(iree_string_view_t directory,
                                           iree_allocator_t host_allocator);

// Formats a child path under |directory| and transfers owned storage to
// |out_path|.
iree_status_t id4_tooling_format_child_path(iree_string_view_t directory,
                                            iree_string_view_t file_name,
                                            iree_allocator_t host_allocator,
                                            iree_string_view_t* out_path);

// Frees an owned path returned by id4_tooling_format_child_path.
void id4_tooling_free_path(iree_string_view_t* path,
                           iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_TOOLING_FILESYSTEM_H_
