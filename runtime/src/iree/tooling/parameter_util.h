// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PARAMETER_UTIL_H_
#define IREE_TOOLING_PARAMETER_UTIL_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_io_parameter_index_t iree_io_parameter_index_t;
typedef struct iree_io_scope_map_t iree_io_scope_map_t;

// Appends parameters from |path| to |index|.
//
// Supported paths are ordinary parameter files and Hugging Face safetensors
// index manifests named `*.safetensors.index.json`. Manifest paths are expanded
// relative to their containing directory and each referenced shard file is
// parsed into the same index.
iree_status_t iree_tooling_append_parameter_file_to_index(
    iree_string_view_t path, iree_io_parameter_index_t* index,
    iree_allocator_t host_allocator);

// Populates |scope_map| with parameter indices as specified by flags.
iree_status_t iree_tooling_build_parameter_indices_from_flags(
    iree_io_scope_map_t* scope_map);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PARAMETER_UTIL_H_
