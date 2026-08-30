// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical loom-compile command artifact manifest serialization.

#ifndef LOOM_TOOLS_LOOM_COMPILE_COMMAND_MANIFEST_H_
#define LOOM_TOOLS_LOOM_COMPILE_COMMAND_MANIFEST_H_

#include "iree/base/api.h"
#include "loom/target/arch/cmd/artifact_set.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stable format name for the loom-compile command artifact manifest.
#define LOOM_COMPILE_COMMAND_MANIFEST_FORMAT "loom-command-set"

// Formats the canonical filename for |program_ordinal| into |buffer|.
iree_status_t loom_compile_command_manifest_format_program_filename(
    iree_host_size_t program_ordinal, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_filename);

// Formats the canonical filename for |entry_ordinal| into |buffer|.
iree_status_t loom_compile_command_manifest_format_kernel_request_filename(
    iree_host_size_t entry_ordinal, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_filename);

// Writes one complete command artifact-set manifest to |stream|.
//
// |source_requirement_indices| is the sorted subset of entry ordinals whose
// independently compilable source requests were emitted with the artifact set.
iree_status_t loom_compile_command_manifest_write(
    const loom_cmd_program_artifact_set_t* artifact_set,
    const uint32_t* source_requirement_indices,
    iree_host_size_t source_requirement_count, loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_COMPILE_COMMAND_MANIFEST_H_
