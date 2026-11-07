// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CMD_COMPILER_H_
#define LOOMC_TARGET_CMD_COMPILER_H_

#include "loomc/link_index.h"

/// @file
/// Reusable command compilation over one immutable source catalog.

#ifdef __cplusplus
extern "C" {
#endif

/// Concurrent command compiler retaining a frozen index and prepared native
/// sources. Each invocation supplies its own workspace, roots, configuration,
/// targets, and publication sink. Published products and requests own their
/// contents independently of the compiler.
typedef struct loomc_cmd_compiler_t loomc_cmd_compiler_t;

/// Command compiler storage policy.
typedef struct loomc_cmd_compiler_options_t {
  /// Structure type. Must be LOOMC_STRUCTURE_TYPE_CMD_COMPILER_OPTIONS when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain; currently must be NULL.
  const void* next;

  /// Maximum retained source bytes, including allocation tracking and unused
  /// arena capacity. Zero disables retention. This excludes the frozen index,
  /// shared target profiles, the fixed source-lookup table, active private
  /// preparations, and underlying allocator fragmentation.
  /// Oversized sources and misses that cannot displace an idle source compile
  /// normally without retention. Passing NULL options selects 64 MiB.
  loomc_host_size_t retained_byte_limit;
} loomc_cmd_compiler_options_t;

/// Creates a compiler retaining |link_index|. Creation inspects no source
/// bodies. Passing NULL options selects the default storage policy.
///
/// The compiler supports concurrent invocations with independent workspaces;
/// callers hold a reference until each synchronous invocation has returned.
///
/// @param link_index Immutable source index retained by the compiler.
/// @param options Storage policy, or `NULL` for defaults. Borrowed for this
/// call.
/// @param allocator Host allocator used for compiler and retained source
/// storage.
/// @param out_compiler Receives one owned compiler reference on success and
///     `NULL` on failure. Release it with `loomc_cmd_compiler_release`.
/// @return OK when the compiler was created.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_compiler_create(
    loomc_link_index_t* link_index, const loomc_cmd_compiler_options_t* options,
    loomc_allocator_t allocator, loomc_cmd_compiler_t** out_compiler);

/// Retains a compiler. Passing NULL is allowed. Thread-safe.
///
/// @param compiler Compiler to retain, or `NULL`.
LOOMC_API_EXPORT void loomc_cmd_compiler_retain(loomc_cmd_compiler_t* compiler);

/// Releases a compiler. Passing NULL is allowed. Thread-safe. The final release
/// destroys prepared sources before releasing their source index and context.
///
/// @param compiler Compiler reference to release, or `NULL`.
LOOMC_API_EXPORT void loomc_cmd_compiler_release(
    loomc_cmd_compiler_t* compiler);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_CMD_COMPILER_H_
