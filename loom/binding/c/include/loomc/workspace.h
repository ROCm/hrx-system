// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_WORKSPACE_H_
#define LOOMC_WORKSPACE_H_

#include "loomc/status.h"

/// @file
/// Per-worker scratch workspace.
///
/// A workspace owns the arena block pool used by parsing, indexing, linking,
/// cloning, and compiling. High-throughput embedders normally create one
/// workspace per worker thread or task-system worker and reuse it across
/// operations.
///
/// @par Example
/// Keep a workspace thread-local and reuse it across independent operations:
///
/// @code{.c}
/// loomc_workspace_t* workspace = NULL;
/// loomc_status_t status =
///     loomc_workspace_create(NULL, loomc_allocator_system(), &workspace);
/// if (!loomc_status_is_ok(status)) {
///   return status;
/// }
///
/// for (size_t i = 0; i < kernel_count; ++i) {
///   // Pass workspace to deserialize, link, compile, or emit invocations for
///   // this kernel. Releasing modules returns their arena blocks to the
///   // workspace for reuse by later operations.
/// }
///
/// // Optionally return idle blocks to the host allocator under memory
/// // pressure or before entering a long idle period.
/// loomc_workspace_trim(workspace);
///
/// loomc_workspace_release(workspace);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Mutable per-worker scratch workspace.
///
/// @thread_safety
/// A workspace is not internally synchronized. Calls that mutate the same
/// workspace require external synchronization. Distinct workspaces may be used
/// concurrently by different threads.
typedef struct loomc_workspace_t loomc_workspace_t;

/// Workspace creation options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_WORKSPACE_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_workspace_options_t)`, and fill the requested fields.
typedef struct loomc_workspace_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_WORKSPACE_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future workspace options.
  const void* next;

  /// Arena block size used for transient compiler/linker storage.
  loomc_host_size_t block_size;
} loomc_workspace_options_t;

/// Creates a mutable workspace for one worker at a time.
///
/// @param options Workspace allocation options, or `NULL` for defaults.
/// @param allocator Host allocator used for workspace-owned storage.
/// @param out_workspace Receives the created workspace on success.
/// @return OK when the workspace was created.
///
/// @ownership
/// The caller owns the returned workspace and releases it with
/// `loomc_workspace_release`.
///
/// @lifetime
/// Sources, retained results, frozen link indexes, prepared linkers, and
/// prepared compilers do not use workspace arena storage unless an operation
/// descriptor explicitly says so. Modules created, linked, cloned, or
/// deserialized with a workspace retain that workspace and return their arena
/// blocks when released.
///
/// @thread_safety
/// The returned workspace is mutable scratch and is intended for one worker at
/// a time.
LOOMC_API_EXPORT loomc_status_t loomc_workspace_create(
    const loomc_workspace_options_t* options, loomc_allocator_t allocator,
    loomc_workspace_t** out_workspace);

/// Retains `workspace` for another owner.
///
/// @param workspace Workspace to retain. Passing `NULL` is allowed.
///
/// @thread_safety
/// Retain/release operations are safe from multiple threads.
LOOMC_API_EXPORT void loomc_workspace_retain(loomc_workspace_t* workspace);

/// Trims idle arena blocks owned by `workspace`.
///
/// @param workspace Workspace to trim.
///
/// @lifetime
/// Trim returns blocks that are already idle in the workspace block pool to the
/// host allocator. Retained modules created from this workspace remain valid
/// across trim, but their active arena blocks cannot be reused until those
/// modules are released. Results, sources, frozen indexes, and prepared tools
/// remain valid.
///
/// @thread_safety
/// External synchronization is required if another thread may be using the same
/// workspace.
LOOMC_API_EXPORT void loomc_workspace_trim(loomc_workspace_t* workspace);

/// Releases `workspace` and all storage it owns.
///
/// @param workspace Workspace to release. Passing `NULL` is allowed.
///
/// @thread_safety
/// Retain/release operations are safe from multiple threads. The workspace is
/// destroyed when the final reference is released; no thread may be mutating it
/// at that moment.
LOOMC_API_EXPORT void loomc_workspace_release(loomc_workspace_t* workspace);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_WORKSPACE_H_
