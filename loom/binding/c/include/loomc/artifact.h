// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_ARTIFACT_H_
#define LOOMC_ARTIFACT_H_

#include <stdio.h>

#include "loomc/byte_sequence.h"
#include "loomc/source.h"

/// @file
/// In-memory output artifacts produced by compiler operations.
///
/// Artifacts are structured objects, not filesystem side effects. They carry
/// format, identity, and bytes so language bindings can expose native byte
/// arrays or typed artifact objects.
///
/// Filesystem persistence is explicit and composable. JIT integrations can keep
/// artifact bytes in memory for loading or cache lookup, while tools and
/// release builds can write the same artifact object to a `FILE*` or path.
///
/// @par Example
/// Save the kernel payload from a result:
///
/// @code{.c}
/// for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result);
///      ++i) {
///   const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
///   if (!loomc_string_view_equal(
///           artifact->role,
///           loomc_make_cstring_view(LOOMC_ARTIFACT_ROLE_KERNEL))) continue;
///   return loomc_artifact_write_to_path(
///       artifact, loomc_make_cstring_view("kernel.hsaco"),
///       loomc_allocator_system());
/// }
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Textual Loom module artifact format.
#define LOOMC_ARTIFACT_FORMAT_LOOM_TEXT "loom-text"

/// Binary Loom bytecode module artifact format.
#define LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE "loombc"

/// JSON report artifact format.
#define LOOMC_ARTIFACT_FORMAT_JSON "json"

/// Loadable or inspectable kernel payload role.
#define LOOMC_ARTIFACT_ROLE_KERNEL "kernel"

/// Portable command-program payload role.
#define LOOMC_ARTIFACT_ROLE_COMMAND_PROGRAM "command-program"

/// Loom source or bytecode module role.
#define LOOMC_ARTIFACT_ROLE_MODULE "module"

/// Compiled kernel launch-configuration program role.
#define LOOMC_ARTIFACT_ROLE_LAUNCH_CONFIG "launch-config"

/// Structured artifact-manifest role.
#define LOOMC_ARTIFACT_ROLE_ARTIFACT_MANIFEST "artifact-manifest"

/// Structured compile-report role.
#define LOOMC_ARTIFACT_ROLE_COMPILE_REPORT "compile-report"

/// Structured link-dependency report role.
#define LOOMC_ARTIFACT_ROLE_LINK_DEPENDENCY_REPORT "link-dependency-report"

/// Target-owned human-readable listing role.
#define LOOMC_ARTIFACT_ROLE_LISTING "listing"

/// Borrowed artifact view owned by a producer-specific result or product.
///
/// @lifetime
/// Artifact strings and the contents reference are owned by the
/// operation-specific object that returned this view. They remain valid until
/// that owner is released. Callers retaining contents independently use
/// `loomc_byte_sequence_retain`.
typedef struct loomc_artifact_t {
  /// Semantic role within the owning product or operation result.
  ///
  /// Roles are open stable names. They describe how an artifact participates
  /// in its product independently of whether its format is text, binary, an
  /// object container, or a loadable image.
  loomc_string_view_t role;

  /// Stable format string, such as `amdgpu-hsaco`, `spirv-binary`,
  /// `LOOMC_ARTIFACT_FORMAT_LOOM_TEXT`, or `LOOMC_ARTIFACT_FORMAT_JSON`.
  loomc_string_view_t format;

  /// Human-readable artifact identifier.
  loomc_string_view_t identifier;

  /// Immutable artifact bytes. Valid artifacts always provide a non-NULL
  /// sequence, including for empty contents.
  loomc_byte_sequence_t* contents;
} loomc_artifact_t;

/// Creates an immutable source handle from an artifact.
///
/// @param artifact Artifact whose bytes should become source contents.
/// @param format Source format for the returned handle. Unknown infers Loom
/// text and bytecode from well-known Loom artifact format strings and leaves
/// other artifact formats unknown.
/// @param allocator Host allocator used for source-owned storage.
/// @param out_source Receives one retained source on success.
/// @return OK when the source was created.
///
/// @ownership
/// The returned source retains contiguous artifact storage without copying.
/// Segmented storage is coalesced once. The caller releases the source with
/// `loomc_source_release`.
LOOMC_API_EXPORT loomc_status_t loomc_artifact_create_source(
    const loomc_artifact_t* artifact, loomc_source_format_t format,
    loomc_allocator_t allocator, loomc_source_t** out_source);

/// Writes artifact bytes to an open C `FILE*`.
///
/// @param artifact Artifact to write.
/// @param file Open writable file handle, such as `stdout`.
/// @return OK when all artifact bytes were written.
///
/// @ownership
/// The caller retains ownership of `file`; this function does not close it.
/// On Windows the underlying file descriptor is switched to binary mode and
/// remains in that mode so artifact bytes are not newline-translated.
LOOMC_API_EXPORT loomc_status_t
loomc_artifact_write_to_file(const loomc_artifact_t* artifact, FILE* file);

/// Writes artifact bytes to a filesystem path.
///
/// @param artifact Artifact to write.
/// @param path Output file path. The path is a borrowed byte view and does not
/// need to be NUL-terminated.
/// @param allocator Host allocator used for transient file path/stream storage.
/// @return OK when all artifact bytes were written.
LOOMC_API_EXPORT loomc_status_t loomc_artifact_write_to_path(
    const loomc_artifact_t* artifact, loomc_string_view_t path,
    loomc_allocator_t allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_ARTIFACT_H_
