// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_ARTIFACT_H_
#define LOOMC_ARTIFACT_H_

#include <stdio.h>

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
/// Save the first executable artifact from a result:
///
/// @code{.c}
/// for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result);
///      ++i) {
///   const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
///   if (artifact->kind != LOOMC_ARTIFACT_KIND_EXECUTABLE) continue;
///   return loomc_artifact_write_to_path(
///       artifact, loomc_make_cstring_view("kernel.vmfb"),
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

/// Artifact category.
typedef enum loomc_artifact_kind_e {
  /// Executable or loadable binary artifact.
  LOOMC_ARTIFACT_KIND_EXECUTABLE = 0,

  /// Human-readable textual artifact.
  LOOMC_ARTIFACT_KIND_TEXT = 1,

  /// Machine-readable report artifact.
  LOOMC_ARTIFACT_KIND_REPORT = 2,

  /// Loom module artifact. The format indicates text or bytecode.
  LOOMC_ARTIFACT_KIND_MODULE = 3,
} loomc_artifact_kind_t;

/// Borrowed artifact view owned by a result object.
///
/// @lifetime
/// Artifact strings and bytes are owned by the result that returned this view.
/// They remain valid until that result is released.
typedef struct loomc_artifact_t {
  /// Artifact kind.
  loomc_artifact_kind_t kind;

  /// Stable format string, such as `vmfb`, `spirv`,
  /// `LOOMC_ARTIFACT_FORMAT_LOOM_TEXT`, or `LOOMC_ARTIFACT_FORMAT_JSON`.
  loomc_string_view_t format;

  /// Human-readable artifact identifier.
  loomc_string_view_t identifier;

  /// Artifact bytes.
  loomc_byte_span_t contents;
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
/// The returned source owns a copy of the artifact bytes and identifier. The
/// caller releases it with `loomc_source_release`.
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
