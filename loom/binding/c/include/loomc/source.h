// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_SOURCE_H_
#define LOOMC_SOURCE_H_

#include <stdio.h>

#include "loomc/status.h"

/// @file
/// Immutable source handles used by parsing, indexing, linking, and compiling.
///
/// Sources carry bytes plus stable identity. They are distinct from parsed
/// modules and from output artifacts. Source identity is used for diagnostics,
/// cache keys, and deterministic link-index provenance.
///
/// @par Example
/// Copy source bytes when the caller does not want source lifetime to depend on
/// an external buffer:
///
/// @code{.c}
/// const char* source_text = "...";
/// loomc_source_options_t options = {
///     .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
///     .structure_size = sizeof(loomc_source_options_t),
///     .format = LOOMC_SOURCE_FORMAT_TEXT,
///     .identifier = loomc_make_cstring_view("generated.loom"),
///     .contents = loomc_make_byte_span(source_text, strlen(source_text)),
///     .storage = LOOMC_SOURCE_STORAGE_COPY,
/// };
///
/// loomc_source_t* source = NULL;
/// loomc_status_t status =
///     loomc_source_create(&options, loomc_allocator_system(), &source);
/// if (!loomc_status_is_ok(status)) {
///   return status;
/// }
///
/// // Use source from any worker thread after sharing or retaining it.
///
/// loomc_source_release(source);
/// @endcode
///
/// @par Example
/// Load source bytes from a path and let the source own the loaded file buffer:
///
/// @code{.c}
/// loomc_source_load_options_t options = {
///     .type = LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS,
///     .structure_size = sizeof(loomc_source_load_options_t),
///     .format = LOOMC_SOURCE_FORMAT_UNKNOWN,
/// };
///
/// loomc_source_t* source = NULL;
/// loomc_status_t status = loomc_source_create_from_path(
///     loomc_make_cstring_view("profile.json"), &options,
///     loomc_allocator_system(), &source);
/// if (!loomc_status_is_ok(status)) {
///   return status;
/// }
///
/// // Pass source to a parser, importer, linker index builder, or compiler.
///
/// loomc_source_release(source);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Immutable source handle.
///
/// @thread_safety
/// Sources are immutable after creation. Retained source handles may be shared
/// across threads.
typedef struct loomc_source_t loomc_source_t;

/// Source byte format.
typedef enum loomc_source_format_e {
  /// Source format is unknown and must be inferred by the consumer.
  LOOMC_SOURCE_FORMAT_UNKNOWN = 0,

  /// Textual `.loom` source.
  LOOMC_SOURCE_FORMAT_TEXT = 1,

  /// Binary `.loombc` source.
  LOOMC_SOURCE_FORMAT_BYTECODE = 2,
} loomc_source_format_t;

/// Storage policy for source bytes.
typedef enum loomc_source_storage_e {
  /// Source bytes are borrowed from the caller for the lifetime of the source.
  LOOMC_SOURCE_STORAGE_BORROWED = 0,

  /// Source bytes are copied into storage owned by the source.
  LOOMC_SOURCE_STORAGE_COPY = 1,

  /// Source bytes are owned externally and released by a caller callback.
  LOOMC_SOURCE_STORAGE_EXTERNAL = 2,
} loomc_source_storage_t;

/// Callback used to release externally owned source bytes.
///
/// @param user_data Caller-provided value from `loomc_source_options_t`.
/// @param contents Source bytes being released.
///
/// @thread_safety
/// The callback is invoked when the last source reference is released. Callers
/// that share sources across threads must make the callback safe for whichever
/// thread drops the final reference.
typedef void(LOOMC_API_PTR* loomc_source_release_fn_t)(
    void* user_data, loomc_byte_span_t contents);

/// Source creation options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_source_options_t)`, and fill the requested fields.
typedef struct loomc_source_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future source options.
  const void* next;

  /// Input source format.
  loomc_source_format_t format;

  /// Stable identifier used in diagnostics and cache keys.
  loomc_string_view_t identifier;

  /// Input bytes.
  loomc_byte_span_t contents;

  /// Storage policy for `contents`.
  loomc_source_storage_t storage;

  /// Callback used when `storage` is `LOOMC_SOURCE_STORAGE_EXTERNAL`.
  loomc_source_release_fn_t release;

  /// User data passed to `release`.
  void* release_user_data;
} loomc_source_options_t;

/// Source load options for host file and path constructors.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_source_load_options_t)`, and fill the requested fields.
typedef struct loomc_source_load_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  const void* next;

  /// Loaded source format.
  loomc_source_format_t format;

  /// Stable identifier used in diagnostics and cache keys.
  ///
  /// Empty uses no identifier for open-file loads and uses `path` for path
  /// loads.
  loomc_string_view_t identifier;
} loomc_source_load_options_t;

/// Creates an immutable source handle.
///
/// @param options Source format, identity, bytes, and storage policy.
/// @param allocator Host allocator used for source-owned storage.
/// @param out_source Receives one retained source on success.
/// @return OK when the source was created. Invalid argument is returned for
/// malformed descriptors or impossible storage/release combinations.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_source_release`.
///
/// @lifetime
/// The source always copies `options->identifier`. Source contents are
/// borrowed, copied, or externally released according to `options->storage`.
/// Borrowed contents must outlive the source.
///
/// @thread_safety
/// The returned source is immutable and may be shared across threads.
LOOMC_API_EXPORT loomc_status_t
loomc_source_create(const loomc_source_options_t* options,
                    loomc_allocator_t allocator, loomc_source_t** out_source);

/// Creates an immutable source by reading an open file.
///
/// @param file Open file positioned at the first byte to read.
/// @param options Source format and identity metadata, or `NULL` for defaults.
/// @param allocator Host allocator used for source and byte storage.
/// @param out_source Receives one retained source on success.
/// @return OK when all remaining file bytes were read into source-owned memory.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_source_release`. The caller still owns `file` and is responsible for
/// closing it.
///
/// @lifetime
/// The returned source owns a copy of the bytes read from `file`; the file can
/// be closed or reused after this call returns.
///
/// @thread_safety
/// The returned source is immutable and may be shared across threads. This call
/// performs ordinary C `FILE*` reads and does not synchronize access to `file`.
LOOMC_API_EXPORT loomc_status_t loomc_source_create_from_file(
    FILE* file, const loomc_source_load_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source);

/// Creates an immutable source by reading a filesystem path.
///
/// @param path Path to read. The path view need not be NUL-terminated.
/// @param options Source format and identity metadata, or `NULL` for defaults.
/// @param allocator Host allocator used for source and byte storage.
/// @param out_source Receives one retained source on success.
/// @return OK when the path contents were read into source-owned memory.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_source_release`.
///
/// @lifetime
/// The returned source owns the loaded bytes and copies the source identifier.
/// When `options->identifier` is empty, `path` is copied as the identifier.
///
/// @thread_safety
/// The returned source is immutable and may be shared across threads.
LOOMC_API_EXPORT loomc_status_t loomc_source_create_from_path(
    loomc_string_view_t path, const loomc_source_load_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source);

/// Retains `source` for another owner.
///
/// @param source Source to retain.
///
/// @thread_safety
/// Retain/release operations are safe to perform from multiple threads.
LOOMC_API_EXPORT void loomc_source_retain(loomc_source_t* source);

/// Releases `source` from one owner.
///
/// @param source Source to release. Passing `NULL` is allowed.
///
/// @thread_safety
/// Retain/release operations are safe to perform from multiple threads. The
/// source is destroyed when the final reference is released.
LOOMC_API_EXPORT void loomc_source_release(loomc_source_t* source);

/// Returns the source format.
///
/// @param source Source to inspect.
/// @return Format recorded when the source was created.
LOOMC_API_EXPORT loomc_source_format_t
loomc_source_format(const loomc_source_t* source);

/// Returns the source identifier.
///
/// @param source Source to inspect.
/// @return Borrowed identifier view owned by `source`.
///
/// @lifetime
/// The returned view remains valid until `source` is released.
LOOMC_API_EXPORT loomc_string_view_t
loomc_source_identifier(const loomc_source_t* source);

/// Returns the source contents.
///
/// @param source Source to inspect.
/// @return Borrowed byte span owned or referenced by `source`.
///
/// @lifetime
/// The returned span remains valid until `source` is released. For borrowed
/// source storage, the caller-provided bytes must also remain alive.
LOOMC_API_EXPORT loomc_byte_span_t
loomc_source_contents(const loomc_source_t* source);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_SOURCE_H_
