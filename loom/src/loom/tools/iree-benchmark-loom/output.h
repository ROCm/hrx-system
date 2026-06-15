// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Output paths, event streaming, and artifact bundle ownership.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_OUTPUT_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_OUTPUT_H_

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/io/stream.h"
#include "loom/tools/iree-benchmark-loom/options.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_benchmark_loom_run_identity_t {
  // Process-local run identifier copied into result output records.
  iree_string_view_t run_id;
  // User-provided source path, or "<stdin>" when reading standard input.
  iree_string_view_t source;
  // Effective result output path, or "-" when writing to stdout.
  iree_string_view_t results_path;
  // Directory receiving check.file.write.* outputs for this run.
  iree_string_view_t file_output_dir;
  // Directory receiving raw profile artifacts by default.
  iree_string_view_t profile_artifacts_dir;
  // Directory owning the artifact bundle, or empty when bundling is disabled.
  iree_string_view_t artifact_bundle_dir;
  // Active artifact bundle policy name, or "none" when bundling is disabled.
  iree_string_view_t artifact_bundle_policy;
} iree_benchmark_loom_run_identity_t;

typedef struct iree_benchmark_loom_jsonl_sink_t {
  // Host allocator used for scratch row storage.
  iree_allocator_t host_allocator;
  // Open FILE receiving the JSONL stream.
  FILE* file;
  // True when |file| is owned by this sink and must be closed.
  bool owns_file;
  // Scratch storage used to assemble complete rows before writing.
  iree_string_builder_t row_builder;
} iree_benchmark_loom_jsonl_sink_t;

typedef enum iree_benchmark_loom_bundle_file_kind_e {
  IREE_BENCHMARK_LOOM_BUNDLE_FILE_FIXTURE_READ = 0,
  IREE_BENCHMARK_LOOM_BUNDLE_FILE_OUTPUT = 1,
  IREE_BENCHMARK_LOOM_BUNDLE_FILE_PROFILE = 2,
  IREE_BENCHMARK_LOOM_BUNDLE_FILE_COMPILE_REPORT = 3,
  IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_ARTIFACT = 4,
  IREE_BENCHMARK_LOOM_BUNDLE_FILE_HAL_EXECUTABLE = 5,
  IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_LISTING = 6,
  IREE_BENCHMARK_LOOM_BUNDLE_FILE_ARTIFACT_MANIFEST = 7,
} iree_benchmark_loom_bundle_file_kind_t;

typedef struct iree_benchmark_loom_bundle_file_entry_t {
  // Manifest bucket that owns this file reference.
  iree_benchmark_loom_bundle_file_kind_t kind;
  // Owned resolved filesystem path.
  char* path;
} iree_benchmark_loom_bundle_file_entry_t;

typedef struct iree_benchmark_loom_artifact_bundle_t {
  // Host allocator used for owned path strings.
  iree_allocator_t host_allocator;
  // True when --artifact-bundle-dir requested a bundle.
  bool enabled;
  // Parsed bundle policy controlling which artifact classes are expected.
  iree_benchmark_loom_artifact_bundle_policy_t policy;
  // Borrowed view into |dir_storage|.
  iree_string_view_t dir;
  // Owned bundle root directory.
  char* dir_storage;
  // Borrowed view into |results_path_storage|.
  iree_string_view_t results_path;
  // Owned default result path inside |dir|.
  char* results_path_storage;
  // Borrowed view into |manifest_path_storage|.
  iree_string_view_t manifest_path;
  // Owned manifest path inside |dir|.
  char* manifest_path_storage;
  // Borrowed view into |file_output_dir_storage|.
  iree_string_view_t file_output_dir;
  // Owned default file-output directory inside |dir|.
  char* file_output_dir_storage;
  // Borrowed view into |profile_artifacts_dir_storage|.
  iree_string_view_t profile_artifacts_dir;
  // Owned default profile-artifacts directory inside |dir|.
  char* profile_artifacts_dir_storage;
  // Borrowed view into |compile_report_dir_storage|.
  iree_string_view_t compile_report_dir;
  // Owned debug compile-report directory inside |dir|.
  char* compile_report_dir_storage;
  // Borrowed view into |artifact_manifest_dir_storage|.
  iree_string_view_t artifact_manifest_dir;
  // Owned debug artifact-manifest directory inside |dir|.
  char* artifact_manifest_dir_storage;
  // Borrowed view into |target_artifact_dir_storage|.
  iree_string_view_t target_artifact_dir;
  // Owned debug target-native artifact directory inside |dir|.
  char* target_artifact_dir_storage;
  // Borrowed view into |target_listing_dir_storage|.
  iree_string_view_t target_listing_dir;
  // Owned debug target-native listing directory inside |dir|.
  char* target_listing_dir_storage;
  // Borrowed view into |hal_executable_dir_storage|.
  iree_string_view_t hal_executable_dir;
  // Owned debug HAL executable artifact directory inside |dir|.
  char* hal_executable_dir_storage;
  // Owned file references observed while the run executed.
  iree_benchmark_loom_bundle_file_entry_t* file_entries;
  // Number of populated entries in |file_entries|.
  iree_host_size_t file_entry_count;
  // Allocated capacity of |file_entries|.
  iree_host_size_t file_entry_capacity;
} iree_benchmark_loom_artifact_bundle_t;

typedef struct iree_benchmark_loom_artifact_bundle_options_t {
  // Trimmed bundle directory requested by the CLI, or empty when disabled.
  iree_string_view_t dir;
  // Parsed bundle policy requested by the CLI.
  iree_benchmark_loom_artifact_bundle_policy_t policy;
  // Result file format used to pick the bundle default result filename.
  iree_benchmark_loom_output_format_t output_format;
} iree_benchmark_loom_artifact_bundle_options_t;

typedef struct iree_benchmark_loom_file_provider_t {
  // Host allocator used only for path strings and stream-owned storage.
  iree_allocator_t host_allocator;
  // Optional bundle receiving opened fixture/output file references.
  iree_benchmark_loom_artifact_bundle_t* artifact_bundle;
  // Borrowed directory containing the input module for relative fixture reads.
  iree_string_view_t input_dir;
  // Borrowed view into |output_dir_storage|.
  iree_string_view_t output_dir;
  // Owned directory receiving relative file outputs.
  char* output_dir_storage;
} iree_benchmark_loom_file_provider_t;

// Duplicates a string view into NUL-terminated storage owned by |allocator|.
iree_status_t iree_benchmark_loom_dup_string_view(iree_string_view_t value,
                                                  iree_allocator_t allocator,
                                                  char** out_storage);

// Appends |value| with path-unsafe characters rewritten for artifact leaves.
iree_status_t iree_benchmark_loom_append_sanitized_path_component(
    iree_string_view_t value, iree_string_builder_t* builder);

// Joins two filesystem path components into allocator-owned storage.
iree_status_t iree_benchmark_loom_join_path(iree_string_view_t lhs,
                                            iree_string_view_t rhs,
                                            iree_allocator_t allocator,
                                            char** out_path);

// Creates |path| and any missing parent directories.
iree_status_t iree_benchmark_loom_create_directory_if_needed(
    iree_string_view_t path, iree_allocator_t allocator);

// Creates the parent directory for |path| when it is non-empty.
iree_status_t iree_benchmark_loom_create_parent_directory(
    iree_string_view_t path, iree_allocator_t allocator);

// Records a file observed by the run artifact bundle.
iree_status_t iree_benchmark_loom_artifact_bundle_record_file(
    iree_benchmark_loom_artifact_bundle_t* bundle,
    iree_benchmark_loom_bundle_file_kind_t kind, iree_string_view_t path);

// Counts files observed in one artifact bundle manifest bucket.
iree_host_size_t iree_benchmark_loom_artifact_bundle_file_count(
    const iree_benchmark_loom_artifact_bundle_t* bundle,
    iree_benchmark_loom_bundle_file_kind_t kind);

// Initializes an artifact bundle path layout from parsed CLI options.
iree_status_t iree_benchmark_loom_artifact_bundle_initialize(
    const iree_benchmark_loom_artifact_bundle_options_t* options,
    iree_allocator_t allocator,
    iree_benchmark_loom_artifact_bundle_t* out_bundle);

// Releases all storage owned by |bundle|.
void iree_benchmark_loom_artifact_bundle_deinitialize(
    iree_benchmark_loom_artifact_bundle_t* bundle);

// Returns true when |bundle| should receive debug target artifacts.
bool iree_benchmark_loom_artifact_bundle_wants_debug_artifacts(
    const iree_benchmark_loom_artifact_bundle_t* bundle);

// Returns true when |bundle| should receive compile-report artifacts.
bool iree_benchmark_loom_artifact_bundle_wants_compile_reports(
    const iree_benchmark_loom_artifact_bundle_t* bundle);

// Returns true when |bundle| should receive artifact-manifest artifacts.
bool iree_benchmark_loom_artifact_bundle_wants_artifact_manifests(
    const iree_benchmark_loom_artifact_bundle_t* bundle);

// Resolves the result output path after applying bundle defaults.
iree_string_view_t iree_benchmark_loom_effective_results_output_path(
    iree_string_view_t explicit_output,
    const iree_benchmark_loom_artifact_bundle_t* bundle);

// Resolves the profile artifact directory after applying bundle defaults.
iree_string_view_t iree_benchmark_loom_effective_profile_artifacts_dir(
    iree_string_view_t explicit_artifacts_dir,
    const iree_benchmark_loom_artifact_bundle_t* bundle);

// Appends the effective profile artifact directory for |run| and profile data.
iree_status_t iree_benchmark_loom_append_effective_profile_artifacts_dir(
    const iree_benchmark_loom_run_identity_t* run,
    iree_hal_device_profiling_data_families_t profile_data_families,
    iree_string_builder_t* artifact_dir);

// Initializes a file provider used by check.file.read/write materialization.
iree_status_t iree_benchmark_loom_file_provider_initialize(
    iree_string_view_t filename, iree_string_view_t run_id,
    iree_string_view_t explicit_output_dir,
    iree_string_view_t default_output_dir,
    iree_benchmark_loom_artifact_bundle_t* artifact_bundle,
    iree_allocator_t allocator,
    iree_benchmark_loom_file_provider_t* out_provider);

// Releases storage owned by |provider|.
void iree_benchmark_loom_file_provider_deinitialize(
    iree_benchmark_loom_file_provider_t* provider);

// Opens a check.file.read path for the testbench materializer.
iree_status_t iree_benchmark_loom_open_file_for_read(
    void* user_data, iree_string_view_t path, iree_io_stream_t** out_stream);

// Opens a check.file.write path for the testbench materializer.
iree_status_t iree_benchmark_loom_open_file_for_write(
    void* user_data, iree_string_view_t path, iree_io_stream_t** out_stream);

// Initializes a JSONL sink.
iree_status_t iree_benchmark_loom_jsonl_sink_initialize(
    iree_string_view_t path, iree_allocator_t allocator,
    iree_benchmark_loom_jsonl_sink_t* out_sink);

// Releases storage and any owned file handle held by |sink|.
void iree_benchmark_loom_jsonl_sink_deinitialize(
    iree_benchmark_loom_jsonl_sink_t* sink);

// Begins a new JSONL row in |sink|'s scratch builder.
iree_string_builder_t* iree_benchmark_loom_jsonl_sink_begin(
    iree_benchmark_loom_jsonl_sink_t* sink);

// Flushes the current row when |row_status| succeeded, or drops it otherwise.
iree_status_t iree_benchmark_loom_jsonl_sink_end(
    iree_benchmark_loom_jsonl_sink_t* sink, iree_status_t row_status);

// Closes an owned JSONL output file before the final manifest is written.
iree_status_t iree_benchmark_loom_jsonl_sink_close(
    iree_benchmark_loom_jsonl_sink_t* sink);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_OUTPUT_H_
