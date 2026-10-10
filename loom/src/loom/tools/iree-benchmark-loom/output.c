// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/output.h"

#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/path.h"
#include "iree/io/stdio_stream.h"
#include "loom/tooling/io/file.h"

static bool iree_benchmark_loom_is_safe_output_path_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-' || c == '/';
}

static uint64_t iree_benchmark_loom_hash_string_view(uint64_t hash,
                                                     iree_string_view_t value) {
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    hash ^= (uint8_t)value.data[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

iree_status_t iree_benchmark_loom_append_sanitized_path_component(
    iree_string_view_t value, iree_string_builder_t* builder) {
  if (iree_string_view_equal(value, IREE_SV("<stdin>"))) {
    return iree_string_builder_append_cstring(builder, "stdin");
  }
  bool appended = false;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    char c = value.data[i];
    if (!iree_benchmark_loom_is_safe_output_path_char(c) || c == '/') {
      c = '_';
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, iree_make_string_view(&c, 1)));
    appended = true;
  }
  if (!appended) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "input"));
  }
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_dup_string_view(iree_string_view_t value,
                                                  iree_allocator_t allocator,
                                                  char** out_value) {
  *out_value = NULL;
  iree_host_size_t storage_size = 0;
  if (!iree_host_size_checked_add(value.size, 1, &storage_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "string storage size overflow");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, storage_size, (void**)&storage));
  if (value.size != 0) {
    memcpy(storage, value.data, value.size);
  }
  storage[value.size] = '\0';
  *out_value = storage;
  return iree_ok_status();
}

static iree_string_view_t iree_benchmark_loom_tmp_dir(void) {
  const char* tmp_dir = getenv("TMPDIR");
  if (tmp_dir == NULL || tmp_dir[0] == '\0') {
    tmp_dir = getenv("TEMP");
  }
  if (tmp_dir == NULL || tmp_dir[0] == '\0') {
    tmp_dir = getenv("TMP");
  }
  if (tmp_dir == NULL || tmp_dir[0] == '\0') {
    tmp_dir = "/tmp";
  }
  return iree_make_cstring_view(tmp_dir);
}

static iree_status_t iree_benchmark_loom_make_default_file_output_dir(
    iree_string_view_t filename, iree_string_view_t run_id,
    iree_allocator_t allocator, char** out_path) {
  *out_path = NULL;

  uint64_t hash = 1469598103934665603ull;
  hash = iree_benchmark_loom_hash_string_view(hash, filename);
  hash = iree_benchmark_loom_hash_string_view(hash, run_id);

  iree_string_builder_t leaf_builder;
  iree_string_builder_initialize(allocator, &leaf_builder);
  iree_string_view_t source_name = iree_file_path_basename(filename);
  iree_status_t status = iree_benchmark_loom_append_sanitized_path_component(
      source_name, &leaf_builder);
  if (iree_status_is_ok(status)) {
    status =
        iree_string_builder_append_format(&leaf_builder, "_%016" PRIx64, hash);
  }
  iree_string_view_t leaf = iree_string_builder_view(&leaf_builder);

  char* root_path = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_tooling_file_path_join(iree_benchmark_loom_tmp_dir(),
                                         IREE_SV("iree-loom-benchmark"),
                                         allocator, &root_path);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_file_path_join(iree_make_cstring_view(root_path),
                                         leaf, allocator, out_path);
  }
  iree_allocator_free(allocator, root_path);
  iree_string_builder_deinitialize(&leaf_builder);
  return status;
}

static bool iree_benchmark_loom_artifact_bundle_has_file(
    const iree_benchmark_loom_artifact_bundle_t* bundle,
    iree_benchmark_loom_bundle_file_kind_t kind, iree_string_view_t path) {
  for (iree_host_size_t i = 0; i < bundle->file_entry_count; ++i) {
    const iree_benchmark_loom_bundle_file_entry_t* entry =
        &bundle->file_entries[i];
    if (entry->kind == kind &&
        iree_string_view_equal(iree_make_cstring_view(entry->path), path)) {
      return true;
    }
  }
  return false;
}

iree_status_t iree_benchmark_loom_artifact_bundle_record_file(
    iree_benchmark_loom_artifact_bundle_t* bundle,
    iree_benchmark_loom_bundle_file_kind_t kind, iree_string_view_t path) {
  if (bundle == NULL || !bundle->enabled) {
    return iree_ok_status();
  }
  path = iree_string_view_trim(path);
  if (iree_string_view_is_empty(path) ||
      loom_tooling_file_path_is_stdio(path) ||
      iree_benchmark_loom_artifact_bundle_has_file(bundle, kind, path)) {
    return iree_ok_status();
  }

  if (bundle->file_entry_count == bundle->file_entry_capacity) {
    iree_host_size_t new_capacity = 8;
    if (bundle->file_entry_capacity != 0 &&
        !iree_host_size_checked_mul(bundle->file_entry_capacity, 2,
                                    &new_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "too many artifact bundle file entries");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
        bundle->host_allocator, new_capacity, sizeof(bundle->file_entries[0]),
        (void**)&bundle->file_entries));
    bundle->file_entry_capacity = new_capacity;
  }

  iree_benchmark_loom_bundle_file_entry_t* entry =
      &bundle->file_entries[bundle->file_entry_count];
  entry->kind = kind;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_dup_string_view(
      path, bundle->host_allocator, &entry->path));
  ++bundle->file_entry_count;
  return iree_ok_status();
}

iree_host_size_t iree_benchmark_loom_artifact_bundle_file_count(
    const iree_benchmark_loom_artifact_bundle_t* bundle,
    iree_benchmark_loom_bundle_file_kind_t kind) {
  if (bundle == NULL || !bundle->enabled) {
    return 0;
  }
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < bundle->file_entry_count; ++i) {
    if (bundle->file_entries[i].kind == kind) {
      ++count;
    }
  }
  return count;
}

iree_status_t iree_benchmark_loom_file_provider_initialize(
    iree_string_view_t filename, iree_string_view_t run_id,
    iree_string_view_t explicit_output_dir,
    iree_string_view_t default_output_dir,
    iree_benchmark_loom_artifact_bundle_t* artifact_bundle,
    iree_allocator_t allocator,
    iree_benchmark_loom_file_provider_t* out_provider) {
  memset(out_provider, 0, sizeof(*out_provider));
  out_provider->host_allocator = allocator;
  out_provider->artifact_bundle = artifact_bundle;
  if (!iree_string_view_equal(filename, IREE_SV("<stdin>"))) {
    out_provider->input_dir = iree_file_path_dirname(filename);
  }

  iree_string_view_t output_dir = iree_string_view_trim(explicit_output_dir);
  if (!iree_string_view_is_empty(output_dir)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_dup_string_view(
        output_dir, allocator, &out_provider->output_dir_storage));
  } else if (!iree_string_view_is_empty(default_output_dir)) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_dup_string_view(
        default_output_dir, allocator, &out_provider->output_dir_storage));
  } else {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_make_default_file_output_dir(
        filename, run_id, allocator, &out_provider->output_dir_storage));
  }
  out_provider->output_dir =
      iree_make_cstring_view(out_provider->output_dir_storage);
  return iree_ok_status();
}

void iree_benchmark_loom_file_provider_deinitialize(
    iree_benchmark_loom_file_provider_t* provider) {
  if (provider == NULL) {
    return;
  }
  iree_allocator_free(provider->host_allocator, provider->output_dir_storage);
  memset(provider, 0, sizeof(*provider));
}

iree_status_t iree_benchmark_loom_jsonl_sink_initialize(
    iree_string_view_t path, iree_allocator_t allocator,
    iree_benchmark_loom_jsonl_sink_t* out_sink) {
  memset(out_sink, 0, sizeof(*out_sink));
  out_sink->host_allocator = allocator;
  iree_string_builder_initialize(allocator, &out_sink->row_builder);

  path = iree_string_view_trim(path);
  if (loom_tooling_file_path_is_stdio(path)) {
    out_sink->file = stdout;
    return iree_ok_status();
  }

  iree_status_t status = loom_tooling_create_parent_directory(path, allocator);
  if (iree_status_is_ok(status)) {
    status = iree_io_stdio_file_open(path, "wb", allocator, &out_sink->file);
  }
  if (iree_status_is_ok(status)) {
    out_sink->owns_file = true;
  } else {
    iree_string_builder_deinitialize(&out_sink->row_builder);
    memset(out_sink, 0, sizeof(*out_sink));
  }
  return status;
}

void iree_benchmark_loom_jsonl_sink_deinitialize(
    iree_benchmark_loom_jsonl_sink_t* sink) {
  if (sink == NULL) {
    return;
  }
  if (sink->owns_file && sink->file != NULL) {
    fclose(sink->file);
  }
  iree_string_builder_deinitialize(&sink->row_builder);
  memset(sink, 0, sizeof(*sink));
}

iree_string_builder_t* iree_benchmark_loom_jsonl_sink_begin(
    iree_benchmark_loom_jsonl_sink_t* sink) {
  iree_string_builder_reset(&sink->row_builder);
  return &sink->row_builder;
}

static iree_status_t iree_benchmark_loom_jsonl_sink_flush(
    iree_benchmark_loom_jsonl_sink_t* sink) {
  iree_string_view_t contents = iree_string_builder_view(&sink->row_builder);
  if (iree_string_view_is_empty(contents)) {
    return iree_ok_status();
  }
  const size_t written = fwrite(contents.data, 1, contents.size, sink->file);
  if (written != contents.size) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to write %" PRIhsz
                            " bytes to JSONL output: %s",
                            contents.size, strerror(errno));
  }
  if (fflush(sink->file) != 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to flush JSONL output: %s",
                            strerror(errno));
  }
  iree_string_builder_reset(&sink->row_builder);
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_jsonl_sink_close(
    iree_benchmark_loom_jsonl_sink_t* sink) {
  if (!sink->owns_file || sink->file == NULL) {
    return iree_ok_status();
  }
  if (fclose(sink->file) != 0) {
    sink->file = NULL;
    sink->owns_file = false;
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to close JSONL output: %s",
                            strerror(errno));
  }
  sink->file = NULL;
  sink->owns_file = false;
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_jsonl_sink_end(
    iree_benchmark_loom_jsonl_sink_t* sink, iree_status_t row_status) {
  if (!iree_status_is_ok(row_status)) {
    iree_string_builder_reset(&sink->row_builder);
    return row_status;
  }
  return iree_benchmark_loom_jsonl_sink_flush(sink);
}

static iree_status_t iree_benchmark_loom_join_bundle_path(
    iree_string_view_t bundle_dir, iree_string_view_t child,
    iree_allocator_t allocator, char** out_path) {
  return loom_tooling_file_path_join(bundle_dir, child, allocator, out_path);
}

iree_status_t iree_benchmark_loom_artifact_bundle_initialize(
    const iree_benchmark_loom_artifact_bundle_options_t* options,
    iree_allocator_t allocator,
    iree_benchmark_loom_artifact_bundle_t* out_bundle) {
  IREE_ASSERT_ARGUMENT(options);
  memset(out_bundle, 0, sizeof(*out_bundle));
  out_bundle->host_allocator = allocator;
  out_bundle->policy = options->policy;

  const iree_string_view_t bundle_dir = iree_string_view_trim(options->dir);
  if (iree_string_view_is_empty(bundle_dir)) {
    out_bundle->policy = IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_NONE;
    return iree_ok_status();
  }
  if (loom_tooling_file_path_is_stdio(bundle_dir)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--artifact-bundle-dir must name a directory");
  }
  if (out_bundle->policy == IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--artifact-bundle-policy=none conflicts with "
                            "--artifact-bundle-dir");
  }

  iree_status_t status = iree_benchmark_loom_dup_string_view(
      bundle_dir, allocator, &out_bundle->dir_storage);
  if (iree_status_is_ok(status)) {
    out_bundle->dir = iree_make_cstring_view(out_bundle->dir_storage);
    status =
        loom_tooling_create_directory_if_needed(out_bundle->dir, allocator);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t results_leaf = IREE_SV("results.json");
    if (out_bundle->policy != IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_NONE &&
        options->output_format == IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_JSONL) {
      results_leaf = IREE_SV("results.jsonl");
    }
    status = iree_benchmark_loom_join_bundle_path(
        out_bundle->dir, results_leaf, allocator,
        &out_bundle->results_path_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("manifest.json"), allocator,
        &out_bundle->manifest_path_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("outputs"), allocator,
        &out_bundle->file_output_dir_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("profiles"), allocator,
        &out_bundle->profile_artifacts_dir_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("compile_reports"), allocator,
        &out_bundle->compile_report_dir_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("artifact_manifests"), allocator,
        &out_bundle->artifact_manifest_dir_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("target_artifacts"), allocator,
        &out_bundle->target_artifact_dir_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("target_listings"), allocator,
        &out_bundle->target_listing_dir_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("hal_executables"), allocator,
        &out_bundle->hal_executable_dir_storage);
  }
  if (iree_status_is_ok(status)) {
    out_bundle->results_path =
        iree_make_cstring_view(out_bundle->results_path_storage);
    out_bundle->manifest_path =
        iree_make_cstring_view(out_bundle->manifest_path_storage);
    out_bundle->file_output_dir =
        iree_make_cstring_view(out_bundle->file_output_dir_storage);
    out_bundle->profile_artifacts_dir =
        iree_make_cstring_view(out_bundle->profile_artifacts_dir_storage);
    out_bundle->compile_report_dir =
        iree_make_cstring_view(out_bundle->compile_report_dir_storage);
    out_bundle->artifact_manifest_dir =
        iree_make_cstring_view(out_bundle->artifact_manifest_dir_storage);
    out_bundle->target_artifact_dir =
        iree_make_cstring_view(out_bundle->target_artifact_dir_storage);
    out_bundle->target_listing_dir =
        iree_make_cstring_view(out_bundle->target_listing_dir_storage);
    out_bundle->hal_executable_dir =
        iree_make_cstring_view(out_bundle->hal_executable_dir_storage);
    out_bundle->enabled = true;
  } else {
    iree_benchmark_loom_artifact_bundle_deinitialize(out_bundle);
  }
  return status;
}

void iree_benchmark_loom_artifact_bundle_deinitialize(
    iree_benchmark_loom_artifact_bundle_t* bundle) {
  if (bundle == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < bundle->file_entry_count; ++i) {
    iree_allocator_free(bundle->host_allocator, bundle->file_entries[i].path);
  }
  iree_allocator_free(bundle->host_allocator, bundle->file_entries);
  iree_allocator_free(bundle->host_allocator,
                      bundle->hal_executable_dir_storage);
  iree_allocator_free(bundle->host_allocator,
                      bundle->target_artifact_dir_storage);
  iree_allocator_free(bundle->host_allocator,
                      bundle->target_listing_dir_storage);
  iree_allocator_free(bundle->host_allocator,
                      bundle->artifact_manifest_dir_storage);
  iree_allocator_free(bundle->host_allocator,
                      bundle->compile_report_dir_storage);
  iree_allocator_free(bundle->host_allocator,
                      bundle->profile_artifacts_dir_storage);
  iree_allocator_free(bundle->host_allocator, bundle->file_output_dir_storage);
  iree_allocator_free(bundle->host_allocator, bundle->manifest_path_storage);
  iree_allocator_free(bundle->host_allocator, bundle->results_path_storage);
  iree_allocator_free(bundle->host_allocator, bundle->dir_storage);
  *bundle = (iree_benchmark_loom_artifact_bundle_t){0};
}

bool iree_benchmark_loom_artifact_bundle_wants_debug_artifacts(
    const iree_benchmark_loom_artifact_bundle_t* bundle) {
  return bundle != NULL && bundle->enabled &&
         bundle->policy >= IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG;
}

bool iree_benchmark_loom_artifact_bundle_wants_compile_reports(
    const iree_benchmark_loom_artifact_bundle_t* bundle) {
  return iree_benchmark_loom_artifact_bundle_wants_debug_artifacts(bundle);
}

bool iree_benchmark_loom_artifact_bundle_wants_artifact_manifests(
    const iree_benchmark_loom_artifact_bundle_t* bundle) {
  return iree_benchmark_loom_artifact_bundle_wants_debug_artifacts(bundle);
}

iree_string_view_t iree_benchmark_loom_effective_results_output_path(
    iree_string_view_t explicit_output,
    const iree_benchmark_loom_artifact_bundle_t* bundle) {
  explicit_output = iree_string_view_trim(explicit_output);
  if (!iree_string_view_is_empty(explicit_output)) {
    return explicit_output;
  }
  if (bundle->enabled) {
    return bundle->results_path;
  }
  return iree_string_view_empty();
}

iree_string_view_t iree_benchmark_loom_effective_profile_artifacts_dir(
    iree_string_view_t explicit_artifacts_dir,
    const iree_benchmark_loom_artifact_bundle_t* bundle) {
  explicit_artifacts_dir = iree_string_view_trim(explicit_artifacts_dir);
  if (!iree_string_view_is_empty(explicit_artifacts_dir)) {
    return explicit_artifacts_dir;
  }
  return bundle->enabled ? bundle->profile_artifacts_dir
                         : iree_string_view_empty();
}

iree_status_t iree_benchmark_loom_append_effective_profile_artifacts_dir(
    const iree_benchmark_loom_run_identity_t* run,
    iree_hal_device_profiling_data_families_t profile_data_families,
    iree_string_builder_t* artifact_dir) {
  if (!iree_string_view_is_empty(run->profile_artifacts_dir)) {
    return iree_string_builder_append_string(artifact_dir,
                                             run->profile_artifacts_dir);
  }
  if (!iree_benchmark_loom_profile_data_needs_artifact_data(
          profile_data_families)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(artifact_dir, run->file_output_dir));
  if (!iree_string_view_ends_with(run->file_output_dir, IREE_SV("/"))) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(artifact_dir, "/"));
  }
  return iree_string_builder_append_cstring(artifact_dir, "profiles");
}

iree_status_t iree_benchmark_loom_open_file_for_read(
    void* user_data, iree_string_view_t path, iree_io_stream_t** out_stream) {
  *out_stream = NULL;
  iree_benchmark_loom_file_provider_t* provider =
      (iree_benchmark_loom_file_provider_t*)user_data;
  char* resolved_path = NULL;
  IREE_RETURN_IF_ERROR(loom_tooling_resolve_file_read_path(
      provider->input_dir, path, provider->host_allocator, &resolved_path));
  iree_io_stream_t* stream = NULL;
  iree_status_t status = iree_io_stdio_stream_open(
      IREE_IO_STDIO_STREAM_MODE_READ, iree_make_cstring_view(resolved_path),
      provider->host_allocator, &stream);
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_artifact_bundle_record_file(
        provider->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_FIXTURE_READ,
        iree_make_cstring_view(resolved_path));
  }
  iree_allocator_free(provider->host_allocator, resolved_path);
  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    iree_io_stream_release(stream);
  }
  return status;
}

iree_status_t iree_benchmark_loom_open_file_for_write(
    void* user_data, iree_string_view_t path, iree_io_stream_t** out_stream) {
  *out_stream = NULL;
  iree_benchmark_loom_file_provider_t* provider =
      (iree_benchmark_loom_file_provider_t*)user_data;
  char* resolved_path = NULL;
  IREE_RETURN_IF_ERROR(loom_tooling_resolve_file_write_path(
      provider->output_dir, path, provider->host_allocator, &resolved_path));
  iree_string_view_t resolved_path_view = iree_make_cstring_view(resolved_path);
  iree_io_stream_t* stream = NULL;
  iree_status_t status = loom_tooling_create_parent_directory(
      resolved_path_view, provider->host_allocator);
  if (iree_status_is_ok(status)) {
    status = iree_io_stdio_stream_open(
        IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_DISCARD,
        resolved_path_view, provider->host_allocator, &stream);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_artifact_bundle_record_file(
        provider->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_OUTPUT,
        resolved_path_view);
  }
  iree_allocator_free(provider->host_allocator, resolved_path);
  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    iree_io_stream_release(stream);
  }
  return status;
}
