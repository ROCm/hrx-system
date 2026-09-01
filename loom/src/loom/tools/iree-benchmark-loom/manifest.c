// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/manifest.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#if defined(IREE_PLATFORM_WINDOWS)
#include <direct.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif  // defined(IREE_PLATFORM_WINDOWS)

#include "loom/tooling/io/file.h"
#include "loom/tools/iree-benchmark-loom/output.h"
#include "loom/tools/iree-benchmark-loom/report.h"
#include "loom/util/json.h"

static const char* const iree_benchmark_loom_manifest_environment_variables[] =
    {
        "TMPDIR",
        "TEMP",
        "TMP",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "ONEAPI_DEVICE_SELECTOR",
        "GPU_DEVICE_ORDINAL",
        "HSA_OVERRIDE_GFX_VERSION",
        "HSA_ENABLE_SDMA",
        "HSA_TOOLS_LIB",
        "IREE_TRACY_CAPTURE",
};

static iree_status_t iree_benchmark_loom_write_manifest_environment_json(
    loom_json_object_writer_t* parent_object) {
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(parent_object, IREE_SV("environment")));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(parent_object->stream, &object));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_benchmark_loom_manifest_environment_variables);
       ++i) {
    const char* name = iree_benchmark_loom_manifest_environment_variables[i];
    const char* value = getenv(name);
    if (value == NULL || value[0] == '\0') {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, iree_make_cstring_view(name), iree_make_cstring_view(value)));
  }
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_file_stat_error_json(
    int error_number, loom_output_stream_t* stream) {
  const iree_status_code_t code = iree_status_code_from_errno(error_number);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("state"), IREE_SV("stat_failed")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_status_field_json(
      code, iree_make_cstring_view(strerror(error_number)), &object));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_file_identity_json(
    iree_string_view_t path, iree_allocator_t allocator,
    loom_output_stream_t* stream) {
  if (iree_string_view_equal(path, IREE_SV("<stdin>"))) {
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("state"), IREE_SV("stdin")));
    return loom_json_object_end(&object);
  }
  if (loom_tooling_file_path_is_stdio(path)) {
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("state"), IREE_SV("stdio")));
    return loom_json_object_end(&object);
  }

  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_dup_string_view(path, allocator, &storage));
#if defined(IREE_PLATFORM_WINDOWS)
  struct _stat64 file_stat = {0};
  const int stat_result = _stat64(storage, &file_stat);
  const bool is_regular_file =
      stat_result == 0 && (file_stat.st_mode & _S_IFMT) == _S_IFREG;
  const int64_t modified_time_seconds = (int64_t)file_stat.st_mtime;
  const int32_t modified_time_nanoseconds = 0;
#else
  struct stat file_stat = {0};
  const int stat_result = stat(storage, &file_stat);
  const bool is_regular_file = stat_result == 0 && S_ISREG(file_stat.st_mode);
#if defined(IREE_PLATFORM_APPLE)
  const int64_t modified_time_seconds = (int64_t)file_stat.st_mtimespec.tv_sec;
  const int32_t modified_time_nanoseconds =
      (int32_t)file_stat.st_mtimespec.tv_nsec;
#else
  const int64_t modified_time_seconds = (int64_t)file_stat.st_mtim.tv_sec;
  const int32_t modified_time_nanoseconds = (int32_t)file_stat.st_mtim.tv_nsec;
#endif  // defined(IREE_PLATFORM_APPLE)
#endif  // defined(IREE_PLATFORM_WINDOWS)
  const int stat_error_number = errno;
  iree_allocator_free(allocator, storage);

  if (stat_result != 0) {
    return iree_benchmark_loom_write_file_stat_error_json(stat_error_number,
                                                          stream);
  }

  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("state"),
      is_regular_file ? IREE_SV("ok") : IREE_SV("not_regular")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("byte_count"), (uint64_t)file_stat.st_size));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("modified_time")));
  loom_json_object_writer_t modified_time_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &modified_time_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int64_field(
      &modified_time_object, IREE_SV("unix_seconds"), modified_time_seconds));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int32_field(
      &modified_time_object, IREE_SV("nanoseconds"),
      modified_time_nanoseconds));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&modified_time_object));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_manifest_file_reference_json(
    iree_string_view_t path, iree_allocator_t allocator,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("path"), path));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("identity")));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_file_identity_json(path, allocator, stream));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_write_manifest_file_array_json(
    const iree_benchmark_loom_artifact_bundle_t* bundle,
    iree_benchmark_loom_bundle_file_kind_t kind, iree_allocator_t allocator,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < bundle->file_entry_count; ++i) {
    const iree_benchmark_loom_bundle_file_entry_t* entry =
        &bundle->file_entries[i];
    if (entry->kind != kind) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_reference_json(
        iree_make_cstring_view(entry->path), allocator, stream));
  }
  return loom_json_array_end(&array);
}

static iree_status_t iree_benchmark_loom_write_manifest_files_json(
    const iree_benchmark_loom_artifact_bundle_t* bundle,
    const iree_benchmark_loom_run_identity_t* run, iree_allocator_t allocator,
    loom_json_object_writer_t* parent_object) {
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(parent_object, IREE_SV("files")));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(parent_object->stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("results")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_reference_json(
      run->results_path, allocator, object.stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("fixture_reads")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_FIXTURE_READ, allocator,
      object.stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("file_outputs")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_OUTPUT, allocator,
      object.stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("profiles")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_PROFILE, allocator,
      object.stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("compile_reports")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_COMPILE_REPORT, allocator,
      object.stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("artifact_manifests")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_ARTIFACT_MANIFEST, allocator,
      object.stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("target_artifacts")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_ARTIFACT, allocator,
      object.stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("target_listings")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_LISTING, allocator,
      object.stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("hal_executables")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_HAL_EXECUTABLE, allocator,
      object.stream));
  return loom_json_object_end(&object);
}

static iree_status_t iree_benchmark_loom_append_artifact_bundle_manifest_json(
    const iree_benchmark_loom_artifact_bundle_t* bundle,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_hal_context_t* hal_context,
    iree_string_view_t source_text, iree_string_view_t command_line_json,
    bool dry_run, iree_allocator_t allocator, iree_string_builder_t* manifest) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(manifest, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("tool"), IREE_SV("iree-benchmark-loom")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("run_id"), run->run_id));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("source"), run->source));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("source_identity")));
  loom_json_object_writer_t source_identity_object;
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin(&stream, &source_identity_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &source_identity_object, IREE_SV("byte_count"), source_text.size));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&source_identity_object, IREE_SV("file")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_reference_json(
      run->source, allocator, &stream));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&source_identity_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("policy"),
      iree_benchmark_loom_artifact_bundle_policy_name(bundle->policy)));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_bool_field(&object, IREE_SV("dry_run"), dry_run));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("paths")));
  loom_json_object_writer_t paths_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &paths_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &paths_object, IREE_SV("bundle"), bundle->dir));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &paths_object, IREE_SV("results"), run->results_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &paths_object, IREE_SV("manifest"), bundle->manifest_path));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &paths_object, IREE_SV("file_outputs"), run->file_output_dir));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
      &paths_object, IREE_SV("profiles"), run->profile_artifacts_dir));
  if (iree_benchmark_loom_artifact_bundle_wants_compile_reports(bundle)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &paths_object, IREE_SV("compile_reports"), bundle->compile_report_dir));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &paths_object, IREE_SV("artifact_manifests"),
        bundle->artifact_manifest_dir));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &paths_object, IREE_SV("target_artifacts"),
        bundle->target_artifact_dir));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &paths_object, IREE_SV("target_listings"), bundle->target_listing_dir));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &paths_object, IREE_SV("hal_executables"), bundle->hal_executable_dir));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&paths_object));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_files_json(
      bundle, run, allocator, &object));
  if (hal_context->execution.device_provider != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("device")));
    loom_json_object_writer_t device_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &device_object));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_hal_context_identity_fields_json(
            hal_context, &device_object));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&device_object));
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("command_line")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, command_line_json));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_manifest_environment_json(&object));
  char cwd[4096] = {0};
#if defined(IREE_PLATFORM_WINDOWS)
  char* cwd_result = _getcwd(cwd, sizeof(cwd));
#else
  char* cwd_result = getcwd(cwd, sizeof(cwd));
#endif  // defined(IREE_PLATFORM_WINDOWS)
  if (cwd_result != NULL) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("cwd"), iree_make_cstring_view(cwd)));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  return loom_output_stream_write_char(&stream, '\n');
}

iree_status_t iree_benchmark_loom_write_artifact_bundle_manifest(
    const iree_benchmark_loom_artifact_bundle_t* bundle,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_hal_context_t* hal_context,
    iree_string_view_t source_text, iree_string_view_t command_line_json,
    bool dry_run, iree_allocator_t allocator) {
  if (!bundle->enabled) {
    return iree_ok_status();
  }

  iree_string_builder_t manifest;
  iree_string_builder_initialize(allocator, &manifest);
  iree_status_t status =
      iree_benchmark_loom_append_artifact_bundle_manifest_json(
          bundle, run, hal_context, source_text, command_line_json, dry_run,
          allocator, &manifest);
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        bundle->manifest_path, iree_string_builder_view(&manifest), allocator);
  }
  iree_string_builder_deinitialize(&manifest);
  return status;
}

iree_status_t iree_benchmark_loom_append_command_line_json(
    int argc, char** argv, iree_string_builder_t* output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &array));
  for (int i = 0; i < argc; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
        &array, iree_make_cstring_view(argv[i])));
  }
  return loom_json_array_end(&array);
}
