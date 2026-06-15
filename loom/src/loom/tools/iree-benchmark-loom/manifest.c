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
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"environment\":{"));
  bool first_field = true;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_benchmark_loom_manifest_environment_variables);
       ++i) {
    const char* name = iree_benchmark_loom_manifest_environment_variables[i];
    const char* value = getenv(name);
    if (value == NULL || value[0] == '\0') {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
        stream, &first_field, name, iree_make_cstring_view(value)));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_file_stat_error_json(
    int error_number, loom_output_stream_t* stream) {
  const iree_status_code_t code = iree_status_code_from_errno(error_number);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "status", IREE_SV("stat_failed")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u32_field(
      stream, &first_field, "code", (uint32_t)code));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "status_string",
      iree_make_cstring_view(iree_status_code_string(code))));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "message",
      iree_make_cstring_view(strerror(error_number))));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_file_identity_json(
    iree_string_view_t path, iree_allocator_t allocator,
    loom_output_stream_t* stream) {
  if (iree_string_view_equal(path, IREE_SV("<stdin>"))) {
    return loom_output_stream_write_cstring(stream, "{\"status\":\"stdin\"}");
  }
  if (loom_tooling_file_path_is_stdio(path)) {
    return loom_output_stream_write_cstring(stream, "{\"status\":\"stdio\"}");
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

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "status",
      is_regular_file ? IREE_SV("ok") : IREE_SV("not_regular")));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_u64_field(
      stream, &first_field, "byte_count", (uint64_t)file_stat.st_size));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "modified_time"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"unix_seconds\":%" PRIi64 ",\"nanoseconds\":%" PRIi32 "}",
      modified_time_seconds, modified_time_nanoseconds));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_manifest_file_reference_json(
    iree_string_view_t path, iree_allocator_t allocator,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_string_field(
      stream, &first_field, "path", path));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_json_object_field_name(
      stream, &first_field, "identity"));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_file_identity_json(path, allocator, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_write_manifest_file_array_json(
    const iree_benchmark_loom_artifact_bundle_t* bundle,
    iree_benchmark_loom_bundle_file_kind_t kind, iree_allocator_t allocator,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  bool first_entry = true;
  for (iree_host_size_t i = 0; i < bundle->file_entry_count; ++i) {
    const iree_benchmark_loom_bundle_file_entry_t* entry =
        &bundle->file_entries[i];
    if (entry->kind != kind) {
      continue;
    }
    if (!first_entry) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    first_entry = false;
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_reference_json(
        iree_make_cstring_view(entry->path), allocator, stream));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t iree_benchmark_loom_write_manifest_files_json(
    const iree_benchmark_loom_artifact_bundle_t* bundle,
    const iree_benchmark_loom_run_identity_t* run, iree_allocator_t allocator,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"files\":{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"results\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_reference_json(
      run->results_path, allocator, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"fixture_reads\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_FIXTURE_READ, allocator, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"file_outputs\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_OUTPUT, allocator, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"profiles\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_PROFILE, allocator, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"compile_reports\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_COMPILE_REPORT, allocator,
      stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"artifact_manifests\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_ARTIFACT_MANIFEST, allocator,
      stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"target_artifacts\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_ARTIFACT, allocator,
      stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"target_listings\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_TARGET_LISTING, allocator,
      stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"hal_executables\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_array_json(
      bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_HAL_EXECUTABLE, allocator,
      stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_benchmark_loom_append_artifact_bundle_manifest_json(
    const iree_benchmark_loom_artifact_bundle_t* bundle,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_hal_context_t* hal_context,
    iree_string_view_t source_text, iree_string_view_t command_line_json,
    bool dry_run,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode,
    iree_allocator_t allocator, iree_string_builder_t* manifest) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(manifest, &stream);

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "\"tool\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_cstring(&stream, "iree-benchmark-loom"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"run_id\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, run->run_id));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"source\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, run->source));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"source_identity\":{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "\"byte_count\":"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(&stream, "%" PRIhsz, source_text.size));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"file\":"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_file_reference_json(
      run->source, allocator, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"policy\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream,
      iree_benchmark_loom_artifact_bundle_policy_name(bundle->policy)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"dry_run\":"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, dry_run ? "true" : "false"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"sample_compilation\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_benchmark_loom_sample_compilation_mode_name(
                   sample_compilation_mode)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"paths\":{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "\"bundle\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, bundle->dir));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"results\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, run->results_path));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"manifest\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, bundle->manifest_path));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"file_outputs\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, run->file_output_dir));
  if (!iree_string_view_is_empty(run->profile_artifacts_dir)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"profiles\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, run->profile_artifacts_dir));
  }
  if (iree_benchmark_loom_artifact_bundle_wants_compile_reports(bundle)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"compile_reports\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, bundle->compile_report_dir));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"artifact_manifests\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, bundle->artifact_manifest_dir));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"target_artifacts\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, bundle->target_artifact_dir));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"target_listings\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, bundle->target_listing_dir));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"hal_executables\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, bundle->hal_executable_dir));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_write_manifest_files_json(
      bundle, run, allocator, &stream));
  if (hal_context->execution.artifact_provider != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"device\":{"));
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_write_hal_context_identity_fields_json(hal_context,
                                                                   &stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"command_line\":"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, command_line_json));
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_write_manifest_environment_json(&stream));
  char cwd[4096] = {0};
#if defined(IREE_PLATFORM_WINDOWS)
  char* cwd_result = _getcwd(cwd, sizeof(cwd));
#else
  char* cwd_result = getcwd(cwd, sizeof(cwd));
#endif  // defined(IREE_PLATFORM_WINDOWS)
  if (cwd_result != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"cwd\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(&stream, cwd));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_write_artifact_bundle_manifest(
    const iree_benchmark_loom_artifact_bundle_t* bundle,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_hal_context_t* hal_context,
    iree_string_view_t source_text, iree_string_view_t command_line_json,
    bool dry_run,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode,
    iree_allocator_t allocator) {
  if (!bundle->enabled) {
    return iree_ok_status();
  }

  iree_string_builder_t manifest;
  iree_string_builder_initialize(allocator, &manifest);
  iree_status_t status =
      iree_benchmark_loom_append_artifact_bundle_manifest_json(
          bundle, run, hal_context, source_text, command_line_json, dry_run,
          sample_compilation_mode, allocator, &manifest);
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
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "["));
  for (int i = 0; i < argc; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(&stream, argv[i]));
  }
  return loom_output_stream_write_cstring(&stream, "]");
}
