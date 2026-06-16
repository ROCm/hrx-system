// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/options.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/tooling/flags.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/tools/iree-benchmark-loom/module_query.h"
#include "loom/util/json.h"

IREE_FLAG(string, case, "",
          "Optional check.case symbol to benchmark, such as '@smoke'. Empty "
          "keeps all cases referenced by selected benchmarks.");
IREE_FLAG(string, benchmark, "",
          "Optional check.benchmark name to execute, such as '@smoke_time'. "
          "Empty executes all benchmarks in source order.");
IREE_FLAG(int32_t, sample, -1,
          "Optional concrete sample ordinal to execute for selected benchmark "
          "cases. Negative executes all planned samples.");
IREE_FLAG(string, measure, "case_end_to_end",
          "Measurement mode. Use 'case_end_to_end', 'end_to_end', or "
          "'dispatch_complete'.");
IREE_FLAG_NAMED(int32_t, max_samples_per_case, "max-samples-per-case",
                LOOM_TESTBENCH_DEFAULT_MAX_SAMPLES_PER_CASE,
                "Maximum number of samples planned per check.case.");
IREE_FLAG(string, pipeline, "default",
          "Pass pipeline used before HAL candidate emission. Use 'default', "
          "'none', '@symbol', or a comma-separated pass list.");
IREE_FLAG(string, output, "",
          "Output path for benchmark results. Empty or '-' writes to stdout.");
IREE_FLAG_NAMED(
    string, output_format, "output-format", "snapshot",
    "Benchmark result output format. Use 'snapshot' for one compact JSON "
    "document or 'jsonl' for newline-delimited lifecycle events.");
IREE_FLAG_NAMED(string, file_output_dir, "file-output-dir", "",
                "Directory receiving check.file.write.* outputs. Empty uses "
                "$TMPDIR/iree-loom-benchmark/<source>_<hash>/ for this run.");
IREE_FLAG_NAMED(
    string, artifact_bundle_dir, "artifact-bundle-dir", "",
    "Directory receiving a self-contained run bundle. When set and --output "
    "is empty, results are written inside the bundle; check.file.write outputs "
    "and profile artifacts default to bundle subdirectories unless their "
    "explicit flags are set.");
IREE_FLAG_NAMED(string, artifact_bundle_policy, "artifact-bundle-policy",
                "minimal",
                "Artifact bundle policy when --artifact-bundle-dir is set. "
                "Use 'minimal', 'debug', or 'full'.");
IREE_FLAG_NAMED(
    bool, dry_run, "dry-run", false,
    "Reports selected logical benchmarks and deduplicated physical work items "
    "without running correctness, compilation, or measurement.");
IREE_FLAG(bool, agents_md, false,
          "Prints a compact Markdown snippet suitable for AGENTS.md and "
          "exits.");
IREE_FLAG_NAMED(string, compile_report, "compile-report", "summary",
                "Structured compile_report payload embedded in compile rows "
                "and benchmark results. Use 'summary', 'details', or "
                "empty/'none'.");
IREE_FLAG_NAMED(
    string, artifact_manifest, "artifact-manifest", "none",
    "Target-neutral artifact manifest sidecar mode for debug/full artifact "
    "bundles. Use 'summary', 'details', 'analysis', or empty/'none'.");
IREE_FLAG_NAMED(
    string, profile_data, "profile-data", "",
    "HAL profiling data families for the final profiled batch as a "
    "comma-separated list. Empty uses dispatch-events,executable-metadata. "
    "Accepted families match --device_profiling_mode: queue-events, "
    "host-execution, device-queue-events, dispatch-events, memory-events, "
    "device-metrics, command-region-events, counters, counter-ranges, "
    "executable-metadata, executable-traces.");
IREE_FLAG_LIST_NAMED(
    string, profile_counter, "profile-counter",
    "Implementation-specific hardware counter name to request during the final "
    "profiled batch. May be repeated and requires --profile-data to include "
    "counters or counter-ranges.");
IREE_FLAG_NAMED(
    string, profile_artifacts_dir, "profile-artifacts-dir", "",
    "Directory receiving raw IREE HAL profile bundles from final profiled "
    "batches. Setting this implies --profile-final-batch=true unless that flag "
    "was explicitly set false.");
IREE_FLAG_NAMED(string, sample_compilation, "sample-compilation", "once",
                "Sample compilation mode for dispatch_complete benchmarks. "
                "Use 'once' to compile once and pass parameter values at "
                "dispatch time, 'per_sample' to compile each selected sample "
                "with concrete parameter facts, or 'both' to emit both result "
                "sets.");
IREE_FLAG_NAMED(
    int64_t, input_ring_min_bytes, "input-ring-min-bytes",
    IREE_BENCHMARK_LOOM_DEFAULT_INPUT_RING_MIN_BYTES,
    "Minimum total byte size of the device-buffer binding ring used by "
    "dispatch_complete benchmarks. The auto ring count is max(batch-size, "
    "ceil(min bytes / bytes per binding set)); use 0 to record one "
    "hot-reuse binding set.");
IREE_FLAG_NAMED(
    int32_t, input_ring_count, "input-ring-count", 0,
    "Exact number of physical device-buffer binding sets to rotate through "
    "dispatch_complete command buffers. Zero uses --input-ring-min-bytes. Use "
    "1 to force hot-reuse measurements.");
IREE_FLAG(string, compare, "",
          "Comma-separated check.benchmark names to compare in one "
          "interleaved dispatch_complete run, such as '@base,@variant'. "
          "Cannot be combined with --benchmark or --case.");
IREE_FLAG(string, interleave, "ABABA",
          "Interleaving schedule used with --compare. Use 'ABABA' for "
          "baseline-anchored pairwise windows or 'round_robin' for ABCD-style "
          "candidate rotation.");
IREE_FLAG(int32_t, repetitions, 2,
          "Interleaved comparison repetitions. With --interleave=ABABA this "
          "runs A then this many BA pairs; the default produces ABABA.");

static iree_benchmark_loom_i32_flag_t FLAG_iterations = {.value = 10};
IREE_FLAG_CALLBACK(iree_benchmark_loom_parse_i32_flag,
                   iree_benchmark_loom_print_i32_flag, &FLAG_iterations,
                   iterations, "Measured iterations.");
static iree_benchmark_loom_i32_flag_t FLAG_warmup_iterations = {.value = 1};
IREE_FLAG_CALLBACK_NAMED(iree_benchmark_loom_parse_i32_flag,
                         iree_benchmark_loom_print_i32_flag,
                         &FLAG_warmup_iterations, warmup_iterations,
                         "warmup-iterations", "Warmup iterations.");
static iree_benchmark_loom_i32_flag_t FLAG_batch_size = {.value = 1};
IREE_FLAG_CALLBACK_NAMED(
    iree_benchmark_loom_parse_i32_flag, iree_benchmark_loom_print_i32_flag,
    &FLAG_batch_size, batch_size, "batch-size",
    "Number of repeated dispatches recorded into each measured HAL command "
    "buffer batch.");
static iree_benchmark_loom_i32_flag_t FLAG_min_time_ms = {.value = 100};
IREE_FLAG_CALLBACK_NAMED(
    iree_benchmark_loom_parse_i32_flag, iree_benchmark_loom_print_i32_flag,
    &FLAG_min_time_ms, min_time_ms, "min-time-ms",
    "Minimum measured duration for dispatch_complete benchmarks.");
static iree_benchmark_loom_i32_flag_t FLAG_warmup_time_ms = {.value = 0};
IREE_FLAG_CALLBACK_NAMED(
    iree_benchmark_loom_parse_i32_flag, iree_benchmark_loom_print_i32_flag,
    &FLAG_warmup_time_ms, warmup_time_ms, "warmup-time-ms",
    "Minimum warmup duration for dispatch_complete benchmarks.");
static iree_benchmark_loom_i32_flag_t FLAG_max_batches = {.value = 1000};
IREE_FLAG_CALLBACK_NAMED(iree_benchmark_loom_parse_i32_flag,
                         iree_benchmark_loom_print_i32_flag, &FLAG_max_batches,
                         max_batches, "max-batches",
                         "Maximum measured command-buffer batches for "
                         "dispatch_complete benchmarks.");
static iree_benchmark_loom_i32_flag_t FLAG_stable_p90_to_p50_ppm = {
    .value = 100000,
};
IREE_FLAG_CALLBACK_NAMED(
    iree_benchmark_loom_parse_i32_flag, iree_benchmark_loom_print_i32_flag,
    &FLAG_stable_p90_to_p50_ppm, stable_p90_to_p50_ppm, "stable-p90-to-p50-ppm",
    "p90-to-p50 spread threshold in parts per million. Zero stops after the "
    "minimum count and duration are reached.");
static iree_benchmark_loom_bool_flag_t FLAG_profile_final_batch = {.value =
                                                                       false};
IREE_FLAG_CALLBACK_NAMED(
    iree_benchmark_loom_parse_bool_flag, iree_benchmark_loom_print_bool_flag,
    &FLAG_profile_final_batch, profile_final_batch, "profile-final-batch",
    "Runs one final profiled HAL command-buffer batch after measured "
    "dispatch_complete timing.");

void iree_benchmark_loom_options_initialize(
    iree_benchmark_loom_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->sample_ordinal = -1;
  out_options->max_samples_per_case =
      LOOM_TESTBENCH_DEFAULT_MAX_SAMPLES_PER_CASE;
  out_options->pipeline = IREE_SV("default");
  out_options->output_format = IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_SNAPSHOT;
  out_options->artifact_bundle_policy =
      IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_MINIMAL;
  out_options->measure = IREE_SV("case_end_to_end");
  out_options->compile_report = IREE_SV("summary");
  out_options->artifact_manifest = IREE_SV("none");
  out_options->sample_compilation_mode =
      IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE;
  out_options->input_ring_min_bytes =
      IREE_BENCHMARK_LOOM_DEFAULT_INPUT_RING_MIN_BYTES;
  out_options->interleave_mode = IREE_BENCHMARK_LOOM_INTERLEAVE_NONE;
  out_options->repetitions = 2;
  out_options->iterations = 10;
  out_options->warmup_iterations = 1;
  out_options->batch_size = 1;
  out_options->min_time_ms = 100;
  out_options->max_batches = 1000;
  out_options->stable_p90_to_p50_ppm = 100000;
}

bool iree_benchmark_loom_cli_flags_request_agents_md(void) {
  return FLAG_agents_md;
}

static iree_status_t iree_benchmark_loom_positive_i32_to_host_size(
    const char* flag_name, int32_t value, iree_host_size_t* out_value) {
  if (value <= 0 || (uint64_t)value > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--%s must be positive; got %d", flag_name,
                            (int)value);
  }
  *out_value = (iree_host_size_t)value;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_non_negative_i32_to_host_size(
    const char* flag_name, int32_t value, iree_host_size_t* out_value) {
  if (value < 0 || (uint64_t)value > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--%s must be non-negative; got %d", flag_name,
                            (int)value);
  }
  *out_value = (iree_host_size_t)value;
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_options_from_flags(
    iree_benchmark_loom_options_t* out_options) {
  iree_benchmark_loom_options_initialize(out_options);
  out_options->selected_case = iree_benchmark_loom_normalize_selection_name(
      iree_make_cstring_view(FLAG_case));
  out_options->selected_benchmark =
      iree_benchmark_loom_normalize_selection_name(
          iree_make_cstring_view(FLAG_benchmark));
  out_options->sample_ordinal = FLAG_sample;
  out_options->pipeline = iree_make_cstring_view(FLAG_pipeline);
  out_options->output = iree_make_cstring_view(FLAG_output);
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_parse_output_format(
      iree_make_cstring_view(FLAG_output_format), &out_options->output_format));
  out_options->file_output_dir = iree_make_cstring_view(FLAG_file_output_dir);
  out_options->artifact_bundle_dir =
      iree_string_view_trim(iree_make_cstring_view(FLAG_artifact_bundle_dir));
  out_options->dry_run = FLAG_dry_run;
  out_options->measure =
      iree_string_view_trim(iree_make_cstring_view(FLAG_measure));
  out_options->compile_report = iree_make_cstring_view(FLAG_compile_report);
  out_options->artifact_manifest =
      iree_string_view_trim(iree_make_cstring_view(FLAG_artifact_manifest));
  out_options->profile_data =
      iree_string_view_trim(iree_make_cstring_view(FLAG_profile_data));
  out_options->profile_artifacts_dir =
      iree_make_cstring_view(FLAG_profile_artifacts_dir);
  out_options->input_ring_min_bytes = FLAG_input_ring_min_bytes;
  out_options->input_ring_min_bytes_specified =
      FLAG_input_ring_min_bytes !=
      IREE_BENCHMARK_LOOM_DEFAULT_INPUT_RING_MIN_BYTES;
  out_options->compare =
      iree_string_view_trim(iree_make_cstring_view(FLAG_compare));
  out_options->profile_final_batch = FLAG_profile_final_batch.value;
  out_options->profile_final_batch_specified =
      FLAG_profile_final_batch.specified;
  out_options->iterations_specified = FLAG_iterations.specified;
  out_options->warmup_iterations_specified = FLAG_warmup_iterations.specified;
  out_options->batch_size_specified = FLAG_batch_size.specified;
  out_options->min_time_ms_specified = FLAG_min_time_ms.specified;
  out_options->warmup_time_ms_specified = FLAG_warmup_time_ms.specified;
  out_options->max_batches_specified = FLAG_max_batches.specified;
  out_options->stable_p90_to_p50_ppm_specified =
      FLAG_stable_p90_to_p50_ppm.specified;
  out_options->input_ring_count_specified = FLAG_input_ring_count != 0;

  IREE_RETURN_IF_ERROR(iree_benchmark_loom_parse_artifact_bundle_policy(
      iree_make_cstring_view(FLAG_artifact_bundle_policy),
      &out_options->artifact_bundle_policy));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_parse_sample_compilation_mode(
      iree_make_cstring_view(FLAG_sample_compilation),
      &out_options->sample_compilation_mode));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_positive_i32_to_host_size(
      "max-samples-per-case", FLAG_max_samples_per_case,
      &out_options->max_samples_per_case));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_positive_i32_to_host_size(
      "iterations", FLAG_iterations.value, &out_options->iterations));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_non_negative_i32_to_host_size(
      "warmup-iterations", FLAG_warmup_iterations.value,
      &out_options->warmup_iterations));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_positive_i32_to_host_size(
      "batch-size", FLAG_batch_size.value, &out_options->batch_size));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_positive_i32_to_host_size(
      "max-batches", FLAG_max_batches.value, &out_options->max_batches));

  if (FLAG_min_time_ms.value < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--min-time-ms must be non-negative; got %d",
                            (int)FLAG_min_time_ms.value);
  }
  out_options->min_time_ms = FLAG_min_time_ms.value;
  if (FLAG_warmup_time_ms.value < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--warmup-time-ms must be non-negative; got %d",
                            (int)FLAG_warmup_time_ms.value);
  }
  out_options->warmup_time_ms = FLAG_warmup_time_ms.value;
  if (FLAG_stable_p90_to_p50_ppm.value < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--stable-p90-to-p50-ppm must be non-negative; got %d",
        (int)FLAG_stable_p90_to_p50_ppm.value);
  }
  out_options->stable_p90_to_p50_ppm =
      (uint64_t)FLAG_stable_p90_to_p50_ppm.value;
  if (FLAG_input_ring_min_bytes < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--input-ring-min-bytes must be non-negative; got %" PRIi64,
        FLAG_input_ring_min_bytes);
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_non_negative_i32_to_host_size(
      "input-ring-count", FLAG_input_ring_count,
      &out_options->input_ring_count));
  const iree_flag_string_list_t profile_counters = FLAG_profile_counter_list();
  out_options->profile_counters = (iree_string_view_list_t){
      .count = profile_counters.count,
      .values = profile_counters.values,
  };
  out_options->profile_data_requested =
      !iree_string_view_is_empty(out_options->profile_data) &&
      !iree_string_view_equal(out_options->profile_data, IREE_SV("none"));

  const bool compare_requested =
      !iree_string_view_is_empty(out_options->compare);
  if (compare_requested) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_parse_interleave_mode(
        iree_make_cstring_view(FLAG_interleave),
        &out_options->interleave_mode));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_positive_i32_to_host_size(
      "repetitions", FLAG_repetitions, &out_options->repetitions));
  if (compare_requested &&
      !iree_string_view_is_empty(out_options->selected_benchmark)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compare selects benchmarks directly and cannot be combined with "
        "--benchmark");
  }
  if (compare_requested &&
      !iree_string_view_is_empty(out_options->selected_case)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compare selects benchmark/case pairs directly and cannot be "
        "combined with --case");
  }
  if (compare_requested && out_options->sample_compilation_mode ==
                               IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_BOTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compare requires one sample-compilation mode; use "
        "--sample-compilation=once or --sample-compilation=per_sample");
  }
  if (iree_string_view_equal(
          iree_string_view_trim(out_options->file_output_dir), IREE_SV("-"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--file-output-dir must name a directory; '-' is reserved for stdout");
  }
  return iree_ok_status();
}

iree_string_view_t iree_benchmark_loom_artifact_bundle_policy_name(
    iree_benchmark_loom_artifact_bundle_policy_t policy) {
  switch (policy) {
    case IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_NONE:
      return IREE_SV("none");
    case IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_MINIMAL:
      return IREE_SV("minimal");
    case IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG:
      return IREE_SV("debug");
    case IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_FULL:
      return IREE_SV("full");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t iree_benchmark_loom_output_format_name(
    iree_benchmark_loom_output_format_t format) {
  switch (format) {
    case IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_SNAPSHOT:
      return IREE_SV("snapshot");
    case IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_JSONL:
      return IREE_SV("jsonl");
    default:
      return IREE_SV("unknown");
  }
}

iree_status_t iree_benchmark_loom_parse_output_format(
    iree_string_view_t value, iree_benchmark_loom_output_format_t* out_format) {
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("snapshot")) ||
      iree_string_view_equal(value, IREE_SV("json"))) {
    *out_format = IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_SNAPSHOT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("jsonl"))) {
    *out_format = IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_JSONL;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--output-format must be one of snapshot or jsonl; got '%.*s'",
      (int)value.size, value.data);
}

iree_status_t iree_benchmark_loom_parse_artifact_bundle_policy(
    iree_string_view_t value,
    iree_benchmark_loom_artifact_bundle_policy_t* out_policy) {
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    *out_policy = IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_NONE;
    return iree_ok_status();
  }
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("minimal"))) {
    *out_policy = IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_MINIMAL;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("debug"))) {
    *out_policy = IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("full"))) {
    *out_policy = IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_FULL;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--artifact-bundle-policy must be one of minimal, debug, full, or none; "
      "got '%.*s'",
      (int)value.size, value.data);
}

iree_status_t iree_benchmark_loom_parse_sample_compilation_mode(
    iree_string_view_t value,
    iree_benchmark_loom_sample_compilation_mode_t* out_mode) {
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(value, IREE_SV("once"))) {
    *out_mode = IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("per_sample"))) {
    *out_mode = IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_PER_SAMPLE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("both"))) {
    *out_mode = IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_BOTH;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--sample-compilation must be one of once, per_sample, or both; got "
      "'%.*s'",
      (int)value.size, value.data);
}

iree_string_view_t iree_benchmark_loom_sample_compilation_mode_name(
    iree_benchmark_loom_sample_compilation_mode_t mode) {
  switch (mode) {
    case IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE:
      return IREE_SV("once");
    case IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_PER_SAMPLE:
      return IREE_SV("per_sample");
    case IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_BOTH:
      return IREE_SV("both");
    default:
      return IREE_SV("unknown");
  }
}

bool iree_benchmark_loom_sample_compilation_runs_once(
    iree_benchmark_loom_sample_compilation_mode_t mode) {
  return mode == IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE ||
         mode == IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_BOTH;
}

bool iree_benchmark_loom_sample_compilation_runs_per_sample(
    iree_benchmark_loom_sample_compilation_mode_t mode) {
  return mode == IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_PER_SAMPLE ||
         mode == IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_BOTH;
}

iree_status_t iree_benchmark_loom_parse_interleave_mode(
    iree_string_view_t value, iree_benchmark_loom_interleave_mode_t* out_mode) {
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("ABABA")) ||
      iree_string_view_equal(value, IREE_SV("ababa"))) {
    *out_mode = IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("round_robin")) ||
      iree_string_view_equal(value, IREE_SV("round-robin"))) {
    *out_mode = IREE_BENCHMARK_LOOM_INTERLEAVE_ROUND_ROBIN;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--interleave must be one of ABABA or round_robin; got '%.*s'",
      (int)value.size, value.data);
}

iree_string_view_t iree_benchmark_loom_interleave_mode_name(
    iree_benchmark_loom_interleave_mode_t mode) {
  switch (mode) {
    case IREE_BENCHMARK_LOOM_INTERLEAVE_NONE:
      return IREE_SV("none");
    case IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA:
      return IREE_SV("ABABA");
    case IREE_BENCHMARK_LOOM_INTERLEAVE_ROUND_ROBIN:
      return IREE_SV("round_robin");
    default:
      return IREE_SV("unknown");
  }
}

typedef struct iree_benchmark_loom_profile_family_name_t {
  // HAL profiling data-family bit represented by these names.
  iree_hal_device_profiling_data_families_t bit;
  // Command-line family name accepted by --profile-data.
  const char* flag_name;
  // Stable JSON string used for this family.
  const char* json_name;
} iree_benchmark_loom_profile_family_name_t;

static const iree_benchmark_loom_profile_family_name_t
    iree_benchmark_loom_profile_family_names[] = {
        {IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS, "queue-events",
         "queue_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS, "host-execution",
         "host_execution_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS,
         "device-queue-events", "device_queue_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS, "dispatch-events",
         "dispatch_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES, "counters",
         "counter_samples"},
        {IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA,
         "executable-metadata", "executable_metadata"},
        {IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES, "executable-traces",
         "executable_traces"},
        {IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS, "memory-events",
         "memory_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_METRICS, "device-metrics",
         "device_metrics"},
        {IREE_HAL_DEVICE_PROFILING_DATA_COMMAND_REGION_EVENTS,
         "command-region-events", "command_region_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES, "counter-ranges",
         "counter_ranges"},
};

bool iree_benchmark_loom_profile_data_has_counter_data(
    iree_hal_device_profiling_data_families_t profile_data_families) {
  return iree_any_bit_set(profile_data_families,
                          IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
                              IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES);
}

bool iree_benchmark_loom_profile_data_needs_artifact_data(
    iree_hal_device_profiling_data_families_t profile_data_families) {
  return iree_benchmark_loom_profile_data_has_counter_data(
             profile_data_families) ||
         iree_any_bit_set(profile_data_families,
                          IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_METRICS |
                              IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES);
}

iree_status_t iree_benchmark_loom_parse_profile_data_families(
    iree_string_view_t value,
    iree_hal_device_profiling_data_families_t* out_profile_data_families) {
  *out_profile_data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  iree_string_view_t remaining = iree_string_view_trim(value);
  if (iree_string_view_is_empty(remaining)) {
    return iree_ok_status();
  }
  *out_profile_data_families = IREE_HAL_DEVICE_PROFILING_DATA_NONE;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t family_part = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &family_part, &remaining);
    family_part = iree_string_view_trim(family_part);
    if (iree_string_view_is_empty(family_part)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--profile-data contains an empty data family");
    }
    if (iree_string_view_equal(family_part, IREE_SV("none"))) {
      if (*out_profile_data_families != IREE_HAL_DEVICE_PROFILING_DATA_NONE ||
          !iree_string_view_is_empty(iree_string_view_trim(remaining))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--profile-data=none cannot be combined with "
                                "other data families");
      }
      return iree_ok_status();
    }
    bool matched = false;
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(iree_benchmark_loom_profile_family_names); ++i) {
      const iree_benchmark_loom_profile_family_name_t* family =
          &iree_benchmark_loom_profile_family_names[i];
      if (iree_string_view_equal(family_part,
                                 iree_make_cstring_view(family->flag_name)) ||
          iree_string_view_equal(family_part,
                                 iree_make_cstring_view(family->json_name))) {
        *out_profile_data_families |= family->bit;
        matched = true;
        break;
      }
    }
    if (!matched &&
        iree_string_view_equal(family_part, IREE_SV("pmc-ranges"))) {
      *out_profile_data_families |=
          IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES;
      matched = true;
    }
    if (!matched) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported --profile-data family '%.*s'",
                              (int)family_part.size, family_part.data);
    }
    remaining = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_write_profile_family_names_json(
    iree_hal_device_profiling_data_families_t profile_data_families,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  bool first = true;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_benchmark_loom_profile_family_names); ++i) {
    const iree_benchmark_loom_profile_family_name_t* family =
        &iree_benchmark_loom_profile_family_names[i];
    if (!iree_all_bits_set(profile_data_families, family->bit)) {
      continue;
    }
    if (!first) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_cstring(stream, family->json_name));
    first = false;
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_parse_i32_flag(iree_string_view_t flag_name,
                                                 void* storage,
                                                 iree_string_view_t value) {
  iree_benchmark_loom_i32_flag_t* flag =
      (iree_benchmark_loom_i32_flag_t*)storage;
  char* value_end = NULL;
  int64_t parsed_value = 0;
  if (!iree_string_view_is_empty(value)) {
    errno = 0;
    parsed_value = strtoll(value.data, &value_end, 10);
    if (errno == ERANGE) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "flag '--%.*s' value '%.*s' is outside int64 "
                              "range",
                              (int)flag_name.size, flag_name.data,
                              (int)value.size, value.data);
    }
    if (value_end != value.data + value.size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "flag '--%.*s' must be an int32 value; got '%.*s'",
          (int)flag_name.size, flag_name.data, (int)value.size, value.data);
    }
  }
  if (parsed_value < INT32_MIN || parsed_value > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "flag '--%.*s' value %" PRIi64
                            " is outside int32 range",
                            (int)flag_name.size, flag_name.data, parsed_value);
  }
  flag->value = (int32_t)parsed_value;
  flag->specified = true;
  return iree_ok_status();
}

void iree_benchmark_loom_print_i32_flag(iree_string_view_t flag_name,
                                        void* storage, FILE* file) {
  const iree_benchmark_loom_i32_flag_t* flag =
      (const iree_benchmark_loom_i32_flag_t*)storage;
  fprintf(file, "--%.*s=%" PRId32 "\n", (int)flag_name.size, flag_name.data,
          flag->value);
}

iree_status_t iree_benchmark_loom_parse_bool_flag(iree_string_view_t flag_name,
                                                  void* storage,
                                                  iree_string_view_t value) {
  iree_benchmark_loom_bool_flag_t* flag =
      (iree_benchmark_loom_bool_flag_t*)storage;
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("true")) ||
      iree_string_view_equal(value, IREE_SV("1"))) {
    flag->value = true;
  } else if (iree_string_view_equal(value, IREE_SV("false")) ||
             iree_string_view_equal(value, IREE_SV("0"))) {
    flag->value = false;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flag '--%.*s' must be a bool value; got '%.*s'",
                            (int)flag_name.size, flag_name.data,
                            (int)value.size, value.data);
  }
  flag->specified = true;
  return iree_ok_status();
}

void iree_benchmark_loom_print_bool_flag(iree_string_view_t flag_name,
                                         void* storage, FILE* file) {
  const iree_benchmark_loom_bool_flag_t* flag =
      (const iree_benchmark_loom_bool_flag_t*)storage;
  fprintf(file, "--%.*s=%s\n", (int)flag_name.size, flag_name.data,
          flag->value ? "true" : "false");
}
