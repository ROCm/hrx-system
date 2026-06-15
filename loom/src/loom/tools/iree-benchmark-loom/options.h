// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-line option parsing and mode names for iree-benchmark-loom.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_OPTIONS_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_OPTIONS_H_

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum iree_benchmark_loom_sample_compilation_mode_e {
  // Compile once and pass each sample's parameter values dynamically.
  IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE = 0,
  // Compile a separate candidate for each concrete selected sample.
  IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_PER_SAMPLE = 1,
  // Run both once and per-sample compilation modes.
  IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_BOTH = 2,
} iree_benchmark_loom_sample_compilation_mode_t;

typedef enum iree_benchmark_loom_artifact_bundle_policy_e {
  // Artifact bundling is disabled.
  IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_NONE = 0,
  // Bundle only results, manifest, file outputs, and required profile bundles.
  IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_MINIMAL = 1,
  // Bundle debug sidecars such as compile reports and target artifacts.
  IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG = 2,
  // Bundle every currently supported artifact class.
  IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_FULL = 3,
} iree_benchmark_loom_artifact_bundle_policy_t;

typedef enum iree_benchmark_loom_interleave_mode_e {
  // No interleaved comparison was requested.
  IREE_BENCHMARK_LOOM_INTERLEAVE_NONE = 0,
  // Run A once, then repeated B/A pairs for each non-baseline candidate.
  IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA = 1,
  // Rotate all comparison candidates in one repeated round-robin order.
  IREE_BENCHMARK_LOOM_INTERLEAVE_ROUND_ROBIN = 2,
} iree_benchmark_loom_interleave_mode_t;

typedef enum iree_benchmark_loom_output_format_e {
  // Emit one compact JSON document after the benchmark run completes.
  IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_SNAPSHOT = 0,
  // Emit newline-delimited lifecycle event rows as the run proceeds.
  IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_JSONL = 1,
} iree_benchmark_loom_output_format_t;

enum {
  // Default device-buffer ring byte target used to reduce hot-cache artifacts.
  IREE_BENCHMARK_LOOM_DEFAULT_INPUT_RING_MIN_BYTES = 32 * 1024 * 1024,
};

typedef struct iree_benchmark_loom_i32_flag_t {
  // Current flag value.
  int32_t value;
  // True when the flag was provided on the command line or in a flagfile.
  bool specified;
} iree_benchmark_loom_i32_flag_t;

typedef struct iree_benchmark_loom_bool_flag_t {
  // Current flag value.
  bool value;
  // True when the flag was provided on the command line or in a flagfile.
  bool specified;
} iree_benchmark_loom_bool_flag_t;

typedef struct iree_benchmark_loom_options_t {
  // Optional check.case symbol selected for execution.
  iree_string_view_t selected_case;
  // Optional check.benchmark symbol selected for execution.
  iree_string_view_t selected_benchmark;
  // Optional concrete benchmark sample ordinal, or negative to select all.
  int32_t sample_ordinal;
  // Maximum planned concrete samples retained for each check.case.
  iree_host_size_t max_samples_per_case;
  // Pass pipeline used before target artifact emission.
  iree_string_view_t pipeline;
  // Explicit result output path, or empty to use stdout or bundle defaults.
  iree_string_view_t output;
  // Selected result output format.
  iree_benchmark_loom_output_format_t output_format;
  // Explicit directory receiving check.file.write.* outputs.
  iree_string_view_t file_output_dir;
  // Explicit artifact bundle root directory, or empty when disabled.
  iree_string_view_t artifact_bundle_dir;
  // Artifact bundle policy controlling sidecar artifact classes.
  iree_benchmark_loom_artifact_bundle_policy_t artifact_bundle_policy;
  // True when the run stops after planning selected benchmarks.
  bool dry_run;
  // Measurement mode requested for selected benchmarks.
  iree_string_view_t measure;
  // Structured compile-report request.
  iree_string_view_t compile_report;
  // Structured artifact-manifest request.
  iree_string_view_t artifact_manifest;
  // Requested HAL profiling data-family list.
  iree_string_view_t profile_data;
  // Requested HAL profiling counter names.
  iree_string_view_list_t profile_counters;
  // Explicit directory receiving raw HAL profile artifacts.
  iree_string_view_t profile_artifacts_dir;
  // Sample compilation mode for dispatch-complete benchmarks.
  iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode;
  // Minimum total byte size for generated dispatch input rings.
  int64_t input_ring_min_bytes;
  // Exact dispatch input ring count, or zero for byte-size based selection.
  iree_host_size_t input_ring_count;
  // Benchmark list requested for interleaved comparison.
  iree_string_view_t compare;
  // Interleaving policy used when |compare| selects multiple benchmarks.
  iree_benchmark_loom_interleave_mode_t interleave_mode;
  // Number of interleaved comparison repetitions.
  iree_host_size_t repetitions;
  // Number of measured benchmark iterations or batches.
  iree_host_size_t iterations;
  // True when |iterations| was provided by the CLI/user.
  bool iterations_specified;
  // Number of warmup iterations or batches before measurement.
  iree_host_size_t warmup_iterations;
  // True when |warmup_iterations| was provided by the CLI/user.
  bool warmup_iterations_specified;
  // Number of dispatches recorded into each HAL benchmark batch.
  iree_host_size_t batch_size;
  // True when |batch_size| was provided by the CLI/user.
  bool batch_size_specified;
  // Minimum measured duration in milliseconds for dispatch benchmarks.
  int64_t min_time_ms;
  // True when |min_time_ms| was provided by the CLI/user.
  bool min_time_ms_specified;
  // Minimum warmup duration in milliseconds for dispatch benchmarks.
  int64_t warmup_time_ms;
  // True when |warmup_time_ms| was provided by the CLI/user.
  bool warmup_time_ms_specified;
  // Maximum measured HAL command-buffer batches.
  iree_host_size_t max_batches;
  // True when |max_batches| was provided by the CLI/user.
  bool max_batches_specified;
  // Accepted p90-to-p50 spread threshold in parts per million.
  uint64_t stable_p90_to_p50_ppm;
  // True when |stable_p90_to_p50_ppm| was provided by the CLI/user.
  bool stable_p90_to_p50_ppm_specified;
  // True when one final profiled batch should run after timing.
  bool profile_final_batch;
  // True when |profile_final_batch| was provided by the CLI/user.
  bool profile_final_batch_specified;
  // True when |profile_data| was explicitly provided and is not "none".
  bool profile_data_requested;
  // True when |input_ring_min_bytes| differs from the default.
  bool input_ring_min_bytes_specified;
  // True when |input_ring_count| was provided by the CLI/user.
  bool input_ring_count_specified;
} iree_benchmark_loom_options_t;

// Initializes benchmark runner options with production defaults.
void iree_benchmark_loom_options_initialize(
    iree_benchmark_loom_options_t* out_options);

// Returns true when the parsed CLI requested the AGENTS.md help snippet.
bool iree_benchmark_loom_cli_flags_request_agents_md(void);

// Converts parsed command-line flags into structured benchmark options.
iree_status_t iree_benchmark_loom_options_from_flags(
    iree_benchmark_loom_options_t* out_options);

// Returns the stable JSON spelling for an artifact bundle policy.
iree_string_view_t iree_benchmark_loom_artifact_bundle_policy_name(
    iree_benchmark_loom_artifact_bundle_policy_t policy);

// Returns the stable flag and JSON spelling for an output format.
iree_string_view_t iree_benchmark_loom_output_format_name(
    iree_benchmark_loom_output_format_t format);

// Parses an output format flag value.
iree_status_t iree_benchmark_loom_parse_output_format(
    iree_string_view_t value, iree_benchmark_loom_output_format_t* out_format);

// Parses an artifact bundle policy flag value.
iree_status_t iree_benchmark_loom_parse_artifact_bundle_policy(
    iree_string_view_t value,
    iree_benchmark_loom_artifact_bundle_policy_t* out_policy);

// Parses a sample-compilation mode flag value.
iree_status_t iree_benchmark_loom_parse_sample_compilation_mode(
    iree_string_view_t value,
    iree_benchmark_loom_sample_compilation_mode_t* out_mode);

// Returns the stable JSON spelling for a sample-compilation mode.
iree_string_view_t iree_benchmark_loom_sample_compilation_mode_name(
    iree_benchmark_loom_sample_compilation_mode_t mode);

// Returns true when |mode| includes the compile-once pass.
bool iree_benchmark_loom_sample_compilation_runs_once(
    iree_benchmark_loom_sample_compilation_mode_t mode);

// Returns true when |mode| includes the per-sample compilation pass.
bool iree_benchmark_loom_sample_compilation_runs_per_sample(
    iree_benchmark_loom_sample_compilation_mode_t mode);

// Parses a comparison interleave mode flag value.
iree_status_t iree_benchmark_loom_parse_interleave_mode(
    iree_string_view_t value, iree_benchmark_loom_interleave_mode_t* out_mode);

// Returns the stable JSON spelling for an interleave mode.
iree_string_view_t iree_benchmark_loom_interleave_mode_name(
    iree_benchmark_loom_interleave_mode_t mode);

// Returns true when |profile_data_families| includes counter payload data.
bool iree_benchmark_loom_profile_data_has_counter_data(
    iree_hal_device_profiling_data_families_t profile_data_families);

// Returns true when |profile_data_families| requires raw profile artifacts.
bool iree_benchmark_loom_profile_data_needs_artifact_data(
    iree_hal_device_profiling_data_families_t profile_data_families);

// Parses a comma-separated HAL profiling data family list.
iree_status_t iree_benchmark_loom_parse_profile_data_families(
    iree_string_view_t value,
    iree_hal_device_profiling_data_families_t* out_profile_data_families);

// Writes selected profiling data-family JSON names as a JSON array.
iree_status_t iree_benchmark_loom_write_profile_family_names_json(
    iree_hal_device_profiling_data_families_t profile_data_families,
    loom_output_stream_t* stream);

// Parses an int32 flag and records whether it was explicitly specified.
iree_status_t iree_benchmark_loom_parse_i32_flag(iree_string_view_t flag_name,
                                                 void* storage,
                                                 iree_string_view_t value);

// Prints an int32 flag in IREE flagfile form.
void iree_benchmark_loom_print_i32_flag(iree_string_view_t flag_name,
                                        void* storage, FILE* file);

// Parses a bool flag and records whether it was explicitly specified.
iree_status_t iree_benchmark_loom_parse_bool_flag(iree_string_view_t flag_name,
                                                  void* storage,
                                                  iree_string_view_t value);

// Prints a bool flag in IREE flagfile form.
void iree_benchmark_loom_print_bool_flag(iree_string_view_t flag_name,
                                         void* storage, FILE* file);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_OPTIONS_H_
