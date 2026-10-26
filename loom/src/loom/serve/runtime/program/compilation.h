// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_COMPILATION_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_COMPILATION_H_

#include "iree/base/api.h"
#include "loom/util/stream.h"
#include "loomc/result.h"

#ifdef __cplusplus
extern "C" {
#endif

// Copied primary diagnostic. Strings may fill their storage without a NUL;
// lengths retain the original extent so consumers can identify truncation.
// Compiler-owned sources, excerpts and parameter objects are not retained.
typedef struct loom_serve_compilation_diagnostic_t {
  // One loomc_diagnostic_severity_t.
  uint32_t severity;
  // One-based source line, or zero when unavailable.
  uint32_t line;
  // One-based source column, or zero when unavailable.
  uint32_t column;
  // Related locations not copied into this bounded primary diagnostic.
  uint64_t omitted_locations;
  // Named parameters represented only by the rendered message.
  uint64_t omitted_parameters;
  // Original source identifier extent in bytes.
  uint64_t source_length;
  // Original diagnostic code extent in bytes.
  uint64_t code_length;
  // Original rendered message extent in bytes.
  uint64_t message_length;
  // Bounded source identifier, independent of source/result lifetime.
  char source[256];
  // Bounded stable diagnostic code.
  char code[64];
  // Bounded rendered message, without source excerpts.
  char message[512];
} loom_serve_compilation_diagnostic_t;

// Zero-initialized, caller-owned compilation evidence. Producers append on one
// serialized owner; parallel native requests use independent reports and merge
// only after their tasks retire. Capture/merge never allocate, format or write
// to a sink. This value contains no borrowed pointers and survives failed JIT.
typedef struct loom_serve_compilation_report_t {
  // Total diagnostics indexed by loomc_diagnostic_severity_t, even when
  // omitted.
  uint64_t severities[3];
  // Primary diagnostics retained in producer order.
  iree_host_size_t diagnostic_count;
  // Bounded copied evidence; excess diagnostics remain counted above.
  loom_serve_compilation_diagnostic_t diagnostics[8];
  // Completed command compilation attempts, including failures.
  uint64_t commands;
  // Native compilations accepted directly or by command products, excluding
  // exact-input reuse.
  uint64_t kernels;
  // Requests reusing an exact native input within the same JIT catalog.
  uint64_t reused_kernels;
  // Successfully bound native entries across both stage kinds.
  uint64_t entries;
  // Largest shared compiler worker pool used by a stage attempt.
  uint64_t workers;
  // Sum of stage submission-to-completion intervals, including queue and
  // native-dependency waits. Intervals overlap; this is neither preparation
  // elapsed time nor CPU/device work time.
  uint64_t stage_duration_sum_ns;
} loom_serve_compilation_report_t;

// Independently owned evidence retained with a cached compilation result.
// Clean compilations use NULL instead of reserving diagnostic capacity. Work
// counters belong to the submitting job and are not part of this snapshot.
typedef struct loom_serve_compilation_diagnostics_t {
  // All emitted diagnostics indexed by loomc_diagnostic_severity_t.
  uint64_t severities[3];
  // Number of copied records, bounded by the source report's capacity.
  uint8_t count;
  // Exact copied prefix, with strings independent of compiler storage.
  loom_serve_compilation_diagnostic_t values[];
} loom_serve_compilation_diagnostics_t;

// Copies only emitted diagnostic evidence. Returns NULL for a clean report.
// The caller owns the returned allocation and frees it with host_allocator.
iree_status_t loom_serve_compilation_diagnostics_clone(
    const loom_serve_compilation_report_t* report,
    iree_allocator_t host_allocator,
    loom_serve_compilation_diagnostics_t** out_diagnostics);

// Requested host bytes for a snapshot, or zero when diagnostics is NULL.
iree_host_size_t loom_serve_compilation_diagnostics_storage_size(
    const loom_serve_compilation_diagnostics_t* diagnostics);

// Adds retained evidence to a live report without changing its work counters.
void loom_serve_compilation_diagnostics_merge(
    const loom_serve_compilation_diagnostics_t* diagnostics,
    loom_serve_compilation_report_t* report);

// Consumes result on success or failure. Copies bounded diagnostics before
// releasing compiler storage; failure also carries a primary source diagnostic
// in the returned status, independently of any observation consumer.
iree_status_t loom_serve_compilation_result(
    loomc_status_t operation, loomc_result_t* result,
    loom_serve_compilation_report_t* report);

// Adds a retired, disjoint producer's evidence without retaining its storage.
void loom_serve_compilation_merge(
    loom_serve_compilation_report_t* report,
    const loom_serve_compilation_report_t* source);

// Consumer-only JSONL rendering; never invoked by compilation task callbacks.
// Emits nothing for an empty report. Sinks may allocate/block/fail
// independently of the already completed compilation.
iree_status_t loom_serve_compilation_write(
    const loom_serve_compilation_report_t* report,
    loom_output_stream_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_COMPILATION_H_
