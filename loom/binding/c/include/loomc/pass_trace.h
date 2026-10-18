// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PASS_TRACE_H_
#define LOOMC_PASS_TRACE_H_

#include "loomc/base.h"

/// @file
/// Streaming pass-boundary IR traces.
///
/// Pass traces are cold diagnostic output produced while a prepared pass
/// program executes. Loom owns event selection, compiler-state projection, and
/// stable text or JSONL formatting. Callers own only the destination policy by
/// supplying a synchronous write callback.

#ifdef __cplusplus
extern "C" {
#endif

/// Formatted pass trace representation.
typedef enum loomc_pass_trace_format_e {
  /// Human-readable event metadata followed by textual Loom IR.
  LOOMC_PASS_TRACE_FORMAT_TEXT = 0,

  /// One JSON object per event with JSON-escaped textual Loom IR.
  LOOMC_PASS_TRACE_FORMAT_JSONL = 1,
} loomc_pass_trace_format_t;

/// Pass trace selection bits.
typedef enum loomc_pass_trace_flag_bits_e {
  /// Trace the IR immediately before every pass invocation.
  LOOMC_PASS_TRACE_FLAG_BEFORE_ALL = 1u << 0,

  /// Trace the IR immediately after every pass invocation.
  LOOMC_PASS_TRACE_FLAG_AFTER_ALL = 1u << 1,
} loomc_pass_trace_flag_bits_t;

/// Bitmask of `loomc_pass_trace_flag_bits_t`.
typedef uint32_t loomc_pass_trace_flags_t;

/// Writes one ordered trace fragment.
///
/// `fragment` is borrowed only for the callback duration and may contain any
/// number of bytes, including one byte. Returning a non-OK status stops the
/// compile operation and transfers ownership of that status to Loom.
typedef loomc_status_t(LOOMC_API_PTR* loomc_pass_trace_write_fn_t)(
    void* user_data, loomc_string_view_t fragment);

/// Borrowed destination for one sequential pass trace.
typedef struct loomc_pass_trace_sink_t {
  /// Function invoked for each ordered trace fragment.
  loomc_pass_trace_write_fn_t write;

  /// Opaque value passed to `write`.
  void* user_data;
} loomc_pass_trace_sink_t;

/// Pass trace boundary point.
typedef enum loomc_pass_trace_point_e {
  /// Snapshot immediately before the selected pass invocation.
  LOOMC_PASS_TRACE_POINT_BEFORE = 0,

  /// Snapshot immediately after the selected pass invocation.
  LOOMC_PASS_TRACE_POINT_AFTER = 1,
} loomc_pass_trace_point_t;

/// Stable metadata available when opening one per-event artifact.
typedef struct loomc_pass_trace_event_t {
  /// Zero-based ordinal among events emitted by this compile invocation.
  loomc_host_size_t event_ordinal;

  /// Boundary represented by this event.
  loomc_pass_trace_point_t point;

  /// Stable key of the pass invoked at this boundary.
  loomc_string_view_t pass_key;
} loomc_pass_trace_event_t;

/// Caller-owned destination for one per-event trace artifact.
typedef struct loomc_pass_trace_artifact_t {
  /// Non-empty reference written into the sequential trace event.
  ///
  /// The reference may be a bundle-relative path, URI, object-store key, or
  /// another caller-defined identity. Loom never opens or interprets it.
  loomc_string_view_t reference;

  /// Destination receiving this event's textual IR snapshot.
  loomc_pass_trace_sink_t sink;
} loomc_pass_trace_artifact_t;

/// Opens one destination for a selected trace event.
///
/// `event` is borrowed only for the callback duration. On an OK return,
/// `out_artifact` must contain a non-empty reference and write callback that
/// remain live until the matching close callback. A non-OK return retains all
/// destination ownership with the callback and is not followed by close.
typedef loomc_status_t(LOOMC_API_PTR* loomc_pass_trace_artifact_open_fn_t)(
    void* user_data, const loomc_pass_trace_event_t* event,
    loomc_pass_trace_artifact_t* out_artifact);

/// Closes one destination successfully returned from open.
///
/// Close is invoked exactly once after a successful open, including when a
/// sequential or artifact write fails. The callback releases all resources
/// associated with `artifact` and transfers any returned status to Loom.
typedef loomc_status_t(LOOMC_API_PTR* loomc_pass_trace_artifact_close_fn_t)(
    void* user_data, loomc_pass_trace_artifact_t* artifact);

/// Optional per-event artifact destination callbacks.
typedef struct loomc_pass_trace_artifact_sink_t {
  /// Function opening one selected event artifact.
  loomc_pass_trace_artifact_open_fn_t open;

  /// Function closing one successfully opened event artifact.
  loomc_pass_trace_artifact_close_fn_t close;

  /// Opaque value passed to `open` and `close`.
  void* user_data;
} loomc_pass_trace_artifact_sink_t;

/// Pass trace options for one pipeline or compile invocation.
///
/// Attach this descriptor to `loomc_pass_run_options_t::next` or
/// `loomc_compile_options_t::next` or `loomc_compile_artifact_options_t::next`.
/// Before and after filters match a pass key, an authored `pass.pipeline`
/// symbol, or the compiler-defined stage name. The compiler supplies stage
/// names and target-aware textual assembly; callers do not reconstruct either
/// from internal compiler state.
///
/// The descriptor and every borrowed string remain live only for the
/// synchronous compile call. The write callback may be invoked many times and
/// must preserve fragment order.
typedef struct loomc_pass_trace_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_PASS_TRACE_OPTIONS`.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Next invocation option extension.
  const void* next;

  /// Formatted representation written to `sink`.
  loomc_pass_trace_format_t format;

  /// Boundary selection flags.
  loomc_pass_trace_flags_t flags;

  /// Tool identity included in event metadata. Empty uses the format's
  /// unknown-value representation.
  loomc_string_view_t tool_name;

  /// Input identity included in event metadata. Empty uses the format's
  /// unknown-value representation.
  loomc_string_view_t input_identifier;

  /// Pass keys, pipeline symbols, or stage names traced before invocation.
  ///
  /// Compiler-defined stages are `pipeline-text` for textual and empty pass
  /// programs, `pipeline-symbol` for authored pipeline symbols, `source-low`
  /// for source-to-low target programs, and `prepared-low` for complete target
  /// artifact programs.
  const loomc_string_view_t* before_filters;

  /// Number of entries in `before_filters`.
  loomc_host_size_t before_filter_count;

  /// Pass keys, pipeline symbols, or stage names traced after invocation.
  const loomc_string_view_t* after_filters;

  /// Number of entries in `after_filters`.
  loomc_host_size_t after_filter_count;

  /// Destination receiving the complete formatted trace in order.
  loomc_pass_trace_sink_t sink;

  /// Optional destinations for per-event textual IR snapshots.
  ///
  /// When open and close are NULL, the sequential trace embeds each selected
  /// snapshot inline. When both are present, the sequential trace records the
  /// artifact reference and the artifact sink receives the corresponding
  /// snapshot without buffering in Loom.
  loomc_pass_trace_artifact_sink_t artifact_sink;
} loomc_pass_trace_options_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PASS_TRACE_H_
