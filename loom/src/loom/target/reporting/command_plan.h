// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Observations of compiled command order and resource placement.

#ifndef LOOM_TARGET_REPORTING_COMMAND_PLAN_H_
#define LOOM_TARGET_REPORTING_COMMAND_PLAN_H_

#include "iree/base/api.h"
#include "loom/target/reporting/row_list.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_compile_report_t loom_target_compile_report_t;

// Diagnostic categories, independent of command artifact encodings.
typedef enum loom_target_compile_report_command_kind_e {
  LOOM_TARGET_COMPILE_REPORT_COMMAND_FILL,
  LOOM_TARGET_COMPILE_REPORT_COMMAND_COPY,
  LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_DIRECT,
  LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_INDIRECT_STATIC,
  LOOM_TARGET_COMPILE_REPORT_COMMAND_DISPATCH_INDIRECT_DYNAMIC,
  LOOM_TARGET_COMPILE_REPORT_COMMAND_BARRIER,
  LOOM_TARGET_COMPILE_REPORT_COMMAND_TIMESTAMP,
  LOOM_TARGET_COMPILE_REPORT_COMMAND_KIND_COUNT,
} loom_target_compile_report_command_kind_t;

// Resource lifetime selected by command compilation.
typedef enum loom_target_compile_report_command_buffer_role_e {
  // Concrete storage bound when recording the program.
  LOOM_TARGET_COMPILE_REPORT_COMMAND_BUFFER_FIXED,
  // Storage supplied through a binding when issuing the recorded program.
  LOOM_TARGET_COMPILE_REPORT_COMMAND_BUFFER_REBINDABLE,
} loom_target_compile_report_command_buffer_role_t;

typedef struct loom_target_compile_report_command_buffer_t {
  // Resource lifetime of the root.
  loom_target_compile_report_command_buffer_role_t role;
  // Index within the resource table selected by |role|.
  uint32_t root_index;
  // Root-relative byte offset.
  uint64_t byte_offset;
  // Exact byte length, or UINT64_MAX for the remaining root range.
  uint64_t byte_length;
} loom_target_compile_report_command_buffer_t;

enum loom_target_compile_report_command_flag_bits_e {
  // All earlier commands complete before this command's payload begins.
  LOOM_TARGET_COMPILE_REPORT_COMMAND_BARRIER_BEFORE = 1u << 0,
};
typedef uint8_t loom_target_compile_report_command_flags_t;

typedef struct loom_target_compile_report_command_t {
  // Operation performed at this position in the compiled command order.
  loom_target_compile_report_command_kind_t kind;
  // Execution ordering attached to the operation.
  loom_target_compile_report_command_flags_t flags;
  // Selected native entry name for dispatches; empty for other commands.
  iree_string_view_t entry_name;
  // Logical argument count for dispatches; zero for other commands.
  uint32_t argument_count;
  // Operation-specific diagnostic facts.
  union {
    // Exact workgroup counts for a direct dispatch, including zero counts.
    uint32_t workgroup_counts[3];
    // Count-buffer range for either indirect dispatch kind.
    loom_target_compile_report_command_buffer_t indirect_counts;
    // Source and destination of a copy.
    struct {
      // Read range.
      loom_target_compile_report_command_buffer_t source;
      // Written range.
      loom_target_compile_report_command_buffer_t target;
    } copy;
    // Destination and repeated pattern of a fill.
    struct {
      // Written range.
      loom_target_compile_report_command_buffer_t target;
      // Low bytes of the repeated pattern.
      uint32_t pattern;
      // Number of low pattern bytes to repeat.
      uint32_t pattern_byte_count;
    } fill;
    // Destination of a timestamp capture.
    loom_target_compile_report_command_buffer_t timestamp;
  } payload;
} loom_target_compile_report_command_t;

typedef struct loom_target_compile_report_command_binding_t {
  // Resource lifetime selected for this source-signature buffer argument.
  loom_target_compile_report_command_buffer_role_t role;
  // Index in the resource table selected by |role|.
  uint32_t root_index;
} loom_target_compile_report_command_binding_t;

typedef struct loom_target_compile_report_command_storage_t {
  // Fixed or rebindable root index; UINT32_MAX means no storage is required.
  uint32_t root_index;
  // Minimum supplied storage size in bytes.
  uint64_t required_byte_length;
  // Minimum byte alignment of the supplied storage.
  uint64_t minimum_alignment;
} loom_target_compile_report_command_storage_t;

typedef struct loom_target_compile_report_command_parameter_t {
  // Fully substituted immutable parameter key.
  iree_string_view_t key;
  // Fixed resource root containing the payload.
  uint32_t root_index;
  // Root-relative byte offset.
  uint64_t byte_offset;
  // Exact payload size in bytes.
  uint64_t byte_length;
  // Required byte alignment of the placed payload.
  uint64_t minimum_alignment;
} loom_target_compile_report_command_parameter_t;

// One command root, in caller order. Strings borrow the producing plan until
// formatting; owned row arrays can outlive the producer's temporary tables.
typedef struct loom_target_compile_report_command_plan_t {
  // Selected command root name.
  iree_string_view_t root_name;
  // Counts indexed by the diagnostic command kind enum.
  uint32_t command_kind_counts[LOOM_TARGET_COMPILE_REPORT_COMMAND_KIND_COUNT];
  // Number of leading barriers folded into payload commands.
  uint32_t barrier_before_count;
  // Number of fixed resource roots.
  uint32_t fixed_buffer_count;
  // Number of issue-time resource bindings.
  uint32_t rebindable_binding_count;
  // Number of executable requirements.
  uint32_t executable_count;
  // Number of native entry requirements.
  uint32_t entry_count;
  // Issue-time transient slab, absent when its root is UINT32_MAX.
  loom_target_compile_report_command_storage_t transient;
  // Host-produced indirect count table, absent when its root is UINT32_MAX.
  loom_target_compile_report_command_storage_t launch_counts;
  // Total commands, whether or not detail rows were requested.
  uint32_t command_count;
  // Commands in compiled order; NULL in summary mode.
  const loom_target_compile_report_command_t* commands;
  // Number of source launch-buffer arguments.
  uint32_t binding_count;
  // Buffer placements in source signature order; NULL in summary mode.
  const loom_target_compile_report_command_binding_t* bindings;
  // Number of immutable parameter roots.
  uint32_t parameter_root_count;
  // Fixed-root storage requirements; NULL in summary mode.
  const loom_target_compile_report_command_storage_t* parameter_roots;
  // Number of placed immutable parameters.
  uint32_t parameter_count;
  // Parameter placements; NULL in summary mode.
  const loom_target_compile_report_command_parameter_t* parameters;
} loom_target_compile_report_command_plan_t;

// Records one root and copies requested detail arrays. Strings remain borrowed
// from the producing plan, which must live through report formatting.
iree_status_t loom_target_compile_report_record_command_plan(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_command_plan_t* plan);

// Releases the copied detail arrays and their containing row list.
void loom_target_compile_report_command_plan_list_deinitialize(
    loom_target_compile_report_row_list_t* list,
    iree_allocator_t host_allocator);

// Returns a diagnostic operation spelling, independent of wire opcodes.
iree_string_view_t loom_target_compile_report_command_kind_name(
    loom_target_compile_report_command_kind_t kind);

// Returns the diagnostic resource lifetime spelling.
iree_string_view_t loom_target_compile_report_command_buffer_role_name(
    loom_target_compile_report_command_buffer_role_t role);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_COMMAND_PLAN_H_
