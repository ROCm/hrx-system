// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-inserted native packet reporting.

#ifndef LOOM_TARGET_REPORTING_TARGET_INSERTION_H_
#define LOOM_TARGET_REPORTING_TARGET_INSERTION_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_target_compile_report_target_insertion_kind_e {
  // No target insertion kind was recorded.
  LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_NONE = 0,
  // Target-machine state transition inserted around a scheduled packet.
  LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_STATE = 1,
  // Target completion wait inserted around a scheduled packet.
  LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_WAIT = 2,
  // Target delay packet inserted around a scheduled packet.
  LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_DELAY = 3,
  // Target insertion not covered by a more precise stable kind.
  LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_OTHER = 4,
} loom_target_compile_report_target_insertion_kind_t;

typedef uint32_t loom_target_compile_report_target_insertion_flags_t;
enum {
  // No optional row fields are populated.
  LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_FLAG_NONE = 0u,
  // |dynamic_packet_count| is an exact execution count.
  LOOM_TARGET_COMPILE_REPORT_TARGET_INSERTION_FLAG_DYNAMIC_PACKET_COUNT = 1u
                                                                          << 0,
};

// Aggregate target-inserted packet counts.
typedef struct loom_target_compile_report_target_insertion_summary_t {
  // Number of packets inserted into the static native packet stream.
  uint64_t static_packet_count;
  // Number of static packets whose dynamic execution count is exact.
  uint64_t exact_dynamic_packet_count;
  // Number of static packets whose dynamic execution count is unavailable.
  uint64_t unknown_dynamic_packet_count;
  // Sum of exact dynamic execution counts. This is a complete total only when
  // |unknown_dynamic_packet_count| is zero.
  uint64_t dynamic_packet_count;
} loom_target_compile_report_target_insertion_summary_t;

// One target-owned native packet inserted while expanding a scheduled node.
typedef struct loom_target_compile_report_target_insertion_row_t {
  // Optional fields populated for this row.
  loom_target_compile_report_target_insertion_flags_t flags;
  // Target artifact function symbol containing this insertion.
  iree_string_view_t function_name;
  // Stable target-independent insertion kind.
  loom_target_compile_report_target_insertion_kind_t insertion_kind;
  // Stable target packet or descriptor key.
  iree_string_view_t packet_key;
  // Region block label containing the insertion point.
  iree_string_view_t block_name;
  // Region block ordinal containing the insertion point.
  uint32_t block_index;
  // Schedule node whose native expansion contains the insertion.
  uint32_t node_index;
  // Scheduled ordinal of |node_index| within |block_index|.
  uint32_t scheduled_ordinal;
  // Operation mnemonic of the scheduled node owning the insertion.
  iree_string_view_t boundary_operation_name;
  // Descriptor key of the owning node, or empty for structural nodes.
  iree_string_view_t boundary_descriptor_key;
  // Number of native packets represented by this static row.
  uint64_t static_packet_count;
  // Exact execution count when the dynamic-packet-count flag is set.
  uint64_t dynamic_packet_count;
} loom_target_compile_report_target_insertion_row_t;

struct loom_target_compile_report_t;

// Returns the stable text name of |kind|.
iree_string_view_t loom_target_compile_report_target_insertion_kind_name(
    loom_target_compile_report_target_insertion_kind_t kind);

// Accumulates |source| into |target|.
void loom_target_compile_report_accumulate_target_insertion_summary(
    loom_target_compile_report_target_insertion_summary_t* target,
    const loom_target_compile_report_target_insertion_summary_t* source);

// Records one target-owned native packet insertion. Summary counts are always
// retained; the row is retained only when target-insertion details were
// requested.
iree_status_t loom_target_compile_report_record_target_insertion_row(
    struct loom_target_compile_report_t* report,
    const loom_target_compile_report_target_insertion_row_t* row);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_TARGET_INSERTION_H_
