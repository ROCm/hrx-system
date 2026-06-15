// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Machine-readable JSON formatting for target-owned packet hazard sidecars.
//
// The schema is common and target-neutral: packet identity, producer/consumer
// placement, progress accounting, and diagnostic categories are common fields.
// Target packages provide only stable ids and names for progress classes and
// reasons plus the predicates that emitted the rows.

#ifndef LOOM_CODEGEN_LOW_PACKET_HAZARD_PLAN_JSON_H_
#define LOOM_CODEGEN_LOW_PACKET_HAZARD_PLAN_JSON_H_

#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/packet_hazard_plan.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the stable JSON spelling for a packet-progress action.
iree_string_view_t loom_low_packet_progress_action_name(
    loom_low_packet_progress_action_t action);

// Returns the stable JSON spelling for a packet hazard-plan record kind.
iree_string_view_t loom_low_packet_hazard_plan_record_kind_name(
    loom_low_packet_hazard_plan_record_kind_t kind);

// Writes the JSON array for |progress| records to |stream|. A NULL progress
// table writes an empty array.
iree_status_t loom_low_packet_progress_write_json_array(
    const loom_low_packet_progress_table_t* progress,
    loom_output_stream_t* stream);

// Writes the JSON array for |plan| hazard records to |stream|. A NULL plan
// writes an empty array.
iree_status_t loom_low_packet_hazard_plan_write_json_array(
    const loom_low_packet_hazard_plan_t* plan, loom_output_stream_t* stream);

// Writes count and row-array object members for |plan|. This is used by target
// overlays that need their own top-level object but should not own the common
// progress/hazard row schema.
iree_status_t loom_low_packet_hazard_plan_write_json_members(
    const loom_low_packet_hazard_plan_t* plan, loom_output_stream_t* stream);

// Appends the canonical common packet hazard-plan JSON object to |builder|.
iree_status_t loom_low_packet_hazard_plan_format_json(
    const loom_low_packet_hazard_plan_t* plan, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PACKET_HAZARD_PLAN_JSON_H_
