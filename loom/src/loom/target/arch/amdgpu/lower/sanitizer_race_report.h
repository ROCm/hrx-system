// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-low sanitizer race feedback report builders.
//
// Loom models race diagnostics through sanitizer.race operations. The current
// AMDGPU HAL feedback channel exposes those diagnostics through the TSAN packet
// schema, so this file is the narrow boundary that maps Loom race reports onto
// that target runtime ABI.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_REPORT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_REPORT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/location.h"
#include "loom/ir/types.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;
typedef struct loom_block_t loom_block_t;
typedef struct loom_amdgpu_feedback_packet_address_t
    loom_amdgpu_feedback_packet_address_t;

typedef struct loom_amdgpu_sanitizer_race_report_t {
  // TSAN check kind that triggered the report.
  loom_value_id_t check_kind;
  // Report flags describing optional atomic-access properties.
  loom_value_id_t flags;
  // Memory space containing |memory_address|.
  loom_value_id_t memory_space;
  // Access kind performed by the reporting workitem.
  loom_value_id_t current_access_kind;
  // Access kind previously observed for the same memory location.
  loom_value_id_t prior_access_kind;
  // Access size in bytes.
  loom_value_id_t access_size;
  // Compiler-assigned instrumentation site identifier for the current access.
  loom_value_id_t current_site_id;
  // Compiler-assigned instrumentation site identifier for the prior access.
  loom_value_id_t prior_site_id;
  // Address or memory-space-relative byte offset that raced.
  loom_value_id_t memory_address;
  // Shadow address consulted by the check, or zero when unavailable.
  loom_value_id_t shadow_address;
  // Shadow value observed by the check, or zero when unavailable.
  loom_value_id_t shadow_value;
  // X dimension workgroup id that produced the current access report.
  loom_value_id_t current_workgroup_id_x;
  // Y dimension workgroup id that produced the current access report.
  loom_value_id_t current_workgroup_id_y;
  // Z dimension workgroup id that produced the current access report.
  loom_value_id_t current_workgroup_id_z;
  // X dimension workitem id that produced the current access report.
  loom_value_id_t current_workitem_id_x;
  // Y dimension workitem id that produced the current access report.
  loom_value_id_t current_workitem_id_y;
  // Z dimension workitem id that produced the current access report.
  loom_value_id_t current_workitem_id_z;
  // X dimension workgroup id that produced the prior access.
  loom_value_id_t prior_workgroup_id_x;
  // Y dimension workgroup id that produced the prior access.
  loom_value_id_t prior_workgroup_id_y;
  // Z dimension workgroup id that produced the prior access.
  loom_value_id_t prior_workgroup_id_z;
  // X dimension workitem id that produced the prior access.
  loom_value_id_t prior_workitem_id_x;
  // Y dimension workitem id that produced the prior access.
  loom_value_id_t prior_workitem_id_y;
  // Z dimension workitem id that produced the prior access.
  loom_value_id_t prior_workitem_id_z;
} loom_amdgpu_sanitizer_race_report_t;

typedef struct loom_amdgpu_sanitizer_race_report_island_t {
  // Entry block accepting the source/report tuple for one failed race site.
  loom_block_t* entry_block;
  // Terminal block reached after optional reporting or when reporting is not
  // available.
  loom_block_t* terminal_block;
  // Block arguments carrying current source coordinates in |entry_block|.
  loom_amdgpu_feedback_packet_source_t source_args;
  // Block arguments carrying race report values in |entry_block|.
  loom_amdgpu_sanitizer_race_report_t report_args;
} loom_amdgpu_sanitizer_race_report_island_t;

typedef struct loom_amdgpu_sanitizer_race_report_failure_branch_t {
  // Per-site cold block that canonicalizes report values and enters the island.
  loom_block_t* failure_block;
  // Hot continuation block reached when no race was observed.
  loom_block_t* continuation_block;
} loom_amdgpu_sanitizer_race_report_failure_branch_t;

// Emits the AMDGPU sanitizer race report payload into a reserved feedback
// packet.
//
// The generic feedback packet header must be emitted separately with kind
// LOOM_AMDGPU_FEEDBACK_PACKET_KIND_TSAN and a payload length of
// LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH. This helper writes only the payload
// bytes beginning at LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH.
iree_status_t loom_amdgpu_build_sanitizer_race_report_payload(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location);

// Emits a terminal AMDGPU sanitizer race report producer.
//
// The builder must be positioned at the end of a target-low block. This helper
// emits a cold CFG that loads the runtime feedback config, branches around
// feedback channel dereferences when feedback is disabled, reserves packet
// storage when possible, writes and publishes one TSAN-kind feedback packet on
// successful reservation, and terminates the current wave on all paths. The
// builder is left in the terminal block after the return terminator has been
// emitted.
iree_status_t loom_amdgpu_build_sanitizer_race_report_terminate(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t feedback_config_symbol,
    const loom_amdgpu_feedback_packet_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location);

// Builds a shared cold report island for failed race observations.
//
// The island accepts one failed race site's report tuple as block arguments,
// attempts to enqueue a feedback packet, and terminates the current wave
// whether reporting succeeds, is disabled, or cannot reserve packet storage.
// Call loom_amdgpu_build_sanitizer_race_report_branch from each per-site cold
// block to enter the island. Leaves the builder positioned at the island's
// terminal block after the return terminator has been emitted.
iree_status_t loom_amdgpu_build_sanitizer_race_report_island(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_block_t* after_block, loom_symbol_ref_t feedback_config_symbol,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_race_report_island_t* out_island);

// Terminates the current cold block with a branch into |island|.
//
// Values are converted to the island's canonical block-argument register
// classes in the current block, so this should only be used on the
// already-failing path.
iree_status_t loom_amdgpu_build_sanitizer_race_report_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_report_island_t* island,
    const loom_amdgpu_feedback_packet_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location);

// Splits the current hot block on an EXEC-width failure mask and routes
// failures to |island|.
//
// |failure_mask| must be an SGPRx2 native lane mask where set bits identify
// lanes that observed a race. The hot block only compares the mask against zero
// and conditionally branches. The per-site cold block narrows EXEC to the
// failed lanes before converting the already-built report tuple to |island|
// arguments. Since the island terminates the failed wave, the saved EXEC value
// is intentionally not restored. Leaves the builder positioned at the
// continuation block.
iree_status_t loom_amdgpu_build_sanitizer_race_report_failure_mask_branch(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_sanitizer_race_report_island_t* island,
    loom_value_id_t failure_mask,
    const loom_amdgpu_feedback_packet_source_t* source,
    const loom_amdgpu_sanitizer_race_report_t* report,
    loom_location_id_t location,
    loom_amdgpu_sanitizer_race_report_failure_branch_t* out_branch);

// Splits the current hot block on an EXEC-width failure mask and enters the
// cold per-site failure block.
//
// |failure_mask| must be an SGPRx2 native lane mask where set bits identify
// lanes that observed a race. The hot block only compares the mask against zero
// and conditionally branches. The builder is left positioned in the per-site
// cold failure block after EXEC has been narrowed to the failed lanes. The
// caller must terminate that block, usually with
// loom_amdgpu_build_sanitizer_race_report_branch, and then move the builder to
// |out_branch->continuation_block|.
iree_status_t loom_amdgpu_build_sanitizer_race_report_failure_mask_split(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t failure_mask, loom_location_id_t location,
    loom_amdgpu_sanitizer_race_report_failure_branch_t* out_branch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_REPORT_H_
