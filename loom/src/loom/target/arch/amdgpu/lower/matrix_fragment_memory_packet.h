// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU packet selection and report classification for fragment memory.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_PACKET_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_PACKET_H_

#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |payload_form| loads FP8 into packed 16-bit registers.
bool loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
    loom_amdgpu_fragment_memory_payload_form_t payload_form);

// Returns the physical 16-bit result element type for an FP8 load form.
loom_scalar_type_t loom_amdgpu_fragment_memory_load_fp8_result_element_type(
    loom_amdgpu_fragment_memory_payload_form_t payload_form);

// Returns true when the target packet table covers the requested access.
bool loom_amdgpu_fragment_memory_space_supports_access(
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_fragment_memory_packetization_t packetization,
    loom_amdgpu_fragment_memory_payload_form_t payload_form);

// Returns true when the descriptor set can materialize |payload_form|.
bool loom_amdgpu_fragment_memory_payload_form_has_descriptors(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_fragment_memory_payload_form_t payload_form);

// Selects direct memory packets for an analyzed fragment memory plan. Returns
// false and sets an optional stable constraint key when no packetization is
// available.
bool loom_amdgpu_fragment_memory_plan_packets(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_amdgpu_fragment_memory_plan_t* plan,
    iree_string_view_t* out_constraint_key);

// Annotates selected FP8 load packets with the exact target decode strategy.
void loom_amdgpu_fragment_memory_apply_fp8_load_strategy_flags(
    const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* plan);

// Returns true when |strategy| exchanges values across lanes before packing.
bool loom_amdgpu_fragment_memory_epilogue_strategy_is_crosslane_packed_b16(
    loom_amdgpu_fragment_memory_epilogue_strategy_t strategy);

// Returns true when |strategy| exchanges values with DPP instructions.
bool loom_amdgpu_fragment_memory_epilogue_strategy_uses_dpp(
    loom_amdgpu_fragment_memory_epilogue_strategy_t strategy);

// Returns the compile-report plan key for a selected fragment memory plan.
iree_string_view_t loom_amdgpu_fragment_memory_plan_key(
    const loom_amdgpu_fragment_memory_plan_t* plan);

typedef struct loom_amdgpu_fragment_memory_packet_report_t {
  // Stable strategy key derived from the selected packet plan.
  iree_string_view_t strategy_key;
  // Stable reason key describing why a wider strategy was not selected.
  iree_string_view_t fallback_reason;
} loom_amdgpu_fragment_memory_packet_report_t;

// Queries cold-path report classification for one selected memory packet.
void loom_amdgpu_fragment_memory_query_packet_report(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_amdgpu_fragment_memory_packet_report_t* out_report);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_PACKET_H_
