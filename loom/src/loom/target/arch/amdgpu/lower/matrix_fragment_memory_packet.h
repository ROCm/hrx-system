// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU packet selection and report classification for fragment memory.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_PACKET_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_PACKET_H_

#include "loom/codegen/low/representation_plan.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Maximum number of 32-bit registers carried by one fragment memory packet.
  LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_PACKET_REGISTERS = 4,
  // Bits reserved for a publication packet's result-register count.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PUBLICATION_PACKET_COUNT_BITS = 4,
  // Mask selecting a publication packet's result-register count.
  LOOM_AMDGPU_FRAGMENT_MEMORY_PUBLICATION_PACKET_COUNT_MASK =
      (1u << LOOM_AMDGPU_FRAGMENT_MEMORY_PUBLICATION_PACKET_COUNT_BITS) - 1u,
};
static_assert(2u * LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_PACKET_REGISTERS <=
                  LOOM_AMDGPU_FRAGMENT_MEMORY_PUBLICATION_PACKET_COUNT_MASK,
              "publication packet widths must fit their compact encoding");

// Returns true when |payload_form| loads FP8 into packed 16-bit registers.
bool loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
    loom_amdgpu_fragment_memory_payload_form_t payload_form);

// Returns true when |payload_form| narrows f32 result lanes for a 16-bit store.
bool loom_amdgpu_fragment_memory_payload_form_is_store_narrow_f32_to_16bit(
    loom_amdgpu_fragment_memory_payload_form_t payload_form);

// Returns the physical 16-bit store element type for a narrowed f32 form.
loom_scalar_type_t loom_amdgpu_fragment_memory_store_narrow_result_element_type(
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

typedef enum loom_amdgpu_fragment_publication_source_flag_bits_e {
  // No optional narrowed-result source form is available.
  LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_NONE = 0u,
  // A proven rounded source can be packed without slicing constraints.
  LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_ROUNDED = 1u << 0,
  // Each rounded source lane is scaled before narrowing.
  LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_SCALED = 1u << 1,
  // A packed 16-bit source is reused directly.
  LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_PACKED = 1u << 2,
} loom_amdgpu_fragment_publication_source_flag_bits_t;
typedef uint8_t loom_amdgpu_fragment_publication_source_flags_t;

// Static facts sufficient to compare exact narrowed-result publications
// without constructing or retaining a complete fragment memory plan.
typedef struct loom_amdgpu_fragment_memory_publication_query_t {
  // Descriptor capabilities available on the selected target.
  const loom_low_descriptor_set_t* descriptor_set;
  // Exact native fragment layout being published.
  const loom_amdgpu_matrix_fragment_layout_t* layout;
  // Compiled lane/register address coefficients for the exact layout.
  const loom_amdgpu_fragment_memory_address_layout_t* address_layout;
  // Runtime view-axis coefficients for the exact layout.
  const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axes;
  // Exact byte strides for static source-view axes.
  const uint32_t* static_axis_byte_strides;
  // Destination memory space.
  loom_value_fact_memory_space_t memory_space;
  // Matrix fragment role being published.
  loom_contract_operand_role_t role;
  // Source-to-native coordinate interpretation for the representation.
  loom_amdgpu_matrix_result_representation_flags_t representation_flags;
  // Narrowed result payload form.
  loom_amdgpu_fragment_memory_payload_form_t payload_form;
  // Number of native result registers.
  uint16_t register_count;
  // Destination element byte count.
  uint16_t element_byte_count;
  // Destination view rank.
  uint8_t view_rank;
  // Optional narrowed-result source forms available to the publication.
  loom_amdgpu_fragment_publication_source_flags_t source_flags;
} loom_amdgpu_fragment_memory_publication_query_t;

// Packed exact descriptor reference and result-register count retained by
// publication selection. Register indices, repetition, and packet flags are
// derived from sequence position and the selected strategy during plan
// materialization.
typedef uint16_t loom_amdgpu_fragment_memory_publication_packet_t;
static_assert(LOOM_AMDGPU_DESCRIPTOR_REF_COUNT <=
                  (UINT16_MAX >>
                   LOOM_AMDGPU_FRAGMENT_MEMORY_PUBLICATION_PACKET_COUNT_BITS) +
                      1u,
              "descriptor references must fit compact publication packets");

static inline loom_amdgpu_fragment_memory_publication_packet_t
loom_amdgpu_fragment_memory_publication_packet_make(
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    uint16_t result_register_count) {
  return (
      loom_amdgpu_fragment_memory_publication_packet_t)((descriptor_ref
                                                         << LOOM_AMDGPU_FRAGMENT_MEMORY_PUBLICATION_PACKET_COUNT_BITS) |
                                                        result_register_count);
}

static inline loom_amdgpu_descriptor_ref_t
loom_amdgpu_fragment_memory_publication_packet_descriptor_ref(
    loom_amdgpu_fragment_memory_publication_packet_t packet) {
  return (
      loom_amdgpu_descriptor_ref_t)(packet >>
                                    LOOM_AMDGPU_FRAGMENT_MEMORY_PUBLICATION_PACKET_COUNT_BITS);
}

static inline uint16_t
loom_amdgpu_fragment_memory_publication_packet_result_register_count(
    loom_amdgpu_fragment_memory_publication_packet_t packet) {
  return packet & LOOM_AMDGPU_FRAGMENT_MEMORY_PUBLICATION_PACKET_COUNT_MASK;
}

// Cheapest exact publication selected from packet capabilities and issue
// costs.
typedef struct loom_amdgpu_fragment_memory_publication_choice_t {
  // Cross-lane publication recipe, or NULL for direct publication.
  const loom_matrix_fragment_packed_b16_publication_t* packed_b16_publication;
  // Target cost contributed to physical representation selection.
  loom_low_representation_cost_t cost;
  // Register starts where a wider direct packet was rejected because the
  // fragment coordinates were not contiguous in memory.
  uint32_t wider_noncontiguous_register_bits;
  // Exact compact packet sequence, or the repeated packet for scalar and
  // cross-lane strategies, consumed by fragment memory planning.
  loom_amdgpu_fragment_memory_publication_packet_t
      packets[LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS];
  // Concrete result epilogue strategy.
  loom_amdgpu_fragment_memory_epilogue_strategy_t strategy;
  // Number of direct packets, or one repeated scalar/cross-lane packet.
  uint8_t packet_count;
} loom_amdgpu_fragment_memory_publication_choice_t;
static_assert(sizeof(loom_amdgpu_fragment_memory_publication_choice_t) <= 88,
              "publication choices must stay compact");

// Selects the cheapest exact narrowed-result publication. |query| must
// describe a validated narrowed f32-to-16-bit store plan. Returns false when
// no target packet sequence covers the query.
bool loom_amdgpu_fragment_memory_select_publication(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    loom_amdgpu_fragment_memory_publication_choice_t* out_choice);

// Materializes direct memory packets for an analyzed fragment memory plan.
// |publication_choice| must be the exact choice selected for a narrowed
// f32-to-16-bit store and is ignored for other payload forms. Returns false and
// sets an optional stable constraint key when no packetization is available.
bool loom_amdgpu_fragment_memory_plan_packets(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_publication_choice_t* publication_choice,
    loom_amdgpu_fragment_memory_plan_t* plan,
    iree_string_view_t* out_constraint_key);

// Selects the exact FP8 load decode plan. Returns false when the target cannot
// emit any valid strategy for an FP8 payload accepted by earlier analysis.
bool loom_amdgpu_fragment_memory_select_fp8_load_decode_plan(
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
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_amdgpu_fragment_memory_packet_report_t* out_report);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_PACKET_H_
