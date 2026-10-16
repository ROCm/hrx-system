// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Selected rule programs retained independently of value-analysis scratch.
//
// Rule selection resolves descriptor rows, attributes, source origins, and
// result carriers while source facts are available. Emission consumes these
// concrete choices directly.

#ifndef LOOM_CODEGEN_LOW_LOWER_RULE_PLAN_H_
#define LOOM_CODEGEN_LOW_LOWER_RULE_PLAN_H_

#include "iree/base/internal/math.h"
#include "loom/codegen/low/lower/module_state.h"
#include "loom/codegen/low/lower/rules.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_selected_plan_t loom_low_lower_selected_plan_t;
typedef struct loom_low_lower_rule_source_memory_plan_builder_t
    loom_low_lower_rule_source_memory_plan_builder_t;

// Present only for rows whose static emit has read-only data attributes.
// Followed by attr_copy_count private data IDs, indexed by attribute ordinal.
// Only IDs selected by attribute_mask are initialized or consumed.
typedef struct loom_low_lower_read_only_attributes_t {
  // Attribute ordinals whose planned null symbols are filled during execution.
  uint32_t attribute_mask;
} loom_low_lower_read_only_attributes_t;

// Present only for descriptor lane forms. Followed by operand_ref_count slice
// type IDs and result_ref_count packet type IDs. Operand IDs are initialized
// only where slice_operand_mask is set. Bound aggregate result types remain in
// the ordinary result payload.
typedef struct loom_low_lower_lane_plan_t {
  // Number of lanes consumed by this row. A sequence's final row owns its
  // shared iteration count; preceding rows may consume broadcast operands.
  uint32_t lane_count;
  // Operand ordinals requiring slices. Sequence slices precede copies; other
  // forms slice the copied aggregate. Clear bits forward the whole operand.
  uint32_t slice_operand_mask;
} loom_low_lower_lane_plan_t;

typedef struct loom_low_lower_resolved_emit_t {
  // Shared emission behavior selected by planning.
  const loom_low_lower_emit_t* emit;
  // Program-position descriptor resolved during planning.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Source access semantics for this memory packet; zero for address setup.
  loom_memory_access_flags_t access_flags;
  // Result carriers resolved from source values, type patterns, descriptors,
  // and rule-local transfers. Elided recipes have no result payload.
  uint8_t result_type_mask;
  // Operand ordinals with retained materializer recipes in the row payload.
  uint8_t materializer_mask;
  // Fact-derived source references retained in the row payload.
  // Bits address operand refs; set bits have packed value IDs.
  uint8_t source_value_mask;
  // Byte offset from this row to materializer recipes, attributes, optional
  // read-only attribute IDs, copied operand type IDs, canonical result type
  // IDs, optional lane plan, fact-derived source values, and an optional
  // complete-address coordinate type ID, in that order.
  // Zero for an empty payload. All rows and their aligned payloads share one
  // function-arena allocation. Elided recipes retain descriptor identity
  // without executable payloads.
  uint32_t data_offset;
} loom_low_lower_resolved_emit_t;
static_assert(sizeof(loom_low_lower_resolved_emit_t) == 2 * sizeof(void*) + 8,
              "resolved emit rows must retain compact operand masks");

// Aligns the following attributes even on hosts with smaller pointers.
static inline uint32_t loom_low_lower_rule_materializer_data_size(
    uint8_t mask) {
  return (uint32_t)iree_host_align(
      iree_math_count_ones_u32(mask) * sizeof(void*),
      iree_alignof(loom_named_attr_t));
}

// Returns the packed opaque recipes for prepared operand materializers.
static inline const void* const* loom_low_lower_resolved_emit_materializers(
    const loom_low_lower_resolved_emit_t* resolved) {
  return (const void* const*)((const uint8_t*)resolved + resolved->data_offset);
}

// Returns the recipe selected for an operand whose materializer bit is set.
static inline const void* loom_low_lower_resolved_emit_materializer(
    const loom_low_lower_resolved_emit_t* resolved, uint16_t ordinal) {
  const uint32_t preceding_mask = (UINT32_C(1) << ordinal) - 1u;
  return loom_low_lower_resolved_emit_materializers(
      resolved)[iree_math_count_ones_u32(resolved->materializer_mask &
                                         preceding_mask)];
}

// Returns the optional private resource payload size without growing ordinary
// emit rows or their attribute storage.
static inline uint32_t loom_low_lower_rule_read_only_attributes_size(
    const loom_low_lower_emit_t* emit) {
  return emit->has_read_only_data_attributes
             ? sizeof(loom_low_lower_read_only_attributes_t) +
                   emit->attr_copy_count *
                       sizeof(loom_low_lower_read_only_data_id_t)
             : 0;
}

// Ordinary descriptor and structural rows retain no lane payload.
static inline uint32_t loom_low_lower_rule_lane_plan_size(
    const loom_low_lower_emit_t* emit) {
  switch (emit->kind) {
    case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_FIRST_LANE:
    case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE:
    case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE:
    case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES:
      return sizeof(loom_low_lower_lane_plan_t) +
             (emit->operand_ref_count + emit->result_ref_count) *
                 sizeof(loom_type_id_t);
    default:
      return 0;
  }
}

// Returns the attributes selected for this emit row, in generated table order.
static inline loom_named_attr_slice_t loom_low_lower_resolved_emit_attributes(
    const loom_low_lower_resolved_emit_t* resolved) {
  return loom_make_named_attr_slice(
      resolved->data_offset
          ? (const loom_named_attr_t*)((const uint8_t*)resolved +
                                       resolved->data_offset +
                                       loom_low_lower_rule_materializer_data_size(
                                           resolved->materializer_mask))
          : NULL,
      resolved->emit->attr_copy_count);
}

// Returns the resource IDs following the attributes of a row carrying them.
static inline const loom_low_lower_read_only_attributes_t*
loom_low_lower_resolved_emit_read_only_attributes(
    const loom_low_lower_resolved_emit_t* resolved) {
  return (
      const loom_low_lower_read_only_attributes_t*)((const uint8_t*)resolved +
                                                    resolved->data_offset +
                                                    loom_low_lower_rule_materializer_data_size(
                                                        resolved
                                                            ->materializer_mask) +
                                                    resolved->emit
                                                            ->attr_copy_count *
                                                        sizeof(
                                                            loom_named_attr_t));
}

// Returns the packed copy carriers following attribute payloads. Static copy
// bits select their operand ordinals; no storage is reserved for other
// operands.
static inline const loom_type_id_t* loom_low_lower_resolved_emit_copy_types(
    const loom_low_lower_resolved_emit_t* resolved) {
  return (const loom_type_id_t*)((const uint8_t*)resolved +
                                 resolved->data_offset +
                                 loom_low_lower_rule_materializer_data_size(
                                     resolved->materializer_mask) +
                                 resolved->emit->attr_copy_count *
                                     sizeof(loom_named_attr_t) +
                                 loom_low_lower_rule_read_only_attributes_size(
                                     resolved->emit));
}

// Returns the packed canonical result carriers following operand copy types.
static inline const loom_type_id_t* loom_low_lower_resolved_emit_result_types(
    const loom_low_lower_resolved_emit_t* resolved) {
  return loom_low_lower_resolved_emit_copy_types(resolved) +
         iree_math_count_ones_u32(resolved->emit->copy_operand_mask);
}

// Returns the canonical carrier ID for a result fixed by planning.
static inline loom_type_id_t loom_low_lower_resolved_emit_result_type_id(
    const loom_low_lower_resolved_emit_t* resolved, uint16_t result_ordinal) {
  const loom_type_id_t* result_types =
      loom_low_lower_resolved_emit_result_types(resolved);
  const uint32_t preceding_mask = (UINT32_C(1) << result_ordinal) - 1u;
  return result_types[iree_math_count_ones_u32(resolved->result_type_mask &
                                               preceding_mask)];
}

// Returns the lane recipe of a descriptor lane row.
static inline const loom_low_lower_lane_plan_t*
loom_low_lower_resolved_emit_lane_plan(
    const loom_low_lower_resolved_emit_t* resolved) {
  return (
      const loom_low_lower_lane_plan_t*)(loom_low_lower_resolved_emit_result_types(
                                             resolved) +
                                         iree_math_count_ones_u32(
                                             resolved->result_type_mask));
}

// Returns the canonical packet carrier, distinct from an aggregate binding.
static inline loom_type_id_t loom_low_lower_resolved_emit_lane_result_type_id(
    const loom_low_lower_resolved_emit_t* resolved, uint16_t result_ordinal) {
  const loom_type_id_t* type_ids =
      (const loom_type_id_t*)(loom_low_lower_resolved_emit_lane_plan(resolved) +
                              1);
  return type_ids[resolved->emit->operand_ref_count + result_ordinal];
}

// Returns fact-derived source IDs after the optional lane recipe.
static inline const loom_value_id_t* loom_low_lower_resolved_emit_source_values(
    const loom_low_lower_resolved_emit_t* resolved) {
  const uint8_t* lane_payload =
      (const uint8_t*)(loom_low_lower_resolved_emit_result_types(resolved) +
                       iree_math_count_ones_u32(resolved->result_type_mask));
  return (const loom_value_id_t*)(lane_payload +
                                  loom_low_lower_rule_lane_plan_size(
                                      resolved->emit));
}

// Returns the retained source for a fact-derived operand reference.
// The selected recipe establishes that the corresponding mask bit is set.
static inline loom_value_id_t loom_low_lower_resolved_emit_source_value(
    const loom_low_lower_resolved_emit_t* resolved,
    uint16_t reference_ordinal) {
  const loom_value_id_t* source_values =
      loom_low_lower_resolved_emit_source_values(resolved);
  const uint32_t preceding_mask = (UINT32_C(1) << reference_ordinal) - 1u;
  return source_values[iree_math_count_ones_u32(resolved->source_value_mask &
                                                preceding_mask)];
}

// Returns the semantic coordinate carrier for a complete-address materializer.
// The selected source-memory row establishes that this payload is present.
static inline loom_type_id_t
loom_low_lower_resolved_emit_address_coordinate_type_id(
    const loom_low_lower_resolved_emit_t* resolved) {
  const loom_value_id_t* source_values =
      loom_low_lower_resolved_emit_source_values(resolved);
  const loom_type_id_t* coordinate_type =
      (const loom_type_id_t*)(source_values + iree_math_count_ones_u32(
                                                  resolved->source_value_mask));
  return *coordinate_type;
}

// Finalizes a selected rule's descriptors, source access semantics, attributes,
// carrier and lane recipes, and fact-derived operands. Source graphs may be
// borrowed during this call; payloads reference only module storage or the
// retained program allocation. Ordinary source operands remain direct IR field
// reads. Returned rows belong to the function arena and outlive construction
// scratch.
iree_status_t loom_low_lower_rule_plan_finalize(
    loom_low_lower_context_t* context,
    loom_low_lower_rule_source_memory_plan_builder_t* source_memory_builder,
    loom_low_lower_selected_plan_t* selected_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_RULE_PLAN_H_
