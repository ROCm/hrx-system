// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for vector fragment memory source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_view_region_table_t loom_view_region_table_t;

enum {
  // Logical rank of a one-block matrix-fragment source or destination view.
  LOOM_AMDGPU_FRAGMENT_UNBLOCKED_VIEW_RANK = 2,
  // Logical rank of a blocked matrix-fragment source or destination view.
  LOOM_AMDGPU_FRAGMENT_BLOCKED_VIEW_RANK =
      LOOM_AMDGPU_FRAGMENT_MEMORY_VIEW_RANK_CAPACITY,
  // Physical byte width of one matrix-fragment register.
  LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT = 4,
  // Number of packed 16-bit elements carried by one fragment register.
  LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT = 2,
  // Physical bit width of one packed 16-bit fragment element.
  LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_BIT_COUNT = 16,
};

typedef struct loom_amdgpu_matrix_fragment_contract_candidates_t {
  // Descriptor set used to filter descriptors in this list.
  const loom_low_descriptor_set_t* descriptor_set;
  // Matrix features used to filter descriptors in this list.
  loom_amdgpu_matrix_feature_bits_t feature_bits;
  // Wave size used to filter descriptors in this list.
  uint32_t wave_size;
  // Stable built-in ordinals of matrix contracts available to the target.
  const uint16_t* descriptor_ordinals;
  // Number of entries in |descriptor_ordinals|.
  iree_host_size_t descriptor_count;
  // Exact canonical result representations available on this target.
  uint64_t canonical_result_representation_bits;
  // Canonical and operand-exchanged result representations available on this
  // target.
  uint64_t exact_result_representation_bits;
} loom_amdgpu_matrix_fragment_contract_candidates_t;

typedef struct loom_amdgpu_matrix_fragment_role_storage_t {
  // Matrix descriptor payload selected for the fragment role.
  loom_amdgpu_matrix_payload_shape_t payload;
  // Physical scalar type used to move each logical payload element.
  loom_scalar_type_t element_type;
} loom_amdgpu_matrix_fragment_role_storage_t;

typedef struct loom_amdgpu_matrix_fragment_lane_ids_t {
  // Full subgroup lane id.
  loom_value_id_t lane;
  // Lane id modulo the fragment row/column tile when materialized.
  loom_value_id_t lane_mod;
  // Lane id divided by the fragment row/column tile when materialized.
  loom_value_id_t lane_div;
} loom_amdgpu_matrix_fragment_lane_ids_t;

// Returns true when |blocks|, |rows|, and |columns| match |role| in |shape|.
bool loom_amdgpu_matrix_fragment_tile_shape_matches(
    const loom_value_fact_table_t* fact_table,
    loom_amdgpu_matrix_tile_shape_t shape, loom_contract_operand_role_t role,
    loom_value_id_t blocks, loom_value_id_t rows, loom_value_id_t columns);

// Returns true when |role| carries a matrix accumulator or result fragment.
bool loom_amdgpu_matrix_fragment_role_is_result_like(
    loom_contract_operand_role_t role);

// Maps one native fragment axis to its physical source-view axis. Exact
// transposed result representations exchange M/N while input roles retain
// their source coordinates.
uint8_t loom_amdgpu_matrix_fragment_role_view_axis(
    loom_contract_operand_role_t role, uint8_t view_rank,
    loom_amdgpu_matrix_result_representation_flags_t representation_flags,
    loom_matrix_fragment_axis_t native_axis);

// Returns the logical source tile shape for a native fragment
// representation.
loom_amdgpu_matrix_tile_shape_t loom_amdgpu_matrix_fragment_source_tile_shape(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role,
    loom_amdgpu_matrix_result_representation_flags_t representation_flags);

// Returns true when one source value carries the exact shape and payload of a
// generated result representation.
bool loom_amdgpu_matrix_result_representation_matches_value(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id,
    loom_amdgpu_matrix_result_representation_id_t representation_id);

// Returns the physical elements carried by each register in |role_layout|.
uint16_t loom_amdgpu_matrix_fragment_payload_elements_per_register(
    const loom_matrix_fragment_role_layout_t* role_layout);

// Returns true when one logical role element occupies a low register subword.
bool loom_amdgpu_matrix_fragment_role_layout_uses_low_subword(
    const loom_matrix_fragment_role_layout_t* role_layout);

// Returns true when one result-role register stores two semantic b16 elements.
bool loom_amdgpu_matrix_fragment_role_layout_uses_packed_b16_elements(
    loom_contract_operand_role_t role,
    const loom_matrix_fragment_role_layout_t* role_layout);

// Returns true when |blocks|, |rows|, and |columns| match the logical shape of
// |role| in |layout|.
bool loom_amdgpu_matrix_fragment_shape_matches(
    const loom_value_fact_table_t* fact_table,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role, loom_value_id_t blocks,
    loom_value_id_t rows, loom_value_id_t columns);

// Returns true when |payload_type| matches the physical storage selected for
// one matrix-fragment role.
bool loom_amdgpu_matrix_fragment_payload_matches_role_storage(
    loom_type_t payload_type, loom_scalar_type_t expected_element_type,
    const loom_matrix_fragment_role_layout_t* role_layout);

// Returns the physical storage selected for |role| by |descriptor|.
bool loom_amdgpu_matrix_fragment_descriptor_role_storage(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_contract_operand_role_t role,
    loom_amdgpu_matrix_fragment_role_storage_t* out_storage);

// Returns true when |descriptor| is available for the selected target.
bool loom_amdgpu_matrix_fragment_contract_is_available(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_matrix_feature_bits_t feature_bits, uint32_t wave_size);

// Returns the number of cached target-compatible descriptors, or the complete
// matrix descriptor count when |candidates| is NULL.
iree_host_size_t loom_amdgpu_matrix_fragment_contract_candidate_count(
    const loom_amdgpu_matrix_fragment_contract_candidates_t* candidates);

// Returns one cached target-compatible descriptor, or one descriptor from the
// complete matrix table when |candidates| is NULL.
const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_fragment_contract_candidate_at(
    const loom_amdgpu_matrix_fragment_contract_candidates_t* candidates,
    iree_host_size_t index);

// Maps one matrix numeric type to its physical scalar storage type.
bool loom_amdgpu_matrix_fragment_scalar_type_from_numeric(
    loom_amdgpu_matrix_numeric_type_t numeric_type,
    loom_scalar_type_t* out_element_type);

// Returns matrix feature bits for the selected AMDGPU target facts.
loom_amdgpu_matrix_feature_bits_t loom_amdgpu_matrix_fragment_feature_bits(
    const loom_amdgpu_target_facts_t* target_facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_H_
