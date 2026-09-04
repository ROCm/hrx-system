// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Function-local AMDGPU matrix-fragment lowering state.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_STATE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_STATE_H_

#include "loom/target/arch/amdgpu/lower/matrix_fragment.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_matrix_fragment_lane_id_cache_t {
  // Low block where lane ids were materialized.
  loom_block_t* block;
  // Register type used for the materialized lane ids.
  loom_type_t vgpr_type;
  // Tile row/column divisor used to derive lane_mod and lane_div.
  uint16_t lane_divisor;
  // Cached lane ids valid in block append order.
  loom_amdgpu_matrix_fragment_lane_ids_t lane_ids;
} loom_amdgpu_matrix_fragment_lane_id_cache_t;

typedef enum loom_amdgpu_fragment_memory_address_register_kind_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE = 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR,
  LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR,
} loom_amdgpu_fragment_memory_address_register_kind_t;

typedef struct loom_amdgpu_fragment_memory_address_accumulator_t {
  // Low scalar register containing the accumulated byte address.
  loom_value_id_t value;
  // Register class of value.
  loom_amdgpu_fragment_memory_address_register_kind_t register_kind;
} loom_amdgpu_fragment_memory_address_accumulator_t;

#define LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_PRODUCT_VALUE_CAPACITY \
  (LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY + 1)

typedef struct loom_amdgpu_fragment_memory_address_product_key_t {
  // Source SSA values multiplied to materialize the address product.
  loom_value_id_t
      values[LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_PRODUCT_VALUE_CAPACITY];
  // Number of populated source SSA values.
  uint8_t value_count;
  // Static byte coefficient multiplied into the source values.
  int64_t static_byte_coefficient;
} loom_amdgpu_fragment_memory_address_product_key_t;

typedef struct loom_amdgpu_fragment_memory_runtime_axis_base_key_t {
  // Power-of-two divisor applied to the subgroup lane ID.
  uint16_t lane_divisor;
  // Optional power-of-two modulus applied after division; zero omits it.
  uint16_t lane_modulus;
  // Logical coordinate scale applied to the resulting lane digit.
  uint16_t lane_coordinate_scale;
  // Runtime byte-stride product materialized for the axis.
  loom_amdgpu_fragment_memory_address_product_key_t byte_stride;
} loom_amdgpu_fragment_memory_runtime_axis_base_key_t;

typedef struct loom_amdgpu_fragment_memory_address_base_key_t {
  // Low block where the base address was materialized.
  loom_block_t* block;
  // Register type used for any VGPR address terms.
  loom_type_t vgpr_type;
  // Number of lane coordinate terms in the key.
  uint8_t lane_term_count;
  // Lane coordinate terms in the compiled address layout.
  loom_amdgpu_fragment_memory_lane_term_t
      lane_terms[LOOM_MATRIX_FRAGMENT_AXIS_COUNT];
  // Number of dynamic source terms in the key.
  uint8_t dynamic_term_count;
  // Source products used for dynamic byte-address terms.
  loom_amdgpu_fragment_memory_address_product_key_t
      dynamic_terms[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Runtime products indexed by physical view axis. Empty products have no
  // runtime contribution.
  loom_amdgpu_fragment_memory_runtime_axis_base_key_t
      runtime_axes[LOOM_AMDGPU_FRAGMENT_MEMORY_VIEW_RANK_CAPACITY];
} loom_amdgpu_fragment_memory_address_base_key_t;

typedef struct loom_amdgpu_fragment_memory_address_base_cache_t {
  // Whether the cache contains a complete base and runtime-axis products.
  bool valid;
  // Key for the cached base accumulator.
  loom_amdgpu_fragment_memory_address_base_key_t key;
  // Cached low address accumulator; NONE is a valid zero base.
  loom_amdgpu_fragment_memory_address_accumulator_t accumulator;
  // Cached byte-stride product for each runtime physical axis.
  loom_amdgpu_fragment_memory_address_accumulator_t
      runtime_axis_byte_strides[LOOM_AMDGPU_FRAGMENT_MEMORY_VIEW_RANK_CAPACITY];
} loom_amdgpu_fragment_memory_address_base_cache_t;

typedef struct loom_amdgpu_matrix_fragment_state_t {
  // Cached subgroup lane ids for adjacent fragment operations.
  loom_amdgpu_matrix_fragment_lane_id_cache_t lane_ids;
  // Cached base address shared by adjacent fragment memory operations.
  loom_amdgpu_fragment_memory_address_base_cache_t address_base;
  // Cached matrix contract descriptors available to this source function.
  loom_amdgpu_matrix_fragment_contract_candidates_t contract_candidates;
  // Canonical matrix contract ordinal by source value ordinal. Allocated only
  // when exact result-representation planning observes a matrix producer.
  uint16_t* contract_ordinals_by_value_ordinal;
  // Number of entries in |contract_ordinals_by_value_ordinal|.
  loom_value_ordinal_t contract_ordinal_count;
} loom_amdgpu_matrix_fragment_state_t;

// Returns the required function-local matrix-fragment lowering state.
iree_status_t loom_amdgpu_matrix_fragment_state(
    loom_low_lower_context_t* context,
    loom_amdgpu_matrix_fragment_state_t** out_state);

// Returns the function-local target-compatible matrix contract candidates.
iree_status_t loom_amdgpu_matrix_fragment_contract_candidates(
    loom_low_lower_context_t* context,
    const loom_amdgpu_matrix_fragment_contract_candidates_t** out_candidates);

// Records the canonical matrix contract producing one source result value.
iree_status_t loom_amdgpu_matrix_fragment_record_contract_ordinal(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    uint16_t contract_ordinal);

// Queries a recorded canonical contract ordinal through a contract query
// scope. Returns NONE when no producer was recorded for the value.
iree_status_t loom_amdgpu_matrix_fragment_query_contract_ordinal(
    const loom_target_contract_query_environment_t* environment,
    loom_value_id_t source_value_id, uint16_t* out_contract_ordinal);

// Emits or reuses the subgroup lane id for a matrix-fragment operation.
iree_status_t loom_amdgpu_emit_matrix_fragment_lane_ids(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t lane_divisor, loom_type_t vgpr_type,
    loom_amdgpu_matrix_fragment_lane_ids_t* out_lane_ids);

// Materializes the lane id modulo |lane_divisor| when not already available.
iree_status_t loom_amdgpu_ensure_matrix_fragment_lane_mod(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t lane_divisor, loom_type_t vgpr_type,
    loom_amdgpu_matrix_fragment_lane_ids_t* inout_lane_ids);

// Materializes the lane id divided by |lane_divisor| when not already
// available.
iree_status_t loom_amdgpu_ensure_matrix_fragment_lane_div(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t lane_divisor, loom_type_t vgpr_type,
    loom_amdgpu_matrix_fragment_lane_ids_t* inout_lane_ids);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_STATE_H_
