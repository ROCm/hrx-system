// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent operand preparation facts for contractions.

#ifndef LOOM_ANALYSIS_CONTRACT_PREPARATION_H_
#define LOOM_ANALYSIS_CONTRACT_PREPARATION_H_

#include "iree/base/api.h"
#include "loom/analysis/contract_storage.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_contract_preparation_family_e {
  // Unknown or uninitialized preparation family.
  LOOM_CONTRACT_PREPARATION_FAMILY_UNKNOWN = 0,
  // The source operand is consumed in its current logical/storage layout.
  LOOM_CONTRACT_PREPARATION_FAMILY_NONE = 1,
  // Right-hand operand is physically prepared for MMT4D-style N-major blocks.
  LOOM_CONTRACT_PREPARATION_FAMILY_MMT4D_RHS_N_MAJOR_BLOCKED = 2,
  // Operand is prepared as a subgroup-owned matrix fragment.
  LOOM_CONTRACT_PREPARATION_FAMILY_SUBGROUP_MATRIX_FRAGMENT = 3,
  // Operand requires an explicit numeric transform before contract selection.
  LOOM_CONTRACT_PREPARATION_FAMILY_NUMERIC_TRANSFORM = 4,
} loom_contract_preparation_family_t;

typedef enum loom_contract_preparation_availability_e {
  // No availability decision has been made.
  LOOM_CONTRACT_PREPARATION_AVAILABILITY_UNKNOWN = 0,
  // The requested preparation can be inserted or is already materialized.
  LOOM_CONTRACT_PREPARATION_AVAILABILITY_AVAILABLE = 1,
  // The requested preparation is not available for the current target snapshot.
  LOOM_CONTRACT_PREPARATION_AVAILABILITY_UNAVAILABLE = 2,
  // The requested preparation would need to have happened earlier.
  LOOM_CONTRACT_PREPARATION_AVAILABILITY_TOO_LATE = 3,
} loom_contract_preparation_availability_t;

typedef enum loom_contract_numeric_transform_e {
  // Unknown or uninitialized numeric transform state.
  LOOM_CONTRACT_NUMERIC_TRANSFORM_UNKNOWN = 0,
  // Physical preparation preserves the operand's numeric interpretation.
  LOOM_CONTRACT_NUMERIC_TRANSFORM_NONE = 1,
  // A decode must materialize the logical payload before contraction.
  LOOM_CONTRACT_NUMERIC_TRANSFORM_DECODE = 2,
  // A decode and repack must materialize a different payload representation.
  LOOM_CONTRACT_NUMERIC_TRANSFORM_DECODE_REPACK = 3,
} loom_contract_numeric_transform_t;

typedef enum loom_contract_preparation_flag_bits_e {
  // The preparation result carries explicit address-layout facts.
  LOOM_CONTRACT_PREPARATION_HAS_ADDRESS_LAYOUT = 1u << 0,
  // The source storage schema is preserved by this preparation.
  LOOM_CONTRACT_PREPARATION_PRESERVES_STORAGE_SCHEMA = 1u << 1,
  // The logical RHS of the algebra is transposed before contraction.
  LOOM_CONTRACT_PREPARATION_LOGICAL_RHS_TRANSPOSE = 1u << 2,
  // The physical operand layout is packed independently of algebraic transpose.
  LOOM_CONTRACT_PREPARATION_PHYSICAL_PACKING = 1u << 3,
  // The physical RHS layout is N-major blocked for MMT4D-style kernels.
  LOOM_CONTRACT_PREPARATION_PHYSICAL_N_MAJOR_BLOCKED = 1u << 4,
  // The preparation constrains fragment or lane ownership.
  LOOM_CONTRACT_PREPARATION_FRAGMENT_OWNERSHIP = 1u << 5,
} loom_contract_preparation_flag_bits_t;

// Bitset of loom_contract_preparation_flag_bits_t values.
typedef uint32_t loom_contract_preparation_flags_t;

enum loom_contract_preparation_rejection_bits_e {
  // No preparation rejection was recorded.
  LOOM_CONTRACT_PREPARATION_REJECTION_NONE = 0u,
  // The requested preparation was incomplete or inconsistent.
  LOOM_CONTRACT_PREPARATION_REJECTION_INVALID_REQUEST = 1u << 0,
  // The requested preparation family cannot apply to the operand role.
  LOOM_CONTRACT_PREPARATION_REJECTION_ROLE = 1u << 1,
  // The requested preparation family is unavailable.
  LOOM_CONTRACT_PREPARATION_REJECTION_UNAVAILABLE = 1u << 2,
  // The requested preparation would have needed to be inserted earlier.
  LOOM_CONTRACT_PREPARATION_REJECTION_TOO_LATE = 1u << 3,
  // Optimized lowering policy does not permit the available fallback.
  LOOM_CONTRACT_PREPARATION_REJECTION_POLICY = 1u << 4,
  // The requested physical preparation is missing address-layout facts.
  LOOM_CONTRACT_PREPARATION_REJECTION_ADDRESS_LAYOUT = 1u << 5,
  // The requested numeric transform is missing or inconsistent.
  LOOM_CONTRACT_PREPARATION_REJECTION_NUMERIC_TRANSFORM = 1u << 6,
};

// Bitset of loom_contract_preparation_rejection_bits_e values.
typedef uint32_t loom_contract_preparation_rejection_bits_t;

typedef struct loom_contract_preparation_diagnostic_t {
  // Bitset of loom_contract_preparation_rejection_bits_t values.
  loom_contract_preparation_rejection_bits_t rejection_bits;
} loom_contract_preparation_diagnostic_t;

typedef struct loom_contract_operand_preparation_t {
  // Source operand role being prepared.
  loom_contract_operand_role_t role;

  // Selected preparation family.
  loom_contract_preparation_family_t family;

  // Bitset of loom_contract_preparation_flags_t values.
  loom_contract_preparation_flags_t flags;

  // Numeric transform required before contraction.
  loom_contract_numeric_transform_t numeric_transform;

  // Payload facts before preparation.
  loom_contract_view_payload_t source_payload;

  // Address-layout facts after preparation when HAS_ADDRESS_LAYOUT is set.
  loom_value_fact_address_layout_t address_layout;

  // Storage-schema facts preserved from source_payload.
  loom_value_fact_storage_schema_t storage_schema;
} loom_contract_operand_preparation_t;

typedef struct loom_contract_operand_preparation_options_t {
  // Source operand role being prepared.
  loom_contract_operand_role_t role;

  // Desired preparation family for the selected target snapshot.
  loom_contract_preparation_family_t family;

  // Availability of family at the current lowering point.
  loom_contract_preparation_availability_t availability;

  // Fallback and optimized-path selection policy.
  loom_lowering_policy_t policy;

  // Payload facts before preparation.
  loom_contract_view_payload_t source_payload;

  // Additional logical transform facts already proven by the caller.
  loom_contract_preparation_flags_t logical_flags;

  // Address-layout facts after preparation.
  loom_value_fact_address_layout_t address_layout;

  // Numeric transform required before contraction.
  loom_contract_numeric_transform_t numeric_transform;
} loom_contract_operand_preparation_options_t;

// Selects operand preparation facts for a target-independent contract.
//
// The selection keeps algebra, physical packing, and numeric transforms
// separate. If the desired family is unavailable and policy permits reference
// fallback, the result is a NONE preparation. Optimized-required policies fail
// instead of silently losing the requested preparation.
bool loom_contract_operand_preparation_select(
    const loom_contract_operand_preparation_options_t* options,
    loom_contract_operand_preparation_t* out_preparation,
    loom_contract_preparation_diagnostic_t* out_diagnostic);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_CONTRACT_PREPARATION_H_
