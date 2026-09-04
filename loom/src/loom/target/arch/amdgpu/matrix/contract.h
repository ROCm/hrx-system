// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU matrix target contracts.
//
// This file describes the target-native matrix primitives that Loom can select
// from a higher-level tile.contract after shapes, encodings, layouts, and value
// facts are refined enough to make the choice structural. The descriptors are
// intentionally data-only: lowering code can query exact shape/type/feature
// requirements without hard-coding AMDGPU intrinsic names throughout generic
// tile/vector passes.

#ifndef LOOM_TARGET_ARCH_AMDGPU_MATRIX_CONTRACT_H_
#define LOOM_TARGET_ARCH_AMDGPU_MATRIX_CONTRACT_H_

#include "loom/target/arch/amdgpu/matrix/types.h"
#include "loom/util/numeric_format.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the stable display name for a matrix family.
iree_string_view_t loom_amdgpu_matrix_family_name(
    loom_amdgpu_matrix_family_t family);

// Returns the stable display name for a matrix numeric type.
iree_string_view_t loom_amdgpu_matrix_numeric_type_name(
    loom_amdgpu_matrix_numeric_type_t numeric_type);

// Returns the stable display name for a scale kind.
iree_string_view_t loom_amdgpu_matrix_scale_kind_name(
    loom_amdgpu_matrix_scale_kind_t scale_kind);

// Maps an encoded scale numeric format to the target selector value.
bool loom_amdgpu_matrix_scale_format_selector_from_numeric_format(
    loom_value_fact_numeric_format_flags_t format, int64_t* out_value);

// Returns a target-owned fragment layout by kind, or NULL when unknown.
const loom_amdgpu_matrix_fragment_layout_t*
loom_amdgpu_matrix_fragment_layout_for_kind(
    loom_amdgpu_matrix_fragment_layout_kind_t kind);

// Returns the target-owned fragment layout attached to |descriptor|, or NULL
// when the descriptor has no reusable lane/register facts yet.
const loom_amdgpu_matrix_fragment_layout_t*
loom_amdgpu_matrix_contract_descriptor_fragment_layout(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor);

// Maps a matrix feature profile enum to matrix feature bits.
bool loom_amdgpu_matrix_feature_bits_from_profile(
    loom_amdgpu_matrix_feature_profile_t profile,
    loom_amdgpu_matrix_feature_bits_t* out_feature_bits);

// Returns the number of named matrix feature bits.
iree_host_size_t loom_amdgpu_matrix_feature_info_count(void);

// Returns a named matrix feature bit by ordinal, or NULL when |index| is out
// of range.
const loom_amdgpu_matrix_feature_info_t* loom_amdgpu_matrix_feature_info_at(
    iree_host_size_t index);

// Maps a processor name such as "gfx942" or "gfx1250" to matrix feature bits.
iree_status_t loom_amdgpu_matrix_feature_bits_from_processor(
    iree_string_view_t processor,
    loom_amdgpu_matrix_feature_bits_t* out_feature_bits);

// Returns the number of built-in AMDGPU matrix contract descriptors.
iree_host_size_t loom_amdgpu_matrix_contract_descriptor_count(void);

// Returns a built-in descriptor by ordinal, or NULL when |index| is out of
// range.
const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_contract_descriptor_at(iree_host_size_t index);

// Returns generated physical-representation choices for a built-in matrix
// contract ordinal, or NULL when |index| is out of range. The
// operand-exchanged choice applies the identity
// A*B=transpose(transpose(B)*transpose(A)).
const loom_amdgpu_matrix_contract_realization_choices_t*
loom_amdgpu_matrix_contract_realization_choices_at(iree_host_size_t index);

// Returns an exact result representation by ID, or NULL for NONE or an ID
// outside the generated catalog.
const loom_amdgpu_matrix_result_representation_t*
loom_amdgpu_matrix_result_representation_at(
    loom_amdgpu_matrix_result_representation_id_t representation_id);

// Returns a built-in descriptor with the same wait-state behavior as
// |low_descriptor_ref|, or NULL when the descriptor ref is not a matrix
// contract.
const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_contract_wait_state_descriptor_for_low_descriptor_ref(
    loom_amdgpu_descriptor_ref_t low_descriptor_ref);

// Returns whether a descriptor is legal for a processor feature set and wave
// size. Pass wave_size=0 to ignore wave-size filtering.
bool loom_amdgpu_matrix_contract_is_available(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_amdgpu_matrix_feature_bits_t feature_bits, uint32_t wave_size);

// Selects the first descriptor that satisfies a fully structural match request.
// Returns NULL when no descriptor matches and optionally populates
// |out_diagnostic| with the first filter that rejected all candidates.
const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_contract_select(
    const loom_amdgpu_matrix_contract_match_request_t* request,
    loom_amdgpu_matrix_contract_match_diagnostic_t* out_diagnostic);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_MATRIX_CONTRACT_H_
