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

// Returns a target-owned fragment layout by kind, or NULL when unknown.
const loom_amdgpu_matrix_fragment_layout_t*
loom_amdgpu_matrix_fragment_layout_for_kind(
    loom_amdgpu_matrix_fragment_layout_kind_t kind);

// Returns the target-owned fragment layout attached to |descriptor|, or NULL
// when the descriptor has no reusable lane/register facts yet.
const loom_amdgpu_matrix_fragment_layout_t*
loom_amdgpu_matrix_contract_descriptor_fragment_layout(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor);

// Returns the role layout within |layout|, or NULL when the role is not
// modeled.
const loom_amdgpu_matrix_fragment_role_layout_t*
loom_amdgpu_matrix_fragment_role_layout(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role);

// Maps a lane-local payload register element to a logical matrix coordinate.
bool loom_amdgpu_matrix_fragment_coordinate(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role, uint16_t lane, uint16_t register_index,
    uint16_t element_index,
    loom_amdgpu_matrix_fragment_coordinate_t* out_coordinate);

// Maps a matrix feature profile enum to matrix feature bits.
bool loom_amdgpu_matrix_feature_bits_from_profile(
    loom_amdgpu_matrix_feature_profile_t profile,
    loom_amdgpu_matrix_feature_bits_t* out_feature_bits);

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
