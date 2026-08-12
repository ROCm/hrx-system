// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable descriptor-derived AMDGPU VOPD planning data.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_VOPD_DATA_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_VOPD_DATA_H_

#include "iree/base/api.h"
#include "loom/codegen/low/placement_pair.h"
#include "loom/target/arch/amdgpu/planning/vopd_component.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_vopd_component_source_bits_e {
  // Component has no register source operands.
  LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_NONE = 0u,
  // Component source 0 is a VGPR and participates in VOPD constraints.
  LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0 = 1u << 0,
  // Component source 1 is a VGPR and participates in VOPD constraints.
  LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_VSRC1 = 1u << 1,
  // Component has both VOPD source operands modeled as VGPRs.
  LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY =
      LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0 |
      LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_VSRC1,
} loom_amdgpu_vopd_component_source_bits_t;
typedef uint8_t loom_amdgpu_vopd_component_source_mask_t;

typedef enum loom_amdgpu_vopd_component_rule_flag_bits_e {
  // Component rule has no additional planning capabilities.
  LOOM_AMDGPU_VOPD_COMPONENT_FLAG_NONE = 0u,
  // SRC0 and VSRC1 may be exchanged without changing component semantics.
  LOOM_AMDGPU_VOPD_COMPONENT_FLAG_COMMUTABLE_SOURCES = 1u << 0,
  // A same-op dual move routes its Y source through the SRC2 cache.
  LOOM_AMDGPU_VOPD_COMPONENT_FLAG_DUAL_MOV_SRC2_CACHE = 1u << 1,
} loom_amdgpu_vopd_component_rule_flag_bits_t;
typedef uint8_t loom_amdgpu_vopd_component_rule_flags_t;

// Operand layout for VOPD component forms whose sources are row-defined.
typedef struct loom_amdgpu_vopd_component_operand_layout_t {
  // Operand index of the accumulator tied to the result register.
  uint8_t accumulator_index;
  // Operand index of the source encoded in the VOPD SRC0 field.
  uint8_t src0_index;
  // Operand index of the source encoded in the VOPD VSRC1 field.
  uint8_t vsrc1_index;
} loom_amdgpu_vopd_component_operand_layout_t;

typedef struct loom_amdgpu_vopd_component_rule_t {
  // Canonical native facts for the component opcode.
  const loom_amdgpu_vopd_component_info_t* info;
  // Operand/register form selected by this source descriptor.
  loom_amdgpu_vopd_component_form_t form;
  // Operand indexes interpreted by forms with row-defined source layout.
  loom_amdgpu_vopd_component_operand_layout_t operands;
  // Source operand slots that contain real VGPRs.
  loom_amdgpu_vopd_component_source_mask_t source_register_mask;
  // Descriptor-derived component planning capabilities.
  loom_amdgpu_vopd_component_rule_flags_t flags;
} loom_amdgpu_vopd_component_rule_t;

typedef struct loom_amdgpu_vopd_component_descriptor_lookup_range_t {
  // First descriptor-ordinal lookup row for this descriptor-set ordinal.
  uint16_t first_descriptor_lookup;
  // Number of descriptor-ordinal lookup rows for this descriptor-set ordinal.
  uint16_t descriptor_lookup_count;
} loom_amdgpu_vopd_component_descriptor_lookup_range_t;

typedef struct loom_amdgpu_vopd_pair_affinity_range_t {
  // First pair-affinity row for this descriptor-set ordinal.
  uint16_t first_pair_affinity;
  // Number of pair-affinity rows for this descriptor-set ordinal.
  uint16_t pair_affinity_count;
} loom_amdgpu_vopd_pair_affinity_range_t;

typedef struct loom_amdgpu_vopd_pair_affinity_row_t {
  // Descriptor ordinal for the first scheduled packet.
  uint16_t first_descriptor_ordinal;
  // Descriptor ordinal for the second scheduled packet.
  uint16_t second_descriptor_ordinal;
  // Scheduler priority for this descriptor pair.
  uint16_t priority;
  // Pair-placement recipe index + 1, or zero when absent.
  uint16_t placement_recipe_index_plus_one;
} loom_amdgpu_vopd_pair_affinity_row_t;

// Canonical native facts indexed indirectly by component opcode and reason.
extern const loom_amdgpu_vopd_component_info_t
    loom_amdgpu_vopd_component_infos[];

// Descriptor-specific component decoding rules.
extern const loom_amdgpu_vopd_component_rule_t
    loom_amdgpu_vopd_component_rules[];

// Dense descriptor-ordinal to component-rule index+1 lookups.
extern const uint8_t loom_amdgpu_vopd_component_descriptor_lookups[];

// Descriptor lookup ranges indexed by descriptor-set ordinal.
extern const loom_amdgpu_vopd_component_descriptor_lookup_range_t
    loom_amdgpu_vopd_component_descriptor_lookup_ranges[];

// Pair-affinity ranges indexed by descriptor-set ordinal.
extern const loom_amdgpu_vopd_pair_affinity_range_t
    loom_amdgpu_vopd_pair_affinity_ranges[];

// Descriptor pairs worth making adjacent during target-low scheduling.
extern const loom_amdgpu_vopd_pair_affinity_row_t
    loom_amdgpu_vopd_pair_affinities[];

// Placement recipes referenced by pair-affinity rows.
extern const loom_low_placement_pair_recipe_t
    loom_amdgpu_vopd_pair_placement_recipes[];

// Number of rows in |loom_amdgpu_vopd_pair_placement_recipes|.
extern const iree_host_size_t loom_amdgpu_vopd_pair_placement_recipe_count;

// Dense component opcode to component-info index+1 lookup.
extern const uint8_t
    loom_amdgpu_vopd_component_info_indices_by_op[LOOM_AMDGPU_VOPD_OP_MIN_I32 +
                                                  1];

// Dense same-op pair reason to component-info index+1 lookup.
extern const uint8_t loom_amdgpu_vopd_component_info_indices_by_same_op_reason
    [LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_CNDMASK_B32 + 1];

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_VOPD_DATA_H_
