// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"

bool loom_amdgpu_descriptor_uses_vector_alu(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return iree_any_bit_set(
      loom_amdgpu_descriptor_traits(descriptor_set, descriptor),
      LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU);
}

bool loom_amdgpu_descriptor_uses_scalar_alu(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return iree_any_bit_set(
      loom_amdgpu_descriptor_traits(descriptor_set, descriptor),
      LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU);
}

bool loom_amdgpu_descriptor_uses_matrix(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return iree_any_bit_set(
      loom_amdgpu_descriptor_traits(descriptor_set, descriptor),
      LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX);
}

bool loom_amdgpu_descriptor_uses_vector_memory(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return iree_any_bit_set(
      loom_amdgpu_descriptor_traits(descriptor_set, descriptor),
      LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY);
}

bool loom_amdgpu_descriptor_is_transcendental(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return iree_any_bit_set(
      loom_amdgpu_descriptor_traits(descriptor_set, descriptor),
      LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL);
}

bool loom_amdgpu_descriptor_is_dpp(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return iree_any_bit_set(
      loom_amdgpu_descriptor_traits(descriptor_set, descriptor),
      LOOM_AMDGPU_DESCRIPTOR_TRAIT_DPP);
}

bool loom_amdgpu_descriptor_is_readfirstlane(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return iree_any_bit_set(
      loom_amdgpu_descriptor_traits(descriptor_set, descriptor),
      LOOM_AMDGPU_DESCRIPTOR_TRAIT_READFIRSTLANE);
}

bool loom_amdgpu_descriptor_is_sdwa(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return iree_any_bit_set(
      loom_amdgpu_descriptor_traits(descriptor_set, descriptor),
      LOOM_AMDGPU_DESCRIPTOR_TRAIT_SDWA);
}

bool loom_amdgpu_descriptor_implicitly_drains_xcnt(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  return iree_any_bit_set(
      loom_amdgpu_descriptor_traits(descriptor_set, descriptor),
      LOOM_AMDGPU_DESCRIPTOR_TRAIT_XCNT_IMPLICIT_DRAIN);
}

iree_status_t loom_amdgpu_descriptor_build_structural_state_reads(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_schedule_structural_state_read_list_t* out_state_reads) {
  *out_state_reads = loom_low_schedule_structural_state_read_list_empty();

  loom_low_schedule_structural_state_read_t* state_reads = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, 1, sizeof(*state_reads),
                                                 (void**)&state_reads));
  state_reads[0] = (loom_low_schedule_structural_state_read_t){
      .result_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      .state_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_EXEC,
  };
  *out_state_reads = (loom_low_schedule_structural_state_read_list_t){
      .values = state_reads,
      .count = 1,
  };
  return iree_ok_status();
}
