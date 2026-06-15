// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"

#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static bool loom_amdgpu_descriptor_matches_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  return descriptor != NULL &&
         descriptor == loom_amdgpu_descriptor_ref_descriptor(descriptor_set,
                                                             descriptor_ref);
}

static bool loom_amdgpu_descriptor_matches_any_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    const loom_amdgpu_descriptor_ref_t* descriptor_refs,
    iree_host_size_t descriptor_ref_count) {
  for (iree_host_size_t i = 0; i < descriptor_ref_count; ++i) {
    if (loom_amdgpu_descriptor_matches_ref(descriptor_set, descriptor,
                                           descriptor_refs[i])) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_encoding_format_is_vector_memory(uint16_t format_id) {
  switch (format_id) {
    case LOOM_AMDGPU_ENCODING_FORMAT_MUBUF:
    case LOOM_AMDGPU_ENCODING_FORMAT_VBUFFER:
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT:
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLBL:
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLOBAL:
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_SCRATCH:
    case LOOM_AMDGPU_ENCODING_FORMAT_VFLAT:
    case LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL:
    case LOOM_AMDGPU_ENCODING_FORMAT_VSCRATCH:
      return true;
    default:
      return false;
  }
}

bool loom_amdgpu_descriptor_uses_resource_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_resource_kind_t kind) {
  if (descriptor_set == NULL || descriptor == NULL ||
      descriptor->schedule_class_id >= descriptor_set->schedule_class_count) {
    return false;
  }
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[descriptor->schedule_class_id];
  const uint32_t end = (uint32_t)schedule_class->issue_use_start +
                       schedule_class->issue_use_count;
  if (end > descriptor_set->issue_use_count) {
    return false;
  }
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &descriptor_set->issue_uses[schedule_class->issue_use_start + i];
    if (issue_use->resource_id >= descriptor_set->resource_count) {
      continue;
    }
    if (descriptor_set->resources[issue_use->resource_id].kind == kind) {
      return true;
    }
  }
  return false;
}

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

loom_amdgpu_descriptor_traits_t loom_amdgpu_descriptor_traits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor_set == NULL || descriptor == NULL) {
    return 0;
  }
  loom_amdgpu_descriptor_traits_t traits = 0;
  if (loom_amdgpu_descriptor_uses_resource_kind(
          descriptor_set, descriptor, LOOM_LOW_RESOURCE_KIND_VECTOR_ALU)) {
    traits |= LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU;
  }
  if (loom_amdgpu_descriptor_uses_resource_kind(
          descriptor_set, descriptor, LOOM_LOW_RESOURCE_KIND_SCALAR_ALU)) {
    traits |= LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU;
  }
  if (loom_amdgpu_encoding_format_is_vector_memory(
          descriptor->encoding_format_id)) {
    traits |= LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY;
  }
  switch (descriptor->encoding_format_id) {
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1_SDWA:
      traits |= LOOM_AMDGPU_DESCRIPTOR_TRAIT_SDWA;
      break;
    default:
      break;
  }
  static const loom_amdgpu_descriptor_ref_t kTranscendentalDescriptorRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_EXP_F32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LOG_F32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_SIN_F32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_COS_F32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_SQRT_F32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_RSQ_F32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_RCP_F32,
  };
  if (loom_amdgpu_descriptor_matches_any_ref(
          descriptor_set, descriptor, kTranscendentalDescriptorRefs,
          IREE_ARRAYSIZE(kTranscendentalDescriptorRefs))) {
    traits |= LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL;
  }
  static const loom_amdgpu_descriptor_ref_t kDppDescriptorRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP16,
  };
  if (loom_amdgpu_descriptor_matches_any_ref(
          descriptor_set, descriptor, kDppDescriptorRefs,
          IREE_ARRAYSIZE(kDppDescriptorRefs))) {
    traits |= LOOM_AMDGPU_DESCRIPTOR_TRAIT_DPP;
  }
  static const loom_amdgpu_descriptor_ref_t kReadfirstlaneDescriptorRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32,
  };
  if (loom_amdgpu_descriptor_matches_any_ref(
          descriptor_set, descriptor, kReadfirstlaneDescriptorRefs,
          IREE_ARRAYSIZE(kReadfirstlaneDescriptorRefs))) {
    traits |= LOOM_AMDGPU_DESCRIPTOR_TRAIT_READFIRSTLANE;
  }
  return traits;
}
