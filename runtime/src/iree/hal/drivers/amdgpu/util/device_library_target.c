// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/device_library_target.h"

typedef struct iree_hal_amdgpu_device_library_target_variant_t {
  iree_string_view_t artifact;
  iree_string_view_t target;
} iree_hal_amdgpu_device_library_target_variant_t;

static const iree_hal_amdgpu_device_library_target_variant_t
    iree_hal_amdgpu_device_library_target_variants[] = {
#define IREE_AMDGPU_DEVICE_LIBRARY_TARGET_VARIANT(artifact, target) \
  {IREE_SVL(artifact), IREE_SVL(target)},
#include "iree/hal/drivers/amdgpu/util/device_library_target_map.inl"
#undef IREE_AMDGPU_DEVICE_LIBRARY_TARGET_VARIANT
};

bool iree_hal_amdgpu_device_library_target_matches_file_arch(
    iree_string_view_t file_arch, iree_string_view_t target) {
  if (iree_string_view_is_empty(target)) return false;
  if (!iree_string_view_starts_with(file_arch, target)) {
    return false;
  }
  iree_string_view_t suffix =
      iree_string_view_remove_prefix(file_arch, target.size);
  return iree_string_view_is_empty(suffix) ||
         iree_string_view_starts_with(suffix, IREE_SV("."));
}

static iree_status_t
iree_hal_amdgpu_device_library_target_append_unique_candidate(
    iree_string_view_t target,
    iree_hal_amdgpu_device_library_target_candidate_list_t* candidates) {
  if (iree_string_view_is_empty(target)) return iree_ok_status();
  for (iree_host_size_t i = 0; i < candidates->count; ++i) {
    if (iree_string_view_equal(target, candidates->values[i].value)) {
      return iree_ok_status();
    }
  }
  if (candidates->count >= IREE_ARRAYSIZE(candidates->values)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU device library target candidate list capacity %" PRIhsz
        " exceeded",
        IREE_ARRAYSIZE(candidates->values));
  }
  iree_hal_amdgpu_device_library_target_candidate_t* candidate =
      &candidates->values[candidates->count];
  if (target.size >= sizeof(candidate->storage)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU device library target candidate length %" PRIhsz " exceeded",
        sizeof(candidate->storage) - 1);
  }
  memcpy(candidate->storage, target.data, target.size);
  candidate->storage[target.size] = 0;
  candidate->value = iree_make_string_view(candidate->storage, target.size);
  ++candidates->count;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_device_library_target_append_target_id_candidate(
    const iree_hal_amdgpu_target_id_t* target_id,
    iree_hal_amdgpu_device_library_target_candidate_list_t* candidates) {
  char target_id_buffer[64] = {0};
  iree_host_size_t target_id_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_target_id_format(target_id, sizeof(target_id_buffer),
                                       target_id_buffer, &target_id_length));
  return iree_hal_amdgpu_device_library_target_append_unique_candidate(
      iree_make_string_view(target_id_buffer, target_id_length), candidates);
}

static bool
iree_hal_amdgpu_device_library_target_feature_requires_qualification(
    iree_hal_amdgpu_target_feature_state_t variant_state,
    iree_hal_amdgpu_target_feature_state_t physical_state) {
  const bool variant_is_qualified =
      variant_state == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON ||
      variant_state == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF;
  const bool physical_is_qualified =
      physical_state == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON ||
      physical_state == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF;
  return variant_is_qualified && !physical_is_qualified;
}

static iree_status_t
iree_hal_amdgpu_device_library_target_lookup_variant_for_physical_target(
    const iree_hal_amdgpu_target_id_t* physical_target_id,
    iree_string_view_t* out_artifact, bool* out_requires_qualification) {
  *out_artifact = iree_string_view_empty();
  *out_requires_qualification = false;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amdgpu_device_library_target_variants);
       ++i) {
    const iree_hal_amdgpu_device_library_target_variant_t* variant =
        &iree_hal_amdgpu_device_library_target_variants[i];
    iree_hal_amdgpu_target_id_t variant_target_id;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_id_parse(
        variant->target,
        IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_ARCH_ONLY |
            IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_FEATURE_SUFFIXES,
        &variant_target_id));
    if (!iree_string_view_equal(variant_target_id.processor,
                                physical_target_id->processor)) {
      continue;
    }
    if (iree_hal_amdgpu_target_id_equal(&variant_target_id,
                                        physical_target_id)) {
      *out_artifact = variant->artifact;
      return iree_ok_status();
    }
    *out_requires_qualification |=
        iree_hal_amdgpu_device_library_target_feature_requires_qualification(
            variant_target_id.sramecc, physical_target_id->sramecc) ||
        iree_hal_amdgpu_device_library_target_feature_requires_qualification(
            variant_target_id.xnack, physical_target_id->xnack) ||
        iree_hal_amdgpu_device_library_target_feature_requires_qualification(
            variant_target_id.gfx1250_b0_specific,
            physical_target_id->gfx1250_b0_specific);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_device_library_target_candidates_from_agent_isa(
    const iree_hal_amdgpu_target_id_t* physical_target_id,
    const iree_hal_amdgpu_target_id_t* isa_target_id,
    iree_hal_amdgpu_device_library_target_candidate_list_t* out_candidates) {
  IREE_ASSERT_ARGUMENT(physical_target_id);
  IREE_ASSERT_ARGUMENT(isa_target_id);
  IREE_ASSERT_ARGUMENT(out_candidates);
  memset(out_candidates, 0, sizeof(*out_candidates));

  // A qualified artifact suppresses every fallback ISA for its physical
  // target. Otherwise a lower-priority generic ISA could reintroduce the
  // incompatible family artifact this variant exists to replace.
  iree_string_view_t variant_artifact = iree_string_view_empty();
  bool requires_qualification = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_device_library_target_lookup_variant_for_physical_target(
          physical_target_id, &variant_artifact, &requires_qualification));
  if (!iree_string_view_is_empty(variant_artifact)) {
    if (iree_hal_amdgpu_target_id_equal(physical_target_id, isa_target_id)) {
      return iree_hal_amdgpu_device_library_target_append_unique_candidate(
          variant_artifact, out_candidates);
    }
    return iree_ok_status();
  }
  if (requires_qualification) {
    char target_id_buffer[64] = {0};
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_id_format(
        physical_target_id, sizeof(target_id_buffer), target_id_buffer,
        /*out_buffer_length=*/NULL));
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "physical target `%s` requires feature qualification for device "
        "library selection",
        target_id_buffer);
  }

  // Try the most specific runtime binary names first. Direct arch names beat
  // code-object target fallbacks because a concrete code object is preferable
  // to a family-generic code object when both are bundled into the runtime.
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_device_library_target_append_target_id_candidate(
          isa_target_id, out_candidates));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_device_library_target_append_unique_candidate(
          isa_target_id->processor, out_candidates));
  if (isa_target_id->kind == IREE_HAL_AMDGPU_TARGET_KIND_EXACT) {
    iree_hal_amdgpu_target_id_t code_object_target_id;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_id_lookup_code_object_target(
        isa_target_id, &code_object_target_id));
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_device_library_target_append_target_id_candidate(
            &code_object_target_id, out_candidates));
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_device_library_target_append_unique_candidate(
            code_object_target_id.processor, out_candidates));
  }
  return iree_ok_status();
}
