// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/device_library_target.h"

typedef struct iree_hal_amdgpu_device_library_target_variant_t {
  // Embedded device-library artifact suffix.
  iree_string_view_t artifact;
  // Canonical target selecting the artifact.
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
iree_hal_amdgpu_device_library_target_append_identity_candidate(
    const iree_hal_amdgpu_target_identity_t* identity,
    iree_hal_amdgpu_device_library_target_candidate_list_t* candidates) {
  char artifact_key_buffer[64] = {0};
  iree_host_size_t artifact_key_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_format_artifact_key(
      identity, sizeof(artifact_key_buffer), artifact_key_buffer,
      &artifact_key_length));
  return iree_hal_amdgpu_device_library_target_append_unique_candidate(
      iree_make_string_view(artifact_key_buffer, artifact_key_length),
      candidates);
}

static iree_string_view_t
iree_hal_amdgpu_device_library_target_lookup_variant_for_physical_target(
    const iree_hal_amdgpu_target_identity_t* physical_identity) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amdgpu_device_library_target_variants);
       ++i) {
    const iree_hal_amdgpu_device_library_target_variant_t* variant =
        &iree_hal_amdgpu_device_library_target_variants[i];
    if (iree_string_view_equal(variant->target, physical_identity->target)) {
      return variant->artifact;
    }
  }
  return iree_string_view_empty();
}

iree_status_t iree_hal_amdgpu_device_library_target_candidates_from_agent_isa(
    const iree_hal_amdgpu_target_identity_t* physical_identity,
    const iree_hal_amdgpu_target_identity_t* isa_identity,
    iree_hal_amdgpu_device_library_target_candidate_list_t* out_candidates) {
  IREE_ASSERT_ARGUMENT(physical_identity);
  IREE_ASSERT_ARGUMENT(isa_identity);
  IREE_ASSERT_ARGUMENT(out_candidates);
  memset(out_candidates, 0, sizeof(*out_candidates));

  // A target-overlay artifact suppresses every fallback ISA for its physical
  // target. Otherwise a lower-priority generic ISA could reintroduce the
  // incompatible family artifact this variant exists to replace.
  const iree_string_view_t variant_artifact =
      iree_hal_amdgpu_device_library_target_lookup_variant_for_physical_target(
          physical_identity);
  if (!iree_string_view_is_empty(variant_artifact)) {
    if (iree_hal_amdgpu_target_identity_equal(physical_identity,
                                              isa_identity)) {
      return iree_hal_amdgpu_device_library_target_append_unique_candidate(
          variant_artifact, out_candidates);
    }
    return iree_ok_status();
  }

  // Try the most specific runtime binary names first. Direct arch names beat
  // code-object target fallbacks because a concrete code object is preferable
  // to a family-generic code object when both are bundled into the runtime.
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_device_library_target_append_identity_candidate(
          isa_identity, out_candidates));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_device_library_target_append_unique_candidate(
          isa_identity->processor, out_candidates));
  if (isa_identity->kind == IREE_HAL_AMDGPU_TARGET_KIND_EXACT) {
    iree_hal_amdgpu_target_identity_t code_object_identity;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_project_code_object(
        isa_identity, &code_object_identity));
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_device_library_target_append_identity_candidate(
            &code_object_identity, out_candidates));
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_device_library_target_append_unique_candidate(
            code_object_identity.processor, out_candidates));
  }
  return iree_ok_status();
}
