// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/executable_target.h"

enum {
  IREE_HAL_AMDGPU_EXECUTABLE_TARGET_PRIORITY_EXACT = 100,
  IREE_HAL_AMDGPU_EXECUTABLE_TARGET_PRIORITY_GENERIC = 50,
};

static iree_status_t iree_hal_amdgpu_device_spec_builder_add_executable_target(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_amdgpu_target_id_t* target_id,
    iree_hal_executable_target_kind_t kind, uint32_t priority,
    iree_hal_physical_device_affinity_t physical_device_affinity) {
  char target_key_storage[128] = {0};
  iree_host_size_t target_key_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_target_id_format(target_id, sizeof(target_key_storage),
                                       target_key_storage, &target_key_length));
  const iree_hal_executable_target_t target = {
      .family = IREE_SV("amdgpu"),
      .target_key =
          iree_make_string_view(target_key_storage, target_key_length),
      .kind = kind,
      .priority = priority,
      .physical_device_affinity = physical_device_affinity,
      .flags = IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  return iree_hal_device_spec_builder_add_executable_target(builder, &target);
}

iree_status_t iree_hal_amdgpu_device_spec_builder_add_executable_targets(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_amdgpu_target_id_t* exact_target_id,
    iree_hal_physical_device_affinity_t physical_device_affinity) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_device_spec_builder_add_executable_target(
          builder, exact_target_id, IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
          IREE_HAL_AMDGPU_EXECUTABLE_TARGET_PRIORITY_EXACT,
          physical_device_affinity));

  iree_hal_amdgpu_target_id_t code_object_target_id;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_id_lookup_code_object_target(
      exact_target_id, &code_object_target_id));
  if (iree_string_view_equal(exact_target_id->processor,
                             code_object_target_id.processor) &&
      exact_target_id->sramecc == code_object_target_id.sramecc &&
      exact_target_id->xnack == code_object_target_id.xnack) {
    return iree_ok_status();
  }
  return iree_hal_amdgpu_device_spec_builder_add_executable_target(
      builder, &code_object_target_id, IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
      IREE_HAL_AMDGPU_EXECUTABLE_TARGET_PRIORITY_GENERIC,
      physical_device_affinity);
}

static iree_hal_executable_target_kind_t iree_hal_amdgpu_executable_target_kind(
    const iree_hal_amdgpu_target_id_t* target_id) {
  return target_id->kind == IREE_HAL_AMDGPU_TARGET_KIND_GENERIC
             ? IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC
             : IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT;
}

iree_status_t iree_hal_amdgpu_device_spec_select_executable_target(
    const iree_hal_device_spec_t* device_spec,
    iree_string_view_t artifact_target_key,
    iree_hal_physical_device_affinity_t physical_device_affinity,
    iree_hal_executable_target_selection_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(device_spec);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (iree_hal_executable_target_selection_result_t){
      .outcome = IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH,
      .target = NULL,
      .target_ordinal = IREE_HOST_SIZE_MAX,
  };

  iree_hal_amdgpu_target_id_t artifact_target_id;
  iree_status_t status = iree_hal_amdgpu_target_id_parse(
      artifact_target_key,
      IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_ARCH_ONLY |
          IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_FEATURE_SUFFIXES,
      &artifact_target_id);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(
        status, "parsing AMDGPU executable artifact target '%.*s'",
        (int)artifact_target_key.size, artifact_target_key.data);
  }
  const iree_hal_executable_target_kind_t artifact_target_kind =
      iree_hal_amdgpu_executable_target_kind(&artifact_target_id);

  const iree_hal_device_executable_spec_t* executable_spec =
      iree_hal_device_spec_executables(device_spec);
  const iree_hal_executable_target_t* selected_target = NULL;
  iree_host_size_t selected_ordinal = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < executable_spec->target_count; ++i) {
    const iree_hal_executable_target_t* candidate =
        &executable_spec->targets[i];
    if (!iree_string_view_equal(candidate->family, IREE_SV("amdgpu")) ||
        candidate->kind != artifact_target_kind) {
      continue;
    }
    if (physical_device_affinity != 0 &&
        !iree_all_bits_set(candidate->physical_device_affinity,
                           physical_device_affinity)) {
      continue;
    }

    iree_hal_amdgpu_target_id_t candidate_target_id;
    status = iree_hal_amdgpu_target_id_parse(
        candidate->target_key,
        IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_ARCH_ONLY |
            IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_FEATURE_SUFFIXES,
        &candidate_target_id);
    if (!iree_status_is_ok(status)) {
      return iree_status_annotate_f(
          status, "parsing AMDGPU device target '%.*s'",
          (int)candidate->target_key.size, candidate->target_key.data);
    }
    if (iree_hal_amdgpu_executable_target_kind(&candidate_target_id) !=
        candidate->kind) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU device target '%.*s' kind does not match its target key",
          (int)candidate->target_key.size, candidate->target_key.data);
    }
    if (iree_hal_amdgpu_target_id_check_compatible(&artifact_target_id,
                                                   &candidate_target_id) !=
        IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE) {
      continue;
    }

    if (selected_target == NULL ||
        candidate->priority > selected_target->priority) {
      out_result->outcome =
          IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED;
      selected_target = candidate;
      selected_ordinal = i;
    } else if (candidate->priority == selected_target->priority) {
      out_result->outcome =
          IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS;
    }
  }

  if (out_result->outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    out_result->target = selected_target;
    out_result->target_ordinal = selected_ordinal;
  }
  return iree_ok_status();
}
