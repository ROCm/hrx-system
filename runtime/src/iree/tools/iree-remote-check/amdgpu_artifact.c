// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tools/iree-remote-check/amdgpu_artifact.h"

#include <stdio.h>
#include <string.h>

#include "build_tools/amdgpu/target_map.h"
#include "iree/hal/executable/amdgpu/executable_target.h"
#include "iree/hal/executable/amdgpu/target_id.h"
#include "iree/tools/iree-remote-check/amdgpu_artifacts.h"

static const iree_file_toc_t* iree_remote_check_find_amdgpu_file(
    iree_string_view_t artifact_target_key) {
  char target_key[64];
  if (artifact_target_key.size >= sizeof(target_key)) return NULL;
  memcpy(target_key, artifact_target_key.data, artifact_target_key.size);
  target_key[artifact_target_key.size] = 0;

  char target_fragment[64];
  if (!iree_amdgpu_target_label_fragment(target_key, target_fragment,
                                         sizeof(target_fragment))) {
    return NULL;
  }

  char file_name[128];
  int file_name_length =
      snprintf(file_name, sizeof(file_name), "iree_remote_check_kernel_%s.so",
               target_fragment);
  if (file_name_length < 0 ||
      (iree_host_size_t)file_name_length >= sizeof(file_name)) {
    return NULL;
  }

  const iree_file_toc_t* files = iree_remote_check_amdgpu_artifacts_create();
  for (iree_host_size_t i = 0; i < iree_remote_check_amdgpu_artifacts_size();
       ++i) {
    if (strcmp(files[i].name, file_name) == 0) return &files[i];
  }
  return NULL;
}

static iree_status_t iree_remote_check_amdgpu_artifact_for_target(
    const iree_hal_executable_target_t* device_target,
    const iree_file_toc_t** out_file,
    iree_string_view_t* out_artifact_target_key) {
  *out_file = NULL;
  *out_artifact_target_key = iree_string_view_empty();

  iree_hal_amdgpu_target_id_t target_id;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_id_parse(
      device_target->target_key,
      IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_ARCH_ONLY |
          IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_FEATURE_SUFFIXES,
      &target_id));

  const iree_file_toc_t* file =
      iree_remote_check_find_amdgpu_file(target_id.processor);
  iree_string_view_t artifact_target_key = target_id.processor;
  if (!file && device_target->kind == IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT) {
    char exact_target[64];
    if (target_id.processor.size >= sizeof(exact_target)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "AMDGPU target key is too long");
    }
    memcpy(exact_target, target_id.processor.data, target_id.processor.size);
    exact_target[target_id.processor.size] = 0;
    const char* generic_target =
        iree_amdgpu_code_object_target_for_exact(exact_target);
    if (generic_target) {
      artifact_target_key = iree_make_cstring_view(generic_target);
      file = iree_remote_check_find_amdgpu_file(artifact_target_key);
    }
  }

  *out_file = file;
  *out_artifact_target_key = artifact_target_key;
  return iree_ok_status();
}

iree_status_t iree_remote_check_select_amdgpu_artifact(
    const iree_hal_device_spec_t* device_spec,
    iree_remote_check_artifact_t* out_artifact) {
  IREE_ASSERT_ARGUMENT(device_spec);
  IREE_ASSERT_ARGUMENT(out_artifact);
  memset(out_artifact, 0, sizeof(*out_artifact));

  const iree_hal_device_executable_spec_t* executable_spec =
      iree_hal_device_spec_executables(device_spec);
  static const int32_t addend = 0;
  uint32_t selected_priority = 0;
  for (iree_host_size_t i = 0; i < executable_spec->target_count; ++i) {
    const iree_hal_executable_target_t* device_target =
        &executable_spec->targets[i];
    if (!iree_string_view_equal(device_target->family, IREE_SV("amdgpu"))) {
      continue;
    }

    const iree_file_toc_t* file = NULL;
    iree_string_view_t artifact_target_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_remote_check_amdgpu_artifact_for_target(
        device_target, &file, &artifact_target_key));
    if (!file) continue;

    iree_hal_executable_target_selection_result_t target_result;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_device_spec_select_executable_target(
        device_spec, artifact_target_key,
        /*physical_device_affinity=*/0, &target_result));
    if (target_result.outcome ==
        IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "remote device has ambiguous targets for AMDGPU artifact '%.*s'",
          (int)artifact_target_key.size, artifact_target_key.data);
    }
    if (target_result.outcome !=
        IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
      continue;
    }
    if (out_artifact->executable_target &&
        target_result.target->priority <= selected_priority) {
      continue;
    }

    selected_priority = target_result.target->priority;
    out_artifact->executable_target = target_result.target;
    out_artifact->file_name = iree_make_cstring_view(file->name);
    out_artifact->artifact_target_key = artifact_target_key;
    out_artifact->executable_data = iree_make_const_byte_span(
        (const uint8_t*)file->data, (iree_host_size_t)file->size);
    out_artifact->entry_point = IREE_SV("iree_remote_check_add7");
    out_artifact->dispatch_constants =
        iree_make_const_byte_span(&addend, sizeof(addend));
    out_artifact->dispatch_config =
        iree_hal_make_static_dispatch_config(1, 1, 1);
  }
  return iree_ok_status();
}
