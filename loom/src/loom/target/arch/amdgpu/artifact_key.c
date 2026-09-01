// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/artifact_key.h"

#include "loom/target/arch/amdgpu/amdhsa_target_id.h"
#include "loom/target/arch/amdgpu/target_info.h"

iree_status_t loom_amdgpu_artifact_key_parse(
    iree_string_view_t value, loom_amdgpu_target_identity_t* out_identity) {
  IREE_ASSERT_ARGUMENT(out_identity);
  *out_identity = (loom_amdgpu_target_identity_t){0};
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU artifact key is required");
  }

  iree_string_view_t target_name = value;
  iree_string_view_t feature_suffix = iree_string_view_empty();
  if (iree_string_view_split(value, ':', &target_name, &feature_suffix) != -1 &&
      (iree_string_view_is_empty(feature_suffix) ||
       value.data[value.size - 1] == ':')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU artifact key has an empty feature suffix");
  }

  const loom_amdgpu_target_info_t* target = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_lookup_target(target_name, &target));
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(target);
  IREE_ASSERT(processor != NULL);

  loom_amdgpu_target_identity_t identity = {0};
  identity.target = target;
  IREE_RETURN_IF_ERROR(loom_amdgpu_amdhsa_feature_suffix_parse(
      processor, feature_suffix, &identity.amdhsa_features));
  *out_identity = identity;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_artifact_key_append(
    const loom_amdgpu_target_identity_t* identity,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, identity->target->name));
  return loom_amdgpu_amdhsa_feature_suffix_append(&identity->amdhsa_features,
                                                  builder);
}

iree_status_t loom_amdgpu_artifact_key_format(
    const loom_amdgpu_target_identity_t* identity,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_string_view_t* out_artifact_key) {
  *out_artifact_key = iree_string_view_empty();
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(buffer, buffer_capacity,
                                              &builder);
  const iree_status_t status =
      loom_amdgpu_artifact_key_append(identity, &builder);
  if (iree_status_is_ok(status)) {
    *out_artifact_key = iree_string_builder_view(&builder);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

iree_status_t loom_amdgpu_artifact_key_format_arena(
    const loom_amdgpu_target_identity_t* identity,
    iree_arena_allocator_t* arena, iree_string_view_t* out_artifact_key) {
  *out_artifact_key = iree_string_view_empty();

  iree_string_builder_t measure_builder;
  iree_string_builder_initialize(iree_allocator_null(), &measure_builder);
  iree_status_t status =
      loom_amdgpu_artifact_key_append(identity, &measure_builder);
  const iree_host_size_t artifact_key_length =
      iree_string_builder_size(&measure_builder);
  iree_string_builder_deinitialize(&measure_builder);
  IREE_RETURN_IF_ERROR(status);

  iree_host_size_t storage_length = 0;
  if (!iree_host_size_checked_add(artifact_key_length, 1, &storage_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU artifact-key length overflows");
  }
  char* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, storage_length, (void**)&data));
  iree_string_builder_t builder;
  iree_string_builder_initialize_with_storage(data, storage_length, &builder);
  status = loom_amdgpu_artifact_key_append(identity, &builder);
  if (iree_status_is_ok(status)) {
    *out_artifact_key = iree_string_builder_view(&builder);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}
