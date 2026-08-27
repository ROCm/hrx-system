// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/profile.h"

#include <string.h>

static iree_status_t loom_spirv_target_profile_allocate_copy(
    iree_arena_allocator_t* arena, const void* source,
    iree_host_size_t byte_length, void** out_copy) {
  *out_copy = NULL;
  if (byte_length == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, byte_length, out_copy));
  memcpy(*out_copy, source, byte_length);
  return iree_ok_status();
}

static bool loom_spirv_target_profile_accumulate_name_bytes(
    const iree_string_view_t name, iree_host_size_t* total_byte_length) {
  return iree_host_size_checked_add(*total_byte_length, name.size,
                                    total_byte_length);
}

static void loom_spirv_target_profile_copy_name(iree_string_view_t source,
                                                char** storage_cursor,
                                                iree_string_view_t* out_name) {
  if (iree_string_view_is_empty(source)) {
    *out_name = iree_string_view_empty();
    return;
  }
  memcpy(*storage_cursor, source.data, source.size);
  *out_name = iree_make_string_view(*storage_cursor, source.size);
  *storage_cursor += source.size;
}

static iree_status_t loom_spirv_target_profile_copy_cooperative_properties(
    const loom_spirv_cooperative_property_set_t* source,
    iree_arena_allocator_t* arena,
    loom_spirv_cooperative_property_set_t* out_properties) {
  *out_properties = (loom_spirv_cooperative_property_set_t){
      .feature_bits = source->feature_bits,
      .matrix_property_count = source->matrix_property_count,
      .matrix_shape_span_count = source->matrix_shape_span_count,
      .vector_property_count = source->vector_property_count,
      .vector_shape_span_count = source->vector_shape_span_count,
  };

  IREE_RETURN_IF_ERROR(loom_spirv_target_profile_allocate_copy(
      arena, source->matrix_properties,
      source->matrix_property_count * sizeof(*source->matrix_properties),
      (void**)&out_properties->matrix_properties));
  IREE_RETURN_IF_ERROR(loom_spirv_target_profile_allocate_copy(
      arena, source->matrix_shape_spans,
      source->matrix_shape_span_count * sizeof(*source->matrix_shape_spans),
      (void**)&out_properties->matrix_shape_spans));
  IREE_RETURN_IF_ERROR(loom_spirv_target_profile_allocate_copy(
      arena, source->vector_properties,
      source->vector_property_count * sizeof(*source->vector_properties),
      (void**)&out_properties->vector_properties));
  IREE_RETURN_IF_ERROR(loom_spirv_target_profile_allocate_copy(
      arena, source->vector_shape_spans,
      source->vector_shape_span_count * sizeof(*source->vector_shape_spans),
      (void**)&out_properties->vector_shape_spans));

  iree_host_size_t name_storage_byte_length = 0;
  for (uint16_t i = 0; i < source->matrix_property_count; ++i) {
    if (!loom_spirv_target_profile_accumulate_name_bytes(
            source->matrix_properties[i].name, &name_storage_byte_length)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "SPIR-V cooperative property names exceed addressable storage");
    }
  }
  for (uint16_t i = 0; i < source->vector_property_count; ++i) {
    if (!loom_spirv_target_profile_accumulate_name_bytes(
            source->vector_properties[i].name, &name_storage_byte_length)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "SPIR-V cooperative property names exceed addressable storage");
    }
  }
  char* name_storage = NULL;
  if (name_storage_byte_length != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, name_storage_byte_length,
                                             (void**)&name_storage));
  }
  char* name_storage_cursor = name_storage;
  loom_spirv_cooperative_matrix_property_t* matrix_properties =
      (loom_spirv_cooperative_matrix_property_t*)
          out_properties->matrix_properties;
  for (uint16_t i = 0; i < out_properties->matrix_property_count; ++i) {
    loom_spirv_target_profile_copy_name(matrix_properties[i].name,
                                        &name_storage_cursor,
                                        &matrix_properties[i].name);
  }
  loom_spirv_cooperative_vector_property_t* vector_properties =
      (loom_spirv_cooperative_vector_property_t*)
          out_properties->vector_properties;
  for (uint16_t i = 0; i < out_properties->vector_property_count; ++i) {
    loom_spirv_target_profile_copy_name(vector_properties[i].name,
                                        &name_storage_cursor,
                                        &vector_properties[i].name);
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_target_profile_project_facts(
    const loom_target_profile_t* base_profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* base_facts) {
  const loom_spirv_target_profile_t* profile =
      loom_spirv_target_profile_cast(base_profile);
  IREE_ASSERT(profile != NULL);
  loom_spirv_target_facts_t* facts = (loom_spirv_target_facts_t*)base_facts;
  facts->base.selector = LOOM_SPIRV_TARGET_KIND_VULKAN1_3;
  if (profile->cooperative_properties != NULL) {
    return loom_spirv_target_profile_copy_cooperative_properties(
        profile->cooperative_properties, arena, &facts->cooperative_properties);
  }

  loom_spirv_feature_set_t feature_set = {
      .atom_bits = facts->base.storage.config.contract_feature_bits,
  };
  loom_spirv_cooperative_property_set_prepare(&feature_set,
                                              &facts->cooperative_properties);
  return iree_ok_status();
}

const loom_target_profile_type_t loom_spirv_target_profile_type = {
    .name = IREE_SVL("spirv"),
    .fact_type = &loom_spirv_target_fact_type,
    .project_facts = loom_spirv_target_profile_project_facts,
};

void loom_spirv_target_profile_initialize(
    const loom_target_bundle_t* target_bundle,
    const loom_spirv_cooperative_property_set_t* cooperative_properties,
    loom_spirv_target_profile_t* out_profile) {
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = (loom_spirv_target_profile_t){
      .base =
          {
              .type = &loom_spirv_target_profile_type,
              .target_bundle = target_bundle,
          },
      .cooperative_properties = cooperative_properties,
  };
}
