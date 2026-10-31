// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/profile.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/target/arch/spirv/records/target_records.h"

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
  facts->subgroup = profile->subgroup;
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

static bool loom_spirv_profile_matrix_equal(
    const loom_spirv_cooperative_matrix_property_t* lhs,
    const loom_spirv_cooperative_matrix_property_t* rhs) {
  return iree_string_view_equal(lhs->name, rhs->name) &&
         lhs->required_feature_bits == rhs->required_feature_bits &&
         lhs->m_size == rhs->m_size && lhs->n_size == rhs->n_size &&
         lhs->k_size == rhs->k_size && lhs->lhs_type == rhs->lhs_type &&
         lhs->rhs_type == rhs->rhs_type &&
         lhs->accumulator_type == rhs->accumulator_type &&
         lhs->result_type == rhs->result_type && lhs->scope == rhs->scope &&
         lhs->layout_flags == rhs->layout_flags &&
         lhs->storage_class_flags == rhs->storage_class_flags &&
         lhs->operand_flags == rhs->operand_flags &&
         lhs->mul_add_descriptor_ref == rhs->mul_add_descriptor_ref;
}

static bool loom_spirv_profile_vector_equal(
    const loom_spirv_cooperative_vector_property_t* lhs,
    const loom_spirv_cooperative_vector_property_t* rhs) {
  return iree_string_view_equal(lhs->name, rhs->name) &&
         lhs->required_feature_bits == rhs->required_feature_bits &&
         lhs->m_size == rhs->m_size && lhs->k_size == rhs->k_size &&
         lhs->input_type == rhs->input_type &&
         lhs->input_interpretation == rhs->input_interpretation &&
         lhs->matrix_interpretation == rhs->matrix_interpretation &&
         lhs->bias_interpretation == rhs->bias_interpretation &&
         lhs->result_type == rhs->result_type &&
         lhs->matrix_layout_flags == rhs->matrix_layout_flags &&
         lhs->storage_class_flags == rhs->storage_class_flags &&
         lhs->flags == rhs->flags;
}

static bool loom_spirv_profile_spans_equal(
    uint16_t count, const loom_spirv_cooperative_property_span_t* lhs,
    const loom_spirv_cooperative_property_span_t* rhs) {
  for (uint16_t i = 0; i < count; ++i) {
    if (lhs[i].shape_key != rhs[i].shape_key || lhs[i].start != rhs[i].start ||
        lhs[i].count != rhs[i].count) {
      return false;
    }
  }
  return true;
}

static bool loom_spirv_profile_cooperative_equal(
    const loom_spirv_cooperative_property_set_t* lhs,
    const loom_spirv_cooperative_property_set_t* rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (!lhs || !rhs || lhs->feature_bits != rhs->feature_bits ||
      lhs->matrix_property_count != rhs->matrix_property_count ||
      lhs->matrix_shape_span_count != rhs->matrix_shape_span_count ||
      lhs->vector_property_count != rhs->vector_property_count ||
      lhs->vector_shape_span_count != rhs->vector_shape_span_count) {
    return false;
  }
  for (uint16_t i = 0; i < lhs->matrix_property_count; ++i) {
    if (!loom_spirv_profile_matrix_equal(&lhs->matrix_properties[i],
                                         &rhs->matrix_properties[i])) {
      return false;
    }
  }
  for (uint16_t i = 0; i < lhs->vector_property_count; ++i) {
    if (!loom_spirv_profile_vector_equal(&lhs->vector_properties[i],
                                         &rhs->vector_properties[i])) {
      return false;
    }
  }
  return loom_spirv_profile_spans_equal(lhs->matrix_shape_span_count,
                                        lhs->matrix_shape_spans,
                                        rhs->matrix_shape_spans) &&
         loom_spirv_profile_spans_equal(lhs->vector_shape_span_count,
                                        lhs->vector_shape_spans,
                                        rhs->vector_shape_spans);
}

static bool loom_spirv_target_profile_equal(const loom_target_profile_t* lhs,
                                            const loom_target_profile_t* rhs) {
  const loom_spirv_target_profile_t* lhs_profile =
      (const loom_spirv_target_profile_t*)lhs;
  const loom_spirv_target_profile_t* rhs_profile =
      (const loom_spirv_target_profile_t*)rhs;
  return loom_target_profile_equal_common(lhs, rhs) &&
         lhs_profile->subgroup.default_size ==
             rhs_profile->subgroup.default_size &&
         lhs_profile->subgroup.minimum_size ==
             rhs_profile->subgroup.minimum_size &&
         lhs_profile->subgroup.maximum_size ==
             rhs_profile->subgroup.maximum_size &&
         lhs_profile->subgroup.maximum_workgroup_subgroup_count ==
             rhs_profile->subgroup.maximum_workgroup_subgroup_count &&
         loom_spirv_profile_cooperative_equal(
             lhs_profile->cooperative_properties,
             rhs_profile->cooperative_properties);
}

const loom_target_profile_type_t loom_spirv_target_profile_type = {
    .name = IREE_SVL("spirv"),
    .fact_type = &loom_spirv_target_fact_type,
    .project_facts = loom_spirv_target_profile_project_facts,
    .equal = loom_spirv_target_profile_equal,
};

static const loom_spirv_target_profile_t kSpirvVulkan13BdaTargetProfile = {
    .base =
        {
            .type = &loom_spirv_target_profile_type,
            .target_bundle = &loom_spirv_low_target_bundle_vulkan1_3,
        },
};

static const loom_spirv_target_profile_t kSpirvExtendedTypesProfile = {
    .base =
        {
            .type = &loom_spirv_target_profile_type,
            .target_bundle = &loom_spirv_low_target_bundle_extended_types,
            .explicit_fields = UINT64_C(1)
                               << LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS,
        },
};

static const loom_spirv_target_profile_t kSpirvHalKernelProfile = {
    .base =
        {
            .type = &loom_spirv_target_profile_type,
            .target_bundle = &loom_spirv_low_target_bundle_hal_kernel,
            .explicit_fields = UINT64_C(1) << LOOM_TARGET_FACT_FIELD_ABI,
        },
};

static const loom_spirv_target_profile_t kSpirvSubgroup32BallotHalProfile = {
    .base =
        {
            .type = &loom_spirv_target_profile_type,
            .target_bundle =
                &loom_spirv_low_target_bundle_subgroup32_ballot_hal,
            .explicit_fields =
                (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE) |
                (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_ABI) |
                (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS),
        },
};

static const loom_spirv_target_profile_t kSpirvSubgroup32PredicateHalProfile = {
    .base =
        {
            .type = &loom_spirv_target_profile_type,
            .target_bundle =
                &loom_spirv_low_target_bundle_subgroup32_predicate_hal,
            .explicit_fields =
                (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE) |
                (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_ABI) |
                (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS),
        },
};

static const loom_spirv_target_profile_t kSpirvSubgroup32ReductionHalProfile = {
    .base =
        {
            .type = &loom_spirv_target_profile_type,
            .target_bundle =
                &loom_spirv_low_target_bundle_subgroup32_reduction_hal,
            .explicit_fields =
                (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE) |
                (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_ABI) |
                (UINT64_C(1) << LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS),
        },
};

void loom_spirv_target_profile_initialize(
    const loom_target_bundle_t* target_bundle,
    loom_target_fact_field_set_t explicit_fields,
    loom_spirv_subgroup_properties_t subgroup,
    const loom_spirv_cooperative_property_set_t* cooperative_properties,
    loom_spirv_target_profile_t* out_profile) {
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = (loom_spirv_target_profile_t){
      .base =
          {
              .type = &loom_spirv_target_profile_type,
              .target_bundle = target_bundle,
              .explicit_fields = explicit_fields,
          },
      .subgroup = subgroup,
      .cooperative_properties = cooperative_properties,
  };
}

iree_status_t loom_spirv_target_profile_select(
    iree_string_view_t selector,
    const loom_spirv_target_profile_t** out_profile) {
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = NULL;
  if (iree_string_view_equal(selector, IREE_SV("vulkan1.3+bda"))) {
    *out_profile = &kSpirvVulkan13BdaTargetProfile;
  } else if (iree_string_view_equal(selector,
                                    IREE_SV("vulkan1.3+bda+extended-types"))) {
    *out_profile = &kSpirvExtendedTypesProfile;
  } else if (iree_string_view_equal(selector, IREE_SV("vulkan1.3+bda+hal"))) {
    *out_profile = &kSpirvHalKernelProfile;
  } else if (iree_string_view_equal(
                 selector, IREE_SV("vulkan1.3+bda+subgroup32+ballot+hal"))) {
    *out_profile = &kSpirvSubgroup32BallotHalProfile;
  } else if (iree_string_view_equal(
                 selector,
                 IREE_SV("vulkan1.3+bda+subgroup32+ballot+vote+hal"))) {
    *out_profile = &kSpirvSubgroup32PredicateHalProfile;
  } else if (iree_string_view_equal(
                 selector,
                 IREE_SV("vulkan1.3+bda+subgroup32+reductions+hal"))) {
    *out_profile = &kSpirvSubgroup32ReductionHalProfile;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unknown SPIR-V target selector '%.*s'; expected vulkan1.3+bda, "
        "vulkan1.3+bda+extended-types, vulkan1.3+bda+hal, "
        "vulkan1.3+bda+subgroup32+ballot+hal, "
        "vulkan1.3+bda+subgroup32+ballot+vote+hal, or "
        "vulkan1.3+bda+subgroup32+reductions+hal",
        (int)selector.size, selector.data);
  }
  return iree_ok_status();
}
