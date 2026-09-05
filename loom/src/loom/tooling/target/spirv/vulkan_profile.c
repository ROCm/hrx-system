// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/vulkan_profile.h"

#include <stdint.h>

#include "iree/hal/drivers/vulkan/device_spec.h"
#include "loom/target/arch/spirv/cooperative_properties.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/records/target_records.h"

typedef struct loom_spirv_vulkan_hal_feature_row_t {
  // Vulkan HAL features required for this profile fact.
  iree_hal_vulkan_features_t required_features;
  // Compact fact flag set when the HAL reports the feature as available.
  loom_spirv_vulkan_hal_profile_flag_bits_t flag;
  // Additional fact flags required before projecting |feature_bits|.
  loom_spirv_vulkan_hal_profile_flags_t required_flags;
  // SPIR-V feature atoms implied by this Vulkan feature.
  loom_spirv_feature_bits_t feature_bits;
} loom_spirv_vulkan_hal_feature_row_t;

static const loom_spirv_vulkan_hal_feature_row_t kVulkanFeatureRows[] = {
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_BUFFER_DEVICE_ADDRESS,
        .feature_bits = 0,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SUBGROUP_SIZE_CONTROL,
        .feature_bits = 0,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR,
        .feature_bits = LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_8BIT_ACCESS,
        .feature_bits = LOOM_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_16BIT_ACCESS,
        .feature_bits = LOOM_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT16,
        .feature_bits = LOOM_SPIRV_FEATURE_FLOAT16,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT64,
        .feature_bits = LOOM_SPIRV_FEATURE_FLOAT64,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_TYPE,
        .feature_bits = LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_DOT_PRODUCT,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_DOT_PRODUCT,
        .required_flags =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_TYPE,
        .feature_bits = LOOM_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_COOPERATIVE_MATRIX,
        .flag =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_COOPERATIVE_MATRIX,
        .required_flags =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR |
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_TYPE,
        .feature_bits = LOOM_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR,
    },
    {
        .required_features.general = IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT8,
        .feature_bits = LOOM_SPIRV_FEATURE_INT8,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT16,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT16,
        .feature_bits = LOOM_SPIRV_FEATURE_INT16,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT64,
        .feature_bits = LOOM_SPIRV_FEATURE_INT64,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INTEGER_DOT_PRODUCT,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INTEGER_DOT_PRODUCT,
        .feature_bits = 0,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_VULKAN_MEMORY_MODEL,
        .feature_bits = 0,
    },
    {
        .required_features.general =
            IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
        .flag =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
        .feature_bits = LOOM_SPIRV_FEATURE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_INT64,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_INT64_ATOMICS,
        .required_flags = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT64,
        .feature_bits = LOOM_SPIRV_FEATURE_STORAGE_BUFFER_INT64_ATOMICS,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_INT64,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_INT64_ATOMICS,
        .required_flags = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT64,
        .feature_bits = LOOM_SPIRV_FEATURE_WORKGROUP_INT64_ATOMICS,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16,
        .flag =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT16_ATOMICS,
        .required_flags = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT16,
        .feature_bits = LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMICS,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16_ADD,
        .flag =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD,
        .required_flags = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT16,
        .feature_bits = LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT16,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT16_ATOMICS,
        .required_flags = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT16,
        .feature_bits = LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMICS,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT16_ADD,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT16_ATOMIC_ADD,
        .required_flags = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT16,
        .feature_bits = LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMIC_ADD,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32,
        .flag =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT32_ATOMICS,
        .feature_bits = LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMICS,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32_ADD,
        .flag =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD,
        .feature_bits = LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT32,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT32_ATOMICS,
        .feature_bits = LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMICS,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT32_ADD,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT32_ATOMIC_ADD,
        .feature_bits = LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMIC_ADD,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT64,
        .flag =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT64_ATOMICS,
        .required_flags = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT64,
        .feature_bits = LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMICS,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT64_ADD,
        .flag =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD,
        .required_flags = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT64,
        .feature_bits = LOOM_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT64,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT64_ATOMICS,
        .required_flags = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT64,
        .feature_bits = LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMICS,
    },
    {
        .required_features.atomics =
            IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT64_ADD,
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_WORKGROUP_FLOAT64_ATOMIC_ADD,
        .required_flags = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT64,
        .feature_bits = LOOM_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMIC_ADD,
    },
};

static void loom_spirv_vulkan_hal_profile_project_feature_flag(
    const iree_hal_vulkan_device_spec_t* vulkan_spec,
    const loom_spirv_vulkan_hal_feature_row_t* row,
    loom_spirv_vulkan_hal_profile_facts_t* facts) {
  if (iree_all_bits_set(vulkan_spec->enabled_features.general,
                        row->required_features.general) &&
      iree_all_bits_set(vulkan_spec->enabled_features.atomics,
                        row->required_features.atomics)) {
    facts->flags |= row->flag;
  }
}

static iree_status_t loom_spirv_vulkan_hal_profile_decode_device_spec(
    iree_hal_device_t* device, const iree_hal_device_spec_t** out_device_spec,
    iree_hal_vulkan_device_spec_t* out_vulkan_spec) {
  *out_device_spec = iree_hal_device_spec(device);
  *out_vulkan_spec = (iree_hal_vulkan_device_spec_t){0};
  if (*out_device_spec == NULL) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "HAL device does not expose immutable device facts");
  }
  const iree_hal_device_spec_facet_t* vulkan_facet =
      iree_hal_vulkan_device_spec_find_facet(*out_device_spec);
  if (vulkan_facet == NULL) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "HAL device spec does not expose Vulkan device facts");
  }
  return iree_hal_vulkan_device_spec_decode_facet(vulkan_facet,
                                                  out_vulkan_spec);
}

iree_status_t loom_spirv_vulkan_hal_profile_query(
    iree_hal_device_t* device,
    loom_spirv_vulkan_hal_profile_facts_t* out_facts) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_facts);

  *out_facts = (loom_spirv_vulkan_hal_profile_facts_t){0};
  const iree_hal_device_spec_t* device_spec = NULL;
  iree_hal_vulkan_device_spec_t vulkan_spec = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_decode_device_spec(
      device, &device_spec, &vulkan_spec));
  const iree_hal_executable_target_selection_t target_selection = {
      .family = IREE_SV("spirv"),
      .target_key = IREE_SV("vulkan1.3+bda"),
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC,
  };
  const iree_hal_executable_target_selection_result_t target_result =
      iree_hal_device_spec_select_executable_target(device_spec,
                                                    &target_selection);
  if (target_result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL device reports ambiguous Vulkan SPIR-V executable targets");
  } else if (target_result.outcome ==
             IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    out_facts->flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_RAW_BDA_EXECUTABLE;
  }
  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(device_spec);
  if (dispatch == NULL) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "HAL device spec does not expose dispatch capability facts");
  }
  out_facts->api_version = vulkan_spec.api_version;
  out_facts->subgroup_size = dispatch->subgroup.default_size;
  out_facts->max_compute_workgroup_invocations =
      dispatch->launch.maximum_workgroup_invocations;
  out_facts->max_compute_workgroup_size.x =
      dispatch->launch.maximum_workgroup_size[0];
  out_facts->max_compute_workgroup_size.y =
      dispatch->launch.maximum_workgroup_size[1];
  out_facts->max_compute_workgroup_size.z =
      dispatch->launch.maximum_workgroup_size[2];
  out_facts->max_compute_workgroup_count.x =
      dispatch->launch.maximum_workgroup_count[0];
  out_facts->max_compute_workgroup_count.y =
      dispatch->launch.maximum_workgroup_count[1];
  out_facts->max_compute_workgroup_count.z =
      dispatch->launch.maximum_workgroup_count[2];

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kVulkanFeatureRows); ++i) {
    loom_spirv_vulkan_hal_profile_project_feature_flag(
        &vulkan_spec, &kVulkanFeatureRows[i], out_facts);
  }

  return iree_ok_status();
}

static iree_status_t loom_spirv_vulkan_hal_profile_require_flag(
    const loom_spirv_vulkan_hal_profile_facts_t* facts,
    loom_spirv_vulkan_hal_profile_flag_bits_t flag,
    iree_string_view_t message) {
  if (iree_all_bits_set(facts->flags, flag)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "%.*s", (int)message.size,
                          message.data);
}

static loom_spirv_feature_bits_t loom_spirv_vulkan_hal_profile_feature_bits(
    const loom_spirv_vulkan_hal_profile_facts_t* facts) {
  loom_spirv_feature_bits_t feature_bits =
      LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kVulkanFeatureRows); ++i) {
    const loom_spirv_vulkan_hal_feature_row_t* row = &kVulkanFeatureRows[i];
    if (iree_any_bit_set(facts->flags, row->flag) &&
        iree_all_bits_set(facts->flags, row->required_flags)) {
      feature_bits |= row->feature_bits;
    }
  }
  return feature_bits;
}

static bool loom_spirv_vulkan_hal_profile_scalar_type_matches_component(
    loom_spirv_scalar_type_t scalar_type, uint32_t component_type) {
  switch ((loom_spirv_component_type_t)component_type) {
    case LOOM_SPIRV_COMPONENT_TYPE_FLOAT16_NV:
      return scalar_type == LOOM_SPIRV_SCALAR_TYPE_F16;
    case LOOM_SPIRV_COMPONENT_TYPE_FLOAT32_NV:
      return scalar_type == LOOM_SPIRV_SCALAR_TYPE_F32;
    case LOOM_SPIRV_COMPONENT_TYPE_FLOAT64_NV:
      return scalar_type == LOOM_SPIRV_SCALAR_TYPE_F64;
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV:
      return scalar_type == LOOM_SPIRV_SCALAR_TYPE_S8;
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT16_NV:
      return scalar_type == LOOM_SPIRV_SCALAR_TYPE_S16;
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV:
      return scalar_type == LOOM_SPIRV_SCALAR_TYPE_S32;
    case LOOM_SPIRV_COMPONENT_TYPE_SIGNED_INT64_NV:
      return scalar_type == LOOM_SPIRV_SCALAR_TYPE_S64;
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV:
      return scalar_type == LOOM_SPIRV_SCALAR_TYPE_U8;
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT16_NV:
      return scalar_type == LOOM_SPIRV_SCALAR_TYPE_U16;
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV:
      return scalar_type == LOOM_SPIRV_SCALAR_TYPE_U32;
    case LOOM_SPIRV_COMPONENT_TYPE_UNSIGNED_INT64_NV:
      return scalar_type == LOOM_SPIRV_SCALAR_TYPE_U64;
    default:
      return false;
  }
}

static bool loom_spirv_vulkan_hal_profile_model_row_matches_device_row(
    const loom_spirv_cooperative_matrix_property_t* model_row,
    const iree_hal_vulkan_cooperative_matrix_property_t* device_row) {
  if (model_row->m_size != device_row->m_size ||
      model_row->n_size != device_row->n_size ||
      model_row->k_size != device_row->k_size ||
      (uint32_t)model_row->scope != device_row->scope) {
    return false;
  }
  if (!loom_spirv_vulkan_hal_profile_scalar_type_matches_component(
          model_row->lhs_type, device_row->a_type) ||
      !loom_spirv_vulkan_hal_profile_scalar_type_matches_component(
          model_row->rhs_type, device_row->b_type) ||
      !loom_spirv_vulkan_hal_profile_scalar_type_matches_component(
          model_row->accumulator_type, device_row->c_type) ||
      !loom_spirv_vulkan_hal_profile_scalar_type_matches_component(
          model_row->result_type, device_row->result_type)) {
    return false;
  }
  const bool model_requires_saturation = iree_any_bit_set(
      model_row->operand_flags,
      LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_SATURATING_ACCUMULATION);
  return !model_requires_saturation || device_row->saturating_accumulation != 0;
}

static bool loom_spirv_vulkan_hal_profile_model_row_supported(
    const loom_spirv_cooperative_matrix_property_t* model_row,
    const iree_hal_vulkan_device_spec_t* vulkan_spec) {
  for (iree_host_size_t i = 0; i < vulkan_spec->cooperative_matrix.count; ++i) {
    iree_hal_vulkan_cooperative_matrix_property_t device_row = {0};
    const bool row_read =
        iree_hal_vulkan_device_spec_read_cooperative_matrix_property(
            vulkan_spec, i, &device_row);
    IREE_ASSERT_TRUE(row_read);
    if (loom_spirv_vulkan_hal_profile_model_row_matches_device_row(
            model_row, &device_row)) {
      return true;
    }
  }
  return false;
}

static const loom_spirv_cooperative_matrix_property_t*
loom_spirv_vulkan_hal_profile_find_model_row(iree_string_view_t row_name) {
  iree_host_size_t model_row_count = 0;
  const loom_spirv_cooperative_matrix_property_t* model_rows =
      loom_spirv_cooperative_matrix_model_properties(&model_row_count);
  for (iree_host_size_t i = 0; i < model_row_count; ++i) {
    if (iree_string_view_equal(model_rows[i].name, row_name)) {
      return &model_rows[i];
    }
  }
  return NULL;
}

iree_status_t loom_spirv_vulkan_hal_profile_project_target_facts(
    iree_hal_device_t* device, iree_arena_allocator_t* arena,
    loom_spirv_target_facts_t* out_facts) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_facts);

  out_facts->base.selector = LOOM_SPIRV_TARGET_KIND_VULKAN1_3;
  loom_spirv_cooperative_property_set_t* property_set =
      &out_facts->cooperative_properties;
  property_set->feature_bits =
      out_facts->base.storage.config.contract_feature_bits;
  if (!iree_any_bit_set(property_set->feature_bits,
                        LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR)) {
    return iree_ok_status();
  }

  const iree_hal_device_spec_t* device_spec = NULL;
  iree_hal_vulkan_device_spec_t vulkan_spec = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_decode_device_spec(
      device, &device_spec, &vulkan_spec));
  (void)device_spec;

  iree_host_size_t model_row_count = 0;
  const loom_spirv_cooperative_matrix_property_t* model_rows =
      loom_spirv_cooperative_matrix_model_properties(&model_row_count);
  if (model_row_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "SPIR-V cooperative matrix model exceeds uint16_t capacity");
  }

  iree_host_size_t selected_row_count = 0;
  for (iree_host_size_t i = 0; i < model_row_count; ++i) {
    if (loom_spirv_vulkan_hal_profile_model_row_supported(&model_rows[i],
                                                          &vulkan_spec)) {
      ++selected_row_count;
    }
  }
  if (selected_row_count == 0) {
    return iree_ok_status();
  }

  loom_spirv_cooperative_matrix_property_t* selected_rows = NULL;
  loom_spirv_cooperative_property_span_t* shape_spans = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, selected_row_count,
                                                 sizeof(*selected_rows),
                                                 (void**)&selected_rows));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, selected_row_count, sizeof(*shape_spans), (void**)&shape_spans));

  uint64_t previous_shape_key = 0;
  uint16_t row_ordinal = 0;
  uint16_t shape_span_count = 0;
  for (iree_host_size_t i = 0; i < model_row_count; ++i) {
    const loom_spirv_cooperative_matrix_property_t* model_row = &model_rows[i];
    if (!loom_spirv_vulkan_hal_profile_model_row_supported(model_row,
                                                           &vulkan_spec)) {
      continue;
    }
    selected_rows[row_ordinal] = *model_row;
    property_set->feature_bits |= model_row->required_feature_bits;
    const uint64_t shape_key = loom_spirv_cooperative_matrix_shape_key(
        model_row->m_size, model_row->n_size, model_row->k_size);
    if (row_ordinal == 0 || shape_key != previous_shape_key) {
      shape_spans[shape_span_count++] =
          (loom_spirv_cooperative_property_span_t){
              .shape_key = shape_key,
              .start = row_ordinal,
              .count = 1,
          };
    } else {
      ++shape_spans[shape_span_count - 1].count;
    }
    previous_shape_key = shape_key;
    ++row_ordinal;
  }
  IREE_ASSERT(row_ordinal == selected_row_count);

  property_set->matrix_properties = selected_rows;
  property_set->matrix_property_count = row_ordinal;
  property_set->matrix_shape_spans = shape_spans;
  property_set->matrix_shape_span_count = shape_span_count;
  return iree_ok_status();
}

iree_status_t loom_spirv_vulkan_hal_profile_supports_cooperative_matrix_row(
    iree_hal_device_t* device, iree_string_view_t row_name,
    bool* out_supported) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_supported);
  *out_supported = false;

  const loom_spirv_cooperative_matrix_property_t* model_row =
      loom_spirv_vulkan_hal_profile_find_model_row(row_name);
  if (model_row == NULL) {
    return iree_ok_status();
  }
  const iree_hal_device_spec_t* device_spec = NULL;
  iree_hal_vulkan_device_spec_t vulkan_spec = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_decode_device_spec(
      device, &device_spec, &vulkan_spec));
  (void)device_spec;
  if (!iree_all_bits_set(vulkan_spec.enabled_features.general,
                         IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX)) {
    return iree_ok_status();
  }
  *out_supported = loom_spirv_vulkan_hal_profile_model_row_supported(
      model_row, &vulkan_spec);
  return iree_ok_status();
}

iree_status_t loom_spirv_vulkan_hal_profile_initialize_target_bundle(
    const loom_spirv_vulkan_hal_profile_facts_t* facts,
    loom_target_bundle_storage_t* out_storage) {
  IREE_ASSERT_ARGUMENT(facts);
  IREE_ASSERT_ARGUMENT(out_storage);

  *out_storage = (loom_target_bundle_storage_t){0};

  if (facts->api_version < LOOM_SPIRV_VULKAN_API_VERSION_1_3) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan SPIR-V raw-BDA profile requires Vulkan 1.3");
  }
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_require_flag(
      facts, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_RAW_BDA_EXECUTABLE,
      IREE_SV("Vulkan HAL device does not support the vulkan1.3+bda "
              "SPIR-V target")));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_require_flag(
      facts, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_BUFFER_DEVICE_ADDRESS,
      IREE_SV("Vulkan device does not expose buffer device addresses")));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_require_flag(
      facts, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT64,
      IREE_SV("Vulkan device does not expose shaderInt64 required by raw BDA "
              "SPIR-V")));

  *out_storage = (loom_target_bundle_storage_t){
      .snapshot = *loom_spirv_low_target_bundle_vulkan1_3.snapshot,
      .export_plan = *loom_spirv_low_target_bundle_vulkan1_3.export_plan,
      .config = *loom_spirv_low_target_bundle_vulkan1_3.config,
      .bundle = loom_spirv_low_target_bundle_vulkan1_3,
  };
  loom_target_bundle_storage_rebind(out_storage);

  out_storage->bundle.name = IREE_SV("spirv-vulkan1.3-bda-hal");
  out_storage->snapshot.name = IREE_SV("spirv-vulkan1.3-bda");
  out_storage->snapshot.max_workgroup_size = facts->max_compute_workgroup_size;
  out_storage->snapshot.max_flat_workgroup_size =
      facts->max_compute_workgroup_invocations;
  out_storage->snapshot.subgroup_size = facts->subgroup_size;
  out_storage->snapshot.max_workgroup_count =
      facts->max_compute_workgroup_count;
  out_storage->export_plan.name = IREE_SV("spirv-hal-kernel");
  out_storage->export_plan.abi_kind = LOOM_TARGET_ABI_HAL_KERNEL;
  out_storage->config.name = IREE_SV("spirv.logical.core.vulkan1.3.bda");
  out_storage->config.contract_feature_bits =
      loom_spirv_vulkan_hal_profile_feature_bits(facts);
  return iree_ok_status();
}
