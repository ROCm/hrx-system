// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/vulkan_profile.h"

#include <stdint.h>

#include "loom/target/arch/spirv/cooperative_properties.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/records/target_records.h"

typedef struct loom_spirv_vulkan_hal_feature_row_t {
  // HAL device-query key under the vulkan.feature category.
  iree_string_view_t query_key;
  // Compact fact flag set when the HAL reports the feature as available.
  loom_spirv_vulkan_hal_profile_flag_bits_t flag;
  // Additional fact flags required before projecting |feature_bits|.
  loom_spirv_vulkan_hal_profile_flags_t required_flags;
  // SPIR-V feature atoms implied by this Vulkan feature.
  loom_spirv_feature_bits_t feature_bits;
} loom_spirv_vulkan_hal_feature_row_t;

static const loom_spirv_vulkan_hal_feature_row_t kVulkanFeatureRows[] = {
    {
        .query_key = IREE_SVL("buffer_device_address"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_BUFFER_DEVICE_ADDRESS,
        .feature_bits = 0,
    },
    {
        .query_key = IREE_SVL("subgroup_size_control"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SUBGROUP_SIZE_CONTROL,
        .feature_bits = 0,
    },
    {
        .query_key = IREE_SVL("cooperative_matrix_khr"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR,
        .feature_bits = LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
    },
    {
        .query_key = IREE_SVL("storage_buffer_8bit_access"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_8BIT_ACCESS,
        .feature_bits = LOOM_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS,
    },
    {
        .query_key = IREE_SVL("storage_buffer_16bit_access"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_16BIT_ACCESS,
        .feature_bits = LOOM_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS,
    },
    {
        .query_key = IREE_SVL("shader_float16"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT16,
        .feature_bits = LOOM_SPIRV_FEATURE_FLOAT16,
    },
    {
        .query_key = IREE_SVL("shader_float64"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT64,
        .feature_bits = LOOM_SPIRV_FEATURE_FLOAT64,
    },
    {
        .query_key = IREE_SVL("shader_bfloat16_type"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_TYPE,
        .feature_bits = LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR,
    },
    {
        .query_key = IREE_SVL("shader_bfloat16_dot_product"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_DOT_PRODUCT,
        .required_flags =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_TYPE,
        .feature_bits = LOOM_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR,
    },
    {
        .query_key = IREE_SVL("shader_bfloat16_cooperative_matrix"),
        .flag =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_COOPERATIVE_MATRIX,
        .required_flags =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR |
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_BFLOAT16_TYPE,
        .feature_bits = LOOM_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR,
    },
    {
        .query_key = IREE_SVL("shader_int8"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT8,
        .feature_bits = LOOM_SPIRV_FEATURE_INT8,
    },
    {
        .query_key = IREE_SVL("shader_int16"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT16,
        .feature_bits = LOOM_SPIRV_FEATURE_INT16,
    },
    {
        .query_key = IREE_SVL("shader_int64"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT64,
        .feature_bits = LOOM_SPIRV_FEATURE_INT64,
    },
    {
        .query_key = IREE_SVL("shader_integer_dot_product"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INTEGER_DOT_PRODUCT,
        .feature_bits = 0,
    },
    {
        .query_key = IREE_SVL("vulkan_memory_model"),
        .flag = LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_VULKAN_MEMORY_MODEL,
        .feature_bits = 0,
    },
    {
        .query_key = IREE_SVL("vulkan_memory_model_device_scope"),
        .flag =
            LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
        .feature_bits = 0,
    },
};

static iree_status_t loom_spirv_vulkan_hal_profile_query_u32(
    iree_hal_device_t* device, iree_string_view_t category,
    iree_string_view_t key, uint32_t* out_value) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_device_query_i64(device, category, key, &value));
  if (value < 0 || value > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan profile query '%.*s :: %.*s' returned value out of uint32_t "
        "range",
        (int)category.size, category.data, (int)key.size, key.data);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_spirv_vulkan_hal_profile_query_feature_flag(
    iree_hal_device_t* device, const loom_spirv_vulkan_hal_feature_row_t* row,
    loom_spirv_vulkan_hal_profile_facts_t* facts) {
  uint32_t value = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.feature"), row->query_key, &value));
  if (value != 0) {
    facts->flags |= row->flag;
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_vulkan_hal_profile_query(
    iree_hal_device_t* device, iree_hal_executable_cache_t* executable_cache,
    loom_spirv_vulkan_hal_profile_facts_t* out_facts) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(executable_cache);
  IREE_ASSERT_ARGUMENT(out_facts);

  *out_facts = (loom_spirv_vulkan_hal_profile_facts_t){0};
  if (iree_hal_executable_cache_can_prepare_format(
          executable_cache, IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION,
          IREE_SV("vulkan-spirv-bda"))) {
    out_facts->flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_RAW_BDA_EXECUTABLE;
  }

  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"), IREE_SV("api_version"),
      &out_facts->api_version));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"), IREE_SV("subgroup_size"),
      &out_facts->subgroup_size));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"),
      IREE_SV("max_compute_workgroup_invocations"),
      &out_facts->max_compute_workgroup_invocations));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"), IREE_SV("max_compute_workgroup_size_x"),
      &out_facts->max_compute_workgroup_size.x));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"), IREE_SV("max_compute_workgroup_size_y"),
      &out_facts->max_compute_workgroup_size.y));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"), IREE_SV("max_compute_workgroup_size_z"),
      &out_facts->max_compute_workgroup_size.z));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"),
      IREE_SV("max_compute_workgroup_count_x"),
      &out_facts->max_compute_workgroup_count.x));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"),
      IREE_SV("max_compute_workgroup_count_y"),
      &out_facts->max_compute_workgroup_count.y));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"),
      IREE_SV("max_compute_workgroup_count_z"),
      &out_facts->max_compute_workgroup_count.z));

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kVulkanFeatureRows); ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_feature_flag(
        device, &kVulkanFeatureRows[i], out_facts));
  }

  if (iree_any_bit_set(
          out_facts->flags,
          LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR)) {
    IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
        device, IREE_SV("vulkan.cooperative_matrix"), IREE_SV("property_count"),
        &out_facts->cooperative_matrix_property_count));
    IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
        device, IREE_SV("vulkan.cooperative_matrix"),
        IREE_SV("supported_stages"),
        &out_facts->cooperative_matrix_supported_stages));
  }

  return iree_ok_status();
}

iree_status_t loom_spirv_vulkan_hal_query_cooperative_matrix_properties(
    iree_hal_device_t* device, iree_allocator_t allocator,
    iree_hal_vulkan_cooperative_matrix_property_t** out_properties,
    iree_host_size_t* out_property_count) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_properties);
  IREE_ASSERT_ARGUMENT(out_property_count);

  *out_properties = NULL;
  *out_property_count = 0;
  iree_host_size_t property_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_device_query_cooperative_matrix_properties(
          device, /*property_capacity=*/0, &property_count,
          /*out_properties=*/NULL));
  if (property_count == 0) {
    return iree_ok_status();
  }
  iree_hal_vulkan_cooperative_matrix_property_t* properties = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, property_count * sizeof(*properties), (void**)&properties));
  iree_status_t status =
      iree_hal_vulkan_device_query_cooperative_matrix_properties(
          device, property_count, &property_count, properties);
  if (iree_status_is_ok(status)) {
    *out_properties = properties;
    *out_property_count = property_count;
  } else {
    iree_allocator_free(allocator, properties);
  }
  return status;
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
    const iree_hal_vulkan_cooperative_matrix_property_t* device_rows,
    iree_host_size_t device_row_count) {
  for (iree_host_size_t i = 0; i < device_row_count; ++i) {
    if (loom_spirv_vulkan_hal_profile_model_row_matches_device_row(
            model_row, &device_rows[i])) {
      return true;
    }
  }
  return false;
}

static iree_status_t
loom_spirv_vulkan_hal_target_profile_storage_filter_matrix_rows(
    const iree_hal_vulkan_cooperative_matrix_property_t* device_rows,
    iree_host_size_t device_row_count, iree_allocator_t allocator,
    loom_spirv_cooperative_matrix_property_t** out_rows,
    iree_host_size_t* out_row_count) {
  *out_rows = NULL;
  *out_row_count = 0;
  iree_host_size_t model_row_count = 0;
  const loom_spirv_cooperative_matrix_property_t* model_rows =
      loom_spirv_cooperative_matrix_model_properties(&model_row_count);
  if (model_row_count == 0 || device_row_count == 0) {
    return iree_ok_status();
  }
  loom_spirv_cooperative_matrix_property_t* rows = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, model_row_count * sizeof(*rows), (void**)&rows));
  iree_host_size_t row_count = 0;
  for (iree_host_size_t i = 0; i < model_row_count; ++i) {
    if (!loom_spirv_vulkan_hal_profile_model_row_supported(
            &model_rows[i], device_rows, device_row_count)) {
      continue;
    }
    rows[row_count++] = model_rows[i];
  }
  *out_rows = rows;
  *out_row_count = row_count;
  return iree_ok_status();
}

iree_status_t loom_spirv_vulkan_hal_target_profile_storage_initialize(
    const loom_spirv_vulkan_hal_profile_facts_t* facts,
    const iree_hal_vulkan_cooperative_matrix_property_t*
        cooperative_matrix_properties,
    iree_host_size_t cooperative_matrix_property_count,
    iree_allocator_t allocator,
    loom_spirv_vulkan_hal_target_profile_storage_t* out_storage) {
  IREE_ASSERT_ARGUMENT(facts);
  IREE_ASSERT_ARGUMENT(out_storage);
  if (cooperative_matrix_property_count != 0) {
    IREE_ASSERT_ARGUMENT(cooperative_matrix_properties);
  }
  *out_storage = (loom_spirv_vulkan_hal_target_profile_storage_t){0};

  loom_spirv_feature_bits_t feature_bits =
      loom_spirv_vulkan_hal_profile_feature_bits(facts);
  const bool has_cooperative_matrix_khr = iree_any_bit_set(
      facts->flags, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR);

  loom_spirv_cooperative_matrix_property_t* matrix_rows = NULL;
  iree_host_size_t matrix_row_count = 0;
  iree_status_t status = iree_ok_status();
  if (has_cooperative_matrix_khr) {
    status = loom_spirv_vulkan_hal_target_profile_storage_filter_matrix_rows(
        cooperative_matrix_properties, cooperative_matrix_property_count,
        allocator, &matrix_rows, &matrix_row_count);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < matrix_row_count; ++i) {
      feature_bits |= matrix_rows[i].required_feature_bits;
    }
    status = loom_spirv_cooperative_property_storage_initialize(
        feature_bits, matrix_rows, matrix_row_count,
        /*vector_properties=*/NULL, /*vector_property_count=*/0, allocator,
        &out_storage->cooperative_properties);
  }
  iree_allocator_free(allocator, matrix_rows);
  if (!iree_status_is_ok(status)) {
    loom_spirv_vulkan_hal_target_profile_storage_deinitialize(out_storage,
                                                              allocator);
    return status;
  }
  out_storage->profile.cooperative_properties =
      &out_storage->cooperative_properties.set;
  return iree_ok_status();
}

void loom_spirv_vulkan_hal_target_profile_storage_deinitialize(
    loom_spirv_vulkan_hal_target_profile_storage_t* storage,
    iree_allocator_t allocator) {
  if (storage == NULL) {
    return;
  }
  loom_spirv_cooperative_property_storage_deinitialize(
      &storage->cooperative_properties, allocator);
  *storage = (loom_spirv_vulkan_hal_target_profile_storage_t){0};
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
      IREE_SV("Vulkan HAL executable cache does not support "
              "vulkan-spirv-bda")));
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
