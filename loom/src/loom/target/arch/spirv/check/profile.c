// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/check/profile.h"

#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/compile_diagnostics.h"
#include "loomc/iree.h"
#include "loomc/target/spirv/profile.h"

// The checker spells feature IDs using their public enum suffixes. This table
// belongs to fixture setup; production profiles use the typed LoomC API.
#define LOOM_SPIRV_CHECK_FEATURE(name) \
  [LOOMC_SPIRV_FEATURE_##name] = IREE_SVL(#name)
static const iree_string_view_t kFeatureNames[] = {
    LOOM_SPIRV_CHECK_FEATURE(VULKAN_SHADER),
    LOOM_SPIRV_CHECK_FEATURE(PHYSICAL_STORAGE_BUFFER),
    LOOM_SPIRV_CHECK_FEATURE(FLOAT16),
    LOOM_SPIRV_CHECK_FEATURE(FLOAT64),
    LOOM_SPIRV_CHECK_FEATURE(INT8),
    LOOM_SPIRV_CHECK_FEATURE(INT16),
    LOOM_SPIRV_CHECK_FEATURE(STORAGE_BUFFER_8BIT_ACCESS),
    LOOM_SPIRV_CHECK_FEATURE(STORAGE_BUFFER_16BIT_ACCESS),
    LOOM_SPIRV_CHECK_FEATURE(COOPERATIVE_VECTOR_NV),
    LOOM_SPIRV_CHECK_FEATURE(COOPERATIVE_VECTOR_TRAINING_NV),
    LOOM_SPIRV_CHECK_FEATURE(COOPERATIVE_MATRIX_KHR),
    LOOM_SPIRV_CHECK_FEATURE(BFLOAT16_TYPE_KHR),
    LOOM_SPIRV_CHECK_FEATURE(BFLOAT16_DOT_PRODUCT_KHR),
    LOOM_SPIRV_CHECK_FEATURE(BFLOAT16_COOPERATIVE_MATRIX_KHR),
    LOOM_SPIRV_CHECK_FEATURE(INT64),
    LOOM_SPIRV_CHECK_FEATURE(GROUP_NON_UNIFORM),
    LOOM_SPIRV_CHECK_FEATURE(VULKAN_MEMORY_MODEL_DEVICE_SCOPE),
    LOOM_SPIRV_CHECK_FEATURE(STORAGE_BUFFER_INT64_ATOMICS),
    LOOM_SPIRV_CHECK_FEATURE(WORKGROUP_INT64_ATOMICS),
    LOOM_SPIRV_CHECK_FEATURE(STORAGE_BUFFER_FLOAT16_ATOMICS),
    LOOM_SPIRV_CHECK_FEATURE(WORKGROUP_FLOAT16_ATOMICS),
    LOOM_SPIRV_CHECK_FEATURE(STORAGE_BUFFER_FLOAT16_ATOMIC_ADD),
    LOOM_SPIRV_CHECK_FEATURE(WORKGROUP_FLOAT16_ATOMIC_ADD),
    LOOM_SPIRV_CHECK_FEATURE(STORAGE_BUFFER_FLOAT32_ATOMICS),
    LOOM_SPIRV_CHECK_FEATURE(WORKGROUP_FLOAT32_ATOMICS),
    LOOM_SPIRV_CHECK_FEATURE(STORAGE_BUFFER_FLOAT32_ATOMIC_ADD),
    LOOM_SPIRV_CHECK_FEATURE(WORKGROUP_FLOAT32_ATOMIC_ADD),
    LOOM_SPIRV_CHECK_FEATURE(STORAGE_BUFFER_FLOAT64_ATOMICS),
    LOOM_SPIRV_CHECK_FEATURE(WORKGROUP_FLOAT64_ATOMICS),
    LOOM_SPIRV_CHECK_FEATURE(STORAGE_BUFFER_FLOAT64_ATOMIC_ADD),
    LOOM_SPIRV_CHECK_FEATURE(WORKGROUP_FLOAT64_ATOMIC_ADD),
    LOOM_SPIRV_CHECK_FEATURE(FLOAT32_DENORM_PRESERVE),
    LOOM_SPIRV_CHECK_FEATURE(GROUP_NON_UNIFORM_BALLOT),
    LOOM_SPIRV_CHECK_FEATURE(FLOAT16_64_DENORM_PRESERVE),
    LOOM_SPIRV_CHECK_FEATURE(FLOAT16_64_ROUNDING_RTE),
    LOOM_SPIRV_CHECK_FEATURE(FLOAT32_ROUNDING_RTE),
    LOOM_SPIRV_CHECK_FEATURE(FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE),
    LOOM_SPIRV_CHECK_FEATURE(FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE),
    LOOM_SPIRV_CHECK_FEATURE(FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE),
    LOOM_SPIRV_CHECK_FEATURE(FLOAT64_TO_FLOAT16_CONVERSION),
    LOOM_SPIRV_CHECK_FEATURE(GROUP_NON_UNIFORM_VOTE),
    LOOM_SPIRV_CHECK_FEATURE(GROUP_NON_UNIFORM_ARITHMETIC),
    LOOM_SPIRV_CHECK_FEATURE(GROUP_NON_UNIFORM_CLUSTERED),
    LOOM_SPIRV_CHECK_FEATURE(SUBGROUP_EXTENDED_TYPES),
    LOOM_SPIRV_CHECK_FEATURE(GROUP_NON_UNIFORM_SHUFFLE),
    LOOM_SPIRV_CHECK_FEATURE(GROUP_NON_UNIFORM_SHUFFLE_RELATIVE),
    LOOM_SPIRV_CHECK_FEATURE(INTEGER_DOT_PRODUCT),
    LOOM_SPIRV_CHECK_FEATURE(INTEGER_DOT_PRODUCT_INPUT_ALL),
    LOOM_SPIRV_CHECK_FEATURE(INTEGER_DOT_PRODUCT_INPUT4X8_BIT),
    LOOM_SPIRV_CHECK_FEATURE(INTEGER_DOT_PRODUCT_INPUT4X8_BIT_PACKED),
    LOOM_SPIRV_CHECK_FEATURE(FMA_FLOAT16),
    LOOM_SPIRV_CHECK_FEATURE(FMA_FLOAT32),
    LOOM_SPIRV_CHECK_FEATURE(FMA_FLOAT64),
    LOOM_SPIRV_CHECK_FEATURE(FLOAT_CONTROLS2),
};
#undef LOOM_SPIRV_CHECK_FEATURE
static_assert(IREE_ARRAYSIZE(kFeatureNames) == LOOMC_SPIRV_FEATURE_COUNT,
              "all public SPIR-V feature IDs need checker spellings");

#define LOOM_SPIRV_CHECK_LIMIT(name) \
  [LOOMC_SPIRV_LIMIT_##name] = IREE_SVL(#name)
static const iree_string_view_t kLimitNames[] = {
    LOOM_SPIRV_CHECK_LIMIT(MAX_WORKGROUP_SIZE_X),
    LOOM_SPIRV_CHECK_LIMIT(MAX_WORKGROUP_SIZE_Y),
    LOOM_SPIRV_CHECK_LIMIT(MAX_WORKGROUP_SIZE_Z),
    LOOM_SPIRV_CHECK_LIMIT(MAX_FLAT_WORKGROUP_SIZE),
    LOOM_SPIRV_CHECK_LIMIT(MAX_WORKGROUP_STORAGE_BYTES),
    LOOM_SPIRV_CHECK_LIMIT(SUBGROUP_SIZE),
    LOOM_SPIRV_CHECK_LIMIT(MAX_WORKGROUP_COUNT_X),
    LOOM_SPIRV_CHECK_LIMIT(MAX_WORKGROUP_COUNT_Y),
    LOOM_SPIRV_CHECK_LIMIT(MAX_WORKGROUP_COUNT_Z),
    LOOM_SPIRV_CHECK_LIMIT(MAX_GRID_SIZE_X),
    LOOM_SPIRV_CHECK_LIMIT(MAX_GRID_SIZE_Y),
    LOOM_SPIRV_CHECK_LIMIT(MAX_GRID_SIZE_Z),
    LOOM_SPIRV_CHECK_LIMIT(MAX_FLAT_GRID_SIZE),
    LOOM_SPIRV_CHECK_LIMIT(DEFAULT_SUBGROUP_SIZE),
    LOOM_SPIRV_CHECK_LIMIT(MIN_SUBGROUP_SIZE),
    LOOM_SPIRV_CHECK_LIMIT(MAX_SUBGROUP_SIZE),
    LOOM_SPIRV_CHECK_LIMIT(MAX_WORKGROUP_SUBGROUP_COUNT),
};
#undef LOOM_SPIRV_CHECK_LIMIT
static_assert(IREE_ARRAYSIZE(kLimitNames) == LOOMC_SPIRV_LIMIT_COUNT,
              "all public SPIR-V limit IDs need checker spellings");

iree_status_t loom_spirv_check_profile_select(
    const loom_check_emit_provider_request_t* request,
    iree_string_view_t target_specification, iree_string_view_t features,
    iree_string_view_t limit_options, loomc_target_profile_t** out_profile) {
  *out_profile = NULL;
  if (iree_string_view_is_empty(target_specification)) {
    return iree_ok_status();
  }

  loomc_spirv_feature_fact_t facts[LOOMC_SPIRV_FEATURE_COUNT] = {0};
  bool seen[LOOMC_SPIRV_FEATURE_COUNT] = {false};
  iree_host_size_t fact_count = 0;
  iree_string_view_t remaining = features;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t name;
    iree_string_view_split(remaining, ',', &name, &remaining);
    const bool enabled = !iree_string_view_consume_prefix(&name, IREE_SV("-"));
    uint32_t feature = 1;
    for (; feature < IREE_ARRAYSIZE(kFeatureNames); ++feature) {
      if (iree_string_view_equal(name, kFeatureNames[feature])) {
        break;
      }
    }
    if (feature == IREE_ARRAYSIZE(kFeatureNames) || seen[feature]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown or duplicate SPIR-V feature '%.*s'",
                              (int)name.size, name.data);
    }
    seen[feature] = true;
    facts[fact_count++] = (loomc_spirv_feature_fact_t){
        .feature = (loomc_spirv_feature_t)feature,
        .state = enabled ? LOOMC_TARGET_FACT_STATE_TRUE
                         : LOOMC_TARGET_FACT_STATE_FALSE,
        .provenance = loomc_make_cstring_view("SPIR-V checker features option"),
    };
  }

  loomc_spirv_limit_fact_t limits[LOOMC_SPIRV_LIMIT_COUNT] = {0};
  bool seen_limits[LOOMC_SPIRV_LIMIT_COUNT] = {false};
  iree_host_size_t limit_count = 0;
  remaining = limit_options;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t entry, name, value_text;
    iree_string_view_split(remaining, ',', &entry, &remaining);
    iree_string_view_split(entry, ':', &name, &value_text);
    uint32_t limit = 1;
    for (; limit < IREE_ARRAYSIZE(kLimitNames); ++limit) {
      if (iree_string_view_equal(name, kLimitNames[limit])) {
        break;
      }
    }
    if (limit == IREE_ARRAYSIZE(kLimitNames) || seen_limits[limit]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown or duplicate SPIR-V limit '%.*s'",
                              (int)name.size, name.data);
    }
    uint64_t value = 0;
    bool is_decimal = !iree_string_view_is_empty(value_text);
    for (iree_host_size_t i = 0; i < value_text.size && is_decimal; ++i) {
      is_decimal = value_text.data[i] >= '0' && value_text.data[i] <= '9';
    }
    if (!is_decimal ||
        !iree_string_view_atoi_uint64_base(value_text, 10, &value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "SPIR-V limit '%.*s' expects an unsigned integer",
                              (int)name.size, name.data);
    }
    seen_limits[limit] = true;
    limits[limit_count++] = (loomc_spirv_limit_fact_t){
        .limit = (loomc_spirv_limit_t)limit,
        .state = LOOMC_TARGET_FACT_STATE_TRUE,
        .value = value,
        .provenance = loomc_make_cstring_view("SPIR-V checker limits option"),
    };
  }

  loomc_target_profile_t* base_profile = NULL;
  IREE_RETURN_IF_ERROR(loom_check_compile_session_select_target_profile(
      request->environment->compile_session, target_specification,
      &base_profile));
  if (!fact_count && !limit_count) {
    *out_profile = base_profile;
    return iree_ok_status();
  }
  const loomc_spirv_profile_options_t profile_options = {
      .type = LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      .structure_size = sizeof(profile_options),
      .feature_facts = facts,
      .feature_fact_count = fact_count,
      .limit_facts = limits,
      .limit_fact_count = limit_count,
  };
  loomc_result_t* result = NULL;
  iree_status_t status =
      iree_status_from_loomc(loomc_spirv_target_profile_refine(
          base_profile, &profile_options,
          loomc_allocator_from_iree(request->host_allocator), out_profile,
          &result));
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result_diagnostics(
        request->diagnostic_collector, result);
  }
  loomc_result_release(result);
  loomc_target_profile_release(base_profile);
  return status;
}
