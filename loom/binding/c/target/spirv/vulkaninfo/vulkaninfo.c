// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/spirv/vulkaninfo.h"

#include <inttypes.h>
#include <string.h>

#include "diagnostic.h"
#include "iree/base/internal/json.h"
#include "loomc/iree.h"
#include "result.h"

enum {
  LOOMC_SPIRV_VULKANINFO_LIMIT_FACT_CAPACITY = 16,
  LOOMC_SPIRV_VULKANINFO_ENVIRONMENT_FACT_CAPACITY = 4,
  // PCI-SIG vendor identifier reported by AMD Vulkan physical devices.
  LOOMC_SPIRV_VULKANINFO_AMD_VENDOR_ID = 0x1002,
};

typedef struct loomc_spirv_vulkaninfo_import_t {
  // Source whose JSON bytes are being imported.
  const loomc_source_t* source;

  // Result used for importer diagnostics.
  loomc_result_t* result;

  // Feature facts normalized from recognized Vulkan fields.
  loomc_spirv_feature_fact_t feature_facts[LOOMC_SPIRV_FEATURE_COUNT];

  // Number of entries in feature_facts.
  loomc_host_size_t feature_fact_count;

  // Numeric limit facts normalized from recognized Vulkan fields.
  loomc_spirv_limit_fact_t
      limit_facts[LOOMC_SPIRV_VULKANINFO_LIMIT_FACT_CAPACITY];

  // Number of entries in limit_facts.
  loomc_host_size_t limit_fact_count;

  // Environment facts normalized from recognized Vulkan fields.
  loomc_spirv_environment_fact_t
      environment_facts[LOOMC_SPIRV_VULKANINFO_ENVIRONMENT_FACT_CAPACITY];

  // Number of entries in environment_facts.
  loomc_host_size_t environment_fact_count;
} loomc_spirv_vulkaninfo_import_t;

typedef struct loomc_spirv_vulkaninfo_first_member_state_t {
  // Captured member value.
  iree_string_view_t value;
} loomc_spirv_vulkaninfo_first_member_state_t;

typedef struct loomc_spirv_vulkaninfo_flag_state_t {
  // Vulkan Profiles flag spelling to find.
  iree_string_view_t flag;

  // Whether the flag was present in the array.
  bool found;
} loomc_spirv_vulkaninfo_flag_state_t;

typedef struct loomc_spirv_vulkaninfo_feature_field_t {
  // Vulkan feature-struct field name.
  const char* field_name;

  // Public SPIR-V feature represented by the field.
  loomc_spirv_feature_t feature;

  // Stable provenance for the imported field.
  const char* provenance;
} loomc_spirv_vulkaninfo_feature_field_t;

static const char* loomc_spirv_vulkaninfo_string_data(
    loomc_string_view_t value) {
  return value.data != NULL ? value.data : "";
}

static loomc_status_t loomc_spirv_vulkaninfo_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_validate_options(
    const loomc_spirv_vulkaninfo_profile_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_SPIRV_VULKANINFO_PROFILE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V vulkaninfo options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V vulkaninfo options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "SPIR-V vulkaninfo option extensions are not supported");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_vulkaninfo_validate_string_view(options->identifier));
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_vulkaninfo_validate_string_view(options->profile_name));
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_fail_status(
    loomc_spirv_vulkaninfo_import_t* import, loomc_status_t status) {
  if (!loomc_result_succeeded(import->result)) {
    loomc_status_free(status);
    return loomc_ok_status();
  }
  return loomc_result_fail_status_diagnostic_consume(
      import->result, import->source, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      loomc_make_cstring_view("SPIRV/VULKANINFO"), status);
}

static loomc_status_t loomc_spirv_vulkaninfo_fail_iree_status(
    loomc_spirv_vulkaninfo_import_t* import, iree_status_t status) {
  return loomc_spirv_vulkaninfo_fail_status(import,
                                            loomc_status_from_iree(status));
}

static loomc_status_t loomc_spirv_vulkaninfo_fail_message(
    loomc_spirv_vulkaninfo_import_t* import, const char* message) {
  if (!loomc_result_succeeded(import->result)) {
    return loomc_ok_status();
  }
  return loomc_spirv_vulkaninfo_fail_iree_status(
      import, iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message));
}

static bool loomc_spirv_vulkaninfo_is_object(iree_string_view_t value) {
  return value.size >= 2 && value.data[0] == '{';
}

static bool loomc_spirv_vulkaninfo_is_array(iree_string_view_t value) {
  return value.size >= 2 && value.data[0] == '[';
}

static bool loomc_spirv_vulkaninfo_is_null(iree_string_view_t value) {
  return iree_string_view_equal(value, IREE_SV("null"));
}

static bool loomc_spirv_vulkaninfo_contains_char(iree_string_view_t value,
                                                 char c) {
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    if (value.data[i] == c) {
      return true;
    }
  }
  return false;
}

static iree_status_t loomc_spirv_vulkaninfo_try_lookup(
    iree_string_view_t object, iree_string_view_t key,
    iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  if (iree_string_view_is_empty(object)) {
    return iree_ok_status();
  }
  return iree_json_try_lookup_object_value(object, key, out_value);
}

static loomc_status_t loomc_spirv_vulkaninfo_try_lookup_object(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    iree_string_view_t key, iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_status_t status = loomc_spirv_vulkaninfo_try_lookup(object, key, &value);
  if (!iree_status_is_ok(status)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(import, status);
  }
  if (iree_string_view_is_empty(value)) {
    return loomc_ok_status();
  }
  if (!loomc_spirv_vulkaninfo_is_object(value)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(
        import, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                 "vulkaninfo member '%.*s' must be an object",
                                 (int)key.size, key.data));
  }
  *out_value = value;
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_try_lookup_array(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    iree_string_view_t key, iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_status_t status = loomc_spirv_vulkaninfo_try_lookup(object, key, &value);
  if (!iree_status_is_ok(status)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(import, status);
  }
  if (iree_string_view_is_empty(value)) {
    return loomc_ok_status();
  }
  if (!loomc_spirv_vulkaninfo_is_array(value)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(
        import, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                 "vulkaninfo member '%.*s' must be an array",
                                 (int)key.size, key.data));
  }
  *out_value = value;
  return loomc_ok_status();
}

static iree_status_t loomc_spirv_vulkaninfo_first_member_visitor(
    void* user_data, iree_string_view_t key, iree_string_view_t value) {
  loomc_spirv_vulkaninfo_first_member_state_t* state =
      (loomc_spirv_vulkaninfo_first_member_state_t*)user_data;
  (void)key;
  state->value = value;
  return iree_status_from_code(IREE_STATUS_CANCELLED);
}

static loomc_status_t loomc_spirv_vulkaninfo_first_member(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    loomc_spirv_vulkaninfo_first_member_state_t* out_state) {
  *out_state = (loomc_spirv_vulkaninfo_first_member_state_t){0};
  iree_status_t status = iree_json_enumerate_object(
      object, loomc_spirv_vulkaninfo_first_member_visitor, out_state);
  if (!iree_status_is_ok(status)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(import, status);
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_add_feature_fact(
    loomc_spirv_vulkaninfo_import_t* import, loomc_spirv_feature_t feature,
    loomc_target_fact_state_t state, loomc_string_view_t provenance) {
  if (!loomc_result_succeeded(import->result) ||
      state == LOOMC_TARGET_FACT_STATE_UNKNOWN) {
    return loomc_ok_status();
  }
  for (loomc_host_size_t i = 0; i < import->feature_fact_count; ++i) {
    const loomc_spirv_feature_fact_t* existing = &import->feature_facts[i];
    if (existing->feature != feature) {
      continue;
    }
    if (existing->state == state) {
      return loomc_ok_status();
    }
    return loomc_spirv_vulkaninfo_fail_iree_status(
        import,
        iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "vulkaninfo sources '%.*s' and '%.*s' disagree on SPIR-V "
            "feature %u",
            (int)existing->provenance.size,
            loomc_spirv_vulkaninfo_string_data(existing->provenance),
            (int)provenance.size,
            loomc_spirv_vulkaninfo_string_data(provenance), (unsigned)feature));
  }
  if (import->feature_fact_count >= LOOMC_SPIRV_FEATURE_COUNT) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "too many vulkaninfo SPIR-V feature facts");
  }
  import->feature_facts[import->feature_fact_count++] =
      (loomc_spirv_feature_fact_t){
          .feature = feature,
          .state = state,
          .provenance = provenance,
      };
  return loomc_ok_status();
}

static loomc_target_fact_state_t loomc_spirv_vulkaninfo_and_fact_states(
    loomc_target_fact_state_t lhs, loomc_target_fact_state_t rhs) {
  if (lhs == LOOMC_TARGET_FACT_STATE_FALSE ||
      rhs == LOOMC_TARGET_FACT_STATE_FALSE) {
    return LOOMC_TARGET_FACT_STATE_FALSE;
  }
  if (lhs == LOOMC_TARGET_FACT_STATE_TRUE &&
      rhs == LOOMC_TARGET_FACT_STATE_TRUE) {
    return LOOMC_TARGET_FACT_STATE_TRUE;
  }
  return LOOMC_TARGET_FACT_STATE_UNKNOWN;
}

static loomc_status_t loomc_spirv_vulkaninfo_add_limit_fact(
    loomc_spirv_vulkaninfo_import_t* import, loomc_spirv_limit_t limit,
    uint64_t value, loomc_string_view_t provenance) {
  if (!loomc_result_succeeded(import->result)) {
    return loomc_ok_status();
  }
  if (import->limit_fact_count >= LOOMC_SPIRV_VULKANINFO_LIMIT_FACT_CAPACITY) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "too many vulkaninfo SPIR-V limit facts");
  }
  import->limit_facts[import->limit_fact_count++] = (loomc_spirv_limit_fact_t){
      .limit = limit,
      .state = LOOMC_TARGET_FACT_STATE_TRUE,
      .value = value,
      .provenance = provenance,
  };
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_add_environment_fact(
    loomc_spirv_vulkaninfo_import_t* import,
    loomc_spirv_environment_t environment, uint64_t value,
    loomc_string_view_t provenance) {
  if (!loomc_result_succeeded(import->result)) {
    return loomc_ok_status();
  }
  if (import->environment_fact_count >=
      LOOMC_SPIRV_VULKANINFO_ENVIRONMENT_FACT_CAPACITY) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "too many vulkaninfo SPIR-V environment facts");
  }
  import->environment_facts[import->environment_fact_count++] =
      (loomc_spirv_environment_fact_t){
          .environment = environment,
          .state = LOOMC_TARGET_FACT_STATE_TRUE,
          .value = value,
          .provenance = provenance,
      };
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_read_bool_feature(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    iree_string_view_t key, loomc_target_fact_state_t* out_state) {
  *out_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  if (!loomc_result_succeeded(import->result) ||
      iree_string_view_is_empty(object)) {
    return loomc_ok_status();
  }
  iree_string_view_t value = iree_string_view_empty();
  iree_status_t status = loomc_spirv_vulkaninfo_try_lookup(object, key, &value);
  if (!iree_status_is_ok(status)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(import, status);
  }
  if (iree_string_view_is_empty(value) ||
      loomc_spirv_vulkaninfo_is_null(value)) {
    return loomc_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("true"))) {
    *out_state = LOOMC_TARGET_FACT_STATE_TRUE;
    return loomc_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("false"))) {
    *out_state = LOOMC_TARGET_FACT_STATE_FALSE;
    return loomc_ok_status();
  }
  return loomc_spirv_vulkaninfo_fail_iree_status(
      import,
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                       "vulkaninfo feature '%.*s' must be boolean or null",
                       (int)key.size, key.data));
}

static loomc_status_t loomc_spirv_vulkaninfo_add_bool_feature(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    iree_string_view_t key, loomc_spirv_feature_t feature,
    loomc_string_view_t provenance) {
  loomc_target_fact_state_t state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_vulkaninfo_read_bool_feature(import, object, key, &state));
  return loomc_spirv_vulkaninfo_add_feature_fact(import, feature, state,
                                                 provenance);
}

static loomc_status_t loomc_spirv_vulkaninfo_add_bool_pair_feature(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    iree_string_view_t lhs_key, iree_string_view_t rhs_key,
    loomc_spirv_feature_t feature, loomc_string_view_t provenance) {
  loomc_target_fact_state_t lhs_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_bool_feature(
      import, object, lhs_key, &lhs_state));
  loomc_target_fact_state_t rhs_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_bool_feature(
      import, object, rhs_key, &rhs_state));
  return loomc_spirv_vulkaninfo_add_feature_fact(
      import, feature,
      loomc_spirv_vulkaninfo_and_fact_states(lhs_state, rhs_state), provenance);
}

static loomc_status_t loomc_spirv_vulkaninfo_flag_visitor(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  (void)index;
  loomc_spirv_vulkaninfo_flag_state_t* state =
      (loomc_spirv_vulkaninfo_flag_state_t*)user_data;
  if (type != IREE_JSON_VALUE_TYPE_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan Profiles flag arrays require strings");
  }
  state->found |= iree_string_view_equal(value, state->flag);
  return iree_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_read_flag_feature(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    iree_string_view_t key, iree_string_view_t flag,
    loomc_target_fact_state_t* out_state) {
  *out_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  if (!loomc_result_succeeded(import->result) ||
      iree_string_view_is_empty(object)) {
    return loomc_ok_status();
  }
  iree_string_view_t array = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_vulkaninfo_try_lookup_array(import, object, key, &array));
  if (iree_string_view_is_empty(array)) {
    return loomc_ok_status();
  }
  loomc_spirv_vulkaninfo_flag_state_t state = {
      .flag = flag,
  };
  iree_status_t status = iree_json_enumerate_array_typed(
      array, loomc_spirv_vulkaninfo_flag_visitor, &state);
  if (!iree_status_is_ok(status)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(import, status);
  }
  *out_state = state.found ? LOOMC_TARGET_FACT_STATE_TRUE
                           : LOOMC_TARGET_FACT_STATE_FALSE;
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_add_subgroup_features(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    iree_string_view_t key, loomc_string_view_t provenance) {
  loomc_target_fact_state_t basic_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_flag_feature(
      import, object, key, IREE_SV("VK_SUBGROUP_FEATURE_BASIC_BIT"),
      &basic_state));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM, basic_state, provenance));
  loomc_target_fact_state_t ballot_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_flag_feature(
      import, object, key, IREE_SV("VK_SUBGROUP_FEATURE_BALLOT_BIT"),
      &ballot_state));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_BALLOT,
      loomc_spirv_vulkaninfo_and_fact_states(basic_state, ballot_state),
      provenance));
  loomc_target_fact_state_t vote_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_flag_feature(
      import, object, key, IREE_SV("VK_SUBGROUP_FEATURE_VOTE_BIT"),
      &vote_state));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_VOTE,
      loomc_spirv_vulkaninfo_and_fact_states(basic_state, vote_state),
      provenance));
  loomc_target_fact_state_t arithmetic_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_flag_feature(
      import, object, key, IREE_SV("VK_SUBGROUP_FEATURE_ARITHMETIC_BIT"),
      &arithmetic_state));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_ARITHMETIC,
      loomc_spirv_vulkaninfo_and_fact_states(basic_state, arithmetic_state),
      provenance));
  loomc_target_fact_state_t shuffle_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_flag_feature(
      import, object, key, IREE_SV("VK_SUBGROUP_FEATURE_SHUFFLE_BIT"),
      &shuffle_state));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE,
      loomc_spirv_vulkaninfo_and_fact_states(basic_state, shuffle_state),
      provenance));
  loomc_target_fact_state_t shuffle_relative_state =
      LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_flag_feature(
      import, object, key, IREE_SV("VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT"),
      &shuffle_relative_state));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE,
      loomc_spirv_vulkaninfo_and_fact_states(basic_state,
                                             shuffle_relative_state),
      provenance));
  loomc_target_fact_state_t clustered_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_flag_feature(
      import, object, key, IREE_SV("VK_SUBGROUP_FEATURE_CLUSTERED_BIT"),
      &clustered_state));
  return loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_CLUSTERED,
      loomc_spirv_vulkaninfo_and_fact_states(basic_state, clustered_state),
      provenance);
}

static loomc_status_t loomc_spirv_vulkaninfo_read_float_control_independence(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    iree_string_view_t key, loomc_target_fact_state_t* out_state) {
  *out_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  iree_string_view_t independence = iree_string_view_empty();
  iree_status_t status =
      loomc_spirv_vulkaninfo_try_lookup(object, key, &independence);
  if (!iree_status_is_ok(status)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(import, status);
  }
  if (iree_string_view_is_empty(independence) ||
      loomc_spirv_vulkaninfo_is_null(independence)) {
    return loomc_ok_status();
  }
  const bool independent =
      iree_string_view_equal(
          independence,
          IREE_SV("VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY")) ||
      iree_string_view_equal(
          independence,
          IREE_SV("VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY_KHR")) ||
      iree_string_view_equal(
          independence, IREE_SV("VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL")) ||
      iree_string_view_equal(
          independence,
          IREE_SV("VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL_KHR"));
  const bool none =
      iree_string_view_equal(
          independence,
          IREE_SV("VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE")) ||
      iree_string_view_equal(
          independence,
          IREE_SV("VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE_KHR"));
  if (!independent && !none) {
    return loomc_spirv_vulkaninfo_fail_iree_status(
        import,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "vulkaninfo float-control independence '%.*s' is "
                         "invalid for field '%.*s'",
                         (int)independence.size, independence.data,
                         (int)key.size, key.data));
  }
  *out_state = independent ? LOOMC_TARGET_FACT_STATE_TRUE
                           : LOOMC_TARGET_FACT_STATE_FALSE;
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_add_float_control_features(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    loomc_string_view_t provenance) {
  loomc_target_fact_state_t denorm16 = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_target_fact_state_t denorm32 = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_target_fact_state_t denorm64 = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_target_fact_state_t rte16 = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_target_fact_state_t rte32 = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_target_fact_state_t rte64 = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_target_fact_state_t preserve16 = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_target_fact_state_t preserve32 = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_target_fact_state_t preserve64 = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_target_fact_state_t denorm_independence =
      LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_target_fact_state_t rounding_independence =
      LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_bool_feature(
      import, object, IREE_SV("shaderDenormPreserveFloat16"), &denorm16));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_bool_feature(
      import, object, IREE_SV("shaderDenormPreserveFloat32"), &denorm32));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_bool_feature(
      import, object, IREE_SV("shaderDenormPreserveFloat64"), &denorm64));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_bool_feature(
      import, object, IREE_SV("shaderRoundingModeRTEFloat16"), &rte16));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_bool_feature(
      import, object, IREE_SV("shaderRoundingModeRTEFloat32"), &rte32));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_bool_feature(
      import, object, IREE_SV("shaderRoundingModeRTEFloat64"), &rte64));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_bool_feature(
      import, object, IREE_SV("shaderSignedZeroInfNanPreserveFloat16"),
      &preserve16));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_bool_feature(
      import, object, IREE_SV("shaderSignedZeroInfNanPreserveFloat32"),
      &preserve32));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_bool_feature(
      import, object, IREE_SV("shaderSignedZeroInfNanPreserveFloat64"),
      &preserve64));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_float_control_independence(
      import, object, IREE_SV("denormBehaviorIndependence"),
      &denorm_independence));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_float_control_independence(
      import, object, IREE_SV("roundingModeIndependence"),
      &rounding_independence));

  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_FLOAT32_DENORM_PRESERVE,
      loomc_spirv_vulkaninfo_and_fact_states(denorm32, denorm_independence),
      provenance));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_FLOAT16_64_DENORM_PRESERVE,
      loomc_spirv_vulkaninfo_and_fact_states(
          loomc_spirv_vulkaninfo_and_fact_states(denorm16, denorm64),
          denorm_independence),
      provenance));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_FLOAT16_64_ROUNDING_RTE,
      loomc_spirv_vulkaninfo_and_fact_states(
          loomc_spirv_vulkaninfo_and_fact_states(rte16, rte64),
          rounding_independence),
      provenance));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_FLOAT32_ROUNDING_RTE,
      loomc_spirv_vulkaninfo_and_fact_states(rte32, rounding_independence),
      provenance));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE,
      preserve16, provenance));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE,
      preserve32, provenance));
  return loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE,
      preserve64, provenance);
}

static loomc_status_t loomc_spirv_vulkaninfo_add_feature_fields(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    const loomc_spirv_vulkaninfo_feature_field_t* fields,
    loomc_host_size_t field_count) {
  for (loomc_host_size_t i = 0; i < field_count; ++i) {
    const loomc_spirv_vulkaninfo_feature_field_t* field = &fields[i];
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
        import, object, iree_make_cstring_view(field->field_name),
        field->feature, loomc_make_cstring_view(field->provenance)));
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_parse_uint64(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t value,
    loomc_string_view_t provenance, uint64_t* out_value) {
  *out_value = 0;
  iree_status_t status = iree_json_parse_uint64(value, out_value);
  if (!iree_status_is_ok(status)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(
        import, iree_status_annotate_f(
                    status, "vulkaninfo field '%.*s'", (int)provenance.size,
                    loomc_spirv_vulkaninfo_string_data(provenance)));
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_add_float64_to_float16_conversion(
    loomc_spirv_vulkaninfo_import_t* import,
    iree_string_view_t physical_properties) {
  iree_string_view_t value = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup(
      physical_properties, IREE_SV("vendorID"), &value));
  if (iree_string_view_is_empty(value) ||
      loomc_spirv_vulkaninfo_is_null(value)) {
    return loomc_ok_status();
  }
  uint64_t vendor_id = 0;
  const loomc_string_view_t provenance =
      loomc_make_cstring_view("vulkaninfo:vendorID");
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_parse_uint64(
      import, value, provenance, &vendor_id));
  if (vendor_id > UINT32_MAX) {
    return loomc_spirv_vulkaninfo_fail_message(
        import, "vulkaninfo vendorID must fit in 32 bits");
  }
  return loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_FLOAT64_TO_FLOAT16_CONVERSION,
      vendor_id != LOOMC_SPIRV_VULKANINFO_AMD_VENDOR_ID
          ? LOOMC_TARGET_FACT_STATE_TRUE
          : LOOMC_TARGET_FACT_STATE_FALSE,
      provenance);
}

static loomc_status_t loomc_spirv_vulkaninfo_add_limit(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    iree_string_view_t key, loomc_spirv_limit_t limit,
    loomc_string_view_t provenance) {
  if (!loomc_result_succeeded(import->result) ||
      iree_string_view_is_empty(object)) {
    return loomc_ok_status();
  }
  iree_string_view_t value = iree_string_view_empty();
  iree_status_t status = loomc_spirv_vulkaninfo_try_lookup(object, key, &value);
  if (!iree_status_is_ok(status)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(import, status);
  }
  if (iree_string_view_is_empty(value) ||
      loomc_spirv_vulkaninfo_is_null(value)) {
    return loomc_ok_status();
  }
  uint64_t parsed_value = 0;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_parse_uint64(
      import, value, provenance, &parsed_value));
  return loomc_spirv_vulkaninfo_add_limit_fact(import, limit, parsed_value,
                                               provenance);
}

static loomc_status_t loomc_spirv_vulkaninfo_add_limit_array_element(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    iree_string_view_t key, iree_host_size_t index, loomc_spirv_limit_t limit,
    loomc_string_view_t provenance) {
  if (!loomc_result_succeeded(import->result) ||
      iree_string_view_is_empty(object)) {
    return loomc_ok_status();
  }
  iree_string_view_t array = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_vulkaninfo_try_lookup_array(import, object, key, &array));
  if (iree_string_view_is_empty(array)) {
    return loomc_ok_status();
  }
  iree_string_view_t value = iree_string_view_empty();
  iree_status_t status = iree_json_array_get(array, index, &value);
  if (!iree_status_is_ok(status)) {
    const iree_status_code_t status_code = iree_status_code(status);
    iree_status_free(status);
    return loomc_spirv_vulkaninfo_fail_iree_status(
        import,
        iree_make_status(status_code,
                         "vulkaninfo array '%.*s' missing element %" PRIhsz,
                         (int)key.size, key.data, index));
  }
  if (loomc_spirv_vulkaninfo_is_null(value)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(
        import, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                 "vulkaninfo array '%.*s' element %" PRIhsz
                                 " must be numeric",
                                 (int)key.size, key.data, index));
  }
  uint64_t parsed_value = 0;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_parse_uint64(
      import, value, provenance, &parsed_value));
  return loomc_spirv_vulkaninfo_add_limit_fact(import, limit, parsed_value,
                                               provenance);
}

static bool loomc_spirv_vulkaninfo_parse_decimal_component(
    iree_string_view_t value, iree_host_size_t* inout_offset,
    uint32_t* out_value) {
  uint64_t result = 0;
  iree_host_size_t offset = *inout_offset;
  if (offset >= value.size || value.data[offset] < '0' ||
      value.data[offset] > '9') {
    return false;
  }
  while (offset < value.size && value.data[offset] >= '0' &&
         value.data[offset] <= '9') {
    result = result * 10u + (uint32_t)(value.data[offset] - '0');
    if (result > UINT32_MAX) {
      return false;
    }
    ++offset;
  }
  *inout_offset = offset;
  *out_value = (uint32_t)result;
  return true;
}

static bool loomc_spirv_vulkaninfo_parse_dotted_version(
    iree_string_view_t value, uint32_t* out_major, uint32_t* out_minor) {
  iree_host_size_t offset = 0;
  uint32_t major = 0;
  uint32_t minor = 0;
  uint32_t patch = 0;
  if (!loomc_spirv_vulkaninfo_parse_decimal_component(value, &offset, &major)) {
    return false;
  }
  if (offset >= value.size || value.data[offset++] != '.') {
    return false;
  }
  if (!loomc_spirv_vulkaninfo_parse_decimal_component(value, &offset, &minor)) {
    return false;
  }
  if (offset < value.size) {
    if (value.data[offset++] != '.') {
      return false;
    }
    if (!loomc_spirv_vulkaninfo_parse_decimal_component(value, &offset,
                                                        &patch)) {
      return false;
    }
  }
  if (offset != value.size) {
    return false;
  }
  (void)patch;
  *out_major = major;
  *out_minor = minor;
  return true;
}

static loomc_status_t loomc_spirv_vulkaninfo_add_api_version(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    iree_string_view_t key, loomc_string_view_t provenance) {
  if (!loomc_result_succeeded(import->result) ||
      iree_string_view_is_empty(object)) {
    return loomc_ok_status();
  }
  iree_string_view_t value = iree_string_view_empty();
  iree_status_t status = loomc_spirv_vulkaninfo_try_lookup(object, key, &value);
  if (!iree_status_is_ok(status)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(import, status);
  }
  if (iree_string_view_is_empty(value) ||
      loomc_spirv_vulkaninfo_is_null(value)) {
    return loomc_ok_status();
  }

  uint32_t vulkan_major = 0;
  uint32_t vulkan_minor = 0;
  if (loomc_spirv_vulkaninfo_contains_char(value, '.')) {
    if (!loomc_spirv_vulkaninfo_parse_dotted_version(value, &vulkan_major,
                                                     &vulkan_minor)) {
      return loomc_spirv_vulkaninfo_fail_iree_status(
          import, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                   "vulkaninfo apiVersion '%.*s' is malformed",
                                   (int)value.size, value.data));
    }
  } else {
    uint64_t encoded_version = 0;
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_parse_uint64(
        import, value, provenance, &encoded_version));
    vulkan_major = (uint32_t)((encoded_version >> 22) & 0x7Fu);
    vulkan_minor = (uint32_t)((encoded_version >> 12) & 0x3FFu);
  }

  const uint32_t max_spirv_version =
      loomc_spirv_max_version_from_vulkan_api_version(vulkan_major,
                                                      vulkan_minor);
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_environment_fact(
      import, LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION, max_spirv_version,
      provenance));
  if (vulkan_major > 1 || (vulkan_major == 1 && vulkan_minor >= 1)) {
    return loomc_spirv_vulkaninfo_add_feature_fact(
        import, LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM,
        LOOMC_TARGET_FACT_STATE_TRUE, provenance);
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_select_profile_capability(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t root,
    loomc_string_view_t profile_name, iree_string_view_t* out_device) {
  *out_device = iree_string_view_empty();
  iree_string_view_t capabilities = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, root, IREE_SV("capabilities"), &capabilities));
  if (iree_string_view_is_empty(capabilities)) {
    return loomc_ok_status();
  }

  iree_string_view_t capability_name = IREE_SV("device");
  iree_string_view_t profiles = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, root, IREE_SV("profiles"), &profiles));
  if (!iree_string_view_is_empty(profiles)) {
    iree_string_view_t profile = iree_string_view_empty();
    if (!loomc_string_view_is_empty(profile_name)) {
      iree_status_t status = iree_json_try_lookup_object_value(
          profiles, iree_string_view_from_loomc(profile_name), &profile);
      if (!iree_status_is_ok(status)) {
        return loomc_spirv_vulkaninfo_fail_iree_status(import, status);
      }
      if (iree_string_view_is_empty(profile)) {
        return loomc_spirv_vulkaninfo_fail_iree_status(
            import,
            iree_make_status(IREE_STATUS_NOT_FOUND,
                             "vulkaninfo profile '%.*s' was not found",
                             (int)profile_name.size,
                             loomc_spirv_vulkaninfo_string_data(profile_name)));
      }
    } else {
      loomc_spirv_vulkaninfo_first_member_state_t first_profile = {0};
      LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_first_member(
          import, profiles, &first_profile));
      profile = first_profile.value;
    }
    if (!iree_string_view_is_empty(profile)) {
      if (!loomc_spirv_vulkaninfo_is_object(profile)) {
        return loomc_spirv_vulkaninfo_fail_message(
            import, "vulkaninfo profile entry must be an object");
      }
      iree_string_view_t profile_capabilities = iree_string_view_empty();
      LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_array(
          import, profile, IREE_SV("capabilities"), &profile_capabilities));
      if (!iree_string_view_is_empty(profile_capabilities)) {
        iree_status_t status =
            iree_json_array_get(profile_capabilities, 0, &capability_name);
        if (!iree_status_is_ok(status)) {
          const iree_status_code_t status_code = iree_status_code(status);
          iree_status_free(status);
          return loomc_spirv_vulkaninfo_fail_iree_status(
              import, iree_make_status(
                          status_code,
                          "vulkaninfo profile capabilities array is empty"));
        }
      }
    }
  }

  iree_status_t status = iree_json_try_lookup_object_value(
      capabilities, capability_name, out_device);
  if (!iree_status_is_ok(status)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(import, status);
  }
  if (iree_string_view_is_empty(*out_device)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(
        import,
        iree_make_status(IREE_STATUS_NOT_FOUND,
                         "vulkaninfo capability '%.*s' was not found",
                         (int)capability_name.size, capability_name.data));
  }
  if (!loomc_spirv_vulkaninfo_is_object(*out_device)) {
    return loomc_spirv_vulkaninfo_fail_message(
        import, "vulkaninfo selected capability must be an object");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_select_raw_device(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t root,
    uint32_t device_index, iree_string_view_t* out_device) {
  *out_device = iree_string_view_empty();
  iree_string_view_t devices = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_array(
      import, root, IREE_SV("devices"), &devices));
  if (iree_string_view_is_empty(devices)) {
    *out_device = root;
    return loomc_ok_status();
  }
  iree_status_t status = iree_json_array_get(devices, device_index, out_device);
  if (!iree_status_is_ok(status)) {
    const iree_status_code_t status_code = iree_status_code(status);
    iree_status_free(status);
    return loomc_spirv_vulkaninfo_fail_iree_status(
        import,
        iree_make_status(status_code,
                         "vulkaninfo devices array missing index %" PRIu32,
                         device_index));
  }
  if (!loomc_spirv_vulkaninfo_is_object(*out_device)) {
    return loomc_spirv_vulkaninfo_fail_message(
        import, "vulkaninfo selected device must be an object");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_select_device(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t root,
    const loomc_spirv_vulkaninfo_profile_options_t* options,
    iree_string_view_t* out_device) {
  *out_device = iree_string_view_empty();
  const loomc_string_view_t profile_name =
      options ? options->profile_name : loomc_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_select_profile_capability(
      import, root, profile_name, out_device));
  if (!loomc_result_succeeded(import->result) ||
      !iree_string_view_is_empty(*out_device)) {
    return loomc_ok_status();
  }
  if (!loomc_string_view_is_empty(profile_name)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(
        import,
        iree_make_status(IREE_STATUS_NOT_FOUND,
                         "vulkaninfo profile '%.*s' was not found",
                         (int)profile_name.size,
                         loomc_spirv_vulkaninfo_string_data(profile_name)));
  }
  const uint32_t device_index = options ? options->device_index : 0;
  return loomc_spirv_vulkaninfo_select_raw_device(import, root, device_index,
                                                  out_device);
}

static loomc_status_t loomc_spirv_vulkaninfo_lookup_feature_struct(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t features,
    iree_string_view_t extended_features, iree_string_view_t struct_name,
    iree_string_view_t* out_struct) {
  *out_struct = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, features, struct_name, out_struct));
  if (!iree_string_view_is_empty(*out_struct) ||
      iree_string_view_is_empty(extended_features)) {
    return loomc_ok_status();
  }
  return loomc_spirv_vulkaninfo_try_lookup_object(import, extended_features,
                                                  struct_name, out_struct);
}

static loomc_status_t loomc_spirv_vulkaninfo_import_limits(
    loomc_spirv_vulkaninfo_import_t* import,
    iree_string_view_t physical_properties) {
  iree_string_view_t limits = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, physical_properties, IREE_SV("limits"), &limits));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit_array_element(
      import, limits, IREE_SV("maxComputeWorkGroupSize"), 0,
      LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
      loomc_make_cstring_view("vulkaninfo:maxComputeWorkGroupSize[0]")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit_array_element(
      import, limits, IREE_SV("maxComputeWorkGroupSize"), 1,
      LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y,
      loomc_make_cstring_view("vulkaninfo:maxComputeWorkGroupSize[1]")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit_array_element(
      import, limits, IREE_SV("maxComputeWorkGroupSize"), 2,
      LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z,
      loomc_make_cstring_view("vulkaninfo:maxComputeWorkGroupSize[2]")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit(
      import, limits, IREE_SV("maxComputeWorkGroupInvocations"),
      LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE,
      loomc_make_cstring_view("vulkaninfo:maxComputeWorkGroupInvocations")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit(
      import, limits, IREE_SV("maxComputeSharedMemorySize"),
      LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
      loomc_make_cstring_view("vulkaninfo:maxComputeSharedMemorySize")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit_array_element(
      import, limits, IREE_SV("maxComputeWorkGroupCount"), 0,
      LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X,
      loomc_make_cstring_view("vulkaninfo:maxComputeWorkGroupCount[0]")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit_array_element(
      import, limits, IREE_SV("maxComputeWorkGroupCount"), 1,
      LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Y,
      loomc_make_cstring_view("vulkaninfo:maxComputeWorkGroupCount[1]")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit_array_element(
      import, limits, IREE_SV("maxComputeWorkGroupCount"), 2,
      LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Z,
      loomc_make_cstring_view("vulkaninfo:maxComputeWorkGroupCount[2]")));
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_import_subgroup_limits(
    loomc_spirv_vulkaninfo_import_t* import,
    iree_string_view_t subgroup_properties, iree_string_view_t properties,
    iree_string_view_t features, iree_string_view_t extended_features) {
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit(
      import, subgroup_properties, IREE_SV("subgroupSize"),
      LOOMC_SPIRV_LIMIT_DEFAULT_SUBGROUP_SIZE,
      loomc_make_cstring_view("vulkaninfo:subgroupSize")));

  iree_string_view_t control_properties = iree_string_view_empty();
  const char* property_structs[] = {
      "VkPhysicalDeviceVulkan13Properties",
      "VkPhysicalDeviceSubgroupSizeControlProperties",
      "VkPhysicalDeviceSubgroupSizeControlPropertiesEXT",
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(property_structs) &&
                               iree_string_view_is_empty(control_properties);
       ++i) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
        import, properties, iree_make_cstring_view(property_structs[i]),
        &control_properties));
  }
  iree_string_view_t control_features = iree_string_view_empty();
  const char* feature_structs[] = {
      "VkPhysicalDeviceVulkan13Features",
      "VkPhysicalDeviceSubgroupSizeControlFeatures",
      "VkPhysicalDeviceSubgroupSizeControlFeaturesEXT",
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(feature_structs) &&
                               iree_string_view_is_empty(control_features);
       ++i) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
        import, features, extended_features,
        iree_make_cstring_view(feature_structs[i]), &control_features));
  }
  loomc_target_fact_state_t control = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_bool_feature(
      import, control_features, IREE_SV("subgroupSizeControl"), &control));
  loomc_target_fact_state_t compute = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_read_flag_feature(
      import, control_properties, IREE_SV("requiredSubgroupSizeStages"),
      IREE_SV("VK_SHADER_STAGE_COMPUTE_BIT"), &compute));
  const loomc_target_fact_state_t compute_control =
      loomc_spirv_vulkaninfo_and_fact_states(control, compute);
  if (compute_control == LOOMC_TARGET_FACT_STATE_TRUE) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit(
        import, control_properties, IREE_SV("minSubgroupSize"),
        LOOMC_SPIRV_LIMIT_MIN_SUBGROUP_SIZE,
        loomc_make_cstring_view("vulkaninfo:minSubgroupSize")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit(
        import, control_properties, IREE_SV("maxSubgroupSize"),
        LOOMC_SPIRV_LIMIT_MAX_SUBGROUP_SIZE,
        loomc_make_cstring_view("vulkaninfo:maxSubgroupSize")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit(
        import, control_properties, IREE_SV("maxComputeWorkgroupSubgroups"),
        LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SUBGROUP_COUNT,
        loomc_make_cstring_view("vulkaninfo:maxComputeWorkgroupSubgroups")));
  } else if (compute_control == LOOMC_TARGET_FACT_STATE_FALSE) {
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit(
        import, subgroup_properties, IREE_SV("subgroupSize"),
        LOOMC_SPIRV_LIMIT_MIN_SUBGROUP_SIZE,
        loomc_make_cstring_view("vulkaninfo:subgroupSize")));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_limit(
        import, subgroup_properties, IREE_SV("subgroupSize"),
        LOOMC_SPIRV_LIMIT_MAX_SUBGROUP_SIZE,
        loomc_make_cstring_view("vulkaninfo:subgroupSize")));
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_vulkaninfo_import_feature_properties(
    loomc_spirv_vulkaninfo_import_t* import,
    iree_string_view_t vulkan11_properties,
    iree_string_view_t subgroup_properties,
    iree_string_view_t vulkan12_properties,
    iree_string_view_t float_controls_properties) {
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_subgroup_features(
      import, vulkan11_properties, IREE_SV("subgroupSupportedOperations"),
      loomc_make_cstring_view("vulkaninfo:subgroupSupportedOperations")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_subgroup_features(
      import, subgroup_properties, IREE_SV("supportedOperations"),
      loomc_make_cstring_view("vulkaninfo:supportedOperations")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_float_control_features(
      import, vulkan12_properties,
      loomc_make_cstring_view(
          "vulkaninfo:VkPhysicalDeviceVulkan12Properties.floatControls")));
  return loomc_spirv_vulkaninfo_add_float_control_features(
      import, float_controls_properties,
      loomc_make_cstring_view(
          "vulkaninfo:VkPhysicalDeviceFloatControlsProperties.floatControls"));
}

static const loomc_spirv_vulkaninfo_feature_field_t
    kLoomcSpirvVulkaninfoAtomicInt64Fields[] = {
        {
            .field_name = "shaderBufferInt64Atomics",
            .feature = LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_INT64_ATOMICS,
            .provenance = "vulkaninfo:shaderBufferInt64Atomics",
        },
        {
            .field_name = "shaderSharedInt64Atomics",
            .feature = LOOMC_SPIRV_FEATURE_WORKGROUP_INT64_ATOMICS,
            .provenance = "vulkaninfo:shaderSharedInt64Atomics",
        },
};

static const loomc_spirv_vulkaninfo_feature_field_t
    kLoomcSpirvVulkaninfoAtomicFloatFields[] = {
        {
            .field_name = "shaderBufferFloat32Atomics",
            .feature = LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMICS,
            .provenance = "vulkaninfo:shaderBufferFloat32Atomics",
        },
        {
            .field_name = "shaderSharedFloat32Atomics",
            .feature = LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMICS,
            .provenance = "vulkaninfo:shaderSharedFloat32Atomics",
        },
        {
            .field_name = "shaderBufferFloat32AtomicAdd",
            .feature = LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD,
            .provenance = "vulkaninfo:shaderBufferFloat32AtomicAdd",
        },
        {
            .field_name = "shaderSharedFloat32AtomicAdd",
            .feature = LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMIC_ADD,
            .provenance = "vulkaninfo:shaderSharedFloat32AtomicAdd",
        },
        {
            .field_name = "shaderBufferFloat64Atomics",
            .feature = LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMICS,
            .provenance = "vulkaninfo:shaderBufferFloat64Atomics",
        },
        {
            .field_name = "shaderSharedFloat64Atomics",
            .feature = LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMICS,
            .provenance = "vulkaninfo:shaderSharedFloat64Atomics",
        },
        {
            .field_name = "shaderBufferFloat64AtomicAdd",
            .feature = LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD,
            .provenance = "vulkaninfo:shaderBufferFloat64AtomicAdd",
        },
        {
            .field_name = "shaderSharedFloat64AtomicAdd",
            .feature = LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMIC_ADD,
            .provenance = "vulkaninfo:shaderSharedFloat64AtomicAdd",
        },
};

static const loomc_spirv_vulkaninfo_feature_field_t
    kLoomcSpirvVulkaninfoAtomicFloat2Fields[] = {
        {
            .field_name = "shaderBufferFloat16Atomics",
            .feature = LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMICS,
            .provenance = "vulkaninfo:shaderBufferFloat16Atomics",
        },
        {
            .field_name = "shaderSharedFloat16Atomics",
            .feature = LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMICS,
            .provenance = "vulkaninfo:shaderSharedFloat16Atomics",
        },
        {
            .field_name = "shaderBufferFloat16AtomicAdd",
            .feature = LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD,
            .provenance = "vulkaninfo:shaderBufferFloat16AtomicAdd",
        },
        {
            .field_name = "shaderSharedFloat16AtomicAdd",
            .feature = LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMIC_ADD,
            .provenance = "vulkaninfo:shaderSharedFloat16AtomicAdd",
        },
};

static loomc_status_t loomc_spirv_vulkaninfo_import_features_from_structs(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t features,
    iree_string_view_t extended_features) {
  iree_string_view_t core_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
      import, features, extended_features, IREE_SV("VkPhysicalDeviceFeatures"),
      &core_features));
  if (iree_string_view_is_empty(core_features)) {
    core_features = features;
  }
  iree_string_view_t vulkan12_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
      import, features, extended_features,
      IREE_SV("VkPhysicalDeviceVulkan12Features"), &vulkan12_features));
  iree_string_view_t float16_int8_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
      import, features, extended_features,
      IREE_SV("VkPhysicalDeviceShaderFloat16Int8Features"),
      &float16_int8_features));
  iree_string_view_t subgroup_type_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
      import, features, extended_features,
      IREE_SV("VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures"),
      &subgroup_type_features));
  iree_string_view_t storage8_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
      import, features, extended_features,
      IREE_SV("VkPhysicalDevice8BitStorageFeatures"), &storage8_features));
  iree_string_view_t storage16_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
      import, features, extended_features,
      IREE_SV("VkPhysicalDevice16BitStorageFeatures"), &storage16_features));
  iree_string_view_t bda_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
      import, features, extended_features,
      IREE_SV("VkPhysicalDeviceBufferDeviceAddressFeatures"), &bda_features));
  iree_string_view_t cooperative_matrix_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
      import, features, extended_features,
      IREE_SV("VkPhysicalDeviceCooperativeMatrixFeaturesKHR"),
      &cooperative_matrix_features));
  iree_string_view_t cooperative_vector_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
      import, features, extended_features,
      IREE_SV("VkPhysicalDeviceCooperativeVectorFeaturesNV"),
      &cooperative_vector_features));
  iree_string_view_t atomic_int64_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
      import, features, extended_features,
      IREE_SV("VkPhysicalDeviceShaderAtomicInt64Features"),
      &atomic_int64_features));
  iree_string_view_t memory_model_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
      import, features, extended_features,
      IREE_SV("VkPhysicalDeviceVulkanMemoryModelFeatures"),
      &memory_model_features));
  iree_string_view_t atomic_float_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
      import, features, extended_features,
      IREE_SV("VkPhysicalDeviceShaderAtomicFloatFeaturesEXT"),
      &atomic_float_features));
  iree_string_view_t atomic_float2_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_lookup_feature_struct(
      import, features, extended_features,
      IREE_SV("VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT"),
      &atomic_float2_features));

  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, core_features, IREE_SV("shaderFloat64"),
      LOOMC_SPIRV_FEATURE_FLOAT64,
      loomc_make_cstring_view("vulkaninfo:shaderFloat64")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, core_features, IREE_SV("shaderInt16"), LOOMC_SPIRV_FEATURE_INT16,
      loomc_make_cstring_view("vulkaninfo:shaderInt16")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, core_features, IREE_SV("shaderInt64"), LOOMC_SPIRV_FEATURE_INT64,
      loomc_make_cstring_view("vulkaninfo:shaderInt64")));

  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, vulkan12_features, IREE_SV("shaderSubgroupExtendedTypes"),
      LOOMC_SPIRV_FEATURE_SUBGROUP_EXTENDED_TYPES,
      loomc_make_cstring_view("vulkaninfo:shaderSubgroupExtendedTypes")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, subgroup_type_features, IREE_SV("shaderSubgroupExtendedTypes"),
      LOOMC_SPIRV_FEATURE_SUBGROUP_EXTENDED_TYPES,
      loomc_make_cstring_view("vulkaninfo:shaderSubgroupExtendedTypes")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, vulkan12_features, IREE_SV("shaderFloat16"),
      LOOMC_SPIRV_FEATURE_FLOAT16,
      loomc_make_cstring_view("vulkaninfo:shaderFloat16")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, float16_int8_features, IREE_SV("shaderFloat16"),
      LOOMC_SPIRV_FEATURE_FLOAT16,
      loomc_make_cstring_view("vulkaninfo:shaderFloat16")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, vulkan12_features, IREE_SV("shaderInt8"),
      LOOMC_SPIRV_FEATURE_INT8,
      loomc_make_cstring_view("vulkaninfo:shaderInt8")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, float16_int8_features, IREE_SV("shaderInt8"),
      LOOMC_SPIRV_FEATURE_INT8,
      loomc_make_cstring_view("vulkaninfo:shaderInt8")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, vulkan12_features, IREE_SV("storageBuffer8BitAccess"),
      LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS,
      loomc_make_cstring_view("vulkaninfo:storageBuffer8BitAccess")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, storage8_features, IREE_SV("storageBuffer8BitAccess"),
      LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS,
      loomc_make_cstring_view("vulkaninfo:storageBuffer8BitAccess")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, storage16_features, IREE_SV("storageBuffer16BitAccess"),
      LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS,
      loomc_make_cstring_view("vulkaninfo:storageBuffer16BitAccess")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, vulkan12_features, IREE_SV("bufferDeviceAddress"),
      LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
      loomc_make_cstring_view("vulkaninfo:bufferDeviceAddress")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, bda_features, IREE_SV("bufferDeviceAddress"),
      LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
      loomc_make_cstring_view("vulkaninfo:bufferDeviceAddress")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, cooperative_matrix_features, IREE_SV("cooperativeMatrix"),
      LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
      loomc_make_cstring_view("vulkaninfo:cooperativeMatrix")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, cooperative_vector_features, IREE_SV("cooperativeVector"),
      LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
      loomc_make_cstring_view("vulkaninfo:cooperativeVector")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_feature(
      import, cooperative_vector_features, IREE_SV("cooperativeVectorTraining"),
      LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV,
      loomc_make_cstring_view("vulkaninfo:cooperativeVectorTraining")));

  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fields(
      import, vulkan12_features, kLoomcSpirvVulkaninfoAtomicInt64Fields,
      sizeof(kLoomcSpirvVulkaninfoAtomicInt64Fields) /
          sizeof(kLoomcSpirvVulkaninfoAtomicInt64Fields[0])));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fields(
      import, atomic_int64_features, kLoomcSpirvVulkaninfoAtomicInt64Fields,
      sizeof(kLoomcSpirvVulkaninfoAtomicInt64Fields) /
          sizeof(kLoomcSpirvVulkaninfoAtomicInt64Fields[0])));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_pair_feature(
      import, vulkan12_features, IREE_SV("vulkanMemoryModel"),
      IREE_SV("vulkanMemoryModelDeviceScope"),
      LOOMC_SPIRV_FEATURE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
      loomc_make_cstring_view(
          "vulkaninfo:vulkanMemoryModel+vulkanMemoryModelDeviceScope")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_bool_pair_feature(
      import, memory_model_features, IREE_SV("vulkanMemoryModel"),
      IREE_SV("vulkanMemoryModelDeviceScope"),
      LOOMC_SPIRV_FEATURE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
      loomc_make_cstring_view(
          "vulkaninfo:vulkanMemoryModel+vulkanMemoryModelDeviceScope")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fields(
      import, atomic_float_features, kLoomcSpirvVulkaninfoAtomicFloatFields,
      sizeof(kLoomcSpirvVulkaninfoAtomicFloatFields) /
          sizeof(kLoomcSpirvVulkaninfoAtomicFloatFields[0])));
  return loomc_spirv_vulkaninfo_add_feature_fields(
      import, atomic_float2_features, kLoomcSpirvVulkaninfoAtomicFloat2Fields,
      sizeof(kLoomcSpirvVulkaninfoAtomicFloat2Fields) /
          sizeof(kLoomcSpirvVulkaninfoAtomicFloat2Fields[0]));
}

static loomc_status_t loomc_spirv_vulkaninfo_import_device(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t device) {
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_feature_fact(
      import, LOOMC_SPIRV_FEATURE_VULKAN_SHADER, LOOMC_TARGET_FACT_STATE_TRUE,
      loomc_make_cstring_view("vulkaninfo:device")));

  iree_string_view_t properties = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, device, IREE_SV("properties"), &properties));
  if (iree_string_view_is_empty(properties)) {
    properties = device;
  }
  iree_string_view_t physical_properties = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, properties, IREE_SV("VkPhysicalDeviceProperties"),
      &physical_properties));
  if (iree_string_view_is_empty(physical_properties)) {
    physical_properties = properties;
  }
  iree_string_view_t vulkan11_properties = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, properties, IREE_SV("VkPhysicalDeviceVulkan11Properties"),
      &vulkan11_properties));
  iree_string_view_t subgroup_properties = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, properties, IREE_SV("VkPhysicalDeviceSubgroupProperties"),
      &subgroup_properties));
  iree_string_view_t vulkan12_properties = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, properties, IREE_SV("VkPhysicalDeviceVulkan12Properties"),
      &vulkan12_properties));
  iree_string_view_t float_controls_properties = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, properties, IREE_SV("VkPhysicalDeviceFloatControlsProperties"),
      &float_controls_properties));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_api_version(
      import, physical_properties, IREE_SV("apiVersion"),
      loomc_make_cstring_view("vulkaninfo:apiVersion")));
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_vulkaninfo_add_float64_to_float16_conversion(
          import, physical_properties));
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_vulkaninfo_import_limits(import, physical_properties));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_import_feature_properties(
      import, vulkan11_properties, subgroup_properties, vulkan12_properties,
      float_controls_properties));

  iree_string_view_t features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, device, IREE_SV("features"), &features));
  iree_string_view_t extended_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, device, IREE_SV("extended_features"), &extended_features));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_import_subgroup_limits(
      import,
      !iree_string_view_is_empty(vulkan11_properties) ? vulkan11_properties
                                                      : subgroup_properties,
      properties, features, extended_features));
  return loomc_spirv_vulkaninfo_import_features_from_structs(import, features,
                                                             extended_features);
}

static loomc_string_view_t loomc_spirv_vulkaninfo_profile_identifier(
    const loomc_spirv_vulkaninfo_profile_options_t* options,
    const loomc_source_t* source) {
  if (options != NULL && !loomc_string_view_is_empty(options->identifier)) {
    return options->identifier;
  }
  loomc_string_view_t source_identifier = loomc_source_identifier(source);
  if (!loomc_string_view_is_empty(source_identifier)) {
    return source_identifier;
  }
  return loomc_make_cstring_view("<vulkaninfo-profile>");
}

static loomc_status_t loomc_spirv_vulkaninfo_parse_root(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t contents,
    iree_string_view_t* out_root) {
  *out_root = iree_string_view_empty();
  iree_string_view_t cursor = contents;
  iree_status_t status = iree_json_consume_object(&cursor, out_root);
  if (iree_status_is_ok(status)) {
    status = iree_json_consume_insignificant(&cursor);
  }
  if (iree_status_is_ok(status) && !iree_string_view_is_empty(cursor)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unexpected trailing content after vulkaninfo "
                              "JSON object");
  }
  if (!iree_status_is_ok(status)) {
    return loomc_spirv_vulkaninfo_fail_iree_status(import, status);
  }
  return loomc_ok_status();
}

loomc_status_t loomc_target_profile_create_spirv_vulkaninfo(
    loomc_target_environment_t* target_environment,
    const loomc_source_t* source,
    const loomc_spirv_vulkaninfo_profile_options_t* options,
    loomc_allocator_t allocator, loomc_target_profile_t** out_profile,
    loomc_result_t** out_result) {
  if (out_profile != NULL) {
    *out_profile = NULL;
  }
  if (out_result != NULL) {
    *out_result = NULL;
  }
  if (target_environment == NULL || source == NULL || out_profile == NULL ||
      out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target_environment, source, out_profile, and out_result must not be "
        "NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_validate_options(options));

  const loomc_byte_span_t contents = loomc_source_contents(source);
  if (contents.data == NULL && contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source contents have length but no data");
  }

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_create(
      LOOMC_STATUS_OK, LOOMC_SOURCE_RETENTION_EXACT, allocator, &result));
  loomc_spirv_vulkaninfo_import_t import = {
      .source = source,
      .result = result,
  };

  iree_string_view_t root = iree_string_view_empty();
  loomc_status_t status = loomc_spirv_vulkaninfo_parse_root(
      &import,
      iree_make_string_view((const char*)contents.data, contents.data_length),
      &root);
  iree_string_view_t device = iree_string_view_empty();
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status =
        loomc_spirv_vulkaninfo_select_device(&import, root, options, &device);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_spirv_vulkaninfo_import_device(&import, device);
  }

  if (!loomc_status_is_ok(status)) {
    loomc_result_release(result);
    return status;
  }
  if (!loomc_result_succeeded(result)) {
    *out_result = result;
    return loomc_ok_status();
  }
  loomc_result_release(result);

  const loomc_spirv_profile_options_t profile_options = {
      .type = LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      .structure_size = sizeof(profile_options),
      .identifier = loomc_spirv_vulkaninfo_profile_identifier(options, source),
      .preset = LOOMC_SPIRV_PROFILE_PRESET_NONE,
      .feature_facts = import.feature_facts,
      .feature_fact_count = import.feature_fact_count,
      .limit_facts = import.limit_facts,
      .limit_fact_count = import.limit_fact_count,
      .environment_facts = import.environment_facts,
      .environment_fact_count = import.environment_fact_count,
  };
  return loomc_target_profile_create_spirv(target_environment, &profile_options,
                                           allocator, out_profile, out_result);
}
