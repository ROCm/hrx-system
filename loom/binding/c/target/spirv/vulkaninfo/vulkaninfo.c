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
  LOOMC_SPIRV_VULKANINFO_FEATURE_FACT_CAPACITY = 32,
  LOOMC_SPIRV_VULKANINFO_LIMIT_FACT_CAPACITY = 16,
  LOOMC_SPIRV_VULKANINFO_ENVIRONMENT_FACT_CAPACITY = 4,
};

typedef struct loomc_spirv_vulkaninfo_import_t {
  // Source whose JSON bytes are being imported.
  const loomc_source_t* source;

  // Result used for importer diagnostics.
  loomc_result_t* result;

  // Feature facts normalized from recognized Vulkan fields.
  loomc_spirv_feature_fact_t
      feature_facts[LOOMC_SPIRV_VULKANINFO_FEATURE_FACT_CAPACITY];

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
  if (import->feature_fact_count >=
      LOOMC_SPIRV_VULKANINFO_FEATURE_FACT_CAPACITY) {
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

static loomc_status_t loomc_spirv_vulkaninfo_add_bool_feature(
    loomc_spirv_vulkaninfo_import_t* import, iree_string_view_t object,
    iree_string_view_t key, loomc_spirv_feature_t feature,
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
  if (iree_string_view_equal(value, IREE_SV("true"))) {
    return loomc_spirv_vulkaninfo_add_feature_fact(
        import, feature, LOOMC_TARGET_FACT_STATE_TRUE, provenance);
  }
  if (iree_string_view_equal(value, IREE_SV("false"))) {
    return loomc_spirv_vulkaninfo_add_feature_fact(
        import, feature, LOOMC_TARGET_FACT_STATE_FALSE, provenance);
  }
  return loomc_spirv_vulkaninfo_fail_iree_status(
      import,
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                       "vulkaninfo feature '%.*s' must be boolean or null",
                       (int)key.size, key.data));
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
  return loomc_spirv_vulkaninfo_add_environment_fact(
      import, LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION, max_spirv_version,
      provenance);
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
    iree_string_view_t physical_properties,
    iree_string_view_t vulkan11_properties) {
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
  return loomc_spirv_vulkaninfo_add_limit(
      import, vulkan11_properties, IREE_SV("subgroupSize"),
      LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
      loomc_make_cstring_view("vulkaninfo:subgroupSize"));
}

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
  return loomc_spirv_vulkaninfo_add_bool_feature(
      import, cooperative_vector_features, IREE_SV("cooperativeVectorTraining"),
      LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV,
      loomc_make_cstring_view("vulkaninfo:cooperativeVectorTraining"));
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
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_add_api_version(
      import, physical_properties, IREE_SV("apiVersion"),
      loomc_make_cstring_view("vulkaninfo:apiVersion")));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_import_limits(
      import, physical_properties, vulkan11_properties));

  iree_string_view_t features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, device, IREE_SV("features"), &features));
  iree_string_view_t extended_features = iree_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_spirv_vulkaninfo_try_lookup_object(
      import, device, IREE_SV("extended_features"), &extended_features));
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
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));
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
