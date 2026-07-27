// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/spirv/profile.h"

#include <inttypes.h>
#include <string.h>

#include "diagnostic.h"
#include "loom/target/arch/spirv/cooperative_properties.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/profile.h"
#include "loom/target/arch/spirv/records/target_records.h"
#include "loomc/iree.h"
#include "profile_rows.h"
#include "result.h"
#include "target.h"

typedef struct loomc_spirv_feature_state_t {
  // Current observed state.
  loomc_target_fact_state_t state;

  // Provenance for current observed state.
  loomc_string_view_t provenance;
} loomc_spirv_feature_state_t;

typedef struct loomc_spirv_numeric_fact_state_t {
  // Current observed state and value.
  loomc_target_fact_state_t state;

  // Current observed value when state is true.
  uint64_t value;

  // Provenance for current observed state.
  loomc_string_view_t provenance;
} loomc_spirv_numeric_fact_state_t;

typedef struct loomc_spirv_target_profile_payload_t {
  // Opaque SPIR-V profile passed through Loom target selections.
  loom_spirv_target_profile_t profile;

  // Prepared feature set derived from known-true feature facts.
  loom_spirv_feature_set_t feature_set;

  // Cooperative matrix/vector row facts and prepared property views.
  loomc_spirv_cooperative_row_fact_set_t cooperative_row_facts;

  // Materialized compiler-facing target bundle, when the profile facts are
  // concrete enough to refine source-selected target records.
  loom_target_bundle_storage_t bundle_storage;

  // Public tri-state feature facts with owned provenance strings.
  loomc_spirv_feature_state_t feature_states[LOOMC_SPIRV_FEATURE_COUNT];

  // Public tri-state numeric limit facts with owned provenance strings.
  loomc_spirv_numeric_fact_state_t limit_states[LOOMC_SPIRV_LIMIT_COUNT];

  // Public tri-state numeric environment facts with owned provenance strings.
  loomc_spirv_numeric_fact_state_t
      environment_states[LOOMC_SPIRV_ENVIRONMENT_COUNT];

} loomc_spirv_target_profile_payload_t;

static const char kLoomcSpirvTargetProfilePayloadType = 0;

static loomc_status_t loomc_spirv_profile_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static const char* loomc_spirv_profile_string_data(loomc_string_view_t value) {
  return value.data != NULL ? value.data : "";
}

static loomc_string_view_t loomc_spirv_profile_identifier(
    const loomc_spirv_profile_options_t* options) {
  if (options == NULL || loomc_string_view_is_empty(options->identifier)) {
    return loomc_make_cstring_view("<spirv-profile>");
  }
  return options->identifier;
}

static loom_spirv_feature_atom_t loomc_spirv_feature_atom_from_public(
    loomc_spirv_feature_t feature) {
  switch (feature) {
    case LOOMC_SPIRV_FEATURE_VULKAN_SHADER:
      return LOOM_SPIRV_FEATURE_ATOM_VULKAN_SHADER;
    case LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER:
      return LOOM_SPIRV_FEATURE_ATOM_PHYSICAL_STORAGE_BUFFER;
    case LOOMC_SPIRV_FEATURE_FLOAT16:
      return LOOM_SPIRV_FEATURE_ATOM_FLOAT16;
    case LOOMC_SPIRV_FEATURE_FLOAT64:
      return LOOM_SPIRV_FEATURE_ATOM_FLOAT64;
    case LOOMC_SPIRV_FEATURE_INT8:
      return LOOM_SPIRV_FEATURE_ATOM_INT8;
    case LOOMC_SPIRV_FEATURE_INT16:
      return LOOM_SPIRV_FEATURE_ATOM_INT16;
    case LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS:
      return LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_8BIT_ACCESS;
    case LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS:
      return LOOM_SPIRV_FEATURE_ATOM_STORAGE_BUFFER_16BIT_ACCESS;
    case LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV:
      return LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_NV;
    case LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV:
      return LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_TRAINING_NV;
    case LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR:
      return LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR;
    case LOOMC_SPIRV_FEATURE_BFLOAT16_TYPE_KHR:
      return LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_TYPE_KHR;
    case LOOMC_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR:
      return LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_DOT_PRODUCT_KHR;
    case LOOMC_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR:
      return LOOM_SPIRV_FEATURE_ATOM_BFLOAT16_COOPERATIVE_MATRIX_KHR;
    case LOOMC_SPIRV_FEATURE_INT64:
      return LOOM_SPIRV_FEATURE_ATOM_INT64;
    case LOOMC_SPIRV_FEATURE_UNKNOWN:
    case LOOMC_SPIRV_FEATURE_COUNT:
      return LOOM_SPIRV_FEATURE_ATOM_UNKNOWN;
  }
  return LOOM_SPIRV_FEATURE_ATOM_UNKNOWN;
}

static loomc_string_view_t loomc_spirv_profile_feature_name(
    loomc_spirv_feature_t feature) {
  return loomc_string_view_from_iree(loom_spirv_feature_atom_name(
      loomc_spirv_feature_atom_from_public(feature)));
}

static loomc_string_view_t loomc_spirv_profile_limit_name(
    loomc_spirv_limit_t limit) {
  switch (limit) {
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X:
      return loomc_make_cstring_view("spirv.max_workgroup_size_x");
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y:
      return loomc_make_cstring_view("spirv.max_workgroup_size_y");
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z:
      return loomc_make_cstring_view("spirv.max_workgroup_size_z");
    case LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE:
      return loomc_make_cstring_view("spirv.max_flat_workgroup_size");
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES:
      return loomc_make_cstring_view("spirv.max_workgroup_storage_bytes");
    case LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE:
      return loomc_make_cstring_view("spirv.subgroup_size");
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X:
      return loomc_make_cstring_view("spirv.max_workgroup_count_x");
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Y:
      return loomc_make_cstring_view("spirv.max_workgroup_count_y");
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Z:
      return loomc_make_cstring_view("spirv.max_workgroup_count_z");
    case LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_X:
      return loomc_make_cstring_view("spirv.max_grid_size_x");
    case LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Y:
      return loomc_make_cstring_view("spirv.max_grid_size_y");
    case LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Z:
      return loomc_make_cstring_view("spirv.max_grid_size_z");
    case LOOMC_SPIRV_LIMIT_MAX_FLAT_GRID_SIZE:
      return loomc_make_cstring_view("spirv.max_flat_grid_size");
    case LOOMC_SPIRV_LIMIT_UNKNOWN:
    case LOOMC_SPIRV_LIMIT_COUNT:
      return loomc_make_cstring_view("spirv.unknown");
  }
  return loomc_make_cstring_view("spirv.unknown");
}

static uint64_t loomc_spirv_profile_limit_maximum_value(
    loomc_spirv_limit_t limit) {
  switch (limit) {
    case LOOMC_SPIRV_LIMIT_MAX_FLAT_GRID_SIZE:
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES:
      return UINT64_MAX;
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X:
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y:
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z:
    case LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE:
    case LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE:
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X:
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Y:
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Z:
    case LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_X:
    case LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Y:
    case LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Z:
    case LOOMC_SPIRV_LIMIT_UNKNOWN:
    case LOOMC_SPIRV_LIMIT_COUNT:
      return UINT32_MAX;
  }
  return UINT32_MAX;
}

static const char* loomc_spirv_profile_limit_range_name(
    loomc_spirv_limit_t limit) {
  switch (limit) {
    case LOOMC_SPIRV_LIMIT_MAX_FLAT_GRID_SIZE:
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES:
      return "uint64_t";
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X:
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y:
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z:
    case LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE:
    case LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE:
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X:
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Y:
    case LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Z:
    case LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_X:
    case LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Y:
    case LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Z:
    case LOOMC_SPIRV_LIMIT_UNKNOWN:
    case LOOMC_SPIRV_LIMIT_COUNT:
      return "uint32_t";
  }
  return "uint32_t";
}

static loomc_string_view_t loomc_spirv_profile_environment_name(
    loomc_spirv_environment_t environment) {
  switch (environment) {
    case LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION:
      return loomc_make_cstring_view("spirv.max_spirv_version");
    case LOOMC_SPIRV_ENVIRONMENT_UNKNOWN:
    case LOOMC_SPIRV_ENVIRONMENT_COUNT:
      return loomc_make_cstring_view("spirv.unknown");
  }
  return loomc_make_cstring_view("spirv.unknown");
}

static loomc_spirv_feature_bits_t loomc_spirv_profile_feature_bits_from_loom(
    loom_spirv_feature_bits_t feature_bits) {
  loomc_spirv_feature_bits_t public_feature_bits = 0;
  for (uint32_t i = LOOMC_SPIRV_FEATURE_UNKNOWN + 1;
       i < LOOMC_SPIRV_FEATURE_COUNT; ++i) {
    const loomc_spirv_feature_t feature = (loomc_spirv_feature_t)i;
    const loom_spirv_feature_atom_t atom =
        loomc_spirv_feature_atom_from_public(feature);
    if ((feature_bits & loom_spirv_feature_atom_bit(atom)) != 0) {
      public_feature_bits |= loomc_spirv_feature_bit(feature);
    }
  }
  return public_feature_bits;
}

static loomc_status_t loomc_spirv_profile_validate_feature(
    loomc_spirv_feature_t feature) {
  if (feature <= LOOMC_SPIRV_FEATURE_UNKNOWN ||
      feature >= LOOMC_SPIRV_FEATURE_COUNT) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V feature identifier is invalid");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_profile_validate_limit(
    loomc_spirv_limit_t limit) {
  if (limit <= LOOMC_SPIRV_LIMIT_UNKNOWN || limit >= LOOMC_SPIRV_LIMIT_COUNT) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V limit identifier is invalid");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_profile_validate_environment(
    loomc_spirv_environment_t environment) {
  if (environment <= LOOMC_SPIRV_ENVIRONMENT_UNKNOWN ||
      environment >= LOOMC_SPIRV_ENVIRONMENT_COUNT) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V environment identifier is invalid");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_profile_validate_fact_state(
    loomc_target_fact_state_t state) {
  if (state != LOOMC_TARGET_FACT_STATE_UNKNOWN &&
      state != LOOMC_TARGET_FACT_STATE_FALSE &&
      state != LOOMC_TARGET_FACT_STATE_TRUE) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "target fact state is invalid");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_profile_validate_options(
    const loomc_spirv_profile_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V profile options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V profile options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "SPIR-V profile option extensions are not supported");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_profile_validate_string_view(options->identifier));
  if (options->preset > LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "SPIR-V profile preset is invalid");
  }
  if (options->feature_fact_count != 0 && options->feature_facts == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V feature_fact_count is non-zero but feature_facts is NULL");
  }
  for (loomc_host_size_t i = 0; i < options->feature_fact_count; ++i) {
    const loomc_spirv_feature_fact_t* fact = &options->feature_facts[i];
    LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_feature(fact->feature));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_fact_state(fact->state));
    LOOMC_RETURN_IF_ERROR(
        loomc_spirv_profile_validate_string_view(fact->provenance));
  }
  if (options->limit_fact_count != 0 && options->limit_facts == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V limit_fact_count is non-zero but limit_facts is NULL");
  }
  for (loomc_host_size_t i = 0; i < options->limit_fact_count; ++i) {
    const loomc_spirv_limit_fact_t* fact = &options->limit_facts[i];
    LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_limit(fact->limit));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_fact_state(fact->state));
    LOOMC_RETURN_IF_ERROR(
        loomc_spirv_profile_validate_string_view(fact->provenance));
  }
  if (options->environment_fact_count != 0 &&
      options->environment_facts == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "SPIR-V environment_fact_count is non-zero but environment_facts is "
        "NULL");
  }
  for (loomc_host_size_t i = 0; i < options->environment_fact_count; ++i) {
    const loomc_spirv_environment_fact_t* fact = &options->environment_facts[i];
    LOOMC_RETURN_IF_ERROR(
        loomc_spirv_profile_validate_environment(fact->environment));
    LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_fact_state(fact->state));
    LOOMC_RETURN_IF_ERROR(
        loomc_spirv_profile_validate_string_view(fact->provenance));
  }
  return loomc_spirv_profile_validate_cooperative_row_options(options);
}

static loomc_status_t loomc_spirv_profile_fail_status(loomc_result_t* result,
                                                      loomc_string_view_t code,
                                                      loomc_status_t status) {
  return loomc_result_fail_status_diagnostic_consume(
      result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR, code, status);
}

static loomc_status_t loomc_spirv_profile_apply_feature_fact(
    loomc_spirv_feature_t feature, loomc_target_fact_state_t state,
    loomc_string_view_t provenance, loomc_spirv_feature_state_t* feature_states,
    loomc_result_t* result) {
  if (state == LOOMC_TARGET_FACT_STATE_UNKNOWN) {
    return loomc_ok_status();
  }
  loomc_spirv_feature_state_t* current = &feature_states[feature];
  if (current->state == LOOMC_TARGET_FACT_STATE_UNKNOWN) {
    current->state = state;
    current->provenance = provenance;
    return loomc_ok_status();
  }
  if (current->state == state) {
    return loomc_ok_status();
  }

  const loomc_string_view_t feature_name =
      loomc_spirv_profile_feature_name(feature);
  iree_status_t status = iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "SPIR-V feature '%.*s' has contradictory facts: '%.*s' says %s while "
      "'%.*s' says %s",
      (int)feature_name.size, loomc_spirv_profile_string_data(feature_name),
      (int)current->provenance.size,
      loomc_spirv_profile_string_data(current->provenance),
      current->state == LOOMC_TARGET_FACT_STATE_TRUE ? "true" : "false",
      (int)provenance.size, loomc_spirv_profile_string_data(provenance),
      state == LOOMC_TARGET_FACT_STATE_TRUE ? "true" : "false");
  return loomc_spirv_profile_fail_status(
      result, loomc_make_cstring_view("SPIRV/PROFILE"),
      loomc_status_from_iree(status));
}

static loomc_status_t loomc_spirv_profile_apply_numeric_fact(
    const char* fact_kind, loomc_string_view_t fact_name,
    loomc_target_fact_state_t state, uint64_t value, uint64_t maximum_value,
    const char* range_name, loomc_string_view_t provenance,
    loomc_spirv_numeric_fact_state_t* current, loomc_result_t* result) {
  if (state == LOOMC_TARGET_FACT_STATE_UNKNOWN) {
    return loomc_ok_status();
  }

  if (state == LOOMC_TARGET_FACT_STATE_TRUE && value == 0) {
    iree_status_t status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V %s '%.*s' has invalid zero value from '%.*s'", fact_kind,
        (int)fact_name.size, loomc_spirv_profile_string_data(fact_name),
        (int)provenance.size, loomc_spirv_profile_string_data(provenance));
    return loomc_spirv_profile_fail_status(
        result, loomc_make_cstring_view("SPIRV/PROFILE"),
        loomc_status_from_iree(status));
  }
  if (state == LOOMC_TARGET_FACT_STATE_TRUE && value > maximum_value) {
    iree_status_t status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V %s '%.*s' value from '%.*s' exceeds %s range: %" PRIu64,
        fact_kind, (int)fact_name.size,
        loomc_spirv_profile_string_data(fact_name), (int)provenance.size,
        loomc_spirv_profile_string_data(provenance), range_name, value);
    return loomc_spirv_profile_fail_status(
        result, loomc_make_cstring_view("SPIRV/PROFILE"),
        loomc_status_from_iree(status));
  }

  if (current->state == LOOMC_TARGET_FACT_STATE_UNKNOWN) {
    current->state = state;
    current->value =
        state == LOOMC_TARGET_FACT_STATE_TRUE ? value : UINT64_C(0);
    current->provenance = provenance;
    return loomc_ok_status();
  }
  if (current->state == state) {
    if (state != LOOMC_TARGET_FACT_STATE_TRUE || current->value == value) {
      return loomc_ok_status();
    }
    iree_status_t status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V %s '%.*s' has contradictory values: '%.*s' says %" PRIu64
        " while '%.*s' says %" PRIu64,
        fact_kind, (int)fact_name.size,
        loomc_spirv_profile_string_data(fact_name),
        (int)current->provenance.size,
        loomc_spirv_profile_string_data(current->provenance), current->value,
        (int)provenance.size, loomc_spirv_profile_string_data(provenance),
        value);
    return loomc_spirv_profile_fail_status(
        result, loomc_make_cstring_view("SPIRV/PROFILE"),
        loomc_status_from_iree(status));
  }

  iree_status_t status = iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "SPIR-V %s '%.*s' has contradictory facts: '%.*s' says %s while "
      "'%.*s' says %s",
      fact_kind, (int)fact_name.size,
      loomc_spirv_profile_string_data(fact_name), (int)current->provenance.size,
      loomc_spirv_profile_string_data(current->provenance),
      current->state == LOOMC_TARGET_FACT_STATE_TRUE ? "true" : "false",
      (int)provenance.size, loomc_spirv_profile_string_data(provenance),
      state == LOOMC_TARGET_FACT_STATE_TRUE ? "true" : "false");
  return loomc_spirv_profile_fail_status(
      result, loomc_make_cstring_view("SPIRV/PROFILE"),
      loomc_status_from_iree(status));
}

static loomc_status_t loomc_spirv_profile_apply_limit_fact(
    loomc_spirv_limit_t limit, loomc_target_fact_state_t state, uint64_t value,
    loomc_string_view_t provenance,
    loomc_spirv_numeric_fact_state_t* limit_states, loomc_result_t* result) {
  return loomc_spirv_profile_apply_numeric_fact(
      "limit", loomc_spirv_profile_limit_name(limit), state, value,
      loomc_spirv_profile_limit_maximum_value(limit),
      loomc_spirv_profile_limit_range_name(limit), provenance,
      &limit_states[limit], result);
}

static loomc_status_t loomc_spirv_profile_apply_environment_fact(
    loomc_spirv_environment_t environment, loomc_target_fact_state_t state,
    uint64_t value, loomc_string_view_t provenance,
    loomc_spirv_numeric_fact_state_t* environment_states,
    loomc_result_t* result) {
  return loomc_spirv_profile_apply_numeric_fact(
      "environment", loomc_spirv_profile_environment_name(environment), state,
      value, UINT32_MAX, "uint32_t", provenance,
      &environment_states[environment], result);
}

static loomc_status_t loomc_spirv_profile_apply_feature_bits(
    loomc_spirv_feature_bits_t feature_bits, loomc_string_view_t provenance,
    loomc_spirv_feature_state_t* feature_states, loomc_result_t* result) {
  for (uint32_t i = LOOMC_SPIRV_FEATURE_UNKNOWN + 1;
       i < LOOMC_SPIRV_FEATURE_COUNT; ++i) {
    const loomc_spirv_feature_t feature = (loomc_spirv_feature_t)i;
    if ((feature_bits & loomc_spirv_feature_bit(feature)) == 0) {
      continue;
    }
    LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_apply_feature_fact(
        feature, LOOMC_TARGET_FACT_STATE_TRUE, provenance, feature_states,
        result));
    if (!loomc_result_succeeded(result)) {
      return loomc_ok_status();
    }
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_profile_apply_preset(
    loomc_spirv_profile_preset_t preset,
    loomc_spirv_feature_state_t* feature_states, loomc_result_t* result) {
  switch (preset) {
    case LOOMC_SPIRV_PROFILE_PRESET_NONE:
      return loomc_ok_status();
    case LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA:
      return loomc_spirv_profile_apply_feature_bits(
          LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA,
          loomc_make_cstring_view("preset:vulkan1.3-bda"), feature_states,
          result);
  }
  return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                           "SPIR-V profile preset is invalid");
}

static loomc_status_t loomc_spirv_profile_apply_explicit_feature_facts(
    const loomc_spirv_profile_options_t* options,
    loomc_spirv_feature_state_t* feature_states, loomc_result_t* result) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  for (loomc_host_size_t i = 0; i < options->feature_fact_count; ++i) {
    const loomc_spirv_feature_fact_t* fact = &options->feature_facts[i];
    LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_apply_feature_fact(
        fact->feature, fact->state, fact->provenance, feature_states, result));
    if (!loomc_result_succeeded(result)) {
      return loomc_ok_status();
    }
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_profile_apply_explicit_limit_facts(
    const loomc_spirv_profile_options_t* options,
    loomc_spirv_numeric_fact_state_t* limit_states, loomc_result_t* result) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  for (loomc_host_size_t i = 0; i < options->limit_fact_count; ++i) {
    const loomc_spirv_limit_fact_t* fact = &options->limit_facts[i];
    LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_apply_limit_fact(
        fact->limit, fact->state, fact->value, fact->provenance, limit_states,
        result));
    if (!loomc_result_succeeded(result)) {
      return loomc_ok_status();
    }
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_profile_apply_explicit_environment_facts(
    const loomc_spirv_profile_options_t* options,
    loomc_spirv_numeric_fact_state_t* environment_states,
    loomc_result_t* result) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  for (loomc_host_size_t i = 0; i < options->environment_fact_count; ++i) {
    const loomc_spirv_environment_fact_t* fact = &options->environment_facts[i];
    LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_apply_environment_fact(
        fact->environment, fact->state, fact->value, fact->provenance,
        environment_states, result));
    if (!loomc_result_succeeded(result)) {
      return loomc_ok_status();
    }
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_profile_apply_options(
    const loomc_spirv_profile_options_t* options,
    loomc_spirv_feature_state_t* feature_states,
    loomc_spirv_numeric_fact_state_t* limit_states,
    loomc_spirv_numeric_fact_state_t* environment_states,
    loomc_result_t* result) {
  loomc_status_t status = loomc_spirv_profile_apply_preset(
      options ? options->preset : LOOMC_SPIRV_PROFILE_PRESET_NONE,
      feature_states, result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_spirv_profile_apply_explicit_feature_facts(
        options, feature_states, result);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_spirv_profile_apply_explicit_limit_facts(
        options, limit_states, result);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_spirv_profile_apply_explicit_environment_facts(
        options, environment_states, result);
  }
  return status;
}

static loom_spirv_feature_bits_t loomc_spirv_profile_feature_bits(
    const loomc_spirv_feature_state_t* feature_states) {
  loom_spirv_feature_bits_t feature_bits = 0;
  for (uint32_t i = LOOMC_SPIRV_FEATURE_UNKNOWN + 1;
       i < LOOMC_SPIRV_FEATURE_COUNT; ++i) {
    if (feature_states[i].state != LOOMC_TARGET_FACT_STATE_TRUE) {
      continue;
    }
    feature_bits |= loom_spirv_feature_atom_bit(
        loomc_spirv_feature_atom_from_public((loomc_spirv_feature_t)i));
  }
  return feature_bits;
}

static void loomc_spirv_profile_deinitialize_feature_states(
    loomc_allocator_t allocator, loomc_spirv_feature_state_t* states,
    uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    loomc_allocator_free(allocator, (void*)states[i].provenance.data);
    states[i] = (loomc_spirv_feature_state_t){0};
  }
}

static void loomc_spirv_profile_deinitialize_numeric_states(
    loomc_allocator_t allocator, loomc_spirv_numeric_fact_state_t* states,
    uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    loomc_allocator_free(allocator, (void*)states[i].provenance.data);
    states[i] = (loomc_spirv_numeric_fact_state_t){0};
  }
}

static void loomc_spirv_target_profile_payload_deinitialize(
    void* payload, loomc_allocator_t allocator) {
  loomc_spirv_target_profile_payload_t* profile_payload =
      (loomc_spirv_target_profile_payload_t*)payload;
  loomc_spirv_cooperative_row_fact_set_deinitialize(
      &profile_payload->cooperative_row_facts, allocator);
  loomc_spirv_profile_deinitialize_feature_states(
      allocator, profile_payload->feature_states, LOOMC_SPIRV_FEATURE_COUNT);
  loomc_spirv_profile_deinitialize_numeric_states(
      allocator, profile_payload->limit_states, LOOMC_SPIRV_LIMIT_COUNT);
  loomc_spirv_profile_deinitialize_numeric_states(
      allocator, profile_payload->environment_states,
      LOOMC_SPIRV_ENVIRONMENT_COUNT);
  loomc_allocator_free(allocator, profile_payload);
}

static loomc_status_t loomc_spirv_profile_copy_feature_states(
    const loomc_spirv_feature_state_t* source, loomc_allocator_t allocator,
    loomc_spirv_target_profile_payload_t* payload) {
  for (uint32_t i = 0; i < LOOMC_SPIRV_FEATURE_COUNT; ++i) {
    payload->feature_states[i].state = source[i].state;
    LOOMC_RETURN_IF_ERROR(
        loomc_string_view_clone(source[i].provenance, allocator,
                                &payload->feature_states[i].provenance));
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_profile_copy_numeric_states(
    const loomc_spirv_numeric_fact_state_t* source, uint32_t count,
    loomc_allocator_t allocator, loomc_spirv_numeric_fact_state_t* target) {
  for (uint32_t i = 0; i < count; ++i) {
    target[i].state = source[i].state;
    target[i].value = source[i].value;
    LOOMC_RETURN_IF_ERROR(loomc_string_view_clone(
        source[i].provenance, allocator, &target[i].provenance));
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_spirv_profile_copy_limit_states(
    const loomc_spirv_numeric_fact_state_t* source, loomc_allocator_t allocator,
    loomc_spirv_target_profile_payload_t* payload) {
  return loomc_spirv_profile_copy_numeric_states(
      source, LOOMC_SPIRV_LIMIT_COUNT, allocator, payload->limit_states);
}

static loomc_status_t loomc_spirv_profile_copy_environment_states(
    const loomc_spirv_numeric_fact_state_t* source, loomc_allocator_t allocator,
    loomc_spirv_target_profile_payload_t* payload) {
  return loomc_spirv_profile_copy_numeric_states(
      source, LOOMC_SPIRV_ENVIRONMENT_COUNT, allocator,
      payload->environment_states);
}

static loomc_status_t loomc_spirv_profile_validate_environment_constraints(
    const loom_spirv_feature_set_t* feature_set,
    const loomc_spirv_numeric_fact_state_t* environment_states,
    loomc_result_t* result) {
  const loomc_spirv_numeric_fact_state_t* max_spirv_version =
      &environment_states[LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION];
  if (max_spirv_version->state != LOOMC_TARGET_FACT_STATE_TRUE ||
      feature_set->minimum_spirv_version <= max_spirv_version->value) {
    return loomc_ok_status();
  }

  const loomc_string_view_t environment_name =
      loomc_spirv_profile_environment_name(
          LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION);
  iree_status_t status = iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "SPIR-V target profile requires SPIR-V version 0x%08" PRIx32
      ", but environment '%.*s' from '%.*s' allows only 0x%08" PRIx64,
      feature_set->minimum_spirv_version, (int)environment_name.size,
      loomc_spirv_profile_string_data(environment_name),
      (int)max_spirv_version->provenance.size,
      loomc_spirv_profile_string_data(max_spirv_version->provenance),
      max_spirv_version->value);
  return loomc_spirv_profile_fail_status(
      result, loomc_make_cstring_view("SPIRV/PROFILE"),
      loomc_status_from_iree(status));
}

static void loomc_spirv_profile_apply_u32_limit_value(
    const loomc_spirv_numeric_fact_state_t* limit_states,
    loomc_spirv_limit_t limit, uint32_t* out_value) {
  if (limit_states[limit].state != LOOMC_TARGET_FACT_STATE_TRUE) {
    return;
  }
  *out_value = (uint32_t)limit_states[limit].value;
}

static void loomc_spirv_profile_apply_u64_limit_value(
    const loomc_spirv_numeric_fact_state_t* limit_states,
    loomc_spirv_limit_t limit, uint64_t* out_value) {
  if (limit_states[limit].state != LOOMC_TARGET_FACT_STATE_TRUE) {
    return;
  }
  *out_value = limit_states[limit].value;
}

static void loomc_spirv_profile_apply_limit_states_to_bundle(
    const loomc_spirv_numeric_fact_state_t* limit_states,
    loom_target_bundle_storage_t* storage) {
  loomc_spirv_profile_apply_u32_limit_value(
      limit_states, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
      &storage->snapshot.max_workgroup_size.x);
  loomc_spirv_profile_apply_u32_limit_value(
      limit_states, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y,
      &storage->snapshot.max_workgroup_size.y);
  loomc_spirv_profile_apply_u32_limit_value(
      limit_states, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z,
      &storage->snapshot.max_workgroup_size.z);
  loomc_spirv_profile_apply_u32_limit_value(
      limit_states, LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE,
      &storage->snapshot.max_flat_workgroup_size);
  loomc_spirv_profile_apply_u64_limit_value(
      limit_states, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
      &storage->snapshot.max_workgroup_storage_bytes);
  loomc_spirv_profile_apply_u32_limit_value(limit_states,
                                            LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
                                            &storage->snapshot.subgroup_size);
  loomc_spirv_profile_apply_u32_limit_value(
      limit_states, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X,
      &storage->snapshot.max_workgroup_count.x);
  loomc_spirv_profile_apply_u32_limit_value(
      limit_states, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Y,
      &storage->snapshot.max_workgroup_count.y);
  loomc_spirv_profile_apply_u32_limit_value(
      limit_states, LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Z,
      &storage->snapshot.max_workgroup_count.z);
  loomc_spirv_profile_apply_u32_limit_value(limit_states,
                                            LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_X,
                                            &storage->snapshot.max_grid_size.x);
  loomc_spirv_profile_apply_u32_limit_value(limit_states,
                                            LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Y,
                                            &storage->snapshot.max_grid_size.y);
  loomc_spirv_profile_apply_u32_limit_value(limit_states,
                                            LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Z,
                                            &storage->snapshot.max_grid_size.z);
  loomc_spirv_profile_apply_u64_limit_value(
      limit_states, LOOMC_SPIRV_LIMIT_MAX_FLAT_GRID_SIZE,
      &storage->snapshot.max_flat_grid_size);
}

static bool loomc_spirv_profile_can_materialize_vulkan_bda_bundle(
    const loom_spirv_feature_set_t* feature_set) {
  return loom_spirv_feature_set_has_atom(
      feature_set, LOOM_SPIRV_FEATURE_ATOM_PHYSICAL_STORAGE_BUFFER);
}

static void loomc_spirv_profile_initialize_vulkan_bda_bundle(
    const loom_spirv_feature_set_t* feature_set,
    const loomc_spirv_numeric_fact_state_t* limit_states,
    loom_target_bundle_storage_t* out_storage) {
  *out_storage = (loom_target_bundle_storage_t){
      .snapshot = *loom_spirv_low_target_bundle_vulkan1_3.snapshot,
      .export_plan = *loom_spirv_low_target_bundle_vulkan1_3.export_plan,
      .config = *loom_spirv_low_target_bundle_vulkan1_3.config,
      .bundle = loom_spirv_low_target_bundle_vulkan1_3,
  };
  loom_target_bundle_storage_rebind(out_storage);
  out_storage->bundle.name = IREE_SV("spirv-vulkan1.3-bda-profile");
  out_storage->snapshot.name = IREE_SV("spirv-vulkan1.3-bda-profile");
  out_storage->config.name =
      IREE_SV("spirv.logical.core.vulkan1.3.bda.profile");
  out_storage->config.contract_feature_bits = feature_set->atom_bits;
  loomc_spirv_profile_apply_limit_states_to_bundle(limit_states, out_storage);
}

static loomc_status_t loomc_spirv_target_profile_create_from_states(
    loomc_target_environment_t* target_environment,
    loomc_string_view_t identifier,
    const loomc_spirv_feature_state_t* feature_states,
    const loomc_spirv_numeric_fact_state_t* limit_states,
    const loomc_spirv_numeric_fact_state_t* environment_states,
    const loomc_spirv_cooperative_row_fact_set_t* base_row_facts,
    const loomc_spirv_profile_options_t* options, loomc_result_t* result,
    loomc_allocator_t allocator, loomc_target_profile_t** out_profile) {
  loomc_spirv_target_profile_payload_t* payload = NULL;
  loomc_status_t status =
      loomc_allocator_malloc(allocator, sizeof(*payload), (void**)&payload);
  if (loomc_status_is_ok(status)) {
    memset(payload, 0, sizeof(*payload));
    status = loomc_spirv_profile_copy_feature_states(feature_states, allocator,
                                                     payload);
  }
  if (loomc_status_is_ok(status)) {
    status =
        loomc_spirv_profile_copy_limit_states(limit_states, allocator, payload);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_spirv_profile_copy_environment_states(environment_states,
                                                         allocator, payload);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_spirv_cooperative_row_fact_set_initialize(
        base_row_facts, options, result, allocator,
        &payload->cooperative_row_facts);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_status_from_iree(loom_spirv_feature_set_prepare(
        iree_string_view_from_loomc(identifier),
        loomc_spirv_profile_feature_bits(feature_states),
        &payload->feature_set));
    if (!loomc_status_is_ok(status) &&
        loomc_status_is_result_diagnostic(status)) {
      status = loomc_spirv_profile_fail_status(
          result, loomc_make_cstring_view("SPIRV/PROFILE"), status);
    }
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_spirv_profile_validate_environment_constraints(
        &payload->feature_set, environment_states, result);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_spirv_cooperative_row_fact_set_prepare_properties(
        &payload->cooperative_row_facts, &payload->feature_set, allocator,
        &payload->profile.cooperative_properties);
  }
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    const loom_target_selection_t selection = {
        .bundle = loomc_spirv_profile_can_materialize_vulkan_bda_bundle(
                      &payload->feature_set)
                      ? &payload->bundle_storage.bundle
                      : NULL,
        .data = &payload->profile,
    };
    if (selection.bundle != NULL) {
      loomc_spirv_profile_initialize_vulkan_bda_bundle(
          &payload->feature_set, payload->limit_states,
          &payload->bundle_storage);
    }
    loomc_target_profile_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_PROFILE_OPTIONS,
        .structure_size = sizeof(target_options),
        .identifier = identifier,
    };
    status = loomc_target_profile_create_from_selection(
        target_environment, &target_options, selection,
        &kLoomcSpirvTargetProfilePayloadType, payload,
        loomc_spirv_target_profile_payload_deinitialize, allocator,
        out_profile);
    payload = NULL;
  }
  if (payload != NULL) {
    loomc_spirv_target_profile_payload_deinitialize(payload, allocator);
  }
  return status;
}

static const loomc_spirv_target_profile_payload_t*
loomc_spirv_profile_payload_from_profile(
    const loomc_target_profile_t* profile) {
  return (const loomc_spirv_target_profile_payload_t*)
      loomc_target_profile_payload(profile,
                                   &kLoomcSpirvTargetProfilePayloadType);
}

static loomc_status_t loomc_spirv_profile_validate_query(
    const loomc_target_profile_t* profile,
    const loomc_spirv_target_profile_payload_t** out_payload) {
  if (out_payload == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_payload must not be NULL");
  }
  *out_payload = NULL;
  if (profile == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "profile must not be NULL");
  }
  const loomc_spirv_target_profile_payload_t* payload =
      loomc_spirv_profile_payload_from_profile(profile);
  if (payload == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "profile is not a SPIR-V profile");
  }
  *out_payload = payload;
  return loomc_ok_status();
}

static loomc_string_view_t loomc_spirv_profile_refined_identifier(
    const loomc_target_profile_t* base_profile,
    const loomc_spirv_profile_options_t* options) {
  if (options != NULL && !loomc_string_view_is_empty(options->identifier)) {
    return options->identifier;
  }
  return loomc_target_profile_identifier(base_profile);
}

loomc_status_t loomc_target_profile_create_spirv(
    loomc_target_environment_t* target_environment,
    const loomc_spirv_profile_options_t* options, loomc_allocator_t allocator,
    loomc_target_profile_t** out_profile, loomc_result_t** out_result) {
  if (out_profile == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_profile and out_result must not be NULL");
  }
  *out_profile = NULL;
  *out_result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_options(options));

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));

  loomc_spirv_feature_state_t feature_states[LOOMC_SPIRV_FEATURE_COUNT] = {0};
  loomc_spirv_numeric_fact_state_t limit_states[LOOMC_SPIRV_LIMIT_COUNT] = {0};
  loomc_spirv_numeric_fact_state_t
      environment_states[LOOMC_SPIRV_ENVIRONMENT_COUNT] = {0};
  loomc_status_t status = loomc_spirv_profile_apply_options(
      options, feature_states, limit_states, environment_states, result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_spirv_target_profile_create_from_states(
        target_environment, loomc_spirv_profile_identifier(options),
        feature_states, limit_states, environment_states,
        /*base_row_facts=*/NULL, options, result, allocator, out_profile);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    result = NULL;
  }
  loomc_result_release(result);
  return status;
}

loomc_status_t loomc_spirv_target_profile_refine(
    const loomc_target_profile_t* base_profile,
    const loomc_spirv_profile_options_t* options, loomc_allocator_t allocator,
    loomc_target_profile_t** out_profile, loomc_result_t** out_result) {
  if (out_profile == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_profile and out_result must not be NULL");
  }
  *out_profile = NULL;
  *out_result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_options(options));
  const loomc_spirv_target_profile_payload_t* base_payload = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_spirv_profile_validate_query(base_profile, &base_payload));

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));

  loomc_spirv_feature_state_t feature_states[LOOMC_SPIRV_FEATURE_COUNT] = {0};
  loomc_spirv_numeric_fact_state_t limit_states[LOOMC_SPIRV_LIMIT_COUNT] = {0};
  loomc_spirv_numeric_fact_state_t
      environment_states[LOOMC_SPIRV_ENVIRONMENT_COUNT] = {0};
  memcpy(feature_states, base_payload->feature_states, sizeof(feature_states));
  memcpy(limit_states, base_payload->limit_states, sizeof(limit_states));
  memcpy(environment_states, base_payload->environment_states,
         sizeof(environment_states));

  loomc_status_t status = loomc_spirv_profile_apply_options(
      options, feature_states, limit_states, environment_states, result);
  if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
    status = loomc_spirv_target_profile_create_from_states(
        loomc_target_profile_target_environment(base_profile),
        loomc_spirv_profile_refined_identifier(base_profile, options),
        feature_states, limit_states, environment_states,
        &base_payload->cooperative_row_facts, options, result, allocator,
        out_profile);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    result = NULL;
  }
  loomc_result_release(result);
  return status;
}

loomc_status_t loomc_spirv_target_profile_query_feature(
    const loomc_target_profile_t* profile, loomc_spirv_feature_t feature,
    loomc_target_fact_state_t* out_state) {
  if (out_state == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_state must not be NULL");
  }
  *out_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_feature(feature));
  const loomc_spirv_target_profile_payload_t* payload = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_query(profile, &payload));
  *out_state = payload->feature_states[feature].state;
  return loomc_ok_status();
}

loomc_status_t loomc_spirv_target_profile_query_limit(
    const loomc_target_profile_t* profile, loomc_spirv_limit_t limit,
    loomc_spirv_limit_value_t* out_value) {
  if (out_value == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_value must not be NULL");
  }
  *out_value = (loomc_spirv_limit_value_t){0};
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_limit(limit));
  const loomc_spirv_target_profile_payload_t* payload = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_query(profile, &payload));
  *out_value = (loomc_spirv_limit_value_t){
      .state = payload->limit_states[limit].state,
      .value = payload->limit_states[limit].value,
  };
  return loomc_ok_status();
}

loomc_status_t loomc_spirv_target_profile_query_environment(
    const loomc_target_profile_t* profile,
    loomc_spirv_environment_t environment,
    loomc_spirv_environment_value_t* out_value) {
  if (out_value == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_value must not be NULL");
  }
  *out_value = (loomc_spirv_environment_value_t){0};
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_environment(environment));
  const loomc_spirv_target_profile_payload_t* payload = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_query(profile, &payload));
  *out_value = (loomc_spirv_environment_value_t){
      .state = payload->environment_states[environment].state,
      .value = payload->environment_states[environment].value,
  };
  return loomc_ok_status();
}

loomc_status_t loomc_spirv_target_profile_query_info(
    const loomc_target_profile_t* profile,
    loomc_spirv_profile_info_t* out_info) {
  if (out_info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  *out_info = (loomc_spirv_profile_info_t){0};
  const loomc_spirv_target_profile_payload_t* payload = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_query(profile, &payload));
  const loom_spirv_feature_set_t* feature_set = &payload->feature_set;
  *out_info = (loomc_spirv_profile_info_t){
      .minimum_spirv_version = feature_set->minimum_spirv_version,
      .addressing_model = feature_set->addressing_model,
      .memory_model = feature_set->memory_model,
      .extension_count = feature_set->extension_count,
      .capability_count = feature_set->capability_count,
      .opcode_count = feature_set->opcode_count,
      .storage_class_count = feature_set->storage_class_count,
      .decoration_count = feature_set->decoration_count,
      .cooperative_matrix_row_count =
          loomc_spirv_cooperative_row_fact_set_matrix_row_count(
              &payload->cooperative_row_facts),
      .cooperative_vector_row_count =
          loomc_spirv_cooperative_row_fact_set_vector_row_count(
              &payload->cooperative_row_facts),
  };
  return loomc_ok_status();
}

loomc_status_t loomc_spirv_target_profile_extension_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    loomc_string_view_t* out_extension) {
  if (out_extension == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_extension must not be NULL");
  }
  *out_extension = loomc_string_view_empty();
  const loomc_spirv_target_profile_payload_t* payload = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_query(profile, &payload));
  if (index >= payload->feature_set.extension_count) {
    return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                             "SPIR-V extension index is out of range");
  }
  *out_extension =
      loomc_string_view_from_iree(payload->feature_set.extension_names[index]);
  return loomc_ok_status();
}

loomc_status_t loomc_spirv_target_profile_capability_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    uint32_t* out_capability) {
  if (out_capability == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_capability must not be NULL");
  }
  *out_capability = 0;
  const loomc_spirv_target_profile_payload_t* payload = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_query(profile, &payload));
  if (index >= payload->feature_set.capability_count) {
    return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                             "SPIR-V capability index is out of range");
  }
  *out_capability = payload->feature_set.capabilities[index];
  return loomc_ok_status();
}

loomc_status_t loomc_spirv_target_profile_opcode_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    uint32_t* out_opcode) {
  if (out_opcode == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_opcode must not be NULL");
  }
  *out_opcode = 0;
  const loomc_spirv_target_profile_payload_t* payload = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_query(profile, &payload));
  if (index >= payload->feature_set.opcode_count) {
    return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                             "SPIR-V opcode index is out of range");
  }
  *out_opcode = payload->feature_set.opcodes[index];
  return loomc_ok_status();
}

loomc_status_t loomc_spirv_target_profile_storage_class_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    uint32_t* out_storage_class) {
  if (out_storage_class == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_storage_class must not be NULL");
  }
  *out_storage_class = 0;
  const loomc_spirv_target_profile_payload_t* payload = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_query(profile, &payload));
  if (index >= payload->feature_set.storage_class_count) {
    return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                             "SPIR-V storage class index is out of range");
  }
  *out_storage_class = payload->feature_set.storage_classes[index];
  return loomc_ok_status();
}

loomc_status_t loomc_spirv_target_profile_decoration_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    uint32_t* out_decoration) {
  if (out_decoration == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_decoration must not be NULL");
  }
  *out_decoration = 0;
  const loomc_spirv_target_profile_payload_t* payload = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_query(profile, &payload));
  if (index >= payload->feature_set.decoration_count) {
    return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                             "SPIR-V decoration index is out of range");
  }
  *out_decoration = payload->feature_set.decorations[index];
  return loomc_ok_status();
}

loomc_status_t loomc_spirv_target_profile_cooperative_matrix_row_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    loomc_spirv_cooperative_matrix_row_t* out_row) {
  if (out_row == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_row must not be NULL");
  }
  *out_row = (loomc_spirv_cooperative_matrix_row_t){0};
  const loomc_spirv_target_profile_payload_t* payload = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_query(profile, &payload));
  return loomc_spirv_cooperative_row_fact_set_matrix_row_at(
      &payload->cooperative_row_facts, index, out_row);
}

loomc_status_t loomc_spirv_target_profile_cooperative_vector_row_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    loomc_spirv_cooperative_vector_row_t* out_row) {
  if (out_row == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_row must not be NULL");
  }
  *out_row = (loomc_spirv_cooperative_vector_row_t){0};
  const loomc_spirv_target_profile_payload_t* payload = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_spirv_profile_validate_query(profile, &payload));
  return loomc_spirv_cooperative_row_fact_set_vector_row_at(
      &payload->cooperative_row_facts, index, out_row);
}
