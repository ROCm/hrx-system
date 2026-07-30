// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/artifact_manifest.h"

#include <stddef.h>

#include "loom/util/json.h"

void loom_target_artifact_manifest_format_options_initialize(
    loom_target_artifact_manifest_format_options_t* out_options) {
  *out_options = (loom_target_artifact_manifest_format_options_t){
      .mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE,
  };
}

iree_string_view_t loom_target_artifact_manifest_mode_name(
    loom_target_artifact_manifest_mode_t mode) {
  switch (mode) {
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE:
      return IREE_SV("none");
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY:
      return IREE_SV("summary");
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS:
      return IREE_SV("details");
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS:
      return IREE_SV("analysis");
    default:
      return IREE_SV("unknown");
  }
}

iree_status_t loom_target_artifact_manifest_mode_parse(
    iree_string_view_t value, loom_target_artifact_manifest_mode_t* out_mode) {
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("none"))) {
    *out_mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("summary"))) {
    *out_mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("details"))) {
    *out_mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("analysis"))) {
    *out_mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported artifact manifest mode '%.*s'; "
                          "expected 'none', 'summary', 'details', or "
                          "'analysis'",
                          (int)value.size, value.data);
}

static iree_string_view_t loom_target_artifact_manifest_parameter_kind_name(
    loom_target_artifact_manifest_parameter_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_VALUE:
      return IREE_SV("value");
    case LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_POINTER:
      return IREE_SV("pointer");
    case LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_BINDING:
      return IREE_SV("binding");
    case LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_CONSTANT:
      return IREE_SV("constant");
    case LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

static bool loom_target_artifact_manifest_mode_includes_details(
    loom_target_artifact_manifest_mode_t mode) {
  return mode == LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS ||
         mode == LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS;
}

static bool loom_target_artifact_manifest_mode_is_valid(
    loom_target_artifact_manifest_mode_t mode) {
  switch (mode) {
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE:
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY:
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS:
    case LOOM_TARGET_ARTIFACT_MANIFEST_MODE_ANALYSIS:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_target_artifact_manifest_require_string(
    iree_string_view_t value, const char* field_name) {
  if (!iree_string_view_is_empty(value)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "artifact manifest field '%s' must not be empty",
                          field_name);
}

static iree_status_t loom_target_artifact_manifest_require_array(
    const void* values, iree_host_size_t count, const char* field_name) {
  if (count == 0 || values != NULL) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "artifact manifest array '%s' has a count but no "
                          "storage",
                          field_name);
}

static iree_status_t
loom_target_artifact_manifest_json_write_optional_string_field(
    loom_json_object_writer_t* object, const char* name,
    iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) {
    return iree_ok_status();
  }
  return loom_json_object_write_string_field(
      object, iree_make_cstring_view(name), value);
}

static iree_status_t
loom_target_artifact_manifest_json_write_optional_source_name(
    loom_json_object_writer_t* object, iree_string_view_t name,
    iree_string_view_t source_name) {
  if (iree_string_view_is_empty(source_name) ||
      iree_string_view_equal(name, source_name)) {
    return iree_ok_status();
  }
  return loom_json_object_write_string_field(object, IREE_SV("source"),
                                             source_name);
}

static iree_status_t loom_target_artifact_manifest_json_write_dimension3(
    loom_json_object_writer_t* object, iree_string_view_t field_name,
    const uint32_t dimensions[3]) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, field_name));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(object->stream, &array));
  IREE_RETURN_IF_ERROR(
      loom_json_array_write_uint32_element(&array, dimensions[0]));
  IREE_RETURN_IF_ERROR(
      loom_json_array_write_uint32_element(&array, dimensions[1]));
  IREE_RETURN_IF_ERROR(
      loom_json_array_write_uint32_element(&array, dimensions[2]));
  return loom_json_array_end(&array);
}

static bool loom_target_artifact_manifest_workgroup_size_has_limit(
    loom_target_workgroup_size_t value) {
  return value.x != 0 || value.y != 0 || value.z != 0;
}

static bool loom_target_artifact_manifest_grid_size_has_limit(
    loom_target_grid_size_t value) {
  return value.x != 0 || value.y != 0 || value.z != 0;
}

static bool loom_target_artifact_manifest_workgroup_count_has_limit(
    loom_target_workgroup_count_limit_t value) {
  return value.x != 0 || value.y != 0 || value.z != 0;
}

static iree_status_t
loom_target_artifact_manifest_json_write_optional_dimension_field(
    loom_json_object_writer_t* object, const char* name, uint32_t value) {
  if (value == 0) {
    return iree_ok_status();
  }
  return loom_json_object_write_uint32_field(
      object, iree_make_cstring_view(name), value);
}

static iree_status_t
loom_target_artifact_manifest_json_write_workgroup_size_limit_field(
    loom_json_object_writer_t* object, const char* name,
    loom_target_workgroup_size_t value) {
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, iree_make_cstring_view(name)));
  loom_json_object_writer_t dimensions;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &dimensions));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_dimension_field(
          &dimensions, "x", value.x));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_dimension_field(
          &dimensions, "y", value.y));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_dimension_field(
          &dimensions, "z", value.z));
  return loom_json_object_end(&dimensions);
}

static iree_status_t
loom_target_artifact_manifest_json_write_grid_size_limit_field(
    loom_json_object_writer_t* object, const char* name,
    loom_target_grid_size_t value) {
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, iree_make_cstring_view(name)));
  loom_json_object_writer_t dimensions;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &dimensions));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_dimension_field(
          &dimensions, "x", value.x));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_dimension_field(
          &dimensions, "y", value.y));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_dimension_field(
          &dimensions, "z", value.z));
  return loom_json_object_end(&dimensions);
}

static iree_status_t
loom_target_artifact_manifest_json_write_workgroup_count_limit_field(
    loom_json_object_writer_t* object, const char* name,
    loom_target_workgroup_count_limit_t value) {
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, iree_make_cstring_view(name)));
  loom_json_object_writer_t dimensions;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &dimensions));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_dimension_field(
          &dimensions, "x", value.x));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_dimension_field(
          &dimensions, "y", value.y));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_dimension_field(
          &dimensions, "z", value.z));
  return loom_json_object_end(&dimensions);
}

static iree_status_t
loom_target_artifact_manifest_json_write_optional_address_space_field(
    loom_json_object_writer_t* object, const char* name, uint32_t value) {
  if (value == UINT32_MAX) {
    return iree_ok_status();
  }
  return loom_json_object_write_uint32_field(
      object, iree_make_cstring_view(name), value);
}

static iree_status_t
loom_target_artifact_manifest_json_write_address_spaces_field(
    loom_json_object_writer_t* object,
    loom_target_memory_space_map_t memory_spaces) {
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("address_spaces")));
  loom_json_object_writer_t address_spaces;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &address_spaces));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_address_space_field(
          &address_spaces, "generic", memory_spaces.generic));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_address_space_field(
          &address_spaces, "global", memory_spaces.global));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_address_space_field(
          &address_spaces, "workgroup", memory_spaces.workgroup));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_address_space_field(
          &address_spaces, "constant", memory_spaces.constant));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_address_space_field(
          &address_spaces, "private", memory_spaces.private_memory));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_address_space_field(
          &address_spaces, "host", memory_spaces.host));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_address_space_field(
          &address_spaces, "descriptor", memory_spaces.descriptor));
  return loom_json_object_end(&address_spaces);
}

static iree_status_t loom_target_artifact_manifest_json_write_name_array_field(
    loom_json_object_writer_t* object, const char* name,
    const iree_string_view_t* values, iree_host_size_t count) {
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_require_array(values, count, name));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, iree_make_cstring_view(name)));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(object->stream, &array));
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_target_artifact_manifest_require_string(values[i], name));
    IREE_RETURN_IF_ERROR(
        loom_json_array_write_string_element(&array, values[i]));
  }
  return loom_json_array_end(&array);
}

static bool loom_target_artifact_manifest_has_target_name(
    const loom_target_artifact_manifest_t* manifest, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < manifest->target_count; ++i) {
    if (iree_string_view_equal(manifest->targets[i].name, name)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_target_artifact_manifest_validate_target_names(
    const loom_target_artifact_manifest_t* manifest,
    const iree_string_view_t* values, iree_host_size_t count,
    const char* field_name) {
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_require_array(values, count, field_name));
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_target_artifact_manifest_require_string(values[i], field_name));
    if (!loom_target_artifact_manifest_has_target_name(manifest, values[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "artifact manifest target reference '%.*s' in '%s' must match a "
          "declared targets[].name",
          (int)values[i].size, values[i].data, field_name);
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_artifact_manifest_json_write_target_array_field(
    const loom_target_artifact_manifest_t* manifest,
    loom_json_object_writer_t* object, const char* name,
    const iree_string_view_t* values, iree_host_size_t count) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_validate_target_names(
      manifest, values, count, name));
  return loom_target_artifact_manifest_json_write_name_array_field(
      object, name, values, count);
}

static bool loom_target_artifact_manifest_interface_has_data(
    const loom_target_artifact_manifest_interface_t* interface,
    loom_target_artifact_manifest_mode_t mode) {
  return interface->flags != 0 ||
         (loom_target_artifact_manifest_mode_includes_details(mode) &&
          interface->parameter_detail_count != 0);
}

static bool loom_target_artifact_manifest_execution_has_data(
    const loom_target_artifact_manifest_execution_t* execution) {
  return execution->flags != 0;
}

static iree_status_t loom_target_artifact_manifest_format_artifact_json(
    const loom_target_artifact_manifest_artifact_t* artifact,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_string(
      artifact->format, "artifact.format"));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("format"), artifact->format));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          &object, "name", artifact->name));
  if (iree_any_bit_set(
          artifact->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_ARTIFACT_FLAG_BYTE_LENGTH)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("byte_length"), artifact->byte_length));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_artifact_manifest_format_target_json(
    const loom_target_artifact_manifest_target_t* target,
    loom_target_artifact_manifest_mode_t mode, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_string(
      target->name, "target.name"));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("name"), target->name));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          &object, "family", target->family));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          &object, "selector", target->selector));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          &object, "processor", target->processor));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          &object, "triple", target->triple));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          &object, "profile", target->profile));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          &object, "code_object_target", target->code_object_target));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_name_array_field(
          &object, "features", target->feature_names,
          target->feature_name_count));
  if (loom_target_artifact_manifest_mode_includes_details(mode)) {
    if (iree_any_bit_set(
            target->flags,
            LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_DEFAULT_POINTER_BITWIDTH)) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("default_pointer_bitwidth"),
          target->default_pointer_bitwidth));
    }
    if (iree_any_bit_set(
            target->flags,
            LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_INDEX_BITWIDTH)) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("index_bitwidth"), target->index_bitwidth));
    }
    if (iree_any_bit_set(
            target->flags,
            LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_OFFSET_BITWIDTH)) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("offset_bitwidth"), target->offset_bitwidth));
    }
    if (iree_any_bit_set(
            target->flags,
            LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MAX_WORKGROUP_SIZE) &&
        loom_target_artifact_manifest_workgroup_size_has_limit(
            target->max_workgroup_size)) {
      IREE_RETURN_IF_ERROR(
          loom_target_artifact_manifest_json_write_workgroup_size_limit_field(
              &object, "max_workgroup_size", target->max_workgroup_size));
    }
    if (iree_any_bit_set(
            target->flags,
            LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MAX_FLAT_WORKGROUP_SIZE)) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("max_flat_workgroup_size"),
          target->max_flat_workgroup_size));
    }
    if (iree_any_bit_set(
            target->flags,
            LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MAX_WORKGROUP_STORAGE_BYTES)) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &object, IREE_SV("max_workgroup_storage_bytes"),
          target->max_workgroup_storage_bytes));
    }
    if (iree_any_bit_set(
            target->flags,
            LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_SUBGROUP_SIZE)) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("subgroup_size"), target->subgroup_size));
    }
    if (iree_any_bit_set(
            target->flags,
            LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MAX_GRID_SIZE) &&
        loom_target_artifact_manifest_grid_size_has_limit(
            target->max_grid_size)) {
      IREE_RETURN_IF_ERROR(
          loom_target_artifact_manifest_json_write_grid_size_limit_field(
              &object, "max_grid_size", target->max_grid_size));
    }
    if (iree_any_bit_set(
            target->flags,
            LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MAX_FLAT_GRID_SIZE)) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          &object, IREE_SV("max_flat_grid_size"), target->max_flat_grid_size));
    }
    if (iree_any_bit_set(
            target->flags,
            LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MAX_WORKGROUP_COUNT) &&
        loom_target_artifact_manifest_workgroup_count_has_limit(
            target->max_workgroup_count)) {
      IREE_RETURN_IF_ERROR(
          loom_target_artifact_manifest_json_write_workgroup_count_limit_field(
              &object, "max_workgroup_count", target->max_workgroup_count));
    }
    if (iree_any_bit_set(
            target->flags,
            LOOM_TARGET_ARTIFACT_MANIFEST_TARGET_FLAG_MEMORY_SPACES)) {
      IREE_RETURN_IF_ERROR(
          loom_target_artifact_manifest_json_write_address_spaces_field(
              &object, target->memory_spaces));
    }
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_artifact_manifest_format_parameter_json(
    const loom_target_artifact_manifest_parameter_t* parameter,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          &object, "name", parameter->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"),
      loom_target_artifact_manifest_parameter_kind_name(parameter->kind)));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          &object, "type", parameter->type));
  if (iree_any_bit_set(parameter->flags,
                       LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_INDEX)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("index"), parameter->index));
  }
  if (iree_any_bit_set(
          parameter->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_OFFSET)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("byte_offset"), parameter->byte_offset));
  }
  if (iree_any_bit_set(
          parameter->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_LENGTH)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("byte_length"), parameter->byte_length));
  }
  if (iree_any_bit_set(
          parameter->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_ALIGNMENT)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("byte_alignment"), parameter->byte_alignment));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_artifact_manifest_format_parameters_json(
    const loom_target_artifact_manifest_interface_t* interface,
    loom_json_object_writer_t* object) {
  if (interface->parameter_detail_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_array(
      interface->parameters, interface->parameter_detail_count,
      "interface.parameters"));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("parameters")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(object->stream, &array));
  for (iree_host_size_t i = 0; i < interface->parameter_detail_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_parameter_json(
        &interface->parameters[i], object->stream));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_target_artifact_manifest_format_interface_json(
    const loom_target_artifact_manifest_interface_t* interface,
    loom_target_artifact_manifest_mode_t mode, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  if (iree_any_bit_set(
          interface->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_PARAMETER_COUNT)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("parameter_count"), interface->parameter_count));
  }
  if (iree_any_bit_set(
          interface->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_BINDING_COUNT)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("binding_count"), interface->binding_count));
  }
  if (iree_any_bit_set(
          interface->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_CONSTANT_BYTE_LENGTH)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("constant_byte_length"),
        interface->constant_byte_length));
  }
  if (loom_target_artifact_manifest_mode_includes_details(mode)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_parameters_json(
        interface, &object));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_artifact_manifest_format_execution_json(
    const loom_target_artifact_manifest_execution_t* execution,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  if (iree_any_bit_set(
          execution->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_EXECUTION_FLAG_WORKGROUP_SIZE)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_dimension3(
        &object, IREE_SV("workgroup_size"), execution->workgroup_size));
  }
  if (iree_any_bit_set(
          execution->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_EXECUTION_FLAG_SUBGROUP_SIZE)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("subgroup_size"), execution->subgroup_size));
  }
  if (iree_any_bit_set(
          execution->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_EXECUTION_FLAG_CLUSTER_SIZE)) {
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_json_write_dimension3(
        &object, IREE_SV("cluster_size"), execution->cluster_size));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_artifact_manifest_format_function_json(
    const loom_target_artifact_manifest_t* manifest,
    const loom_target_artifact_manifest_function_t* function,
    loom_target_artifact_manifest_mode_t mode, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_string(
      function->name, "function.name"));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("name"), function->name));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_source_name(
          &object, function->name, function->source_name));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_target_array_field(
          manifest, &object, "targets", function->target_names,
          function->target_name_count));
  if (loom_target_artifact_manifest_interface_has_data(&function->interface,
                                                       mode)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("interface")));
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_interface_json(
        &function->interface, mode, stream));
  }
  if (loom_target_artifact_manifest_execution_has_data(&function->execution)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("execution")));
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_execution_json(
        &function->execution, stream));
  }
  if (loom_target_artifact_manifest_mode_includes_details(mode)) {
    IREE_RETURN_IF_ERROR(
        loom_target_artifact_manifest_json_write_name_array_field(
            &object, "uses_globals", function->used_global_names,
            function->used_global_name_count));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_artifact_manifest_format_global_json(
    const loom_target_artifact_manifest_t* manifest,
    const loom_target_artifact_manifest_global_t* global,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_string(
      global->name, "global.name"));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("name"), global->name));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_source_name(
          &object, global->name, global->source_name));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_optional_string_field(
          &object, "type", global->type));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_json_write_target_array_field(
          manifest, &object, "targets", global->target_names,
          global->target_name_count));
  if (iree_any_bit_set(global->flags,
                       LOOM_TARGET_ARTIFACT_MANIFEST_GLOBAL_FLAG_BYTE_LENGTH)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("byte_length"), global->byte_length));
  }
  if (iree_any_bit_set(
          global->flags,
          LOOM_TARGET_ARTIFACT_MANIFEST_GLOBAL_FLAG_BYTE_ALIGNMENT)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("byte_alignment"), global->byte_alignment));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_artifact_manifest_format_targets_json(
    const loom_target_artifact_manifest_t* manifest,
    loom_target_artifact_manifest_mode_t mode,
    loom_json_object_writer_t* object) {
  if (manifest->target_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_array(
      manifest->targets, manifest->target_count, "targets"));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("targets")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(object->stream, &array));
  for (iree_host_size_t i = 0; i < manifest->target_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_target_json(
        &manifest->targets[i], mode, object->stream));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_target_artifact_manifest_format_functions_json(
    const loom_target_artifact_manifest_t* manifest,
    loom_target_artifact_manifest_mode_t mode,
    loom_json_object_writer_t* object) {
  if (manifest->function_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_array(
      manifest->functions, manifest->function_count, "functions"));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("functions")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(object->stream, &array));
  for (iree_host_size_t i = 0; i < manifest->function_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_function_json(
        manifest, &manifest->functions[i], mode, object->stream));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_target_artifact_manifest_format_globals_json(
    const loom_target_artifact_manifest_t* manifest,
    loom_json_object_writer_t* object) {
  if (manifest->global_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_require_array(
      manifest->globals, manifest->global_count, "globals"));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("globals")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(object->stream, &array));
  for (iree_host_size_t i = 0; i < manifest->global_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_global_json(
        manifest, &manifest->globals[i], object->stream));
  }
  return loom_json_array_end(&array);
}

iree_status_t loom_target_artifact_manifest_format_json(
    const loom_target_artifact_manifest_t* manifest,
    const loom_target_artifact_manifest_format_options_t* options,
    loom_output_stream_t* stream) {
  if (options == NULL || stream == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest format options and stream are "
                            "required");
  }
  if (options->mode == LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_ok_status();
  }
  if (manifest == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest is required when formatting is "
                            "enabled");
  }
  if (!loom_target_artifact_manifest_mode_is_valid(options->mode)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported artifact manifest mode value %d",
                            (int)options->mode);
  }

  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"), IREE_SV("loom.artifact_manifest")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("schema_version"),
      LOOM_TARGET_ARTIFACT_MANIFEST_SCHEMA_VERSION));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("mode"),
      loom_target_artifact_manifest_mode_name(options->mode)));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("artifact")));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_artifact_json(
      &manifest->artifact, stream));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_targets_json(
      manifest, options->mode, &object));
  IREE_RETURN_IF_ERROR(loom_target_artifact_manifest_format_functions_json(
      manifest, options->mode, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_artifact_manifest_format_globals_json(manifest, &object));
  return loom_json_object_end(&object);
}

iree_status_t loom_target_artifact_manifest_format_json_bytes(
    const loom_target_artifact_manifest_t* manifest,
    const loom_target_artifact_manifest_format_options_t* options,
    iree_allocator_t allocator,
    loom_target_artifact_manifest_json_t* out_json) {
  if (out_json == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact manifest JSON output is NULL");
  }
  *out_json = (loom_target_artifact_manifest_json_t){0};

  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  iree_status_t status =
      loom_target_artifact_manifest_format_json(manifest, options, &stream);
  if (iree_status_is_ok(status)) {
    const iree_host_size_t storage_length = iree_string_builder_size(&builder);
    char* storage = iree_string_builder_take_storage(&builder);
    out_json->contents =
        iree_make_const_byte_span((const uint8_t*)storage, storage_length);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

void loom_target_artifact_manifest_json_release(
    loom_target_artifact_manifest_json_t* json, iree_allocator_t allocator) {
  if (json == NULL) {
    return;
  }
  iree_allocator_free(allocator, (void*)json->contents.data);
  *json = (loom_target_artifact_manifest_json_t){0};
}
