// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/executable/library/util.h"

iree_status_t iree_hal_executable_library_validate_query_result(
    const iree_hal_executable_library_header_t* const* query_result,
    const iree_hal_executable_library_v0_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  *out_library = NULL;
  if (!query_result || !*query_result) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable does not support this version of the runtime (%08X)",
        IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST);
  }
  const iree_hal_executable_library_header_t* header = *query_result;
  if (header->version != IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable library version %u does not match the supported version "
        "%u",
        header->version, IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST);
  }
  *out_library = iree_hal_executable_library_v0_from_query_result(query_result);
  return iree_ok_status();
}

iree_status_t iree_hal_executable_library_verify(
    const iree_hal_executable_load_params_t* load_params,
    const iree_hal_executable_library_v0_t* library) {
  // Tooling and testing may disable verification to make it easier to define
  // libraries. The compiler should never produce anything that fails
  // verification, though, and should always have it enabled.
  const bool disable_verification = iree_all_bits_set(
      load_params->flags, IREE_HAL_EXECUTABLE_LOAD_FLAG_DISABLE_VERIFICATION);
  if (disable_verification) return iree_ok_status();

  // Check to make sure that the constant table has values for all constants.
  if (library->constants.count != load_params->constant_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable requires %u constants but caller "
                            "provided %" PRIhsz "; must match",
                            library->constants.count,
                            load_params->constant_count);
  }

  // Validate that dispatch attributes are present.
  if (!library->exports.attrs) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable exports must provide dispatch attributes");
  }
  if (library->exports.count && !library->exports.names) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable exports must provide function names");
  }

  // Validate dispatch attributes are in range and names are usable.
  for (uint32_t i = 0; i < library->exports.count; ++i) {
    const iree_hal_executable_dispatch_attrs_v0_t dispatch_attrs =
        library->exports.attrs[i];
    if (dispatch_attrs.constant_byte_length >
        IREE_HAL_EXECUTABLE_MAX_CONSTANT_BYTE_LENGTH) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "dispatch requiring %u constant bytes exceeds limit of %" PRIhsz,
          dispatch_attrs.constant_byte_length,
          (iree_host_size_t)IREE_HAL_EXECUTABLE_MAX_CONSTANT_BYTE_LENGTH);
    }
    if (dispatch_attrs.binding_count > IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "dispatch requiring %u bindings exceeds limit of %d",
          dispatch_attrs.binding_count, IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT);
    }
    if (!library->exports.names[i] || !library->exports.names[i][0]) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "executable export %u missing function name", i);
    }
  }

  return iree_ok_status();
}

iree_host_size_t iree_hal_executable_library_export_count(
    const iree_hal_executable_library_v0_t* library) {
  IREE_ASSERT_ARGUMENT(library);
  return library->exports.count;
}

iree_status_t iree_hal_executable_library_export_info(
    const iree_hal_executable_library_v0_t* library,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_ASSERT_ARGUMENT(out_info);
  memset(out_info, 0, sizeof(*out_info));

  if (!iree_hal_executable_function_is_index_in_range(function,
                                                      library->exports.count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64 " out of range (count: %u)",
                            function.value, library->exports.count);
  }
  const uint32_t export_ordinal = iree_hal_executable_function_index(function);

  // Set the name if available.
  if (library->exports.names && library->exports.names[export_ordinal]) {
    out_info->name =
        iree_make_cstring_view(library->exports.names[export_ordinal]);
  } else {
    out_info->name = iree_string_view_empty();
  }

  // Get dispatch attributes if available.
  if (library->exports.attrs) {
    const iree_hal_executable_dispatch_attrs_v0_t* attrs =
        &library->exports.attrs[export_ordinal];

    if (iree_any_bit_set(attrs->flags,
                         IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_SEQUENTIAL)) {
      out_info->flags |= IREE_HAL_EXECUTABLE_FUNCTION_FLAG_SEQUENTIAL;
    }
    if (iree_any_bit_set(
            attrs->flags,
            IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_WORKGROUP_SIZE_DYNAMIC)) {
      out_info->flags |=
          IREE_HAL_EXECUTABLE_FUNCTION_FLAG_WORKGROUP_SIZE_DYNAMIC;
    }

    out_info->constant_byte_length = attrs->constant_byte_length;
    out_info->binding_count = attrs->binding_count;
    out_info->parameter_count = attrs->parameter_count;

    out_info->workgroup_size[0] = attrs->workgroup_size_x;
    out_info->workgroup_size[1] = attrs->workgroup_size_y;
    out_info->workgroup_size[2] = attrs->workgroup_size_z;

    out_info->resource_usage.provided_flags |=
        IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_WORKGROUP_LOCAL_MEMORY;
    out_info->resource_usage.fixed_workgroup_local_memory_size =
        (uint32_t)attrs->local_memory_pages *
        IREE_HAL_EXECUTABLE_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE;
  }

  // Occupancy info is not yet implemented.
  memset(&out_info->occupancy_info, 0, sizeof(out_info->occupancy_info));

  return iree_ok_status();
}

iree_status_t iree_hal_executable_library_export_parameters(
    const iree_hal_executable_library_v0_t* library,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_ASSERT_ARGUMENT(out_parameters || capacity == 0);

  if (!iree_hal_executable_function_is_index_in_range(function,
                                                      library->exports.count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64 " out of range (count: %u)",
                            function.value, library->exports.count);
  }
  const uint32_t export_ordinal = iree_hal_executable_function_index(function);

  if (!library->exports.attrs || !library->exports.params ||
      !library->exports.params[export_ordinal]) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "reflection data not available");
  }

  const iree_hal_executable_dispatch_attrs_v0_t* attrs =
      &library->exports.attrs[export_ordinal];
  const iree_hal_executable_dispatch_parameter_v0_t** params =
      library->exports.params;
  const iree_hal_executable_dispatch_parameter_v0_t* export_params =
      params[export_ordinal];

  // Copy as many parameters as we can fit.
  iree_host_size_t count =
      iree_min(capacity, (iree_host_size_t)attrs->parameter_count);
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_hal_executable_dispatch_parameter_v0_t* src = &export_params[i];
    iree_hal_executable_function_parameter_t* dst = &out_parameters[i];
    memset(dst, 0, sizeof(*dst));
    switch (src->type) {
      case IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT:
        dst->type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT;
        break;
      case IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING:
        dst->type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING;
        break;
      case IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BUFFER_PTR:
        dst->type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BUFFER_PTR;
        break;
      default:
        dst->type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT;
        break;
    }
    dst->size = src->size;
    dst->flags = src->flags;
    dst->offset = src->offset;
    if (library->exports.parameter_names && src->name < UINT16_MAX &&
        library->exports.parameter_names[src->name]) {
      dst->name =
          iree_make_cstring_view(library->exports.parameter_names[src->name]);
    } else {
      dst->name = iree_string_view_empty();
    }
  }

  return iree_ok_status();
}

iree_status_t iree_hal_executable_library_lookup_export_by_name(
    const iree_hal_executable_library_v0_t* library, iree_string_view_t name,
    iree_hal_executable_function_t* out_export_ordinal) {
  IREE_ASSERT_ARGUMENT(library);
  IREE_ASSERT_ARGUMENT(out_export_ordinal);

  if (!library->exports.names) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "function names not available in library");
  }

  // Linear search through function names.
  for (uint32_t i = 0; i < library->exports.count; ++i) {
    if (library->exports.names[i]) {
      iree_string_view_t export_name =
          iree_make_cstring_view(library->exports.names[i]);
      if (iree_string_view_equal(export_name, name)) {
        *out_export_ordinal = iree_hal_executable_function_from_index(i);
        return iree_ok_status();
      }
    }
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "export '%.*s' not found in library", (int)name.size,
                          name.data);
}

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

void iree_hal_executable_library_publish_source_files(
    const iree_hal_executable_library_v0_t* library) {
  for (uint32_t i = 0; i < library->sources.count; ++i) {
    const iree_hal_executable_source_file_v0_t* source_file =
        &library->sources.files[i];
    IREE_TRACE_PUBLISH_SOURCE_FILE(source_file->path, source_file->path_length,
                                   source_file->content,
                                   source_file->content_length);
  }
}

iree_zone_id_t iree_hal_executable_library_call_zone_begin(
    iree_string_view_t executable_identifier,
    const iree_hal_executable_library_v0_t* library, iree_host_size_t ordinal) {
  iree_string_view_t entry_point_name = iree_string_view_empty();
  if (library->exports.names != NULL && library->exports.names[ordinal]) {
    entry_point_name = iree_make_cstring_view(library->exports.names[ordinal]);
  }
  if (iree_string_view_is_empty(entry_point_name)) {
    entry_point_name = iree_make_cstring_view("unknown_dylib_call");
  }

  const char* source_file = NULL;
  size_t source_file_length = 0;
  uint32_t source_line = 0;
  if (library->exports.stage_locations != NULL) {
    for (uint32_t i = 0; i < library->exports.stage_locations->count; ++i) {
      // TODO(benvanik): a way to select what location is chosen. For now we
      // just pick the first one.
      // const char* name = library->exports.stage_locations->names[i];
      const iree_hal_executable_source_location_v0_t* location =
          &library->exports.stage_locations->locations[i];
      source_file = location->path;
      source_file_length = location->path_length;
      source_line = location->line;
      break;
    }
  }
  if (source_file == NULL) {
    if (library->exports.source_locations != NULL) {
      // We have source location data, so use it.
      const iree_hal_executable_source_location_v0_t* location =
          &library->exports.source_locations[ordinal];
      source_file = location->path;
      source_file_length = location->path_length;
      source_line = location->line;
    } else {
      // No source location data, so make do with what we have.
      source_file = executable_identifier.data;
      source_file_length = executable_identifier.size;
      source_line = ordinal;
    }
  }

  IREE_TRACE_ZONE_BEGIN_EXTERNAL(z0, source_file, source_file_length,
                                 source_line, entry_point_name.data,
                                 entry_point_name.size, NULL, 0);

  if (library->exports.tags != NULL) {
    const char* tag = library->exports.tags[ordinal];
    if (tag) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, tag);
    }
  }

  return z0;
}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
