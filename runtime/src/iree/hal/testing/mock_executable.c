// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/testing/mock_executable.h"

#include <string.h>

static const iree_string_view_t iree_hal_mock_executable_format =
    IREE_SVL("mock-executable");
static const iree_string_view_t iree_hal_mock_executable_simple_format =
    IREE_SVL("mock");

typedef struct iree_hal_mock_executable_function_record_t {
  // Number of 32-bit constants reflected for the function.
  uint8_t constant_count;
  // Number of buffer bindings reflected for the function.
  uint8_t binding_count;
  // Executable function flags byte.
  uint8_t flags;
  // Static workgroup size reflected for the function.
  uint8_t workgroup_size[3];
  // Byte length of the function name in trailing storage.
  uint8_t name_length;
  // Reserved byte; must be zero.
  uint8_t reserved;
} iree_hal_mock_executable_function_record_t;

static const iree_hal_executable_vtable_t iree_hal_mock_executable_vtable;

typedef struct iree_hal_mock_executable_t {
  // Base HAL resource header.
  iree_hal_resource_t resource;
  // Allocator used to free the executable.
  iree_allocator_t host_allocator;
  // Number of function metadata records in functions.
  iree_host_size_t function_count;
  // Reflected function metadata stored in trailing storage.
  iree_hal_executable_function_info_t* functions;
  // Byte length of function_name_storage.
  iree_host_size_t function_name_storage_length;
  // Function names stored in trailing storage.
  char* function_name_storage;
} iree_hal_mock_executable_t;

static iree_hal_mock_executable_t* iree_hal_mock_executable_cast(
    iree_hal_executable_t* base_executable) {
  IREE_HAL_ASSERT_TYPE(base_executable, &iree_hal_mock_executable_vtable);
  return (iree_hal_mock_executable_t*)base_executable;
}

static iree_status_t iree_hal_mock_executable_create_with_function_count(
    iree_host_size_t function_count,
    iree_host_size_t function_name_storage_length,
    iree_allocator_t host_allocator,
    iree_hal_mock_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  iree_host_size_t functions_offset = 0;
  iree_host_size_t function_name_storage_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_mock_executable_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(
          function_count, iree_hal_executable_function_info_t,
          iree_alignof(iree_hal_executable_function_info_t), &functions_offset),
      IREE_STRUCT_FIELD(function_name_storage_length, char,
                        &function_name_storage_offset)));

  iree_hal_mock_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  memset(executable, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_mock_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->function_count = function_count;
  executable->functions =
      (iree_hal_executable_function_info_t*)((uint8_t*)executable +
                                             functions_offset);
  executable->function_name_storage_length = function_name_storage_length;
  executable->function_name_storage =
      (char*)executable + function_name_storage_offset;
  *out_executable = executable;
  return iree_ok_status();
}

static void iree_hal_mock_executable_set_default_functions(
    iree_hal_mock_executable_t* executable) {
  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    executable->functions[i].workgroup_size[0] = 1;
    executable->functions[i].workgroup_size[1] = 1;
    executable->functions[i].workgroup_size[2] = 1;
  }
}

static iree_status_t iree_hal_mock_executable_create_simple(
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  iree_hal_mock_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_mock_executable_create_with_function_count(
      /*function_count=*/1, /*function_name_storage_length=*/0, host_allocator,
      &executable));
  iree_hal_mock_executable_set_default_functions(executable);
  *out_executable = (iree_hal_executable_t*)executable;
  return iree_ok_status();
}

static iree_status_t iree_hal_mock_executable_create_from_metadata(
    const iree_hal_executable_load_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  if (IREE_UNLIKELY(executable_params->executable_data.data_length <
                    sizeof(uint32_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mock executable data is too short");
  }

  uint32_t function_count = 0;
  memcpy(&function_count, executable_params->executable_data.data,
         sizeof(function_count));
  iree_const_byte_span_t function_data = iree_make_const_byte_span(
      executable_params->executable_data.data + sizeof(function_count),
      executable_params->executable_data.data_length - sizeof(function_count));

  iree_host_size_t function_record_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          (iree_host_size_t)function_count,
          sizeof(iree_hal_mock_executable_function_record_t),
          &function_record_length))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "mock executable function metadata is too large");
  }
  if (IREE_UNLIKELY(function_data.data_length < function_record_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mock executable function metadata is too short");
  }

  const iree_hal_mock_executable_function_record_t* function_records =
      (const iree_hal_mock_executable_function_record_t*)function_data.data;
  iree_const_byte_span_t function_name_data = iree_make_const_byte_span(
      function_data.data + function_record_length,
      function_data.data_length - function_record_length);
  iree_host_size_t expected_function_name_length = 0;
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    const iree_hal_mock_executable_function_record_t* record =
        &function_records[i];
    if (IREE_UNLIKELY(record->reserved != 0)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "mock executable function metadata reserved byte "
                              "must be zero");
    }
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            expected_function_name_length, record->name_length,
            &expected_function_name_length))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "mock executable function names are too large");
    }
  }
  if (IREE_UNLIKELY(expected_function_name_length !=
                    function_name_data.data_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "mock executable function name storage length mismatch");
  }

  iree_hal_mock_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_mock_executable_create_with_function_count(
      function_count, function_name_data.data_length, host_allocator,
      &executable));
  if (function_name_data.data_length > 0) {
    memcpy(executable->function_name_storage, function_name_data.data,
           function_name_data.data_length);
  }
  iree_host_size_t function_name_offset = 0;
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    const iree_hal_mock_executable_function_record_t* record =
        &function_records[i];
    executable->functions[i].name = iree_make_string_view(
        executable->function_name_storage + function_name_offset,
        record->name_length);
    executable->functions[i].flags = record->flags;
    executable->functions[i].constant_byte_length =
        record->constant_count * sizeof(uint32_t);
    executable->functions[i].binding_count = record->binding_count;
    executable->functions[i].parameter_count = 0;
    executable->functions[i].workgroup_size[0] = record->workgroup_size[0];
    executable->functions[i].workgroup_size[1] = record->workgroup_size[1];
    executable->functions[i].workgroup_size[2] = record->workgroup_size[2];
    function_name_offset += record->name_length;
  }

  *out_executable = (iree_hal_executable_t*)executable;
  return iree_ok_status();
}

static void iree_hal_mock_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  iree_allocator_free(host_allocator, executable);
}

static iree_host_size_t iree_hal_mock_executable_function_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  return executable->function_count;
}

static iree_status_t iree_hal_mock_executable_function_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
  }
  *out_info =
      executable->functions[iree_hal_executable_function_index(function)];
  return iree_ok_status();
}

static iree_status_t iree_hal_mock_executable_function_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
  }
  (void)capacity;
  (void)out_parameters;
  return iree_ok_status();
}

static iree_status_t iree_hal_mock_executable_lookup_function_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  iree_hal_mock_executable_t* executable =
      iree_hal_mock_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    if (iree_string_view_equal(executable->functions[i].name, name)) {
      *out_function = iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  *out_function = iree_hal_executable_function_invalid();
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "no function named '%.*s' in executable",
                          (int)name.size, name.data);
}

static iree_status_t iree_hal_mock_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  (void)base_executable;
  (void)name;
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t iree_hal_mock_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  (void)base_executable;
  (void)global;
  memset(out_info, 0, sizeof(*out_info));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
}

static iree_status_t iree_hal_mock_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  (void)base_executable;
  (void)global;
  (void)queue_affinity;
  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
}

iree_status_t iree_hal_mock_executable_create(
    const iree_hal_executable_load_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  if (iree_string_view_equal(executable_params->executable_format,
                             iree_hal_mock_executable_simple_format)) {
    return iree_hal_mock_executable_create_simple(host_allocator,
                                                  out_executable);
  } else if (iree_string_view_equal(executable_params->executable_format,
                                    iree_hal_mock_executable_format)) {
    return iree_hal_mock_executable_create_from_metadata(
        executable_params, host_allocator, out_executable);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported mock executable format '%.*s'",
                          (int)executable_params->executable_format.size,
                          executable_params->executable_format.data);
}

static const iree_hal_executable_vtable_t iree_hal_mock_executable_vtable = {
    .destroy = iree_hal_mock_executable_destroy,
    .function_count = iree_hal_mock_executable_function_count,
    .function_info = iree_hal_mock_executable_function_info,
    .function_parameters = iree_hal_mock_executable_function_parameters,
    .lookup_function_by_name = iree_hal_mock_executable_lookup_function_by_name,
    .try_lookup_global_by_name =
        iree_hal_mock_executable_try_lookup_global_by_name,
    .global_info = iree_hal_mock_executable_global_info,
    .global_buffer = iree_hal_mock_executable_global_buffer,
};
