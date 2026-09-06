// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable_global_resolver.h"

#include "iree/hal/drivers/amdgpu/buffer.h"

#define IREE_HAL_AMDGPU_MAX_STACK_GLOBAL_NAME_LENGTH \
  ((iree_host_size_t)(4 * 1024))

static iree_status_t
iree_hal_amdgpu_executable_global_resolver_try_get_symbol_by_name(
    const iree_hal_amdgpu_executable_global_resolver_t* resolver,
    iree_string_view_t name, bool* out_found,
    hsa_executable_symbol_t* out_symbol) {
  *out_found = false;
  memset(out_symbol, 0, sizeof(*out_symbol));
  if (iree_string_view_is_empty(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable global name is empty");
  }
  if (name.size > IREE_HAL_AMDGPU_MAX_STACK_GLOBAL_NAME_LENGTH) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "executable global name `%.*s` exceeds maximum length %" PRIhsz,
        (int)name.size, name.data,
        IREE_HAL_AMDGPU_MAX_STACK_GLOBAL_NAME_LENGTH);
  }

  char* name_storage = (char*)iree_alloca(name.size + 1);
  memcpy(name_storage, name.data, name.size);
  name_storage[name.size] = 0;
  hsa_status_t hsa_status = iree_hsa_executable_get_symbol_by_name_raw(
      resolver->libhsa, resolver->executable, name_storage,
      &resolver->device_agent, out_symbol);
  if (hsa_status == HSA_STATUS_SUCCESS) {
    *out_found = true;
    return iree_ok_status();
  }
  if (hsa_status == HSA_STATUS_ERROR_INVALID_SYMBOL_NAME) {
    return iree_ok_status();
  }
  return iree_status_from_hsa_status(
      __FILE__, __LINE__, hsa_status, "hsa_executable_get_symbol_by_name",
      "failed to resolve executable global symbol");
}

static iree_status_t
iree_hal_amdgpu_executable_global_resolver_try_query_variable(
    const iree_hal_amdgpu_executable_global_resolver_t* resolver,
    hsa_executable_symbol_t symbol, uint64_t* out_address, bool* out_found,
    iree_device_size_t* out_byte_length) {
  *out_found = false;
  if (out_address) *out_address = 0;
  *out_byte_length = 0;

  hsa_symbol_kind_t symbol_kind = HSA_SYMBOL_KIND_KERNEL;
  IREE_RETURN_IF_ERROR(iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(resolver->libhsa), symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE,
      &symbol_kind));
  if (symbol_kind != HSA_SYMBOL_KIND_VARIABLE) {
    return iree_ok_status();
  }

  if (out_address) {
    IREE_RETURN_IF_ERROR(iree_hsa_executable_symbol_get_info(
        IREE_LIBHSA(resolver->libhsa), symbol,
        HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, out_address));
  }

  uint32_t variable_size = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(resolver->libhsa), symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &variable_size));
  *out_byte_length = variable_size;
  *out_found = true;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_global_resolver_query_variable(
    const iree_hal_amdgpu_executable_global_resolver_t* resolver,
    hsa_executable_symbol_t symbol, iree_string_view_t name,
    iree_status_code_t wrong_kind_status_code, uint64_t* out_address,
    iree_device_size_t* out_byte_length) {
  bool found = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_global_resolver_try_query_variable(
          resolver, symbol, out_address, &found, out_byte_length));
  if (!found) {
    return iree_make_status(wrong_kind_status_code,
                            "executable global `%.*s` is not a variable on "
                            "physical device %" PRIhsz,
                            (int)name.size, name.data,
                            resolver->physical_device_ordinal);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_executable_global_resolver_try_verify(
    void* user_data, iree_string_view_t name, bool* out_found,
    iree_device_size_t* out_byte_length) {
  iree_hal_amdgpu_executable_global_resolver_t* resolver =
      (iree_hal_amdgpu_executable_global_resolver_t*)user_data;
  *out_found = false;
  *out_byte_length = 0;

  hsa_executable_symbol_t symbol = {0};
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_global_resolver_try_get_symbol_by_name(
          resolver, name, out_found, &symbol));
  if (!*out_found) return iree_ok_status();

  return iree_hal_amdgpu_executable_global_resolver_try_query_variable(
      resolver, symbol, /*out_address=*/NULL, out_found, out_byte_length);
}

static void iree_hal_amdgpu_executable_global_resolver_release_buffer(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)user_data;
  (void)buffer;
}

static iree_status_t iree_hal_amdgpu_executable_global_resolver_create_buffer(
    void* user_data, iree_string_view_t name,
    iree_device_size_t expected_byte_length, iree_hal_buffer_t** out_buffer) {
  iree_hal_amdgpu_executable_global_resolver_t* resolver =
      (iree_hal_amdgpu_executable_global_resolver_t*)user_data;
  *out_buffer = NULL;

  hsa_executable_symbol_t symbol = {0};
  bool found = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_global_resolver_try_get_symbol_by_name(
          resolver, name, &found, &symbol));
  if (IREE_UNLIKELY(!found)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "verified executable global `%.*s` disappeared on "
                            "physical device %" PRIhsz,
                            (int)name.size, name.data,
                            resolver->physical_device_ordinal);
  }

  uint64_t variable_address = 0;
  iree_device_size_t byte_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_global_resolver_query_variable(
          resolver, symbol, name, IREE_STATUS_INTERNAL, &variable_address,
          &byte_length));
  if (IREE_UNLIKELY(byte_length != expected_byte_length)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "verified executable global `%.*s` changed size on physical device "
        "%" PRIhsz " from %" PRIu64 " to %" PRIu64,
        (int)name.size, name.data, resolver->physical_device_ordinal,
        (uint64_t)expected_byte_length, (uint64_t)byte_length);
  }

  iree_hal_buffer_placement_t placement = {
      .device = resolver->device,
      .queue_family_affinity = iree_hal_make_queue_family_affinity(
          (iree_hal_queue_family_ordinal_t)resolver->physical_device_ordinal),
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  iree_hal_buffer_release_callback_t release_callback = {
      .fn = iree_hal_amdgpu_executable_global_resolver_release_buffer,
      .user_data = NULL,
  };
  return iree_hal_amdgpu_buffer_create(
      resolver->libhsa, placement, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_USAGE_DEFAULT,
      IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_NONE, expected_byte_length,
      expected_byte_length, (void*)(uintptr_t)variable_address,
      release_callback, resolver->host_allocator, out_buffer);
}

iree_status_t iree_hal_amdgpu_executable_global_resolver_initialize_table(
    iree_hal_amdgpu_executable_global_resolver_t* resolver,
    iree_hal_amdgpu_global_table_t* out_table) {
  const iree_hal_amdgpu_global_table_params_t table_params = {
      .host_allocator = resolver->host_allocator,
      .resolver =
          {
              .user_data = resolver,
              .try_verify =
                  iree_hal_amdgpu_executable_global_resolver_try_verify,
              .create_buffer =
                  iree_hal_amdgpu_executable_global_resolver_create_buffer,
          },
  };
  return iree_hal_amdgpu_global_table_initialize(&table_params, out_table);
}
