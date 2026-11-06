// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/wasm/host.h"

#include <stddef.h>

enum {
  LOOM_WASM_HOST_OUTCOME_OK = 0,
  LOOM_WASM_HOST_OUTCOME_ERROR = 1,
  LOOM_WASM_HOST_OUTCOME_TRAP = 2,
};

#if defined(IREE_PLATFORM_WASM)

static_assert(sizeof(loom_wasm_host_memory_region_t) == 12,
              "host memory region ABI must remain stable on wasm32");
static_assert(sizeof(loom_wasm_host_value_type_t) == 4,
              "host value type ABI must remain stable on wasm32");
static_assert(offsetof(loom_wasm_host_memory_region_t, address) == 0,
              "host memory region address offset must remain stable");
static_assert(offsetof(loom_wasm_host_memory_region_t, data) == 4,
              "host memory region data offset must remain stable");
static_assert(offsetof(loom_wasm_host_memory_region_t, data_length) == 8,
              "host memory region length offset must remain stable");

__attribute__((import_module("loom_wasm_host"),
               import_name("module_load"))) extern uint32_t
loom_wasm_host_import_module_load(
    const uint8_t* module_data, uint32_t module_length,
    const char* function_name, uint32_t function_name_length,
    loom_wasm_host_value_type_t* out_parameter_types, uint32_t parameter_count,
    loom_wasm_host_value_type_t* out_result_types, uint32_t result_count,
    loom_wasm_host_module_flags_t* out_flags);

__attribute__((import_module("loom_wasm_host"),
               import_name("module_call"))) extern uint32_t
loom_wasm_host_import_module_call(uint32_t module_handle,
                                  const uint64_t* argument_bits,
                                  uint64_t* result_bits,
                                  const loom_wasm_host_memory_region_t* regions,
                                  uint32_t region_count);

__attribute__((import_module("loom_wasm_host"),
               import_name("module_release"))) extern void
loom_wasm_host_import_module_release(uint32_t module_handle);

__attribute__((import_module("loom_wasm_host"),
               import_name("error_length"))) extern uint32_t
loom_wasm_host_import_error_length(void);

__attribute__((import_module("loom_wasm_host"),
               import_name("error_copy"))) extern uint32_t
loom_wasm_host_import_error_copy(char* buffer, uint32_t capacity);

static bool loom_wasm_host_type_is_callable(loom_wasm_host_value_type_t type) {
  return type == LOOM_WASM_HOST_VALUE_TYPE_I32 ||
         type == LOOM_WASM_HOST_VALUE_TYPE_I64 ||
         type == LOOM_WASM_HOST_VALUE_TYPE_F32 ||
         type == LOOM_WASM_HOST_VALUE_TYPE_F64;
}

static iree_status_t loom_wasm_host_validate_type_list(
    const loom_wasm_host_value_type_t* types, uint32_t type_count,
    const char* list_name) {
  if (type_count != 0 && types == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm host %s type list is missing", list_name);
  }
  for (uint32_t i = 0; i < type_count; ++i) {
    if (!loom_wasm_host_type_is_callable(types[i])) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Wasm host %s %u has non-callable physical type 0x%02X", list_name, i,
          (unsigned)types[i]);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_host_copy_error(
    iree_allocator_t allocator, char** out_storage,
    iree_string_view_t* out_message) {
  *out_storage = NULL;
  *out_message = IREE_SV("WebAssembly host returned no error detail");
  const uint32_t message_length = loom_wasm_host_import_error_length();
  if (message_length == 0) {
    return iree_ok_status();
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      allocator, message_length, (void**)&storage));
  const uint32_t copied_length =
      loom_wasm_host_import_error_copy(storage, message_length);
  if (copied_length != message_length) {
    iree_allocator_free(allocator, storage);
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "WebAssembly host error changed from %u to %u bytes while copying",
        message_length, copied_length);
  }
  *out_storage = storage;
  *out_message = iree_make_string_view(storage, message_length);
  return iree_ok_status();
}

iree_status_t loom_wasm_host_module_load(
    iree_const_byte_span_t module_data, iree_string_view_t function_export_name,
    uint32_t parameter_count, uint32_t result_count,
    loom_wasm_host_value_type_t* out_parameter_types,
    loom_wasm_host_value_type_t* out_result_types,
    loom_wasm_host_module_t* out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = (loom_wasm_host_module_t){0};
  if (module_data.data_length > UINT32_MAX ||
      function_export_name.size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm host module metadata exceeds u32");
  }
  if (module_data.data_length != 0 && module_data.data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm host module data is missing");
  }
  if (iree_string_view_is_empty(function_export_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm host function export name is required");
  }
  if ((parameter_count != 0 && out_parameter_types == NULL) ||
      (result_count != 0 && out_result_types == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm host physical type storage is incomplete");
  }

  loom_wasm_host_module_flags_t flags = 0;
  const uint32_t handle = loom_wasm_host_import_module_load(
      module_data.data, (uint32_t)module_data.data_length,
      function_export_name.data, (uint32_t)function_export_name.size,
      out_parameter_types, parameter_count, out_result_types, result_count,
      &flags);
  if (handle == 0) {
    const iree_allocator_t allocator = iree_allocator_system();
    char* message_storage = NULL;
    iree_string_view_t message = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        loom_wasm_host_copy_error(allocator, &message_storage, &message));
    iree_status_t status = iree_status_allocate(IREE_STATUS_INVALID_ARGUMENT,
                                                __FILE__, __LINE__, message);
    iree_allocator_free(allocator, message_storage);
    return status;
  }

  const loom_wasm_host_module_flags_t known_flags =
      LOOM_WASM_HOST_MODULE_FLAG_HAS_MEMORY;
  iree_status_t status =
      iree_any_bit_set(flags, ~known_flags)
          ? iree_make_status(IREE_STATUS_DATA_LOSS,
                             "WebAssembly host returned unknown flags 0x%08X",
                             (unsigned)flags)
          : iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = loom_wasm_host_validate_type_list(out_parameter_types,
                                               parameter_count, "parameter");
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_host_validate_type_list(out_result_types, result_count,
                                               "result");
  }
  if (iree_status_is_ok(status)) {
    *out_module = (loom_wasm_host_module_t){
        .handle = handle,
        .parameter_count = parameter_count,
        .result_count = result_count,
        .flags = flags,
    };
  } else {
    loom_wasm_host_import_module_release(handle);
  }
  return status;
}

void loom_wasm_host_module_release(loom_wasm_host_module_t* module) {
  if (module == NULL || module->handle == 0) {
    return;
  }
  loom_wasm_host_import_module_release(module->handle);
  *module = (loom_wasm_host_module_t){0};
}

iree_status_t loom_wasm_host_module_call(
    const loom_wasm_host_module_t* module, const uint64_t* argument_bits,
    uint64_t* result_bits, iree_host_size_t region_count,
    const loom_wasm_host_memory_region_t* regions) {
  IREE_ASSERT_ARGUMENT(module);
  if (module->handle == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm host module is not loaded");
  }
  if ((module->parameter_count != 0 && argument_bits == NULL) ||
      (module->result_count != 0 && result_bits == NULL) ||
      (region_count != 0 && regions == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Wasm host call storage is incomplete");
  }
  if (region_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm host memory region count exceeds u32");
  }
  if (region_count != 0 &&
      !iree_any_bit_set(module->flags, LOOM_WASM_HOST_MODULE_FLAG_HAS_MEMORY)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Wasm host call provides roots to a module without memory");
  }
  const uint32_t outcome = loom_wasm_host_import_module_call(
      module->handle, argument_bits, result_bits, regions,
      (uint32_t)region_count);
  if (outcome == LOOM_WASM_HOST_OUTCOME_OK) {
    return iree_ok_status();
  }
  if (outcome != LOOM_WASM_HOST_OUTCOME_ERROR &&
      outcome != LOOM_WASM_HOST_OUTCOME_TRAP) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "WebAssembly host returned unknown outcome %u",
                            outcome);
  }
  const iree_allocator_t allocator = iree_allocator_system();
  char* message_storage = NULL;
  iree_string_view_t message = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_wasm_host_copy_error(allocator, &message_storage, &message));
  const iree_status_code_t code = outcome == LOOM_WASM_HOST_OUTCOME_TRAP
                                      ? IREE_STATUS_ABORTED
                                      : IREE_STATUS_INVALID_ARGUMENT;
  iree_status_t status =
      iree_status_allocate(code, __FILE__, __LINE__, message);
  iree_allocator_free(allocator, message_storage);
  return status;
}

#else

iree_status_t loom_wasm_host_module_load(
    iree_const_byte_span_t module_data, iree_string_view_t function_export_name,
    uint32_t parameter_count, uint32_t result_count,
    loom_wasm_host_value_type_t* out_parameter_types,
    loom_wasm_host_value_type_t* out_result_types,
    loom_wasm_host_module_t* out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = (loom_wasm_host_module_t){0};
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "synchronous WebAssembly module hosting requires a Wasm host");
}

void loom_wasm_host_module_release(loom_wasm_host_module_t* module) {
  if (module != NULL) {
    *module = (loom_wasm_host_module_t){0};
  }
}

iree_status_t loom_wasm_host_module_call(
    const loom_wasm_host_module_t* module, const uint64_t* argument_bits,
    uint64_t* result_bits, iree_host_size_t region_count,
    const loom_wasm_host_memory_region_t* regions) {
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "synchronous WebAssembly module hosting requires a Wasm host");
}

#endif
