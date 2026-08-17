// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <limits.h>
#include <stdint.h>
#include <string.h>

#include "binding/hip/api.h"
#include "binding/hip/binding_internal.h"
#include "common/module.h"
#include "iree/base/threading/call_once.h"

typedef struct iree_hip_library_kernel_t {
  // Streaming symbol represented by this kernel handle. Unowned.
  iree_hal_streaming_symbol_t* symbol;
  // NUL-terminated name valid until the owning library is unloaded.
  char* name;
} iree_hip_library_kernel_t;

struct hipLibrary_st {
  // Retains the wrapper while an API query is in progress.
  iree_atomic_ref_count_t ref_count;
  // Next live library in the process registry. Guarded by the registry mutex.
  struct hipLibrary_st* next;
  // Loaded streaming module, or NULL when loading was deferred with an error.
  iree_hal_streaming_module_t* module;
  // Error reported by the first query after a deferred library load.
  hipError_t deferred_load_error;
  // Function records in module export order.
  iree_hip_library_kernel_t* kernels;
  // Number of entries in |kernels|.
  unsigned int kernel_count;
  // Contiguous storage backing all kernel names.
  char* kernel_name_storage;
  // Allocator used for the wrapper and its owned storage.
  iree_allocator_t host_allocator;
};

static iree_once_flag iree_hip_library_registry_once = IREE_ONCE_FLAG_INIT;
static iree_slim_mutex_t iree_hip_library_registry_mutex;
static hipLibrary_t iree_hip_library_registry_head;

static void iree_hip_library_registry_initialize(void) {
  iree_slim_mutex_initialize(&iree_hip_library_registry_mutex);
}

static void iree_hip_library_registry_ensure_initialized(void) {
  iree_call_once(&iree_hip_library_registry_once,
                 iree_hip_library_registry_initialize);
}

static void iree_hip_library_destroy(hipLibrary_t library) {
  iree_hal_streaming_module_release(library->module);
  iree_allocator_free(library->host_allocator, library->kernel_name_storage);
  iree_allocator_free(library->host_allocator, library->kernels);
  iree_allocator_free(library->host_allocator, library);
}

static void iree_hip_library_retain(hipLibrary_t library) {
  iree_atomic_ref_count_inc(&library->ref_count);
}

static void iree_hip_library_release(hipLibrary_t library) {
  if (library && iree_atomic_ref_count_dec(&library->ref_count) == 1) {
    iree_hip_library_destroy(library);
  }
}

static iree_status_t iree_hip_library_create(
    iree_hal_streaming_module_t* module, hipError_t deferred_load_error,
    hipLibrary_t* out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  *out_library = NULL;

  iree_allocator_t host_allocator = iree_allocator_system();
  hipLibrary_t library = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*library),
                                             (void**)&library));
  memset(library, 0, sizeof(*library));
  iree_atomic_ref_count_init(&library->ref_count);
  library->module = module;
  library->deferred_load_error = deferred_load_error;
  library->host_allocator = host_allocator;

  iree_status_t status = iree_ok_status();
  iree_host_size_t kernel_count = 0;
  iree_host_size_t kernel_name_storage_size = 0;
  if (module) {
    for (iree_host_size_t i = 0; i < module->symbol_count; ++i) {
      const iree_hal_streaming_symbol_t* symbol = &module->symbols[i];
      if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) continue;
      iree_host_size_t name_size = 0;
      if (IREE_UNLIKELY(
              kernel_count == UINT_MAX ||
              !iree_host_size_checked_add(symbol->name.size, 1, &name_size) ||
              !iree_host_size_checked_add(kernel_name_storage_size, name_size,
                                          &kernel_name_storage_size))) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "library kernel metadata size overflow");
        break;
      }
      ++kernel_count;
    }
  }

  iree_host_size_t kernel_records_size = 0;
  if (iree_status_is_ok(status) && kernel_count > 0 &&
      IREE_UNLIKELY(!iree_host_size_checked_mul(
          kernel_count, sizeof(*library->kernels), &kernel_records_size))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "library kernel record size overflow");
  }
  if (iree_status_is_ok(status) && kernel_records_size > 0) {
    status = iree_allocator_malloc(host_allocator, kernel_records_size,
                                   (void**)&library->kernels);
  }
  if (iree_status_is_ok(status) && kernel_name_storage_size > 0) {
    status = iree_allocator_malloc(host_allocator, kernel_name_storage_size,
                                   (void**)&library->kernel_name_storage);
  }

  if (iree_status_is_ok(status) && module) {
    char* next_name = library->kernel_name_storage;
    unsigned int kernel_ordinal = 0;
    for (iree_host_size_t i = 0; i < module->symbol_count; ++i) {
      iree_hal_streaming_symbol_t* symbol = &module->symbols[i];
      if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) continue;
      library->kernels[kernel_ordinal].symbol = symbol;
      library->kernels[kernel_ordinal].name = next_name;
      memcpy(next_name, symbol->name.data, symbol->name.size);
      next_name[symbol->name.size] = '\0';
      next_name += symbol->name.size + 1;
      ++kernel_ordinal;
    }
    library->kernel_count = (unsigned int)kernel_count;
  }

  if (iree_status_is_ok(status)) {
    *out_library = library;
  } else {
    library->module = NULL;
    iree_hip_library_destroy(library);
  }
  return status;
}

static void iree_hip_library_registry_insert(hipLibrary_t library) {
  iree_hip_library_registry_ensure_initialized();
  iree_slim_mutex_lock(&iree_hip_library_registry_mutex);
  library->next = iree_hip_library_registry_head;
  iree_hip_library_registry_head = library;
  iree_slim_mutex_unlock(&iree_hip_library_registry_mutex);
}

static hipError_t iree_hip_library_registry_lookup(hipLibrary_t handle,
                                                   hipLibrary_t* out_library) {
  *out_library = NULL;
  if (!handle) return hipErrorInvalidResourceHandle;

  iree_hip_library_registry_ensure_initialized();
  iree_slim_mutex_lock(&iree_hip_library_registry_mutex);
  for (hipLibrary_t library = iree_hip_library_registry_head; library;
       library = library->next) {
    if (library != handle) continue;
    iree_hip_library_retain(library);
    *out_library = library;
    break;
  }
  iree_slim_mutex_unlock(&iree_hip_library_registry_mutex);
  return *out_library ? hipSuccess : hipErrorInvalidResourceHandle;
}

static hipError_t iree_hip_library_registry_remove(hipLibrary_t handle,
                                                   hipLibrary_t* out_library) {
  *out_library = NULL;
  if (!handle) return hipErrorInvalidResourceHandle;

  iree_hip_library_registry_ensure_initialized();
  iree_slim_mutex_lock(&iree_hip_library_registry_mutex);
  hipLibrary_t* link = &iree_hip_library_registry_head;
  while (*link && *link != handle) link = &(*link)->next;
  if (*link) {
    *out_library = *link;
    *link = (*link)->next;
    (*out_library)->next = NULL;
  }
  iree_slim_mutex_unlock(&iree_hip_library_registry_mutex);
  return *out_library ? hipSuccess : hipErrorInvalidResourceHandle;
}

static hipError_t iree_hip_library_acquire_ready(hipLibrary_t handle,
                                                 hipLibrary_t* out_library) {
  hipError_t result = iree_hip_library_registry_lookup(handle, out_library);
  if (result != hipSuccess) return result;
  if ((*out_library)->deferred_load_error == hipSuccess) return hipSuccess;
  result = (*out_library)->deferred_load_error;
  iree_hip_library_release(*out_library);
  *out_library = NULL;
  return result;
}

// Kernel handles are tagged streaming symbols and deliberately remain direct
// handles on the launch path. Query APIs recover their library ownership from
// the cold-path registry before dereferencing them, which also rejects stale or
// fabricated handles without adding work to dispatch.
static hipError_t iree_hip_library_acquire_for_kernel(
    hipKernel_t kernel, hipError_t null_error, hipLibrary_t* out_library,
    iree_hal_streaming_symbol_t** out_symbol) {
  *out_library = NULL;
  *out_symbol = NULL;
  if (!kernel) return null_error;
  if (!iree_hal_streaming_symbol_has_tag(kernel)) return hipErrorInvalidHandle;

  const uintptr_t candidate =
      (uintptr_t)iree_hal_streaming_symbol_untag(kernel);
  iree_hip_library_registry_ensure_initialized();
  iree_slim_mutex_lock(&iree_hip_library_registry_mutex);
  for (hipLibrary_t library = iree_hip_library_registry_head; library;
       library = library->next) {
    iree_hal_streaming_module_t* module = library->module;
    if (!module || module->symbol_count == 0) continue;
    if (module->symbol_count > UINTPTR_MAX / sizeof(*module->symbols)) continue;
    const uintptr_t begin = (uintptr_t)module->symbols;
    const uintptr_t byte_length =
        module->symbol_count * sizeof(*module->symbols);
    if (begin > UINTPTR_MAX - byte_length) continue;
    const uintptr_t end = begin + byte_length;
    if (candidate < begin || candidate >= end ||
        (candidate - begin) % sizeof(*module->symbols) != 0) {
      continue;
    }
    iree_hip_library_retain(library);
    *out_library = library;
    *out_symbol = (iree_hal_streaming_symbol_t*)candidate;
    break;
  }
  iree_slim_mutex_unlock(&iree_hip_library_registry_mutex);

  if (!*out_symbol ||
      (*out_symbol)->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    iree_hip_library_release(*out_library);
    *out_library = NULL;
    *out_symbol = NULL;
    return hipErrorInvalidHandle;
  }
  return hipSuccess;
}

static hipError_t iree_hip_library_validate_options(const void* options,
                                                    void* const* option_values,
                                                    unsigned int option_count) {
  if (option_count == 0) return hipSuccess;
  if (!options || !option_values) return hipErrorInvalidValue;
  return hipErrorNotSupported;
}

static hipError_t iree_hip_library_u32_attribute(uint32_t attribute,
                                                 int* out_value) {
  if (attribute > INT_MAX) return hipErrorInvalidValue;
  *out_value = (int)attribute;
  return hipSuccess;
}

static iree_status_t iree_hip_library_try_get_managed_global(
    hipLibrary_t library, const char* name, bool* out_found,
    void** out_host_pointer, size_t* out_byte_count) {
  *out_found = false;
  *out_host_pointer = NULL;
  *out_byte_count = 0;

  static const char suffix[] = ".managed";
  const iree_host_size_t name_size = strlen(name);
  iree_host_size_t managed_name_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(name_size, sizeof(suffix),
                                                &managed_name_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "managed global name length overflow");
  }

  char* managed_name = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      library->host_allocator, managed_name_size, (void**)&managed_name));
  memcpy(managed_name, name, name_size);
  memcpy(managed_name + name_size, suffix, sizeof(suffix));

  iree_device_size_t byte_count = 0;
  iree_status_t status =
      iree_hal_streaming_module_try_initialize_managed_global(
          library->module, name, managed_name, out_found, out_host_pointer,
          &byte_count);
  iree_allocator_free(library->host_allocator, managed_name);
  if (iree_status_is_ok(status) && *out_found) {
    if (IREE_UNLIKELY((iree_device_size_t)(size_t)byte_count != byte_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "managed global size exceeds size_t");
    }
    *out_byte_count = (size_t)byte_count;
  }
  return status;
}

static hipError_t iree_hip_library_wrap_loaded_module(
    hipLibrary_t* out_library, hipModule_t module,
    hipError_t deferred_load_error) {
  hipLibrary_t library = NULL;
  iree_status_t status = iree_hip_library_create(
      (iree_hal_streaming_module_t*)module, deferred_load_error, &library);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_module_release((iree_hal_streaming_module_t*)module);
    return iree_hip_status_to_result(status);
  }
  iree_hip_library_registry_insert(library);
  *out_library = library;
  return hipSuccess;
}

HIPAPI hipError_t hipLibraryLoadData(hipLibrary_t* library, const void* code,
                                     hipJitOption* jit_options,
                                     void** jit_option_values,
                                     unsigned int jit_option_count,
                                     hipLibraryOption* library_options,
                                     void** library_option_values,
                                     unsigned int library_option_count) {
  if (!library || !code) return hipErrorInvalidValue;
  *library = NULL;
  hipError_t result = iree_hip_library_validate_options(
      library_options, library_option_values, library_option_count);
  if (result != hipSuccess) return result;

  hipModule_t module = NULL;
  result = hipModuleLoadDataEx(&module, code, jit_option_count, jit_options,
                               jit_option_values);
  if (result == hipSuccess) {
    return iree_hip_library_wrap_loaded_module(library, module, hipSuccess);
  }
  // Library data loading is lazy: malformed images produce a valid library
  // handle and report the image error on the first operation that needs code.
  if (result == hipErrorInvalidImage) {
    return iree_hip_library_wrap_loaded_module(library, NULL,
                                               hipErrorInvalidImage);
  }
  return result;
}

HIPAPI hipError_t hipLibraryLoadFromFile(
    hipLibrary_t* library, const char* file_name, hipJitOption* jit_options,
    void** jit_option_values, unsigned int jit_option_count,
    hipLibraryOption* library_options, void** library_option_values,
    unsigned int library_option_count) {
  if (!library || !file_name || !file_name[0]) return hipErrorInvalidValue;
  *library = NULL;
  hipError_t result = iree_hip_library_validate_options(
      jit_options, jit_option_values, jit_option_count);
  if (result != hipSuccess) return result;
  result = iree_hip_library_validate_options(
      library_options, library_option_values, library_option_count);
  if (result != hipSuccess) return result;

  hipModule_t module = NULL;
  result = hipModuleLoad(&module, file_name);
  if (result != hipSuccess) return result;
  return iree_hip_library_wrap_loaded_module(library, module, hipSuccess);
}

HIPAPI hipError_t hipLibraryUnload(hipLibrary_t library) {
  hipLibrary_t removed_library = NULL;
  hipError_t result =
      iree_hip_library_registry_remove(library, &removed_library);
  if (result != hipSuccess) return result;

  if (removed_library->module) {
    iree_status_t status = iree_hal_streaming_context_synchronize(
        removed_library->module->context);
    result = iree_hip_status_to_result(status);
  }
  iree_hip_library_release(removed_library);
  return result;
}

HIPAPI hipError_t hipLibraryGetKernel(hipKernel_t* kernel, hipLibrary_t library,
                                      const char* name) {
  if (!kernel || !name || !name[0]) return hipErrorInvalidValue;
  hipLibrary_t retained_library = NULL;
  hipError_t result =
      iree_hip_library_acquire_ready(library, &retained_library);
  if (result != hipSuccess) return result;

  iree_hal_streaming_symbol_t* symbol = NULL;
  iree_status_t status = iree_hal_streaming_module_function(
      retained_library->module, name, &symbol);
  if (iree_status_is_ok(status)) {
    *kernel = (hipKernel_t)iree_hal_streaming_symbol_tag(symbol);
  }
  result = iree_hip_status_to_result(status);
  iree_hip_library_release(retained_library);
  return result;
}

HIPAPI hipError_t hipLibraryGetKernelCount(unsigned int* count,
                                           hipLibrary_t library) {
  if (!count) return hipErrorInvalidValue;
  hipLibrary_t retained_library = NULL;
  hipError_t result =
      iree_hip_library_acquire_ready(library, &retained_library);
  if (result != hipSuccess) return result;
  *count = retained_library->kernel_count;
  iree_hip_library_release(retained_library);
  return hipSuccess;
}

HIPAPI hipError_t hipLibraryEnumerateKernels(hipKernel_t* kernels,
                                             unsigned int kernel_count,
                                             hipLibrary_t library) {
  if (kernel_count > 0 && !kernels) return hipErrorInvalidValue;
  hipLibrary_t retained_library = NULL;
  hipError_t result =
      iree_hip_library_acquire_ready(library, &retained_library);
  if (result != hipSuccess) return result;

  const unsigned int write_count =
      iree_min(kernel_count, retained_library->kernel_count);
  for (unsigned int i = 0; i < write_count; ++i) {
    kernels[i] = (hipKernel_t)iree_hal_streaming_symbol_tag(
        retained_library->kernels[i].symbol);
  }
  iree_hip_library_release(retained_library);
  return hipSuccess;
}

HIPAPI hipError_t hipLibraryGetGlobal(void** device_pointer, size_t* byte_count,
                                      hipLibrary_t library, const char* name) {
  if ((!device_pointer && !byte_count) || !name || !name[0]) {
    return hipErrorInvalidValue;
  }
  hipLibrary_t retained_library = NULL;
  hipError_t result =
      iree_hip_library_acquire_ready(library, &retained_library);
  if (result != hipSuccess) return result;

  bool managed_found = false;
  void* managed_pointer = NULL;
  size_t managed_size = 0;
  iree_status_t status = iree_hip_library_try_get_managed_global(
      retained_library, name, &managed_found, &managed_pointer, &managed_size);
  if (iree_status_is_ok(status) && managed_found) {
    if (device_pointer) *device_pointer = managed_pointer;
    if (byte_count) *byte_count = managed_size;
  }
  iree_hal_streaming_deviceptr_t address = 0;
  iree_device_size_t size = 0;
  if (iree_status_is_ok(status) && !managed_found) {
    status = iree_hal_streaming_module_global(retained_library->module, name,
                                              &address, &size);
  }
  if (iree_status_is_ok(status) && !managed_found) {
    if (device_pointer) *device_pointer = (void*)(uintptr_t)address;
    if (byte_count) *byte_count = (size_t)size;
  }
  result = iree_hip_status_to_result(status);
  iree_hip_library_release(retained_library);
  return result;
}

HIPAPI hipError_t hipLibraryGetManaged(void** host_pointer, size_t* byte_count,
                                       hipLibrary_t library, const char* name) {
  if ((!host_pointer && !byte_count) || !name || !name[0]) {
    return hipErrorInvalidValue;
  }
  hipLibrary_t retained_library = NULL;
  hipError_t result =
      iree_hip_library_acquire_ready(library, &retained_library);
  if (result != hipSuccess) return result;

  bool found = false;
  void* managed_pointer = NULL;
  size_t managed_size = 0;
  iree_status_t status = iree_hip_library_try_get_managed_global(
      retained_library, name, &found, &managed_pointer, &managed_size);
  if (iree_status_is_ok(status) && !found) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "managed global '%s' not found", name);
  }
  if (iree_status_is_ok(status)) {
    if (host_pointer) *host_pointer = managed_pointer;
    if (byte_count) *byte_count = managed_size;
  }

  result = iree_hip_status_to_result(status);
  iree_hip_library_release(retained_library);
  return result;
}

HIPAPI hipError_t hipKernelGetFunction(hipFunction_t* function,
                                       hipKernel_t kernel) {
  if (!function) return hipErrorInvalidValue;
  hipLibrary_t library = NULL;
  iree_hal_streaming_symbol_t* symbol = NULL;
  hipError_t result = iree_hip_library_acquire_for_kernel(
      kernel, hipErrorInvalidValue, &library, &symbol);
  if (result != hipSuccess) return result;
  *function = (hipFunction_t)iree_hal_streaming_symbol_tag(symbol);
  iree_hip_library_release(library);
  return hipSuccess;
}

HIPAPI hipError_t hipKernelGetLibrary(hipLibrary_t* library,
                                      hipKernel_t kernel) {
  if (!library) return hipErrorInvalidValue;
  hipLibrary_t owning_library = NULL;
  iree_hal_streaming_symbol_t* symbol = NULL;
  hipError_t result = iree_hip_library_acquire_for_kernel(
      kernel, hipErrorInvalidValue, &owning_library, &symbol);
  if (result != hipSuccess) return result;
  *library = owning_library;
  iree_hip_library_release(owning_library);
  return hipSuccess;
}

HIPAPI hipError_t hipKernelGetName(const char** name, hipKernel_t kernel) {
  if (!name) return hipErrorInvalidValue;
  hipLibrary_t library = NULL;
  iree_hal_streaming_symbol_t* symbol = NULL;
  hipError_t result = iree_hip_library_acquire_for_kernel(
      kernel, hipErrorInvalidValue, &library, &symbol);
  if (result != hipSuccess) return result;

  result = hipErrorInvalidHandle;
  for (unsigned int i = 0; i < library->kernel_count; ++i) {
    if (library->kernels[i].symbol != symbol) continue;
    *name = library->kernels[i].name;
    result = hipSuccess;
    break;
  }
  iree_hip_library_release(library);
  return result;
}

HIPAPI hipError_t hipKernelGetParamInfo(hipKernel_t kernel,
                                        size_t parameter_index,
                                        size_t* parameter_offset,
                                        size_t* parameter_size) {
  if (!parameter_offset || !parameter_size) return hipErrorInvalidValue;
  hipLibrary_t library = NULL;
  iree_hal_streaming_symbol_t* symbol = NULL;
  hipError_t result = iree_hip_library_acquire_for_kernel(
      kernel, hipErrorInvalidValue, &library, &symbol);
  if (result != hipSuccess) return result;

  const iree_hal_streaming_parameter_info_t* parameters = &symbol->parameters;
  const size_t parameter_count =
      (size_t)parameters->copy_count + (size_t)parameters->binding_count;
  result = hipErrorInvalidValue;
  if (parameter_index < parameter_count) {
    for (uint16_t i = 0; i < parameters->copy_count; ++i) {
      const iree_hal_streaming_parameter_copy_op_t* op =
          &parameters->ops[i].copy;
      if (op->source_ordinal != parameter_index) continue;
      *parameter_offset = op->source_offset;
      *parameter_size = op->size;
      result = hipSuccess;
      break;
    }
    for (uint16_t i = 0; result != hipSuccess && i < parameters->binding_count;
         ++i) {
      const iree_hal_streaming_parameter_resolve_op_t* op =
          &parameters->ops[parameters->copy_count + i].resolve;
      if (op->source_ordinal != parameter_index) continue;
      *parameter_offset = op->source_offset;
      *parameter_size = sizeof(void*);
      result = hipSuccess;
      break;
    }
  }
  iree_hip_library_release(library);
  return result;
}

HIPAPI hipError_t hipKernelGetAttribute(int* value,
                                        hipFunction_attribute attribute,
                                        hipKernel_t kernel,
                                        hipDevice_t device) {
  if (!value) return hipErrorInvalidValue;
  hipLibrary_t library = NULL;
  iree_hal_streaming_symbol_t* symbol = NULL;
  hipError_t result = iree_hip_library_acquire_for_kernel(
      kernel, hipErrorInvalidHandle, &library, &symbol);
  if (result != hipSuccess) return result;

  int maximum_shared_memory = 0;
  result =
      hipDeviceGetAttribute(&maximum_shared_memory,
                            hipDeviceAttributeMaxSharedMemoryPerBlock, device);
  if (result != hipSuccess) {
    iree_hip_library_release(library);
    return result;
  }

  switch ((hipFuncAttribute_t)attribute) {
    case hipFuncAttributeMaxThreadsPerBlock:
      result = iree_hip_library_u32_attribute(
          symbol->function_attributes.maximum_threads_per_block, value);
      break;
    case hipFuncAttributeSharedSizeBytes:
      result = iree_hip_library_u32_attribute(
          symbol->function_attributes.fixed_shared_memory_size, value);
      break;
    case hipFuncAttributeConstSizeBytes:
      *value = 0;
      break;
    case hipFuncAttributeLocalSizeBytes:
      result = iree_hip_library_u32_attribute(
          symbol->function_attributes.fixed_local_memory_size, value);
      break;
    case hipFuncAttributeNumRegs:
      result = iree_hip_library_u32_attribute(
          symbol->function_attributes.register_count, value);
      break;
    case hipFuncAttributePtxVersion:
    case hipFuncAttributeBinaryVersion: {
      int major = 0;
      int minor = 0;
      result = hipDeviceGetAttribute(
          &major, hipDeviceAttributeComputeCapabilityMajor, device);
      if (result == hipSuccess) {
        result = hipDeviceGetAttribute(
            &minor, hipDeviceAttributeComputeCapabilityMinor, device);
      }
      if (result == hipSuccess &&
          (major < 0 || minor < 0 || major > (INT_MAX - minor) / 10)) {
        result = hipErrorInvalidValue;
      }
      if (result == hipSuccess) *value = major * 10 + minor;
      break;
    }
    case hipFuncAttributeCacheModeCA:
    case hipFuncAttributePreferredSharedMemoryCarveout:
      *value = 0;
      break;
    case hipFuncAttributeMaxDynamicSharedSizeBytes:
      result = iree_hip_library_u32_attribute(
          iree_hal_streaming_function_attributes_dynamic_shared_memory_size(
              &symbol->function_attributes),
          value);
      break;
    default:
      result = hipErrorInvalidValue;
      break;
  }

  iree_hip_library_release(library);
  return result;
}

HIPAPI hipError_t hipKernelSetAttribute(hipFunction_attribute attribute,
                                        int value, hipKernel_t kernel,
                                        hipDevice_t device) {
  hipLibrary_t library = NULL;
  iree_hal_streaming_symbol_t* symbol = NULL;
  hipError_t result = iree_hip_library_acquire_for_kernel(
      kernel, hipErrorInvalidValue, &library, &symbol);
  if (result != hipSuccess) return result;

  int maximum_shared_memory = 0;
  result =
      hipDeviceGetAttribute(&maximum_shared_memory,
                            hipDeviceAttributeMaxSharedMemoryPerBlock, device);
  if (result == hipSuccess) {
    switch ((hipFuncAttribute_t)attribute) {
      case hipFuncAttributeMaxDynamicSharedSizeBytes: {
        if (value < 0 ||
            (uint32_t)value >
                symbol->function_attributes
                    .maximum_configurable_dynamic_shared_memory_size) {
          result = hipErrorInvalidValue;
        }
        break;
      }
      case hipFuncAttributePreferredSharedMemoryCarveout:
        break;
      default:
        result = hipErrorInvalidValue;
        break;
    }
  }
  if (result == hipSuccess) {
    result = hipFuncSetAttribute(
        (hipFunction_t)iree_hal_streaming_symbol_tag(symbol),
        (hipFuncAttribute_t)attribute, value);
  }
  iree_hip_library_release(library);
  return result;
}
