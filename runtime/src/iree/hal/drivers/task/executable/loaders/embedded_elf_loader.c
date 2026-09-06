// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/executable/loaders/embedded_elf_loader.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/hal/api.h"
#include "iree/hal/drivers/task/executable/data.h"
#include "iree/hal/drivers/task/executable/elf/elf_module.h"
#include "iree/hal/drivers/task/executable/executable.h"
#include "iree/hal/drivers/task/executable/library/abi.h"
#include "iree/hal/drivers/task/executable/library/util.h"
#include "iree/hal/utils/elf_format.h"

//===----------------------------------------------------------------------===//
// iree_hal_elf_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_elf_executable_t {
  iree_hal_task_executable_t base;

  // Loaded ELF module.
  iree_elf_module_t module;

  // Name used for the file field in tracy and debuggers.
  iree_string_view_t identifier;

  // Queried v0 library metadata.
  const iree_hal_executable_library_v0_t* library;
} iree_hal_elf_executable_t;

static const iree_hal_task_executable_vtable_t iree_hal_elf_executable_vtable;

static iree_status_t iree_hal_elf_executable_query_library(
    iree_hal_elf_executable_t* executable) {
  // Get the exported symbol used to get the library metadata.
  iree_hal_executable_library_query_fn_t query_fn = NULL;
  IREE_RETURN_IF_ERROR(iree_elf_module_lookup_export(
      &executable->module, IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME,
      (void**)&query_fn));

  // Query for a compatible version of the library.
  const iree_hal_executable_library_header_t* const* query_result =
      (const iree_hal_executable_library_header_t* const*)iree_elf_call_p_ip(
          query_fn, IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
          &executable->base.environment);
  IREE_RETURN_IF_ERROR(iree_hal_executable_library_validate_query_result(
      query_result, &executable->library));
  const iree_hal_executable_library_header_t* header =
      executable->library->header;

  // Ensure that if the library is built for a particular sanitizer that we also
  // were compiled with that sanitizer enabled.
  switch (header->sanitizer) {
    case IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE:
      // Always safe even if the host has a sanitizer enabled; it just means
      // that we won't be able to catch anything from within the executable,
      // however checks outside will (often) still trigger when guard pages are
      // dirtied/etc.
      break;
    default:
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "executable requires sanitizer but they are not "
                              "yet supported with embedded libraries: %u",
                              (uint32_t)header->sanitizer);
  }

  executable->identifier = iree_make_cstring_view(header->name);
  executable->base.dispatch_attrs = executable->library->exports.attrs;
#if defined(IREE_PLATFORM_WINDOWS) && defined(IREE_ARCH_X86_64)
  // Embedded ELF exports use the SysV x86-64 ABI while the Windows host uses
  // the Microsoft x64 ABI. Calls must go through iree_elf_call_i_ppp so the
  // argument registers are bridged correctly.
  executable->base.dispatch_ptrs = NULL;
#else
  executable->base.dispatch_ptrs = executable->library->exports.ptrs;
#endif  // IREE_PLATFORM_WINDOWS && IREE_ARCH_X86_64
  executable->base.export_count = executable->library->exports.count;
  executable->base.export_names = executable->library->exports.names;
  return iree_ok_status();
}

static iree_status_t iree_hal_elf_executable_create(
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_load_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(executable_params->executable_data.data &&
                       executable_params->executable_data.data_length);
  IREE_ASSERT_ARGUMENT(!executable_params->constant_count ||
                       executable_params->constants);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_elf_executable_t* executable = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t constants_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(sizeof(*executable), &total_size,
                             IREE_STRUCT_FIELD_ALIGNED(
                                 executable_params->constant_count, uint32_t,
                                 iree_alignof(uint32_t), &constants_offset)));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  iree_hal_task_executable_initialize(queue_family,
                                      &iree_hal_elf_executable_vtable,
                                      host_allocator, &executable->base);

  // Copy executable constants so we own them.
  if (executable_params->constant_count > 0) {
    uint32_t* target_constants =
        (uint32_t*)((uint8_t*)executable + constants_offset);
    memcpy(target_constants, executable_params->constants,
           executable_params->constant_count *
               sizeof(*executable_params->constants));
    executable->base.environment.constants = target_constants;
  }

  // Attempt to load the ELF module.
  iree_status_t status = iree_elf_module_initialize_from_memory(
      executable_params->executable_data, host_allocator, &executable->module);

  // Query metadata and get the entry point function pointers.
  if (iree_status_is_ok(status)) {
    status = iree_hal_elf_executable_query_library(executable);
  }

  // Verify that the library matches the executable params.
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_library_verify(executable_params,
                                                executable->library);
  }

  // Publish the executable sources with the tracing infrastructure.
  if (iree_status_is_ok(status)) {
    iree_hal_executable_library_publish_source_files(executable->library);
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_release((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_elf_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_elf_executable_t* executable =
      (iree_hal_elf_executable_t*)base_executable;
  iree_allocator_t host_allocator = executable->base.host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_elf_module_deinitialize(&executable->module);

  iree_hal_task_executable_deinitialize(
      (iree_hal_task_executable_t*)base_executable);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_elf_executable_issue_call(
    iree_hal_task_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state,
    uint32_t worker_id) {
  iree_hal_elf_executable_t* executable =
      (iree_hal_elf_executable_t*)base_executable;
  const iree_hal_executable_library_v0_t* library = executable->library;

  if (IREE_UNLIKELY(ordinal >= library->exports.count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry point ordinal out of bounds");
  }

  IREE_HAL_EXECUTABLE_LIBRARY_CALL_TRACE_ZONE_BEGIN(z0, executable->identifier,
                                                    library, ordinal);
  IREE_HAL_EXECUTABLE_LIBRARY_CALL_HOOK_BEGIN(executable->identifier, library,
                                              ordinal);
  int ret = iree_elf_call_i_ppp(library->exports.ptrs[ordinal],
                                (void*)&base_executable->environment,
                                (void*)dispatch_state, (void*)workgroup_state);
  IREE_HAL_EXECUTABLE_LIBRARY_CALL_HOOK_END(executable->identifier, library,
                                            ordinal);
  IREE_TRACE_ZONE_END(z0);

  return ret == 0 ? iree_ok_status()
                  : iree_make_status(
                        IREE_STATUS_INTERNAL,
                        "executable entry point returned catastrophic error %d",
                        ret);
}

static iree_host_size_t iree_hal_elf_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_elf_executable_t* executable =
      (iree_hal_elf_executable_t*)base_executable;
  return iree_hal_executable_library_export_count(executable->library);
}

static iree_status_t iree_hal_elf_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t export_ordinal,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_elf_executable_t* executable =
      (iree_hal_elf_executable_t*)base_executable;
  return iree_hal_executable_library_export_info(executable->library,
                                                 export_ordinal, out_info);
}

static iree_status_t iree_hal_elf_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t export_ordinal, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  iree_hal_elf_executable_t* executable =
      (iree_hal_elf_executable_t*)base_executable;
  return iree_hal_executable_library_export_parameters(
      executable->library, export_ordinal, capacity, out_parameters);
}

static iree_status_t iree_hal_elf_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_export_ordinal) {
  iree_hal_elf_executable_t* executable =
      (iree_hal_elf_executable_t*)base_executable;
  return iree_hal_executable_library_lookup_export_by_name(
      executable->library, name, out_export_ordinal);
}

static iree_status_t iree_hal_elf_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  (void)base_executable;
  (void)name;
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t iree_hal_elf_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  (void)base_executable;
  (void)global;
  memset(out_info, 0, sizeof(*out_info));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid embedded ELF executable global");
}

static iree_status_t iree_hal_elf_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_buffer_t** out_buffer) {
  (void)base_executable;
  (void)global;
  (void)out_buffer;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid embedded ELF executable global");
}

static const iree_hal_task_executable_vtable_t iree_hal_elf_executable_vtable =
    {
        .base =
            {
                .destroy = iree_hal_elf_executable_destroy,
                .function_count = iree_hal_elf_executable_export_count,
                .function_info = iree_hal_elf_executable_export_info,
                .function_parameters =
                    iree_hal_elf_executable_export_parameters,
                .lookup_function_by_name =
                    iree_hal_elf_executable_lookup_export_by_name,
                .try_lookup_global_by_name =
                    iree_hal_elf_executable_try_lookup_global_by_name,
                .global_info = iree_hal_elf_executable_global_info,
                .global_buffer = iree_hal_elf_executable_global_buffer,
            },
        .issue_call = iree_hal_elf_executable_issue_call,
};

//===----------------------------------------------------------------------===//
// iree_hal_embedded_elf_loader_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_embedded_elf_loader_t {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
} iree_hal_embedded_elf_loader_t;

static const iree_hal_executable_loader_vtable_t
    iree_hal_embedded_elf_loader_vtable;

iree_status_t iree_hal_embedded_elf_loader_create(
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_embedded_elf_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_loader), (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(&iree_hal_embedded_elf_loader_vtable,
                                          &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_embedded_elf_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_embedded_elf_loader_t* executable_loader =
      (iree_hal_embedded_elf_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable_loader);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_embedded_elf_loader_query_target_support(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_target_t* target) {
  return iree_string_view_equal(target->family, IREE_SV("cpu"));
}

static void iree_hal_embedded_elf_loader_query_spec(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_device_executable_spec_t* out_executable_spec) {
  *out_executable_spec = (iree_hal_device_executable_spec_t){0};
}

static bool iree_hal_embedded_elf_loader_claims_executable(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params) {
  if (!iree_hal_elf_data_starts_with_magic(load_params->executable_data)) {
    return false;
  }
  if (!iree_hal_task_executable_data_is_system_library(
          load_params->executable_data)) {
    return true;
  }
  return !iree_any_bit_set(load_params->flags,
                           IREE_HAL_EXECUTABLE_LOAD_FLAG_ENABLE_DEBUGGING) &&
         !iree_hal_task_elf_data_requires_system_loader(
             load_params->executable_data);
}

static iree_status_t iree_hal_embedded_elf_loader_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* executable_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  iree_hal_embedded_elf_loader_t* executable_loader =
      (iree_hal_embedded_elf_loader_t*)base_executable_loader;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Perform the load of the ELF and wrap it in an executable handle.
  iree_status_t status = iree_hal_elf_executable_create(
      queue_family, executable_params, executable_loader->host_allocator,
      out_executable);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_executable_loader_vtable_t
    iree_hal_embedded_elf_loader_vtable = {
        .destroy = iree_hal_embedded_elf_loader_destroy,
        .query_target_support =
            iree_hal_embedded_elf_loader_query_target_support,
        .query_spec = iree_hal_embedded_elf_loader_query_spec,
        .claims_executable = iree_hal_embedded_elf_loader_claims_executable,
        .load = iree_hal_embedded_elf_loader_load,
};
