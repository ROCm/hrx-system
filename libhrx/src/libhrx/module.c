// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "iree/vm/bytecode/module.h"

#include <stdlib.h>
#include <string.h>

#include "hrx_internal.h"

static void hrx_module_destroy_partial(hrx_module_t module) {
  iree_vm_context_release(module->context);
  iree_vm_module_release(module->hal_module);
  iree_vm_module_release(module->bytecode_module);
  hrx_device_release(module->device);
  free(module);
}

static hrx_status_t hrx_module_load_archive(
    hrx_device_t device, iree_const_byte_span_t archive_contents,
    iree_allocator_t archive_allocator, hrx_module_t* module) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_module_load_archive");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, archive_contents.data_length);
  *module = NULL;
  if (archive_contents.data_length == 0) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                                "vmfb_size must be > 0"));
  }

  hrx_shared_state_t* shared = hrx_get_shared_state();
  if (!shared->shared_initialized || !shared->vm_instance) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_UNAVAILABLE,
                                                "runtime is not initialized"));
  }

  hrx_module_t loaded = (hrx_module_t)calloc(1, sizeof(*loaded));
  if (!loaded) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                                                "failed to allocate module"));
  }

  iree_allocator_t alloc = iree_allocator_system();
  iree_status_t status = iree_vm_bytecode_module_create(
      hrx_get_shared_state()->vm_instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      archive_contents, archive_allocator, alloc, &loaded->bytecode_module);
  if (!iree_status_is_ok(status)) {
    hrx_module_destroy_partial(loaded);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  iree_hal_device_group_t* device_group = device->hal_device_group;
  if (!device_group) {
    hrx_module_destroy_partial(loaded);
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_FAILED_PRECONDITION,
                            "device is missing its HAL device group"));
  }
  iree_hal_device_group_retain(device_group);

  status = iree_hal_module_create(hrx_get_shared_state()->vm_instance,
                                  iree_hal_module_device_policy_default(),
                                  device_group, IREE_HAL_MODULE_FLAG_NONE,
                                  iree_hal_module_debug_sink_null(), alloc,
                                  &loaded->hal_module);
  iree_hal_device_group_release(device_group);
  if (!iree_status_is_ok(status)) {
    hrx_module_destroy_partial(loaded);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  iree_vm_module_t* modules[] = {
      loaded->hal_module,
      loaded->bytecode_module,
  };
  status = iree_vm_context_create_with_modules(
      hrx_get_shared_state()->vm_instance, IREE_VM_CONTEXT_FLAG_NONE, 2,
      modules, alloc, &loaded->context);
  if (!iree_status_is_ok(status)) {
    hrx_module_destroy_partial(loaded);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  iree_atomic_ref_count_init(&loaded->ref_count);
  loaded->device = device;
  hrx_device_retain(loaded->device);
  *module = loaded;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}

hrx_status_t hrx_module_load_vmfb(hrx_device_t device, const void* vmfb_data,
                                  size_t vmfb_size, hrx_module_t* module) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_module_load_vmfb");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, vmfb_size);
  if (!device || !vmfb_data || !module) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "device, vmfb_data, or module is NULL"));
  }
  iree_const_byte_span_t archive_contents = {
      .data = (const uint8_t*)vmfb_data,
      .data_length = vmfb_size,
  };
  HRX_RETURN_AND_END_ZONE(
      z0, hrx_module_load_archive(device, archive_contents,
                                  iree_allocator_null(), module));
}

void hrx_module_retain(hrx_module_t module) {
  if (!module) return;
  iree_vm_context_retain(module->context);
  iree_vm_module_retain(module->hal_module);
  iree_vm_module_retain(module->bytecode_module);
  hrx_device_retain(module->device);
  iree_atomic_ref_count_inc(&module->ref_count);
}

void hrx_module_release(hrx_module_t module) {
  if (!module) return;
  iree_vm_context_release(module->context);
  iree_vm_module_release(module->hal_module);
  iree_vm_module_release(module->bytecode_module);
  if (iree_atomic_ref_count_dec(&module->ref_count) == 1) {
    hrx_device_release(module->device);
    free(module);
  } else {
    hrx_device_release(module->device);
  }
}

hrx_status_t hrx_module_lookup_function(hrx_module_t module, const char* name,
                                        hrx_function_t* function) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_module_lookup_function");
  if (name) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, name);
  }
  if (!module || !name || !function) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "module, name, or function is NULL"));
  }
  *function = NULL;

  hrx_function_t resolved = (hrx_function_t)calloc(1, sizeof(*resolved));
  if (!resolved) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                                                "failed to allocate function"));
  }

  iree_string_view_t function_name = iree_make_cstring_view(name);
  iree_status_t status = iree_vm_context_resolve_function(
      module->context, function_name, &resolved->vm_function);
  if (!iree_status_is_ok(status)) {
    free(resolved);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  iree_atomic_ref_count_init(&resolved->ref_count);
  resolved->module = module;
  hrx_module_retain(module);
  *function = resolved;
  HRX_RETURN_AND_END_ZONE(z0, hrx_ok_status());
}

void hrx_function_retain(hrx_function_t function) {
  if (!function) return;
  hrx_module_retain(function->module);
  iree_atomic_ref_count_inc(&function->ref_count);
}

void hrx_function_release(hrx_function_t function) {
  if (!function) return;
  hrx_module_release(function->module);
  if (iree_atomic_ref_count_dec(&function->ref_count) == 1) {
    free(function);
  }
}

hrx_status_t hrx_function_invoke(hrx_module_t module, hrx_function_t function,
                                 hrx_value_list_t args, hrx_value_list_t rets) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_function_invoke");
  if (!module || !function) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                                "module or function is NULL"));
  }
  if (function->module != module) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "function does not belong to module"));
  }

  iree_allocator_t alloc = iree_allocator_system();
  iree_status_t status = iree_vm_invoke(
      module->context, function->vm_function, IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/NULL, args ? args->vm_list : NULL, rets ? rets->vm_list : NULL,
      alloc);
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
}
