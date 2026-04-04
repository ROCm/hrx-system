// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_internal.h"

#include <stdlib.h>
#include <string.h>

static void pyre_module_destroy_partial(pyre_module_t module) {
  if (module->context) {
    iree_vm_context_release(module->context);
  }
  if (module->hal_module) {
    iree_vm_module_release(module->hal_module);
  }
  if (module->bytecode_module) {
    iree_vm_module_release(module->bytecode_module);
  }
  if (module->device) {
    pyre_device_release(module->device);
  }
  free(module);
}

static iree_status_t pyre_compiler_output_archive_allocator_ctl(
    void* self, iree_allocator_command_t command, const void* params,
    void** inout_ptr) {
  (void)params;
  (void)inout_ptr;
  if (command != IREE_ALLOCATOR_COMMAND_FREE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "compiler output archive allocator only supports FREE");
  }
  pyre_compiler_output_release((pyre_compiler_output_t)self);
  return iree_ok_status();
}

static pyre_status_t pyre_module_load_archive(
    pyre_device_t device, iree_const_byte_span_t archive_contents,
    iree_allocator_t archive_allocator, pyre_module_t* module) {
  *module = NULL;
  if (archive_contents.data_length == 0) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "vmfb_size must be > 0");
  }

  pyre_shared_state_t* shared = pyre_get_shared_state();
  if (!shared->shared_initialized || !shared->vm_instance) {
    return pyre_make_status(PYRE_STATUS_UNAVAILABLE,
                            "runtime is not initialized");
  }

  pyre_module_t loaded = (pyre_module_t)calloc(1, sizeof(*loaded));
  if (!loaded) {
    return pyre_make_status(PYRE_STATUS_OUT_OF_MEMORY,
                            "failed to allocate module");
  }

  iree_allocator_t alloc = iree_allocator_system();
  iree_status_t status = iree_vm_bytecode_module_create(
      pyre_get_shared_state()->vm_instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      archive_contents, archive_allocator, alloc,
      &loaded->bytecode_module);
  if (!iree_status_is_ok(status)) {
    pyre_module_destroy_partial(loaded);
    return pyre_status_from_iree(status);
  }

  iree_hal_device_group_t* device_group = NULL;
  status = iree_hal_device_group_create_from_device(
      device->hal_device, alloc, &device_group);
  if (!iree_status_is_ok(status)) {
    pyre_module_destroy_partial(loaded);
    return pyre_status_from_iree(status);
  }

  status = iree_hal_module_create(
      pyre_get_shared_state()->vm_instance,
      iree_hal_module_device_policy_default(), device_group,
      IREE_HAL_MODULE_FLAG_NONE, iree_hal_module_debug_sink_null(), alloc,
      &loaded->hal_module);
  iree_hal_device_group_release(device_group);
  if (!iree_status_is_ok(status)) {
    pyre_module_destroy_partial(loaded);
    return pyre_status_from_iree(status);
  }

  iree_vm_module_t* modules[] = {
      loaded->hal_module,
      loaded->bytecode_module,
  };
  status = iree_vm_context_create_with_modules(
      pyre_get_shared_state()->vm_instance, IREE_VM_CONTEXT_FLAG_NONE, 2,
      modules, alloc, &loaded->context);
  if (!iree_status_is_ok(status)) {
    pyre_module_destroy_partial(loaded);
    return pyre_status_from_iree(status);
  }

  iree_atomic_ref_count_init(&loaded->ref_count);
  loaded->device = device;
  pyre_device_retain(loaded->device);
  *module = loaded;
  return pyre_ok_status();
}

pyre_status_t pyre_module_load_vmfb(pyre_device_t device, const void* vmfb_data,
                                    size_t vmfb_size,
                                    pyre_module_t* module) {
  if (!device || !vmfb_data || !module) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "device, vmfb_data, or module is NULL");
  }
  iree_const_byte_span_t archive_contents = {
      .data = (const uint8_t*)vmfb_data,
      .data_length = vmfb_size,
  };
  return pyre_module_load_archive(
      device, archive_contents, iree_allocator_null(), module);
}

pyre_status_t pyre_module_load_compiler_output(
    pyre_device_t device, pyre_compiler_output_t compiler_output,
    pyre_module_t* module) {
  if (!device || !compiler_output || !module) {
    return pyre_make_status(
        PYRE_STATUS_INVALID_ARGUMENT,
        "device, compiler_output, or module is NULL");
  }
  iree_const_byte_span_t archive_contents = {
      .data = compiler_output->data,
      .data_length = compiler_output->size,
  };
  iree_allocator_t archive_allocator = {
      .self = compiler_output,
      .ctl = pyre_compiler_output_archive_allocator_ctl,
  };
  pyre_compiler_output_retain(compiler_output);
  pyre_status_t status =
      pyre_module_load_archive(device, archive_contents, archive_allocator,
                               module);
  if (!pyre_status_is_ok(status)) {
    pyre_compiler_output_release(compiler_output);
  }
  return status;
}

void pyre_module_retain(pyre_module_t module) {
  iree_vm_context_retain(module->context);
  iree_vm_module_retain(module->hal_module);
  iree_vm_module_retain(module->bytecode_module);
  pyre_device_retain(module->device);
  iree_atomic_ref_count_inc(&module->ref_count);
}

void pyre_module_release(pyre_module_t module) {
  iree_vm_context_release(module->context);
  iree_vm_module_release(module->hal_module);
  iree_vm_module_release(module->bytecode_module);
  if (iree_atomic_ref_count_dec(&module->ref_count) == 1) {
    pyre_device_release(module->device);
    free(module);
  } else {
    pyre_device_release(module->device);
  }
}

pyre_status_t pyre_module_lookup_function(
    pyre_module_t module, const char* name, pyre_function_t* function) {
  if (!module || !name || !function) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "module, name, or function is NULL");
  }
  *function = NULL;

  pyre_function_t resolved = (pyre_function_t)calloc(1, sizeof(*resolved));
  if (!resolved) {
    return pyre_make_status(PYRE_STATUS_OUT_OF_MEMORY,
                            "failed to allocate function");
  }

  iree_string_view_t function_name = iree_make_cstring_view(name);
  iree_status_t status = iree_vm_context_resolve_function(
      module->context, function_name, &resolved->vm_function);
  if (!iree_status_is_ok(status)) {
    free(resolved);
    return pyre_status_from_iree(status);
  }

  iree_atomic_ref_count_init(&resolved->ref_count);
  resolved->module = module;
  pyre_module_retain(module);
  *function = resolved;
  return pyre_ok_status();
}

void pyre_function_retain(pyre_function_t function) {
  pyre_module_retain(function->module);
  iree_atomic_ref_count_inc(&function->ref_count);
}

void pyre_function_release(pyre_function_t function) {
  pyre_module_release(function->module);
  if (iree_atomic_ref_count_dec(&function->ref_count) == 1) {
    free(function);
  }
}

pyre_status_t pyre_function_invoke(pyre_module_t module,
                                   pyre_function_t function,
                                   pyre_value_list_t args,
                                   pyre_value_list_t rets) {
  if (!module || !function) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "module or function is NULL");
  }
  if (function->module != module) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "function does not belong to module");
  }

  iree_allocator_t alloc = iree_allocator_system();
  iree_status_t status = iree_vm_invoke(
      module->context, function->vm_function, IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/NULL, args ? args->vm_list : NULL,
      rets ? rets->vm_list : NULL, alloc);
  return pyre_status_from_iree(status);
}
