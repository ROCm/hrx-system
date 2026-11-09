// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/program/storage.h"

#include "iree/vm/reflection.h"

typedef struct loom_runtime_storage_module_t {
  // Native module prefix.
  iree_vm_module_t base;
  // Allocator owning this capability.
  iree_allocator_t host_allocator;
  // Immutable declarations borrowing the type array below.
  iree_vm_module_descriptor_t descriptor;
  // Canonical HAL and VM buffer identities.
  iree_vm_ref_type_t types[2];
  // Borrowed model roots and feedback; execution is retained independently.
  loom_runtime_storage_module_options_t options;
  // Explicit owner scope for the current synchronous source invocation.
  loom_runtime_storage_access_t access;
} loom_runtime_storage_module_t;

enum loom_runtime_storage_function_e {
  STORAGE_BUFFER,
  STORAGE_COMPLETED,
  STORAGE_FILL,
  STORAGE_COPY,
};

void loom_runtime_storage_module_set_access(
    iree_vm_module_t* base, loom_runtime_storage_access_t access) {
  ((loom_runtime_storage_module_t*)base)->access = access;
}

static iree_status_t loom_runtime_storage_module_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  const loom_runtime_storage_module_t* module =
      (const loom_runtime_storage_module_t*)base;
  const iree_vm_call_packet_t* call = &params->call;
  const iree_host_size_t function = params->function_ordinal;
  const bool completed = function == STORAGE_COMPLETED;
  if (module->access != (completed ? LOOM_RUNTIME_STORAGE_ACCESS_COLLECT
                                   : LOOM_RUNTIME_STORAGE_ACCESS_SUBMIT)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "storage capability is outside its submission or collection scope");
  }
  if (function == STORAGE_BUFFER || completed) {
    const uint32_t ordinal =
        (uint32_t)iree_vm_call_value_argument_load(call, 0);
    const iree_host_size_t count = completed ? module->options.feedback.count
                                             : module->options.buffers.count;
    if (ordinal >= count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "storage ordinal exceeds declared capacity");
    }
    iree_vm_ref_t ref = iree_vm_ref_null();
    if (completed) {
      ref = iree_vm_ref_from_ptr_retained(
          module->options.feedback.values[ordinal], module->types[1]);
    } else {
      ref = iree_vm_ref_from_ptr_retained(
          module->options.buffers.values[ordinal], module->types[0]);
    }
    iree_vm_call_ref_result_store_move(call, 0, &ref);
  } else {
    iree_vm_ref_t source = iree_vm_ref_null();
    iree_vm_call_ref_argument_load_borrow(call, 0, &source);
    if (!source.object) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "storage transfer buffer is null");
    }
    iree_hal_transfer_operation_t operation = {0};
    const uint32_t pattern =
        (uint32_t)iree_vm_call_value_argument_load(call, 2);
    if (function == STORAGE_FILL) {
      operation = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .fill = {.target_buffer = source.object,
                   .target_offset = iree_vm_call_value_argument_load(call, 0),
                   .length = iree_vm_call_value_argument_load(call, 1),
                   .pattern = &pattern,
                   .pattern_length = sizeof(pattern)}};
    } else {
      iree_vm_ref_t target = iree_vm_ref_null();
      iree_vm_call_ref_argument_load_borrow(call, 1, &target);
      if (!target.object) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "storage copy target is null");
      }
      operation = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
          .copy = {.source_buffer = source.object,
                   .source_offset = iree_vm_call_value_argument_load(call, 0),
                   .target_buffer = target.object,
                   .target_offset = iree_vm_call_value_argument_load(call, 1),
                   .length = iree_vm_call_value_argument_load(call, 2)}};
    }
    uint64_t value = 0;
    IREE_RETURN_IF_ERROR(loom_runtime_execution_transfer(
        module->options.execution, 1, &operation, &value));
    iree_vm_call_value_result_store(call, 0, value);
  }
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static const iree_vm_module_signature_type_t ordinal_arguments[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0}};
static const iree_vm_module_signature_type_t device_buffer_result[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
static const iree_vm_module_signature_type_t host_buffer_result[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1}};
static const iree_vm_module_signature_type_t work_result[] = {
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t fill_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0}};
static const iree_vm_module_signature_type_t copy_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0}};

static const iree_vm_module_callable_type_declaration_t storage_callables[] = {
    {.signature = {.arguments = {ordinal_arguments, 1, 1, 0, 0},
                   .results = {device_buffer_result, 1, 0, 1, 0}}},
    {.signature = {.arguments = {ordinal_arguments, 1, 1, 0, 0},
                   .results = {host_buffer_result, 1, 0, 1, 0}}},
    {.signature = {.arguments = {fill_arguments, 4, 3, 1, 0},
                   .results = {work_result, 1, 1, 0, 0}}},
    {.signature = {.arguments = {copy_arguments, 5, 3, 2, 0},
                   .results = {work_result, 1, 1, 0, 0}}},
};

static const iree_vm_module_export_declaration_t storage_exports[] = {
    {.export_name = IREE_SVL("buffer"),
     .callable_type_ordinal = STORAGE_BUFFER,
     .function_ordinal = STORAGE_BUFFER},
    {.export_name = IREE_SVL("completed"),
     .callable_type_ordinal = STORAGE_COMPLETED,
     .function_ordinal = STORAGE_COMPLETED},
    {.export_name = IREE_SVL("copy"),
     .callable_type_ordinal = STORAGE_COPY,
     .function_ordinal = STORAGE_COPY},
    {.export_name = IREE_SVL("fill"),
     .callable_type_ordinal = STORAGE_FILL,
     .function_ordinal = STORAGE_FILL},
};

static void loom_runtime_storage_module_destroy(iree_vm_module_t* base) {
  loom_runtime_storage_module_t* module = (loom_runtime_storage_module_t*)base;
  loom_runtime_execution_release(module->options.execution);
  iree_allocator_free(module->host_allocator, module);
}
static void loom_runtime_storage_module_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = storage_exports[ordinal];
}
static void loom_runtime_storage_module_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = storage_callables[ordinal];
}
static void loom_runtime_storage_module_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("storage module has no imports");
}
static void loom_runtime_storage_module_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("storage module has no imports");
}

static const iree_vm_module_vtable_t storage_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = loom_runtime_storage_module_destroy,
    .function_start = loom_runtime_storage_module_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = loom_runtime_storage_module_query_import_group,
    .query_import = loom_runtime_storage_module_query_import,
    .query_export = loom_runtime_storage_module_query_export,
    .query_callable_type = loom_runtime_storage_module_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t loom_runtime_storage_module_create(
    const iree_hal_vm_types_t* hal_types, const iree_vm_ref_types_t* vm_types,
    loom_runtime_storage_module_options_t options,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  *out_module = NULL;
  loom_runtime_storage_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  module->options = options;
  loom_runtime_execution_retain(options.execution);
  module->types[0] = hal_types->buffer;
  module->types[1] = vm_types->buffer;
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("storage"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {module->types, IREE_ARRAYSIZE(module->types)},
      .counts = {.function_count = IREE_ARRAYSIZE(storage_exports),
                 .export_count = IREE_ARRAYSIZE(storage_exports),
                 .callable_type_count = IREE_ARRAYSIZE(storage_callables),
                 .callable_fields = {.value_count = 10, .ref_count = 5}},
  };
  iree_status_t status = iree_vm_module_initialize(
      &storage_vtable, &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    loom_runtime_storage_module_destroy(&module->base);
  }
  return status;
}
