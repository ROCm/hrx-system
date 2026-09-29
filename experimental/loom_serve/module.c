// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/module.h"

#include <stdlib.h>

#include "iree/vm/reflection.h"

typedef struct loom_serve_module_t {
  // Native module prefix owning all declaration storage below.
  iree_vm_module_t base;
  // Allocator used for module and declaration storage.
  iree_allocator_t host_allocator;
  // Immutable generic module description.
  iree_vm_module_descriptor_t descriptor;
  // Canonical HAL buffer identity borrowed from the VM environment.
  iree_vm_ref_type_t buffer_type;
  // Retained external execution capability, shared with the host's I/O path.
  loom_serve_execution_t* execution;
  // Owned stage array with retained commands and copied names.
  loom_serve_stage_t* stages;
  // Owned name storage used by the stage and export arrays.
  char* names;
  // Owned lexically sorted export directory.
  iree_vm_module_export_declaration_t* exports;
  // Owned unique callable declarations sorted by buffer argument count.
  iree_vm_module_callable_type_declaration_t* callables;
  // Owned reference fields shared by prefixes of the callable signatures.
  iree_vm_module_signature_type_t* arguments;
} loom_serve_module_t;

static const iree_vm_module_signature_type_t loom_serve_module_result = {
    IREE_VM_SCALAR_TYPE_I64, 0};

static void loom_serve_module_destroy(iree_vm_module_t* base) {
  loom_serve_module_t* module = (loom_serve_module_t*)base;
  if (module->stages) {
    for (iree_host_size_t i = 0; i < module->descriptor.counts.function_count;
         ++i) {
      iree_hal_command_buffer_release(module->stages[i].command_buffer);
    }
  }
  loom_serve_execution_release(module->execution);
  iree_allocator_free(module->host_allocator, module->arguments);
  iree_allocator_free(module->host_allocator, module->callables);
  iree_allocator_free(module->host_allocator, module->exports);
  iree_allocator_free(module->host_allocator, module->names);
  iree_allocator_free(module->host_allocator, module->stages);
  iree_allocator_free(module->host_allocator, module);
}

static iree_status_t loom_serve_module_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  const loom_serve_module_t* module = (const loom_serve_module_t*)base;
  const loom_serve_stage_t* stage = &module->stages[params->function_ordinal];
  iree_hal_buffer_binding_t* bindings =
      (iree_hal_buffer_binding_t*)params->execution.process_storage;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < stage->binding_count && iree_status_is_ok(status);
       ++i) {
    iree_vm_ref_t ref = iree_vm_ref_null();
    iree_vm_call_ref_argument_load_borrow(&params->call, i, &ref);
    // The linked callable signature establishes the exact HAL object type.
    // Null is a legal VM reference but cannot satisfy a command binding.
    iree_hal_buffer_t* buffer = (iree_hal_buffer_t*)ref.object;
    if (!buffer) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "runner stage buffer %u is null", i);
    } else {
      bindings[i] = (iree_hal_buffer_binding_t){
          buffer, 0, iree_hal_buffer_byte_length(buffer)};
    }
  }
  uint64_t value = 0;
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_execute(
        module->execution, stage->command_buffer,
        (iree_hal_buffer_binding_table_t){stage->binding_count, bindings},
        &value);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_call_value_result_store(&params->call, 0, value);
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  }
  return status;
}

static void loom_serve_module_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = ((const loom_serve_module_t*)base)->exports[ordinal];
}

static void loom_serve_module_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = ((const loom_serve_module_t*)base)->callables[ordinal];
}

static void loom_serve_module_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("runner module has no imports");
}

static void loom_serve_module_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("runner module has no imports");
}

static const iree_vm_module_vtable_t loom_serve_module_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = loom_serve_module_destroy,
    .function_start = loom_serve_module_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = loom_serve_module_query_import_group,
    .query_import = loom_serve_module_query_import,
    .query_export = loom_serve_module_query_export,
    .query_callable_type = loom_serve_module_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

static int loom_serve_module_compare_exports(const void* lhs, const void* rhs) {
  return iree_string_view_compare(
      ((const iree_vm_module_export_declaration_t*)lhs)->export_name,
      ((const iree_vm_module_export_declaration_t*)rhs)->export_name);
}

iree_status_t loom_serve_module_create(const iree_hal_module_types_t* types,
                                       loom_serve_execution_t* execution,
                                       iree_host_size_t stage_count,
                                       const loom_serve_stage_t* stages,
                                       iree_allocator_t host_allocator,
                                       iree_vm_module_t** out_module) {
  if (!stage_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "runner module requires at least one prepared stage");
  }
  loom_serve_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  module->buffer_type = types->buffer;
  module->execution = execution;
  loom_serve_execution_retain(execution);
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("runner"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {&module->buffer_type, 1},
      .counts = {.function_count = stage_count, .export_count = stage_count},
  };
  uint16_t maximum_bindings = 0;
  iree_host_size_t name_bytes = 0;
  for (iree_host_size_t i = 0; i < stage_count; ++i) {
    maximum_bindings = iree_max(maximum_bindings, stages[i].binding_count);
    name_bytes += stages[i].name.size;
  }
  module->descriptor.process_storage_size =
      maximum_bindings * sizeof(iree_hal_buffer_binding_t);
  iree_status_t status = iree_allocator_malloc(
      host_allocator, stage_count * sizeof(*module->stages),
      (void**)&module->stages);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, iree_max(name_bytes, 1),
                                   (void**)&module->names);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator,
                                   stage_count * sizeof(*module->exports),
                                   (void**)&module->exports);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator,
                                   stage_count * sizeof(*module->callables),
                                   (void**)&module->callables);
  }
  if (iree_status_is_ok(status) && maximum_bindings) {
    status = iree_allocator_malloc(
        host_allocator, maximum_bindings * sizeof(*module->arguments),
        (void**)&module->arguments);
  }
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < maximum_bindings; ++i) {
      module->arguments[i] = (iree_vm_module_signature_type_t){
          IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0};
    }
    char* name_cursor = module->names;
    for (iree_host_size_t i = 0; i < stage_count; ++i) {
      module->stages[i] = stages[i];
      iree_hal_command_buffer_retain(stages[i].command_buffer);
      memcpy(name_cursor, stages[i].name.data, stages[i].name.size);
      module->stages[i].name =
          iree_make_string_view(name_cursor, stages[i].name.size);
      name_cursor += stages[i].name.size;
      const uint16_t count = stages[i].binding_count;
      iree_host_size_t ordinal = 0;
      iree_host_size_t* callable_count =
          &module->descriptor.counts.callable_type_count;
      while (ordinal < *callable_count &&
             module->callables[ordinal].signature.arguments.count < count) {
        ++ordinal;
      }
      if (ordinal == *callable_count ||
          module->callables[ordinal].signature.arguments.count != count) {
        memmove(&module->callables[ordinal + 1], &module->callables[ordinal],
                (*callable_count - ordinal) * sizeof(*module->callables));
        module->callables[ordinal] =
            (iree_vm_module_callable_type_declaration_t){
                .signature =
                    {
                        .arguments = {module->arguments, count, 0, count, 0},
                        .results = {&loom_serve_module_result, 1, 1, 0, 0},
                    },
            };
        ++*callable_count;
        module->descriptor.counts.callable_fields.ref_count += count;
        ++module->descriptor.counts.callable_fields.value_count;
        for (iree_host_size_t j = 0; j < i; ++j) {
          if (module->exports[j].callable_type_ordinal >= ordinal) {
            ++module->exports[j].callable_type_ordinal;
          }
        }
      }
      module->exports[i] = (iree_vm_module_export_declaration_t){
          .export_name = module->stages[i].name,
          .callable_type_ordinal = ordinal,
          .function_ordinal = i,
      };
    }
    qsort(module->exports, stage_count, sizeof(*module->exports),
          loom_serve_module_compare_exports);
    status = iree_vm_module_initialize(&loom_serve_module_vtable,
                                       &module->descriptor, &module->base);
  }
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    loom_serve_module_destroy(&module->base);
  }
  return status;
}
