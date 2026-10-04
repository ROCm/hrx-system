// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/module.h"

#include <stdio.h>
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
  // Number of retained commands, independent of the export arity family.
  iree_host_size_t stage_count;
  // Owned stage array with retained commands.
  loom_serve_stage_t* stages;
  // Copied descriptors borrowing stable host feedback storage.
  iree_byte_span_t* feedback;
  // Number of registered feedback slots.
  iree_host_size_t feedback_count;
  // Owned name storage used by the export directory.
  char* names;
  // Owned lexically sorted export directory.
  iree_vm_module_export_declaration_t* exports;
  // Owned callable declarations in canonical signature order.
  iree_vm_module_callable_type_declaration_t* callables;
  // Stage index followed by reference fields shared by execute_N signatures.
  iree_vm_module_signature_type_t* arguments;
} loom_serve_module_t;

static const iree_vm_module_signature_type_t loom_serve_module_result = {
    IREE_VM_SCALAR_TYPE_I64, 0};

static void loom_serve_module_destroy(iree_vm_module_t* base) {
  loom_serve_module_t* module = (loom_serve_module_t*)base;
  if (module->stages) {
    for (iree_host_size_t i = 0; i < module->stage_count; ++i) {
      iree_hal_command_buffer_release(module->stages[i].command_buffer);
    }
  }
  loom_serve_execution_release(module->execution);
  iree_allocator_free(module->host_allocator, module->arguments);
  iree_allocator_free(module->host_allocator, module->callables);
  iree_allocator_free(module->host_allocator, module->exports);
  iree_allocator_free(module->host_allocator, module->names);
  iree_allocator_free(module->host_allocator, module->feedback);
  iree_allocator_free(module->host_allocator, module->stages);
  iree_allocator_free(module->host_allocator, module);
}

static iree_status_t loom_serve_module_execute(
    const loom_serve_module_t* module,
    const iree_vm_module_function_start_params_t* params, uint64_t* out_value) {
  const uint32_t index =
      (uint32_t)iree_vm_call_value_argument_load(&params->call, 0);
  if (index >= module->stage_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "runner stage %u is not loaded", index);
  }
  const loom_serve_stage_t* stage = &module->stages[index];
  if (stage->binding_count != params->function_ordinal) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "runner stage %u expects %u buffers", index,
                            stage->binding_count);
  }
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
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_execute(
        module->execution, stage->command_buffer,
        (iree_hal_buffer_binding_table_t){stage->binding_count, bindings},
        out_value);
  }
  return status;
}

static iree_status_t loom_serve_module_feedback(
    const loom_serve_module_t* module, const iree_vm_call_packet_t* call,
    uint64_t* out_value) {
  const uint32_t slot = (uint32_t)iree_vm_call_value_argument_load(call, 0);
  const uint64_t offset = iree_vm_call_value_argument_load(call, 1);
  const uint64_t length = iree_vm_call_value_argument_load(call, 2);
  if (slot >= module->feedback_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "runner feedback slot %u is not registered", slot);
  }
  if (length > module->feedback[slot].data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "runner feedback exceeds host destination");
  }
  iree_vm_ref_t ref = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(call, 0, &ref);
  if (!ref.object) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "runner feedback source is null");
  }
  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {.source_buffer = (iree_hal_buffer_t*)ref.object,
                   .source_offset = offset,
                   .target = module->feedback[slot].data,
                   .length = length},
  };
  return loom_serve_execution_feedback(module->execution, 1, &operation,
                                       out_value);
}

static iree_status_t loom_serve_module_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  const loom_serve_module_t* module = (const loom_serve_module_t*)base;
  uint64_t value = 0;
  iree_status_t status =
      params->function_ordinal + 1 == module->descriptor.counts.function_count
          ? loom_serve_module_feedback(module, &params->call, &value)
          : loom_serve_module_execute(module, params, &value);
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
                                       loom_serve_module_options_t options,
                                       iree_allocator_t host_allocator,
                                       iree_vm_module_t** out_module) {
  *out_module = NULL;
  if (!options.stages.count || options.binding_capacity == UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "runner module requires stages and a representable binding capacity");
  }
  for (iree_host_size_t i = 0; i < options.stages.count; ++i) {
    if (options.stages.values[i].binding_count > options.binding_capacity) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "runner stage exceeds binding capacity");
    }
  }
  loom_serve_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  module->buffer_type = types->buffer;
  module->execution = execution;
  loom_serve_execution_retain(execution);
  const iree_host_size_t export_count = options.binding_capacity + 2;
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("runner"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {&module->buffer_type, 1},
      .counts = {.function_count = export_count,
                 .export_count = export_count,
                 .callable_type_count = export_count},
  };
  module->descriptor.process_storage_size =
      options.binding_capacity * sizeof(iree_hal_buffer_binding_t);
  iree_status_t status = iree_allocator_clone(
      host_allocator,
      iree_make_const_byte_span(options.stages.values,
                                options.stages.count * sizeof(*module->stages)),
      (void**)&module->stages);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < options.stages.count; ++i) {
      iree_hal_command_buffer_retain(module->stages[i].command_buffer);
    }
    module->stage_count = options.stages.count;
    status = iree_allocator_malloc(host_allocator, export_count * 16,
                                   (void**)&module->names);
  }
  if (iree_status_is_ok(status) && options.feedback.count) {
    status = iree_allocator_clone(
        host_allocator,
        iree_make_const_byte_span(
            options.feedback.values,
            options.feedback.count * sizeof(*module->feedback)),
        (void**)&module->feedback);
    module->feedback_count = options.feedback.count;
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator,
                                   export_count * sizeof(*module->exports),
                                   (void**)&module->exports);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator,
                                   export_count * sizeof(*module->callables),
                                   (void**)&module->callables);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
        host_allocator,
        (options.binding_capacity + 1) * sizeof(*module->arguments),
        (void**)&module->arguments);
  }
  if (iree_status_is_ok(status)) {
    module->arguments[0] =
        (iree_vm_module_signature_type_t){IREE_VM_SCALAR_TYPE_I32, 0};
    for (uint32_t i = 1; i <= options.binding_capacity; ++i) {
      module->arguments[i] = (iree_vm_module_signature_type_t){
          IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0};
    }
    // Signatures sort by field count, then type. Feedback has four fields
    // and its scalar offset sorts before execute_3's third reference.
    const iree_host_size_t feedback_ordinal =
        iree_min(3, options.binding_capacity + 1);
    for (iree_host_size_t i = 0; i <= options.binding_capacity; ++i) {
      char* name = module->names + i * 16;
      const int length = snprintf(name, 16, "execute_%zu", i);
      const iree_host_size_t ordinal = i + (i >= feedback_ordinal);
      module->callables[ordinal].signature = (iree_vm_module_signature_t){
          .arguments = {module->arguments, i + 1, 1, i, 0},
          .results = {&loom_serve_module_result, 1, 1, 0, 0},
      };
      module->descriptor.counts.callable_fields.ref_count += i;
      module->descriptor.counts.callable_fields.value_count += 2;
      module->exports[i] = (iree_vm_module_export_declaration_t){
          .export_name = iree_make_string_view(name, length),
          .callable_type_ordinal = ordinal,
          .function_ordinal = i,
      };
    }
    static const iree_vm_module_signature_type_t feedback_arguments[] = {
        {IREE_VM_SCALAR_TYPE_I32, 0},
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
        {IREE_VM_SCALAR_TYPE_I64, 0},
        {IREE_VM_SCALAR_TYPE_I64, 0},
    };
    const iree_host_size_t ordinal = export_count - 1;
    module->callables[feedback_ordinal].signature =
        (iree_vm_module_signature_t){
            .arguments = {feedback_arguments, 4, 3, 1, 0},
            .results = {&loom_serve_module_result, 1, 1, 0, 0},
        };
    module->descriptor.counts.callable_fields.ref_count += 1;
    module->descriptor.counts.callable_fields.value_count += 4;
    module->exports[ordinal] = (iree_vm_module_export_declaration_t){
        .export_name = IREE_SVL("feedback"),
        .callable_type_ordinal = feedback_ordinal,
        .function_ordinal = ordinal,
    };
    qsort(module->exports, export_count, sizeof(*module->exports),
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
