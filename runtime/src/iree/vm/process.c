// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/process.h"

#include <string.h>

#include "iree/vm/invocation_storage.h"
#include "iree/vm/process_storage.h"
#include "iree/vm/program_storage.h"

//===----------------------------------------------------------------------===//
// Process Slab Lifetime
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_process_allocate_unpublished(
    iree_vm_program_t* program, iree_allocator_t host_allocator,
    iree_vm_process_t** out_process) {
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(iree_vm_process_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(program->process_storage_size, uint8_t,
                                iree_max_align_t, NULL)));
  iree_vm_process_t* process = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&process));
  memset(process, 0, total_size);
  process->host_allocator = host_allocator;
  process->program = program;
  iree_vm_program_retain(program);
  *out_process = process;
  return iree_ok_status();
}

static void iree_vm_process_detach_modules(iree_vm_process_t* process,
                                           iree_host_size_t attached_count) {
  while (attached_count != 0) {
    const iree_vm_linked_module_t* linked_module =
        &process->program->linked_modules[--attached_count];
    if (linked_module->module->vtable->detach_state) {
      linked_module->module->vtable->detach_state(
          linked_module->module,
          iree_vm_process_module_state(process, linked_module));
    }
  }
}

static void iree_vm_process_free_unpublished(iree_vm_process_t* process,
                                             iree_host_size_t attached_count) {
  if (!process) return;
  iree_vm_process_detach_modules(process, attached_count);
  iree_vm_program_t* program = process->program;
  const iree_allocator_t host_allocator = process->host_allocator;
  iree_vm_program_release(program);
  iree_allocator_free(host_allocator, process);
}

static iree_status_t iree_vm_process_attach_modules(
    iree_vm_process_t* process, iree_host_size_t* out_attached_count) {
  *out_attached_count = 0;
  iree_status_t status = iree_ok_status();
  while (*out_attached_count < process->program->linked_module_count &&
         iree_status_is_ok(status)) {
    const iree_vm_linked_module_t* linked_module =
        &process->program->linked_modules[*out_attached_count];
    if (linked_module->module->vtable->attach_state) {
      status = linked_module->module->vtable->attach_state(
          linked_module->module,
          iree_vm_process_module_state(process, linked_module),
          process->host_allocator);
    }
    if (iree_status_is_ok(status)) ++*out_attached_count;
  }
  return status;
}

static iree_status_t iree_vm_process_seal_modules(iree_vm_process_t* process) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < process->program->linked_module_count && iree_status_is_ok(status);
       ++i) {
    const iree_vm_linked_module_t* linked_module =
        &process->program->linked_modules[i];
    if (linked_module->module->vtable->seal_state) {
      status = linked_module->module->vtable->seal_state(
          linked_module->module,
          iree_vm_process_module_state(process, linked_module));
    }
  }
  return status;
}

IREE_API_EXPORT void iree_vm_process_retain(iree_vm_process_t* process) {
  if (!process) return;
  iree_atomic_ref_count_inc(&process->ref_count);
}

IREE_API_EXPORT void iree_vm_process_release(iree_vm_process_t* process) {
  if (!process) return;
  if (iree_atomic_ref_count_dec(&process->ref_count) == 1) {
    iree_vm_process_detach_modules(process,
                                   process->program->linked_module_count);
    iree_vm_program_t* program = process->program;
    const iree_allocator_t host_allocator = process->host_allocator;
    iree_vm_program_release(program);
    iree_allocator_free(host_allocator, process);
  }
}

//===----------------------------------------------------------------------===//
// Process-Bound Functions
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_process_bind_target(
    iree_vm_process_t* process, uint64_t target_bits,
    iree_vm_function_t* out_function) {
  if (!target_bits || (target_bits & 3u) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function target is invalid");
  }
  const uint16_t module_ordinal =
      iree_vm_program_target_module_ordinal(target_bits);
  const uint16_t function_ordinal =
      iree_vm_program_target_function_ordinal(target_bits);
  if (module_ordinal >= process->program->linked_module_count ||
      function_ordinal >= process->program->linked_modules[module_ordinal]
                              .module->descriptor->counts.function_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function target is out of range");
  }
  if (!iree_vm_program_callable_token_is_valid(
          process->program,
          iree_vm_program_target_callable_token(target_bits))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function callable token is invalid");
  }
  const iree_vm_function_t function = {
      (uint64_t)(uintptr_t)process,
      target_bits,
  };
  *out_function = function;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_function_from_export(
    iree_vm_process_t* process, iree_vm_export_t export_value,
    iree_vm_function_t* out_function) {
  if (!process || !out_function) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "process and out_function are required");
  }
  iree_vm_function_ref_t function_ref = iree_vm_function_ref_null();
  IREE_RETURN_IF_ERROR(iree_vm_function_ref_from_export(
      process->program, export_value, &function_ref));
  return iree_vm_process_bind_target(process, function_ref.target_bits,
                                     out_function);
}

IREE_API_EXPORT iree_status_t iree_vm_function_from_function_ref(
    iree_vm_process_t* process, iree_vm_function_ref_t function_ref,
    iree_vm_function_t* out_function) {
  if (!process || !out_function || iree_vm_function_ref_is_null(function_ref) ||
      function_ref.program_bits != (uint64_t)(uintptr_t)process->program) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function ref does not belong to the process");
  }
  return iree_vm_process_bind_target(process, function_ref.target_bits,
                                     out_function);
}

IREE_API_EXPORT iree_status_t iree_vm_process_lookup_function(
    iree_vm_process_t* process, iree_string_view_t module_name,
    iree_string_view_t export_name, iree_vm_function_t* out_function) {
  if (!process || !out_function) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "process and out_function are required");
  }
  iree_host_size_t module_ordinal = 0;
  const iree_vm_linked_module_t* linked_module =
      iree_vm_program_lookup_linked_module(process->program, module_name,
                                           &module_ordinal);
  if (!linked_module) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND, "process module '%.*s' was not found",
        (int)iree_min(module_name.size, 128), module_name.data);
  }
  iree_vm_export_t export_value = {0};
  IREE_RETURN_IF_ERROR(iree_vm_module_lookup_export(
      linked_module->module, export_name, &export_value));
  return iree_vm_function_from_export(process, export_value, out_function);
}

//===----------------------------------------------------------------------===//
// Asynchronous Process Construction
//===----------------------------------------------------------------------===//

static void iree_vm_process_consume_arguments(
    iree_vm_variant_span_t arguments) {
  if (arguments.data) iree_vm_variant_span_reset(arguments);
}

static iree_status_t iree_vm_process_complete_construction(
    iree_vm_invocation_t* invocation,
    iree_vm_process_create_outcome_t* out_outcome) {
  iree_vm_process_t* process = invocation->process;
  iree_status_t status = iree_vm_process_seal_modules(process);
  iree_vm_cancel_reason_t cancel_reason = IREE_VM_CANCEL_REASON_NONE;
  if (iree_status_is_ok(status) &&
      !iree_vm_invocation_try_claim_completion(invocation, &cancel_reason)) {
    status = iree_vm_invocation_cancel_status(cancel_reason);
  }
  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_init(&process->ref_count);
    iree_vm_invocation_finish(invocation);
    const iree_vm_process_create_outcome_t outcome = {
        .execution_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED,
        .process = process,
    };
    *out_outcome = outcome;
  } else {
    iree_vm_invocation_abort(invocation);
    iree_vm_process_free_unpublished(process,
                                     process->program->linked_module_count);
  }
  return status;
}

static iree_status_t iree_vm_process_finish_without_initializer(
    iree_vm_process_t* process, iree_vm_process_create_outcome_t* out_outcome) {
  iree_status_t status = iree_vm_process_seal_modules(process);
  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_init(&process->ref_count);
    const iree_vm_process_create_outcome_t outcome = {
        .execution_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED,
        .process = process,
    };
    *out_outcome = outcome;
  } else {
    iree_vm_process_free_unpublished(process,
                                     process->program->linked_module_count);
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_vm_process_create_start(
    iree_vm_program_t* program, iree_vm_invocation_t* invocation,
    iree_vm_variant_span_t arguments,
    iree_vm_invocation_wake_callback_t wake_callback,
    iree_allocator_t host_allocator,
    iree_vm_process_create_outcome_t* out_outcome) {
  IREE_RETURN_IF_ERROR(iree_vm_invocation_validate_boundary(
      invocation, arguments, iree_vm_variant_span_empty(),
      iree_make_byte_span(out_outcome, sizeof(*out_outcome))));
  if (!program || !iree_vm_invocation_is_idle(invocation)) {
    iree_vm_process_consume_arguments(arguments);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid process construction boundary");
  }
  if (!program->initializer.target_bits && arguments.count != 0) {
    iree_vm_process_consume_arguments(arguments);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "a program without initialize accepts no process arguments");
  }

  iree_vm_process_t* process = NULL;
  iree_status_t status =
      iree_vm_process_allocate_unpublished(program, host_allocator, &process);
  if (!iree_status_is_ok(status)) {
    iree_vm_process_consume_arguments(arguments);
    return status;
  }

  if (program->initializer.target_bits) {
    const iree_vm_program_callable_t initializer_callable = {
        .argument_types = program->initializer.arguments.data,
        .result_types = NULL,
        .argument_counts = program->initializer.argument_counts,
        .result_counts = {0},
        .signature_module_ordinal = iree_vm_program_target_module_ordinal(
            program->initializer.target_bits),
        .uniform_result_scalar_type = IREE_VM_SCALAR_TYPE_INVALID,
    };
    status = iree_vm_invocation_prepare_root(
        invocation, IREE_VM_INVOCATION_OPERATION_PROCESS_CREATE, process,
        program->initializer.target_bits, &initializer_callable, arguments,
        iree_vm_variant_span_empty(), wake_callback);
    if (!iree_status_is_ok(status)) {
      iree_vm_process_free_unpublished(process, 0);
      return status;
    }
  } else {
    iree_vm_process_consume_arguments(arguments);
  }

  iree_host_size_t attached_count = 0;
  status = iree_vm_process_attach_modules(process, &attached_count);
  if (!iree_status_is_ok(status)) {
    if (program->initializer.target_bits) {
      iree_vm_invocation_abort(invocation);
    }
    iree_vm_process_free_unpublished(process, attached_count);
    return status;
  }
  if (!program->initializer.target_bits) {
    return iree_vm_process_finish_without_initializer(process, out_outcome);
  }

  iree_vm_execution_outcome_t execution_outcome = UINT32_MAX;
  status = iree_vm_invocation_drive_start(invocation, &execution_outcome);
  if (!iree_status_is_ok(status)) {
    iree_vm_invocation_abort(invocation);
    iree_vm_process_free_unpublished(process, program->linked_module_count);
    return status;
  }
  if (execution_outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED) {
    const iree_vm_process_create_outcome_t outcome = {
        .execution_outcome = execution_outcome,
        .process = NULL,
    };
    *out_outcome = outcome;
    return iree_ok_status();
  }
  return iree_vm_process_complete_construction(invocation, out_outcome);
}

IREE_API_EXPORT iree_status_t
iree_vm_process_create_resume(iree_vm_invocation_t* invocation,
                              iree_vm_process_create_outcome_t* out_outcome) {
  if (!invocation || !out_outcome ||
      invocation->state != IREE_VM_INVOCATION_STATE_SUSPENDED ||
      invocation->operation != IREE_VM_INVOCATION_OPERATION_PROCESS_CREATE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid process construction resume boundary");
  }
  IREE_RETURN_IF_ERROR(iree_vm_invocation_validate_boundary(
      invocation, iree_vm_variant_span_empty(), iree_vm_variant_span_empty(),
      iree_make_byte_span(out_outcome, sizeof(*out_outcome))));
  iree_vm_process_t* process = invocation->process;
  iree_vm_execution_outcome_t execution_outcome = UINT32_MAX;
  iree_status_t status =
      iree_vm_invocation_drive_resume(invocation, &execution_outcome);
  if (!iree_status_is_ok(status)) {
    iree_vm_invocation_abort(invocation);
    iree_vm_process_free_unpublished(process,
                                     process->program->linked_module_count);
    return status;
  }
  if (execution_outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED) {
    const iree_vm_process_create_outcome_t outcome = {
        .execution_outcome = execution_outcome,
        .process = NULL,
    };
    *out_outcome = outcome;
    return iree_ok_status();
  }
  return iree_vm_process_complete_construction(invocation, out_outcome);
}
