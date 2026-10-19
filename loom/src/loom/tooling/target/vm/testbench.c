// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/vm/testbench.h"

#include "iree/hal/buffer.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/execution.h"
#include "iree/vm/sync.h"
#include "loomc/artifact.h"
#include "loomc/iree.h"
#include "loomc/target/vm.h"

static void loom_vm_testbench_initialize_storage(
    iree_allocator_t host_allocator, loom_vm_testbench_t* out_testbench) {
  *out_testbench = (loom_vm_testbench_t){
      .host_allocator = host_allocator,
  };
}

iree_status_t loom_vm_testbench_initialize(
    loomc_target_environment_t* target_environment,
    iree_allocator_t host_allocator, loom_vm_testbench_t* out_testbench) {
  loom_vm_testbench_initialize_storage(host_allocator, out_testbench);
  return iree_status_from_loomc(loomc_target_profile_select(
      target_environment, loomc_make_cstring_view("vm:core"),
      loomc_allocator_from_iree(host_allocator),
      &out_testbench->target_profile));
}

void loom_vm_testbench_deinitialize(loom_vm_testbench_t* testbench) {
  iree_vm_invocation_free(testbench->invocation);
  iree_vm_process_release(testbench->process);
  iree_allocator_free(testbench->host_allocator, testbench->arguments);
  iree_allocator_free(testbench->host_allocator,
                      testbench->compile_failure_storage);
  loomc_target_profile_release(testbench->target_profile);
  memset(testbench, 0, sizeof(*testbench));
}

static iree_status_t loom_vm_testbench_record_compile_rejection(
    loom_vm_testbench_t* testbench, const loomc_result_t* result) {
  const loomc_diagnostic_t* diagnostic = NULL;
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* candidate = loomc_result_diagnostic_at(result, i);
    if (candidate->severity == LOOMC_DIAGNOSTIC_SEVERITY_ERROR) {
      diagnostic = candidate;
      break;
    }
  }
  testbench->compile_rejected = true;
  testbench->compile_failure_stage = IREE_SV("compile");
  if (diagnostic == NULL) {
    testbench->compile_failure_kind = IREE_SV("rejected");
    testbench->compile_failure_message =
        IREE_SV("VM compilation rejected the source module");
    return iree_ok_status();
  }

  iree_string_view_t kind = iree_string_view_from_loomc(diagnostic->code);
  if (iree_string_view_is_empty(kind)) {
    kind = IREE_SV("rejected");
  }
  iree_string_view_t message = iree_string_view_from_loomc(diagnostic->message);
  if (iree_string_view_is_empty(message)) {
    message = IREE_SV("VM compilation rejected the source module");
  }
  iree_host_size_t storage_size = 0;
  if (!iree_host_size_checked_add(kind.size, message.size, &storage_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "VM compile diagnostic is too large");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(testbench->host_allocator, storage_size,
                            (void**)&testbench->compile_failure_storage));
  memcpy(testbench->compile_failure_storage, kind.data, kind.size);
  memcpy(testbench->compile_failure_storage + kind.size, message.data,
         message.size);
  testbench->compile_failure_kind =
      iree_make_string_view(testbench->compile_failure_storage, kind.size);
  testbench->compile_failure_message = iree_make_string_view(
      testbench->compile_failure_storage + kind.size, message.size);
  return iree_ok_status();
}

// The compiler copy and all compiler scratch die before the runtime sees the
// image. This exercises the artifact ownership boundary on every test module.
static void loom_vm_testbench_select_invocation_root(
    const loom_module_t* source,
    const loom_testbench_invocation_plan_t* invocation,
    loomc_string_view_t* roots, iree_host_size_t* inout_max_arguments,
    iree_host_size_t* inout_max_results) {
  *inout_max_arguments =
      iree_max(*inout_max_arguments, invocation->input_count);
  *inout_max_results = iree_max(*inout_max_results, invocation->result_count);
  roots[invocation->callee_ref.symbol_id] =
      loomc_string_view_from_iree(loom_string_table_get(
          &source->strings,
          source->symbols.entries[invocation->callee_ref.symbol_id].name_id));
}

static iree_status_t loom_vm_testbench_select_artifact(
    const loomc_result_t* result, const loomc_artifact_t** out_artifact) {
  *out_artifact = NULL;
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact->kind != LOOMC_ARTIFACT_KIND_EXECUTABLE ||
        !loomc_string_view_equal(
            artifact->format,
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM))) {
      continue;
    }
    if (*out_artifact != NULL) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "VM compilation returned more than one executable artifact");
    }
    *out_artifact = artifact;
  }
  return *out_artifact != NULL
             ? iree_ok_status()
             : iree_make_status(
                   IREE_STATUS_INTERNAL,
                   "VM compilation returned no executable artifact");
}

static iree_status_t loom_vm_testbench_compile(
    loom_vm_testbench_t* testbench, const loom_module_t* source,
    const loom_testbench_invocation_plan_t* product_invocation,
    iree_byte_span_t* out_contents) {
  *out_contents = iree_byte_span_empty();
  loomc_string_view_t* roots = NULL;
  iree_host_size_t root_count = 0;
  iree_host_size_t max_arguments = 0;
  iree_host_size_t max_results = 0;
  iree_status_t status = iree_ok_status();
  if (source->symbols.count != 0) {
    status = iree_allocator_malloc_array(testbench->host_allocator,
                                         source->symbols.count, sizeof(*roots),
                                         (void**)&roots);
  }
  if (iree_status_is_ok(status)) {
    if (source->symbols.count != 0) {
      memset(roots, 0, source->symbols.count * sizeof(*roots));
    }
    if (product_invocation != NULL) {
      loom_vm_testbench_select_invocation_root(
          source, product_invocation, roots, &max_arguments, &max_results);
    } else {
      // The runner owns invocation discovery. Direct callees are executable
      // roots; authored helpers are ordinary dependencies within this module.
      for (iree_host_size_t i = 0; i < testbench->invocations.count; ++i) {
        loom_vm_testbench_select_invocation_root(
            source, testbench->invocations.values[i], roots, &max_arguments,
            &max_results);
      }
    }
    for (iree_host_size_t i = 0; i < source->symbols.count; ++i) {
      if (!loomc_string_view_is_empty(roots[i])) {
        roots[root_count++] = roots[i];
      }
    }
  }

  loomc_module_t* module = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_clone(
        testbench->compilation.module, testbench->compilation.workspace,
        loomc_allocator_from_iree(testbench->host_allocator), &module));
  }

  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM),
  };
  const loomc_compile_artifact_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(compile_options),
      .roots = roots,
      .root_count = root_count,
      .target_profile = testbench->target_profile,
      .config = testbench->compilation.config,
      .emit_options = &emit_options,
  };
  loomc_result_t* result = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_compile_artifact(
        testbench->compilation.compiler, testbench->compilation.workspace,
        /*pass_program=*/NULL, module, &compile_options,
        loomc_allocator_from_iree(testbench->host_allocator), &result));
  }
  if (iree_status_is_ok(status)) {
    status = loom_testbench_observe_compile_result(testbench->result_callback,
                                                   result);
  }
  if (iree_status_is_ok(status) && !loomc_result_succeeded(result)) {
    status = loom_vm_testbench_record_compile_rejection(testbench, result);
  }
  const loomc_artifact_t* artifact = NULL;
  if (iree_status_is_ok(status) && !testbench->compile_rejected) {
    status = loom_vm_testbench_select_artifact(result, &artifact);
  }
  if (iree_status_is_ok(status) && artifact != NULL) {
    loomc_byte_span_t cloned_contents = loomc_byte_span_empty();
    status = iree_status_from_loomc(loomc_byte_sequence_clone(
        artifact->contents,
        loomc_allocator_from_iree(testbench->host_allocator),
        &cloned_contents));
    if (iree_status_is_ok(status)) {
      *out_contents = iree_make_byte_span((uint8_t*)cloned_contents.data,
                                          cloned_contents.data_length);
    }
  }
  if (iree_status_is_ok(status) && artifact != NULL) {
    iree_host_size_t total_size = 0;
    iree_host_size_t results_offset = 0;
    iree_host_size_t argument_buffers_offset = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &total_size,
        IREE_STRUCT_FIELD(max_arguments, iree_vm_variant_t, NULL),
        IREE_STRUCT_FIELD(max_results, iree_vm_variant_t, &results_offset),
        IREE_STRUCT_FIELD(max_arguments, iree_vm_buffer_t*,
                          &argument_buffers_offset));
    if (iree_status_is_ok(status) && total_size) {
      status = iree_allocator_malloc(testbench->host_allocator, total_size,
                                     (void**)&testbench->arguments);
      if (iree_status_is_ok(status)) {
        testbench->results =
            (iree_vm_variant_t*)((uint8_t*)testbench->arguments +
                                 results_offset);
        testbench->argument_buffers =
            (iree_vm_buffer_t**)((uint8_t*)testbench->arguments +
                                 argument_buffers_offset);
        memset(testbench->argument_buffers, 0,
               max_arguments * sizeof(*testbench->argument_buffers));
      }
    }
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(testbench->host_allocator, out_contents->data);
      *out_contents = iree_byte_span_empty();
    }
  }
  loomc_result_release(result);
  loomc_module_release(module);
  loomc_workspace_trim(testbench->compilation.workspace);
  iree_allocator_free(testbench->host_allocator, roots);
  return status;
}

static iree_status_t loom_vm_testbench_prepare(
    loom_vm_testbench_t* testbench, const loom_module_t* source,
    const loom_testbench_invocation_plan_t* product_invocation) {
  iree_byte_span_t contents = iree_byte_span_empty();
  iree_status_t status = loom_vm_testbench_compile(
      testbench, source, product_invocation, &contents);
  if (!iree_status_is_ok(status) || testbench->compile_rejected) {
    iree_allocator_free(testbench->host_allocator, contents.data);
    return status;
  }
  iree_vm_environment_t* environment = NULL;
  iree_vm_module_t* module = NULL;
  iree_vm_program_t* program = NULL;
  iree_vm_invocation_t* invocation = NULL;
  iree_vm_process_t* process = NULL;
  status =
      iree_vm_environment_allocate(testbench->host_allocator, &environment);
  if (iree_status_is_ok(status)) {
    status = iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &testbench->ref_types);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_module_create(
        environment, IREE_SV("test"),
        (iree_vm_bytecode_module_storage_t){
            .contents =
                iree_make_const_byte_span(contents.data, contents.data_length),
            .deallocator = testbench->host_allocator},
        testbench->host_allocator, &module);
    if (iree_status_is_ok(status)) {
      contents = iree_byte_span_empty();
    }
  }
  iree_allocator_free(testbench->host_allocator, contents.data);
  iree_vm_environment_free(environment);
  if (iree_status_is_ok(status)) {
    status = iree_vm_program_create(
        (iree_vm_program_modules_t){.executable = module},
        testbench->host_allocator, &program);
  }
  iree_vm_module_release(module);
  if (iree_status_is_ok(status)) {
    status = iree_vm_invocation_allocate(16 * 1024, testbench->host_allocator,
                                         &invocation);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_create(program, invocation,
                                    iree_vm_variant_span_empty(),
                                    testbench->host_allocator, &process);
  }
  iree_vm_program_release(program);
  if (iree_status_is_ok(status)) {
    testbench->invocation = invocation;
    testbench->process = process;
  } else {
    iree_vm_invocation_free(invocation);
    iree_vm_process_release(process);
    iree_allocator_free(testbench->host_allocator, testbench->arguments);
    testbench->arguments = testbench->results = NULL;
  }
  return status;
}

static void loom_vm_testbench_release_hal_buffer(void* user_data,
                                                 iree_byte_span_t storage) {
  iree_hal_buffer_release(user_data);
}

// A persistent mapping borrows the HAL buffer's lifetime. The VM wrapper owns
// that lifetime, including when a returned alias outlives the invocation.
static iree_status_t loom_vm_testbench_import_buffer(
    loom_vm_testbench_t* testbench,
    const iree_tooling_buffer_binding_t* binding,
    iree_vm_variant_t* out_argument) {
  iree_vm_buffer_t* buffer = NULL;
  if (binding->buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
        iree_hal_buffer_memory_type(binding->buffer),
        IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
    const iree_hal_memory_access_t access =
        iree_hal_buffer_allowed_access(binding->buffer);
    iree_hal_buffer_mapping_t mapping = {0};
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        binding->buffer, IREE_HAL_MAPPING_MODE_PERSISTENT, access,
        IREE_HAL_BUFFER_MAP_FLAG_MAY_ALIAS, binding->byte_offset,
        binding->byte_length, &mapping));
    iree_vm_buffer_access_flags_t vm_access = 0;
    if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_READ)) {
      vm_access |= IREE_VM_BUFFER_ACCESS_FLAG_READ;
    }
    if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
      vm_access |= IREE_VM_BUFFER_ACCESS_FLAG_WRITE;
    }
    IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
        vm_access, mapping.contents,
        (iree_vm_buffer_release_callback_t){
            .function = loom_vm_testbench_release_hal_buffer,
            .user_data = binding->buffer},
        testbench->host_allocator, &buffer));
    iree_hal_buffer_retain(binding->buffer);
  }
  *out_argument =
      iree_vm_buffer_variant_from_ptr_move(&testbench->ref_types, &buffer);
  return iree_ok_status();
}

static void loom_vm_testbench_release_vm_buffer(void* user_data,
                                                iree_hal_buffer_t* buffer) {
  iree_vm_buffer_release(user_data);
}

static const uint8_t* loom_vm_testbench_buffer_data(iree_vm_buffer_t* buffer) {
  const uint8_t* data = (const uint8_t*)iree_vm_buffer_const_data(buffer);
  return data ? data : (const uint8_t*)iree_vm_buffer_data(buffer);
}

static iree_status_t loom_vm_testbench_trace_buffer_reference(
    iree_vm_buffer_t* result_buffer, iree_host_size_t input_count,
    const loom_testbench_value_t* inputs,
    iree_vm_buffer_t* const* argument_buffers,
    loom_testbench_buffer_reference_t* out_reference) {
  *out_reference = (loom_testbench_buffer_reference_t){0};
  const iree_host_size_t result_length = iree_vm_buffer_length(result_buffer);
  const uint8_t* result_data = loom_vm_testbench_buffer_data(result_buffer);
  for (iree_host_size_t input_index = 0; input_index < input_count;
       ++input_index) {
    iree_vm_buffer_t* input_buffer = argument_buffers[input_index];
    const loom_testbench_buffer_reference_t* input_reference =
        &inputs[input_index].buffer_reference;
    if (input_buffer == NULL || !input_reference->is_traceable) {
      continue;
    }

    iree_host_size_t relative_offset = 0;
    if (result_buffer != input_buffer) {
      const uint8_t* input_data = loom_vm_testbench_buffer_data(input_buffer);
      const iree_host_size_t input_length = iree_vm_buffer_length(input_buffer);
      const uintptr_t result_address = (uintptr_t)result_data;
      const uintptr_t input_address = (uintptr_t)input_data;
      if (result_data == NULL || input_data == NULL ||
          result_address < input_address) {
        continue;
      }
      const uintptr_t offset = result_address - input_address;
      if (offset > input_length || result_length > input_length - offset) {
        continue;
      }
      relative_offset = (iree_host_size_t)offset;
    }

    iree_device_size_t byte_offset = 0;
    if (!iree_device_size_checked_add(input_reference->byte_offset,
                                      relative_offset, &byte_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "returned VM buffer logical allocation byte offset overflows");
    }
    const loom_testbench_buffer_reference_t candidate = {
        .is_traceable = true,
        .allocation_value_id = input_reference->allocation_value_id,
        .byte_offset = byte_offset,
        .byte_length = result_length,
    };
    if (!out_reference->is_traceable) {
      *out_reference = candidate;
    } else if (out_reference->allocation_value_id !=
                   candidate.allocation_value_id ||
               out_reference->byte_offset != candidate.byte_offset ||
               out_reference->byte_length != candidate.byte_length) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "returned VM buffer maps to inconsistent logical allocations");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_testbench_export_buffer(
    loom_vm_testbench_t* testbench, iree_vm_variant_t result,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_vm_buffer_t* const* argument_buffers,
    loom_testbench_value_t* out_value) {
  iree_vm_buffer_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_ptr_from_variant_borrowed(
      &testbench->ref_types, result, &source));
  if (source == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM function returned a null buffer reference");
  }
  loom_testbench_buffer_reference_t reference = {0};
  IREE_RETURN_IF_ERROR(loom_vm_testbench_trace_buffer_reference(
      source, input_count, inputs, argument_buffers, &reference));

  iree_hal_buffer_t* buffer = NULL;
  const iree_host_size_t length = iree_vm_buffer_length(source);
  const iree_vm_buffer_access_flags_t vm_access = iree_vm_buffer_access(source);
  iree_hal_memory_access_t access = IREE_HAL_MEMORY_ACCESS_NONE;
  if (iree_any_bit_set(vm_access, IREE_VM_BUFFER_ACCESS_FLAG_READ)) {
    access |= IREE_HAL_MEMORY_ACCESS_READ;
  }
  if (iree_any_bit_set(vm_access, IREE_VM_BUFFER_ACCESS_FLAG_WRITE)) {
    access |= IREE_HAL_MEMORY_ACCESS_WRITE;
  }
  void* data = iree_any_bit_set(vm_access, IREE_VM_BUFFER_ACCESS_FLAG_WRITE)
                   ? iree_vm_buffer_data(source)
                   : (void*)iree_vm_buffer_const_data(source);
  IREE_RETURN_IF_ERROR(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      access,
      IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
          IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      length, iree_make_byte_span(data, length),
      (iree_hal_buffer_release_callback_t){
          .fn = loom_vm_testbench_release_vm_buffer, .user_data = source},
      testbench->host_allocator, &buffer));
  iree_vm_buffer_retain(source);
  *out_value = (loom_testbench_value_t){
      .kind = LOOM_TESTBENCH_VALUE_KIND_BUFFER,
      .buffer = {.kind = IREE_TOOLING_BUFFER_BINDING_KIND_STORAGE_BUFFER,
                 .buffer = buffer,
                 .byte_length = length},
  };
  if (reference.is_traceable) {
    loom_testbench_value_set_buffer_reference(reference.allocation_value_id,
                                              reference.byte_offset,
                                              reference.byte_length, out_value);
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_testbench_invoke_prepared(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t workload_count, const loom_testbench_value_t* workloads,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_host_size_t result_count, loom_testbench_value_t* out_results) {
  loom_vm_testbench_t* testbench = user_data;
  const loom_symbol_t* symbol =
      &invocation->module->symbols.entries[invocation->callee_ref.symbol_id];
  const loom_func_like_t function =
      loom_func_like_const_cast(invocation->module, symbol->defining_op);
  uint16_t parameter_count = 0;
  const loom_value_id_t* parameter_ids =
      loom_func_like_arg_ids(function, &parameter_count);
  const loom_string_id_t export_name = loom_func_like_export_symbol(function);
  iree_vm_function_t callee = iree_vm_function_null();
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      testbench->process, IREE_SV("test"),
      loom_string_table_get(&invocation->module->strings,
                            export_name == LOOM_STRING_ID_INVALID
                                ? symbol->name_id
                                : export_name),
      &callee));
  iree_vm_variant_t* arguments = testbench->arguments;
  iree_vm_variant_t* results = testbench->results;
  iree_vm_buffer_t** argument_buffers = testbench->argument_buffers;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < parameter_count && iree_status_is_ok(status);
       ++i) {
    if (inputs[i].kind == LOOM_TESTBENCH_VALUE_KIND_BUFFER) {
      status = loom_vm_testbench_import_buffer(testbench, &inputs[i].buffer,
                                               &arguments[i]);
      if (iree_status_is_ok(status)) {
        status = iree_vm_buffer_ptr_from_variant_borrowed(
            &testbench->ref_types, arguments[i], &argument_buffers[i]);
      }
      if (iree_status_is_ok(status)) {
        iree_vm_buffer_retain(argument_buffers[i]);
      }
    } else if (inputs[i].kind != LOOM_TESTBENCH_VALUE_KIND_SCALAR) {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "VM test input requires scalar or buffer marshalling");
    } else {
      // The shared materializer stores narrow floats as raw words and narrow
      // integers as i32. The source signature supplies their exact VM tags.
      const loom_scalar_type_t scalar_type = loom_type_element_type(
          loom_module_value_type(invocation->module, parameter_ids[i]));
      const iree_tooling_value_t value = inputs[i].scalar;
      switch (scalar_type) {
        case LOOM_SCALAR_TYPE_I8:
          arguments[i] = iree_vm_variant_from_i8((int8_t)value.storage.i32);
          break;
        case LOOM_SCALAR_TYPE_I16:
          arguments[i] = iree_vm_variant_from_i16((int16_t)value.storage.i32);
          break;
        case LOOM_SCALAR_TYPE_I1:
        case LOOM_SCALAR_TYPE_I32:
          arguments[i] = iree_vm_variant_from_i32(value.storage.i32);
          break;
        case LOOM_SCALAR_TYPE_INDEX:
        case LOOM_SCALAR_TYPE_OFFSET:
        case LOOM_SCALAR_TYPE_I64:
          arguments[i] = iree_vm_variant_from_i64(value.storage.i64);
          break;
        case LOOM_SCALAR_TYPE_F8E4M3:
          arguments[i] =
              iree_vm_variant_from_f8e4m3fn_bits((uint8_t)value.storage.u32);
          break;
        case LOOM_SCALAR_TYPE_F8E5M2:
          arguments[i] =
              iree_vm_variant_from_f8e5m2_bits((uint8_t)value.storage.u32);
          break;
        case LOOM_SCALAR_TYPE_F16:
          arguments[i] =
              iree_vm_variant_from_f16_bits((uint16_t)value.storage.u32);
          break;
        case LOOM_SCALAR_TYPE_BF16:
          arguments[i] =
              iree_vm_variant_from_bf16_bits((uint16_t)value.storage.u32);
          break;
        case LOOM_SCALAR_TYPE_F32:
          arguments[i] = iree_vm_variant_from_f32(value.storage.f32);
          break;
        case LOOM_SCALAR_TYPE_F64:
          arguments[i] = iree_vm_variant_from_f64(value.storage.f64);
          break;
        default:
          status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                    "VM test scalar type %u is not implemented",
                                    scalar_type);
          break;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_invoke(testbench->invocation, callee,
                       iree_vm_variant_span_from_ptr(arguments, input_count),
                       iree_vm_variant_span_from_ptr(results, result_count));
  } else {
    iree_vm_variant_span_reset(
        iree_vm_variant_span_from_ptr(arguments, input_count));
  }
  for (iree_host_size_t i = 0; i < result_count && iree_status_is_ok(status);
       ++i) {
    if (iree_vm_variant_is_ref(results[i])) {
      status = loom_vm_testbench_export_buffer(
          testbench, results[i], input_count, inputs, argument_buffers,
          &out_results[i]);
      continue;
    }
    out_results[i].kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
    switch (iree_vm_variant_scalar_type(results[i])) {
      case IREE_VM_SCALAR_TYPE_I8: {
        int8_t value = 0;
        status = iree_vm_i8_from_variant(results[i], &value);
        out_results[i].scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
        out_results[i].scalar.storage.i32 = value;
        break;
      }
      case IREE_VM_SCALAR_TYPE_I16: {
        int16_t value = 0;
        status = iree_vm_i16_from_variant(results[i], &value);
        out_results[i].scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
        out_results[i].scalar.storage.i32 = value;
        break;
      }
      case IREE_VM_SCALAR_TYPE_I32:
        out_results[i].scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
        status = iree_vm_i32_from_variant(results[i],
                                          &out_results[i].scalar.storage.i32);
        break;
      case IREE_VM_SCALAR_TYPE_I64:
        out_results[i].scalar.kind = IREE_TOOLING_VALUE_KIND_I64;
        status = iree_vm_i64_from_variant(results[i],
                                          &out_results[i].scalar.storage.i64);
        break;
      case IREE_VM_SCALAR_TYPE_F8E4M3FN: {
        uint8_t bits = 0;
        status = iree_vm_f8e4m3fn_bits_from_variant(results[i], &bits);
        out_results[i].scalar.kind = IREE_TOOLING_VALUE_KIND_RAW_U32;
        out_results[i].scalar.storage.u32 = bits;
        break;
      }
      case IREE_VM_SCALAR_TYPE_F8E5M2: {
        uint8_t bits = 0;
        status = iree_vm_f8e5m2_bits_from_variant(results[i], &bits);
        out_results[i].scalar.kind = IREE_TOOLING_VALUE_KIND_RAW_U32;
        out_results[i].scalar.storage.u32 = bits;
        break;
      }
      case IREE_VM_SCALAR_TYPE_F16: {
        uint16_t bits = 0;
        status = iree_vm_f16_bits_from_variant(results[i], &bits);
        out_results[i].scalar.kind = IREE_TOOLING_VALUE_KIND_RAW_U32;
        out_results[i].scalar.storage.u32 = bits;
        break;
      }
      case IREE_VM_SCALAR_TYPE_BF16: {
        uint16_t bits = 0;
        status = iree_vm_bf16_bits_from_variant(results[i], &bits);
        out_results[i].scalar.kind = IREE_TOOLING_VALUE_KIND_RAW_U32;
        out_results[i].scalar.storage.u32 = bits;
        break;
      }
      case IREE_VM_SCALAR_TYPE_F32:
        out_results[i].scalar.kind = IREE_TOOLING_VALUE_KIND_F32;
        status = iree_vm_f32_from_variant(results[i],
                                          &out_results[i].scalar.storage.f32);
        break;
      case IREE_VM_SCALAR_TYPE_F64:
        out_results[i].scalar.kind = IREE_TOOLING_VALUE_KIND_F64;
        status = iree_vm_f64_from_variant(results[i],
                                          &out_results[i].scalar.storage.f64);
        break;
      default:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "VM test result requires supported scalar marshalling");
        break;
    }
  }
  iree_vm_variant_span_reset(
      iree_vm_variant_span_from_ptr(results, result_count));
  for (iree_host_size_t i = 0; i < input_count; ++i) {
    iree_vm_buffer_release(argument_buffers[i]);
    argument_buffers[i] = NULL;
  }
  return status;
}

static iree_status_t loom_vm_testbench_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t workload_count, const loom_testbench_value_t* workloads,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_host_size_t result_count, loom_testbench_value_t* out_results) {
  loom_vm_testbench_t* testbench = user_data;
  if (testbench->process == NULL && !testbench->compile_rejected) {
    IREE_RETURN_IF_ERROR(
        loom_vm_testbench_prepare(testbench, invocation->module, NULL));
  }
  if (testbench->compile_rejected) {
    return iree_ok_status();
  }
  return loom_vm_testbench_invoke_prepared(
      testbench, invocation, workload_count, workloads, input_count, inputs,
      result_count, out_results);
}

static iree_status_t loom_vm_testbench_query_issue(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_sample_issue_t* out_issue) {
  (void)invocation;
  *out_issue = (loom_testbench_sample_issue_t){0};
  const loom_vm_testbench_t* testbench = user_data;
  if (testbench->compile_rejected) {
    *out_issue = (loom_testbench_sample_issue_t){
        .category = LOOM_TESTBENCH_SAMPLE_ISSUE_COMPILE_REJECTED,
        .provider = IREE_SV("vm"),
        .stage = testbench->compile_failure_stage,
        .kind = testbench->compile_failure_kind,
        .message = testbench->compile_failure_message,
    };
  }
  return iree_ok_status();
}

static void loom_vm_testbench_product_destroy(void* user_data) {
  loom_vm_testbench_t* product = user_data;
  const iree_allocator_t host_allocator = product->host_allocator;
  loom_vm_testbench_deinitialize(product);
  iree_allocator_free(host_allocator, product);
}

static iree_status_t loom_vm_testbench_product_execute(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  for (iree_host_size_t call_index = 0; call_index < call_count; ++call_index) {
    loom_testbench_product_call_t* call = &calls[call_index];
    iree_status_t status = loom_vm_testbench_invoke_prepared(
        user_data, invocation, invocation->workload_count,
        call->call_parameters, invocation->input_count, call->arguments,
        invocation->result_count, call->results);
    if (!iree_status_is_ok(status)) {
      return iree_status_annotate_f(
          status,
          "executing VM scenario trial configuration %zu domain %zu "
          "ordinal %zu",
          call->identity->configuration_ordinal, call->identity->trial_index,
          call->identity->trial_ordinal);
    }
  }
  return iree_ok_status();
}

static const loom_testbench_product_vtable_t loom_vm_testbench_product_vtable =
    {
        .execute = loom_vm_testbench_product_execute,
};

static iree_status_t loom_vm_testbench_product_prepare(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* configuration,
    iree_allocator_t host_allocator,
    loom_testbench_prepared_product_t* out_product) {
  (void)configuration;
  if (invocation->kind != LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "VM scenario profile requires a semantic function subject");
  }
  if (invocation->workload_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM scenario function subject cannot have call parameters");
  }

  const loom_vm_testbench_t* parent = user_data;
  loom_vm_testbench_t* product = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*product),
                                             (void**)&product));
  loom_vm_testbench_initialize_storage(host_allocator, product);
  product->compilation = parent->compilation;
  product->result_callback = parent->result_callback;
  product->target_profile = parent->target_profile;
  loomc_target_profile_retain(product->target_profile);
  iree_status_t status =
      loom_vm_testbench_prepare(product, invocation->module, invocation);
  if (iree_status_is_ok(status) && product->compile_rejected) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "VM scenario compilation was rejected during %.*s (%.*s): %.*s",
        (int)product->compile_failure_stage.size,
        product->compile_failure_stage.data,
        (int)product->compile_failure_kind.size,
        product->compile_failure_kind.data,
        (int)product->compile_failure_message.size,
        product->compile_failure_message.data);
  }
  if (!iree_status_is_ok(status)) {
    loom_vm_testbench_product_destroy(product);
    return status;
  }
  *out_product = (loom_testbench_prepared_product_t){
      .vtable = &loom_vm_testbench_product_vtable,
      .destroy = loom_vm_testbench_product_destroy,
      .user_data = product,
  };
  return iree_ok_status();
}

loom_testbench_invocation_provider_t loom_vm_testbench_invocation_provider(
    void* user_data, const loom_testbench_compilation_t* compilation,
    loom_testbench_invocation_plan_list_t invocations,
    loom_testbench_compile_result_callback_t result_callback) {
  loom_vm_testbench_t* testbench = user_data;
  testbench->compilation = *compilation;
  testbench->result_callback = result_callback;
  testbench->invocations = invocations;
  return (loom_testbench_invocation_provider_t){
      .invoke = loom_vm_testbench_invoke,
      .query_issue = loom_vm_testbench_query_issue,
      .user_data = testbench,
  };
}

loom_testbench_execution_profile_t loom_vm_testbench_execution_profile(
    void* user_data, const loom_testbench_compilation_t* compilation,
    const loom_source_table_resolver_t* sources,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_testbench_compile_result_callback_t result_callback) {
  (void)sources;
  (void)diagnostic_sink;
  loom_vm_testbench_t* testbench = user_data;
  testbench->compilation = *compilation;
  testbench->result_callback = result_callback;
  return (loom_testbench_execution_profile_t){
      .name = IREE_SV("vm:core"),
      .prepare = loom_vm_testbench_product_prepare,
      .user_data = testbench,
  };
}
