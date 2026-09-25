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
#include "loom/error/error_defs.h"
#include "loom/error/source.h"
#include "loom/link/linker.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/vm/module.h"
#include "loom/target/arch/vm/provider.h"
#include "loom/target/entry_selection.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/config/config.h"

void loom_vm_testbench_initialize(
    const loom_target_environment_t* target_environment,
    const loom_cleanup_pattern_provider_set_t* cleanup_pattern_provider_set,
    iree_allocator_t host_allocator, loom_vm_testbench_t* out_testbench) {
  *out_testbench = (loom_vm_testbench_t){
      .target_environment = target_environment,
      .cleanup_pattern_provider_set = cleanup_pattern_provider_set,
      .host_allocator = host_allocator,
  };
}

void loom_vm_testbench_deinitialize(loom_vm_testbench_t* testbench) {
  iree_vm_invocation_free(testbench->invocation);
  iree_vm_process_release(testbench->process);
  iree_allocator_free(testbench->host_allocator, testbench->arguments);
  memset(testbench, 0, sizeof(*testbench));
}

typedef struct loom_vm_testbench_pipeline_diagnostic_capture_t {
  // First error definition emitted while running the compile pipeline.
  const loom_error_def_t* error;
  // Source-attributing diagnostic sink receiving every record.
  loom_diagnostic_sink_t downstream;
} loom_vm_testbench_pipeline_diagnostic_capture_t;

static iree_status_t loom_vm_testbench_capture_pipeline_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_vm_testbench_pipeline_diagnostic_capture_t* capture = user_data;
  if (capture->error == NULL && diagnostic->severity == LOOM_DIAGNOSTIC_ERROR) {
    capture->error = diagnostic->error;
  }
  return loom_diagnostic_emit(&capture->downstream, diagnostic);
}

typedef struct loom_vm_testbench_emission_diagnostic_capture_t {
  // First error definition emitted while producing VM bytecode.
  const loom_error_def_t* error;
  // Source-attributing diagnostic emitter receiving every record.
  iree_diagnostic_emitter_t downstream;
} loom_vm_testbench_emission_diagnostic_capture_t;

static iree_status_t loom_vm_testbench_capture_emission_diagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loom_vm_testbench_emission_diagnostic_capture_t* capture = user_data;
  if (capture->error == NULL &&
      loom_error_def_severity(emission->error) == LOOM_DIAGNOSTIC_ERROR) {
    capture->error = emission->error;
  }
  return iree_diagnostic_emit(capture->downstream, emission);
}

static void loom_vm_testbench_record_compile_rejection(
    loom_vm_testbench_t* testbench, iree_string_view_t stage,
    iree_string_view_t kind, iree_string_view_t message) {
  testbench->compile_rejected = true;
  testbench->compile_failure_stage = stage;
  testbench->compile_failure_kind = kind;
  testbench->compile_failure_message = message;
}

// The compiler copy and all compiler scratch die before the runtime sees the
// image. This exercises the artifact ownership boundary on every test module.
static iree_status_t loom_vm_testbench_compile(loom_vm_testbench_t* testbench,
                                               const loom_module_t* source,
                                               iree_byte_span_t* out_contents) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(32 * 1024, testbench->host_allocator, &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  const loom_target_profile_t* profile = NULL;
  iree_status_t status =
      loom_vm_target_provider.select_profile(IREE_SV("core"), &profile);
  iree_string_view_t* roots = NULL;
  iree_host_size_t root_count = 0;
  iree_host_size_t max_arguments = 0;
  iree_host_size_t max_results = 0;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&arena, source->symbols.count,
                                       sizeof(*roots), (void**)&roots);
  }
  if (iree_status_is_ok(status)) {
    memset(roots, 0, source->symbols.count * sizeof(*roots));
    // The case planner owns invocation discovery. Its direct callees are the
    // executable roots; authored public helpers are implementation dependencies
    // within this independently compiled execution module.
    for (iree_host_size_t i = 0; i < testbench->cases.count; ++i) {
      const loom_testbench_case_plan_t* case_plan = testbench->cases.values[i];
      if (case_plan->issue_count) {
        continue;
      }
      for (iree_host_size_t j = 0; j < case_plan->invocation_count; ++j) {
        const loom_testbench_invocation_plan_t* call =
            &case_plan->invocations[j];
        if (call->kind != LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL) {
          continue;
        }
        max_arguments = iree_max(max_arguments, call->input_count);
        max_results = iree_max(max_results, call->result_count);
        roots[call->callee_ref.symbol_id] = loom_string_table_get(
            &source->strings,
            source->symbols.entries[call->callee_ref.symbol_id].name_id);
      }
    }
    for (iree_host_size_t i = 0; i < source->symbols.count; ++i) {
      if (!iree_string_view_is_empty(roots[i])) {
        roots[root_count++] = roots[i];
      }
    }
  }
  loom_source_table_projection_t sources = {.table = *testbench->sources,
                                            .arena = &arena};
  loom_module_t* module = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_link_materialized_modules(
        &source, 1,
        &(loom_link_options_t){
            .module_name = IREE_SV("test"),
            .root_symbols = {.count = root_count, .values = roots},
            .source_callback = {.fn = loom_source_table_project,
                                .user_data = &sources},
        },
        &pool, testbench->host_allocator, &module);
  }
  if (iree_status_is_ok(status)) {
    loom_tooling_config_materialize_result_t result;
    status = loom_tooling_config_materialize_module(
        module,
        &(loom_tooling_config_materialize_options_t){
            .config_set = testbench->config_set,
        },
        &pool, &result);
  }
  loom_target_specialization_request_t* requests = NULL;
  iree_host_size_t request_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&arena, root_count, sizeof(*requests),
                                       (void**)&requests);
  }
  if (iree_status_is_ok(status)) {
    // Root retention is produced by the linker. Dependencies have had their
    // public/export/retain surface closed by that same selection boundary.
    // Publish retained invocation roots for VM lookup, including private roots.
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      loom_symbol_t* symbol = &module->symbols.entries[i];
      if (!iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_RETAIN)) {
        continue;
      }
      const loom_func_like_t function =
          loom_func_like_cast(module, symbol->defining_op);
      loom_op_attrs(function.op)[function.vtable->visibility_attr_index] =
          loom_attr_enum(LOOM_FUNC_VISIBILITY_PUBLIC);
      symbol->flags |= LOOM_SYMBOL_FLAG_PUBLIC;
      requests[request_count++] = (loom_target_specialization_request_t){
          .function_name =
              loom_string_table_get(&module->strings, symbol->name_id),
          .target_profile = profile,
      };
    }
  }
  loom_target_low_descriptor_registry_t registry = {0};
  if (iree_status_is_ok(status)) {
    status = loom_target_environment_initialize_low_descriptor_registry(
        testbench->target_environment, &registry);
  }
  loom_compile_pipeline_result_t pipeline = {0};
  loom_compile_pipeline_options_t options;
  loom_compile_pipeline_options_initialize(&options);
  options.target_environment = testbench->target_environment;
  options.source_resolver = (loom_source_resolver_t){
      .fn = loom_source_table_resolve, .user_data = &sources.table};
  options.target_specializations =
      (loom_target_specialization_request_list_t){requests, request_count};
  options.low_descriptor_registry = &registry;
  options.cleanup_pattern_provider_set =
      testbench->cleanup_pattern_provider_set;
  loom_vm_testbench_pipeline_diagnostic_capture_t pipeline_diagnostic = {
      .downstream = options.diagnostic_sink,
  };
  options.diagnostic_sink = (loom_diagnostic_sink_t){
      .fn = loom_vm_testbench_capture_pipeline_diagnostic,
      .user_data = &pipeline_diagnostic,
  };
  if (iree_status_is_ok(status)) {
    status = loom_compile_run_pipeline(module, &options, &pool, &pipeline);
    if (iree_status_is_ok(status) && pipeline.pass.error_count) {
      IREE_ASSERT(pipeline_diagnostic.error != NULL);
      loom_vm_testbench_record_compile_rejection(
          testbench, IREE_SV("pipeline"),
          iree_make_cstring_view(loom_error_def_id(pipeline_diagnostic.error)),
          iree_make_cstring_view(
              loom_error_def_summary(pipeline_diagnostic.error)));
    }
  }
  options.diagnostic_sink = pipeline_diagnostic.downstream;
  loom_target_emit_artifact_t artifact = {0};
  bool artifact_emitted = false;
  loom_vm_testbench_emission_diagnostic_capture_t emission_diagnostic = {0};
  if (iree_status_is_ok(status) && !testbench->compile_rejected) {
    const loom_target_entry_options_t entry_options = {
        .diagnostic_sink = options.diagnostic_sink,
        .source_resolver = options.source_resolver,
        .max_errors = options.max_errors,
    };
    loom_target_entry_diagnostic_emitter_t entry_emitter = {0};
    loom_target_entry_diagnostic_emitter_initialize(
        module, &entry_options, LOOM_EMITTER_VERIFIER, &entry_emitter);
    emission_diagnostic.downstream = loom_target_entry_emitter(&entry_emitter);
    const loom_target_emit_request_t request = {
        .target_environment = testbench->target_environment,
        .low_descriptor_registry = &registry.registry,
        .module = module,
        .function_versions = &pipeline.function_versions.list,
        .diagnostic_emitter =
            {
                .fn = loom_vm_testbench_capture_emission_diagnostic,
                .user_data = &emission_diagnostic,
            },
        .scratch_arena = &arena,
        .allocator = testbench->host_allocator,
    };
    status = loom_vm_module_emit(&request, &artifact_emitted, &artifact);
  }
  if (iree_status_is_ok(status) && !testbench->compile_rejected &&
      !artifact_emitted) {
    if (emission_diagnostic.error != NULL) {
      loom_vm_testbench_record_compile_rejection(
          testbench, IREE_SV("emission"),
          iree_make_cstring_view(loom_error_def_id(emission_diagnostic.error)),
          iree_make_cstring_view(
              loom_error_def_summary(emission_diagnostic.error)));
    } else {
      loom_vm_testbench_record_compile_rejection(
          testbench, IREE_SV("emission"), IREE_SV("rejected"),
          IREE_SV("VM artifact emission rejected the compiled module"));
    }
  }
  if (iree_status_is_ok(status) && artifact_emitted) {
    status = iree_byte_sequence_clone(artifact.contents,
                                      testbench->host_allocator, out_contents);
  }
  if (iree_status_is_ok(status) && artifact_emitted) {
    iree_host_size_t total_size = 0, results_offset = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &total_size,
        IREE_STRUCT_FIELD(max_arguments, iree_vm_variant_t, NULL),
        IREE_STRUCT_FIELD(max_results, iree_vm_variant_t, &results_offset));
    if (iree_status_is_ok(status) && total_size) {
      status = iree_allocator_malloc(testbench->host_allocator, total_size,
                                     (void**)&testbench->arguments);
      if (iree_status_is_ok(status)) {
        testbench->results =
            (iree_vm_variant_t*)((uint8_t*)testbench->arguments +
                                 results_offset);
      }
    }
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(testbench->host_allocator, out_contents->data);
      *out_contents = iree_byte_span_empty();
    }
  }
  loom_target_emit_artifact_release(&artifact);
  loom_compile_pipeline_result_deinitialize(&pipeline);
  loom_module_free(module);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
  return status;
}

static iree_status_t loom_vm_testbench_prepare(loom_vm_testbench_t* testbench,
                                               const loom_module_t* source) {
  iree_byte_span_t contents = iree_byte_span_empty();
  iree_status_t status =
      loom_vm_testbench_compile(testbench, source, &contents);
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
        iree_hal_buffer_allowed_access(binding->buffer) &
        (IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE);
    iree_hal_buffer_mapping_t mapping = {0};
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        binding->buffer, IREE_HAL_MAPPING_MODE_PERSISTENT, access,
        binding->byte_offset, binding->byte_length, &mapping));
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

static iree_status_t loom_vm_testbench_export_buffer(
    loom_vm_testbench_t* testbench, iree_vm_variant_t result,
    loom_testbench_value_t* out_value) {
  iree_vm_buffer_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_ptr_from_variant_borrowed(
      &testbench->ref_types, result, &source));
  iree_hal_buffer_t* buffer = NULL;
  const iree_host_size_t length = iree_vm_buffer_length(source);
  if (source) {
    const iree_vm_buffer_access_flags_t vm_access =
        iree_vm_buffer_access(source);
    iree_hal_memory_access_t access = IREE_HAL_MEMORY_ACCESS_UNALIGNED;
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
  }
  *out_value = (loom_testbench_value_t){
      .kind = LOOM_TESTBENCH_VALUE_KIND_BUFFER,
      .buffer = {.kind = IREE_TOOLING_BUFFER_BINDING_KIND_STORAGE_BUFFER,
                 .buffer = buffer,
                 .byte_length = length},
  };
  return iree_ok_status();
}

static iree_status_t loom_vm_testbench_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t workload_count, const loom_testbench_value_t* workloads,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_host_size_t result_count, loom_testbench_value_t* out_results) {
  loom_vm_testbench_t* testbench = user_data;
  if (!testbench->process && !testbench->compile_rejected) {
    IREE_RETURN_IF_ERROR(
        loom_vm_testbench_prepare(testbench, invocation->module));
  }
  if (testbench->compile_rejected) {
    return iree_ok_status();
  }
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
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < parameter_count && iree_status_is_ok(status);
       ++i) {
    if (inputs[i].kind == LOOM_TESTBENCH_VALUE_KIND_BUFFER) {
      status = loom_vm_testbench_import_buffer(testbench, &inputs[i].buffer,
                                               &arguments[i]);
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
      status = loom_vm_testbench_export_buffer(testbench, results[i],
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
  return status;
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

loom_testbench_invocation_provider_t loom_vm_testbench_invocation_provider(
    void* user_data, loom_testbench_case_plan_list_t cases,
    const loom_source_table_resolver_t* sources,
    const loom_tooling_config_set_t* config_set) {
  loom_vm_testbench_t* testbench = user_data;
  testbench->cases = cases;
  testbench->sources = sources;
  testbench->config_set = config_set;
  return (loom_testbench_invocation_provider_t){
      .invoke = loom_vm_testbench_invoke,
      .query_issue = loom_vm_testbench_query_issue,
      .user_data = testbench,
  };
}
