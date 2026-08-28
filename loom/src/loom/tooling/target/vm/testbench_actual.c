// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/vm/testbench_actual.h"

#include <string.h>

#include "iree/base/byte_sequence.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/environment.h"
#include "iree/vm/invocation.h"
#include "iree/vm/process.h"
#include "iree/vm/program.h"
#include "iree/vm/variant.h"
#include "loom/ir/types.h"
#include "loom/link/linker.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/entry_selection.h"
#include "loom/tooling/compile/pipeline.h"

enum {
  LOOM_VM_TESTBENCH_INVOCATION_STORAGE_SIZE = 16 * 1024,
};

typedef struct loom_vm_testbench_compile_roots_t {
  // Owned root-name array borrowing strings from the source module.
  iree_string_view_t* values;
  // Number of populated entries in |values|.
  iree_host_size_t count;
  // Number of allocated entries in |values|.
  iree_host_size_t capacity;
  // Maximum semantic function-call argument count.
  iree_host_size_t max_argument_count;
  // Maximum semantic function-call result count.
  iree_host_size_t max_result_count;
} loom_vm_testbench_compile_roots_t;

static iree_status_t loom_vm_testbench_symbol_name_from_ref(
    const loom_module_t* module, loom_symbol_ref_t ref,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM testbench call references invalid symbol {module=%u, symbol=%u}",
        (unsigned)ref.module_id, (unsigned)ref.symbol_id);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM testbench call symbol %u has no valid name",
                            (unsigned)ref.symbol_id);
  }
  *out_name = module->strings.entries[symbol->name_id];
  return iree_ok_status();
}

static iree_status_t loom_vm_testbench_resolve_function_export_name(
    const loom_module_t* module, loom_symbol_ref_t ref,
    iree_string_view_t* out_export_name) {
  *out_export_name = iree_string_view_empty();
  iree_string_view_t symbol_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_vm_testbench_symbol_name_from_ref(module, ref, &symbol_name));
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  const loom_func_like_t function =
      loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM testbench callee '@%.*s' is not a function",
                            (int)symbol_name.size, symbol_name.data);
  }
  const iree_string_view_t export_name =
      loom_func_like_export_name(module, symbol, function);
  if (iree_string_view_is_empty(export_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM testbench callee '@%.*s' is not exported",
                            (int)symbol_name.size, symbol_name.data);
  }
  *out_export_name = export_name;
  return iree_ok_status();
}

static void loom_vm_testbench_append_compile_root(
    iree_string_view_t name, loom_vm_testbench_compile_roots_t* roots) {
  for (iree_host_size_t i = 0; i < roots->count; ++i) {
    if (iree_string_view_equal(roots->values[i], name)) {
      return;
    }
  }
  IREE_ASSERT_LT(roots->count, roots->capacity);
  roots->values[roots->count++] = name;
}

static iree_status_t loom_vm_testbench_collect_compile_roots(
    const loom_testbench_module_plan_t* module_plan,
    iree_allocator_t host_allocator,
    loom_vm_testbench_compile_roots_t* out_roots) {
  *out_roots = (loom_vm_testbench_compile_roots_t){0};
  iree_host_size_t function_call_count = 0;
  for (iree_host_size_t case_index = 0; case_index < module_plan->case_count;
       ++case_index) {
    const loom_testbench_case_plan_t* case_plan =
        &module_plan->cases[case_index];
    for (iree_host_size_t invocation_index = 0;
         invocation_index < case_plan->invocation_count; ++invocation_index) {
      const loom_testbench_invocation_plan_t* invocation =
          &case_plan->invocations[invocation_index];
      if (invocation->kind != LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL) {
        continue;
      }
      if (!iree_host_size_checked_add(function_call_count, 1,
                                      &function_call_count)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "VM testbench function-call count overflow");
      }
      out_roots->max_argument_count =
          iree_max(out_roots->max_argument_count, invocation->input_count);
      out_roots->max_result_count =
          iree_max(out_roots->max_result_count, invocation->result_count);
    }
  }
  if (function_call_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM testbench module has no function calls");
  }

  if (!iree_host_size_checked_add(function_call_count, 1,
                                  &out_roots->capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM testbench compile root capacity overflow");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, out_roots->capacity, sizeof(*out_roots->values),
      (void**)&out_roots->values));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t case_index = 0;
       iree_status_is_ok(status) && case_index < module_plan->case_count;
       ++case_index) {
    const loom_testbench_case_plan_t* case_plan =
        &module_plan->cases[case_index];
    for (iree_host_size_t invocation_index = 0;
         iree_status_is_ok(status) &&
         invocation_index < case_plan->invocation_count;
         ++invocation_index) {
      const loom_testbench_invocation_plan_t* invocation =
          &case_plan->invocations[invocation_index];
      if (invocation->kind != LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL) {
        continue;
      }
      iree_string_view_t symbol_name = iree_string_view_empty();
      status = loom_vm_testbench_symbol_name_from_ref(
          module_plan->module, invocation->callee_ref, &symbol_name);
      if (iree_status_is_ok(status)) {
        loom_vm_testbench_append_compile_root(symbol_name, out_roots);
      }
    }
  }

  bool has_initializer = false;
  const loom_module_t* module = module_plan->module;
  for (iree_host_size_t symbol_index = 0;
       iree_status_is_ok(status) && symbol_index < module->symbols.count;
       ++symbol_index) {
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_index];
    const loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_isa(function)) {
      continue;
    }
    if (symbol->name_id >= module->strings.count) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM testbench function %" PRIhsz " has no valid name", symbol_index);
      continue;
    }
    const iree_string_view_t symbol_name =
        module->strings.entries[symbol->name_id];
    if (loom_func_like_cc(function) != LOOM_FUNC_CC_INITIALIZER) {
      continue;
    }
    if (has_initializer) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM testbench module defines more than one initializer");
      break;
    }
    has_initializer = true;
    loom_vm_testbench_append_compile_root(symbol_name, out_roots);
  }

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, out_roots->values);
    *out_roots = (loom_vm_testbench_compile_roots_t){0};
  }
  return status;
}

static iree_string_view_t loom_vm_testbench_module_name(
    const loom_module_t* module) {
  if (module->name_id < module->strings.count) {
    const iree_string_view_t name = module->strings.entries[module->name_id];
    if (!iree_string_view_is_empty(name)) {
      return name;
    }
  }
  return IREE_SV("test");
}

static iree_status_t loom_vm_testbench_select_emitter(
    const loom_target_environment_t* target_environment,
    const loom_target_emitter_t** out_emitter) {
  *out_emitter = NULL;
  const loom_target_emitter_list_t emitters =
      loom_target_environment_emitter_list(target_environment);
  for (iree_host_size_t i = 0; i < emitters.count; ++i) {
    const loom_target_emitter_t* emitter = emitters.values[i];
    if (emitter->target_artifact_format !=
        LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE) {
      continue;
    }
    if (*out_emitter != NULL) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "target environment contributes multiple VM bytecode emitters");
    }
    *out_emitter = emitter;
  }
  if (*out_emitter == NULL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "target environment has no VM bytecode emitter");
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_testbench_compile_program(
    const loom_vm_testbench_actual_options_t* options,
    iree_vm_program_t** out_program, iree_host_size_t* out_argument_capacity,
    iree_host_size_t* out_result_capacity) {
  *out_program = NULL;
  *out_argument_capacity = 0;
  *out_result_capacity = 0;

  loom_vm_testbench_compile_roots_t roots = {0};
  loom_module_t* compile_module = NULL;
  loom_compile_pipeline_result_t pipeline_result = {0};
  loom_target_emit_artifact_t artifact = {0};
  iree_byte_span_t image = iree_byte_span_empty();
  iree_vm_environment_t* environment = NULL;
  iree_vm_module_t* runtime_module = NULL;
  iree_vm_program_t* program = NULL;
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loom_run_session_block_pool(options->session),
                        &scratch_arena);
  loom_compile_pipeline_options_t pipeline_options = {0};
  loom_compile_pipeline_options_initialize(&pipeline_options);
  pipeline_options.pipeline = options->pipeline;
  pipeline_options.target_environment = options->target_environment;
  pipeline_options.low_descriptor_registry =
      loom_run_session_low_descriptor_registry(options->session);
  pipeline_options.source_resolver =
      loom_run_module_source_resolver(options->run_module);

  iree_status_t status = loom_vm_testbench_collect_compile_roots(
      options->module_plan, options->host_allocator, &roots);
  const loom_module_t* source_modules[] = {options->run_module->module};
  const iree_string_view_t module_name =
      loom_vm_testbench_module_name(options->run_module->module);
  if (iree_status_is_ok(status)) {
    const loom_link_options_t link_options = {
        .module_name = module_name,
        .root_symbols =
            {
                .count = roots.count,
                .values = roots.values,
            },
    };
    status = loom_link_materialized_modules(
        source_modules, IREE_ARRAYSIZE(source_modules), &link_options,
        loom_run_session_block_pool(options->session), options->host_allocator,
        &compile_module);
  }
  if (iree_status_is_ok(status)) {
    loom_tooling_config_materialize_options_t config_options = {0};
    loom_tooling_config_materialize_options_initialize(&config_options);
    config_options.config_set = options->config_set;
    status = loom_tooling_config_materialize_module(
        compile_module, &config_options,
        loom_run_session_block_pool(options->session), NULL);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_run_pipeline(
        compile_module, &pipeline_options,
        loom_run_session_block_pool(options->session), &pipeline_result);
    if (iree_status_is_ok(status) && pipeline_result.pass.error_count != 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "VM testbench compilation reported %u errors",
                                pipeline_result.pass.error_count);
    }
  }
  const loom_target_emitter_t* emitter = NULL;
  if (iree_status_is_ok(status)) {
    status =
        loom_vm_testbench_select_emitter(options->target_environment, &emitter);
  }
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter = {0};
  if (iree_status_is_ok(status)) {
    const loom_target_entry_options_t diagnostic_options = {
        .function_versions = &pipeline_result.function_versions.list,
        .diagnostic_sink = pipeline_options.diagnostic_sink,
        .source_resolver = pipeline_options.source_resolver,
        .max_errors = pipeline_options.max_errors,
    };
    loom_target_entry_diagnostic_emitter_initialize(
        compile_module, &diagnostic_options, LOOM_EMITTER_VERIFIER,
        &diagnostic_emitter);
    const loom_target_emit_request_t emit_request = {
        .target_environment = options->target_environment,
        .low_descriptor_registry =
            &loom_run_session_low_descriptor_registry(options->session)
                 ->registry,
        .module = compile_module,
        .function_versions = &pipeline_result.function_versions.list,
        .identifier = IREE_SVL("test.vm"),
        .diagnostic_emitter = loom_target_entry_emitter(&diagnostic_emitter),
        .scratch_arena = &scratch_arena,
        .allocator = options->host_allocator,
    };
    status = emitter->emit(&emit_request, &artifact);
    if (iree_status_is_ok(status) && diagnostic_emitter.error_count != 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "VM testbench emission reported %u errors",
                                diagnostic_emitter.error_count);
    } else if (iree_status_is_ok(status) &&
               (artifact.contents == NULL ||
                iree_byte_sequence_length(artifact.contents) == 0)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "VM bytecode emitter produced no artifact");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_byte_sequence_clone(artifact.contents,
                                      options->host_allocator, &image);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_environment_allocate(options->host_allocator, &environment);
  }
  if (iree_status_is_ok(status)) {
    const iree_vm_bytecode_module_storage_t storage = {
        .contents = iree_make_const_byte_span(image.data, image.data_length),
        .deallocator = options->host_allocator,
    };
    status = iree_vm_bytecode_module_create(environment, module_name, storage,
                                            options->host_allocator,
                                            &runtime_module);
    if (iree_status_is_ok(status)) {
      image = iree_byte_span_empty();
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_vm_program_modules_t modules = {
        .executable = runtime_module,
        .libraries = iree_vm_module_span_empty(),
    };
    status = iree_vm_program_create(modules, options->host_allocator, &program);
  }

  if (iree_status_is_ok(status)) {
    *out_program = program;
    *out_argument_capacity = roots.max_argument_count;
    *out_result_capacity = roots.max_result_count;
    program = NULL;
  }
  iree_vm_program_release(program);
  iree_vm_module_release(runtime_module);
  iree_vm_environment_free(environment);
  iree_allocator_free(options->host_allocator, image.data);
  loom_target_emit_artifact_release(&artifact);
  loom_compile_pipeline_result_deinitialize(&pipeline_result);
  loom_module_free(compile_module);
  iree_allocator_free(options->host_allocator, roots.values);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

static iree_status_t loom_vm_testbench_vm_scalar_type(
    loom_scalar_type_t scalar_type, iree_vm_scalar_type_t* out_vm_scalar_type) {
  *out_vm_scalar_type = loom_vm_call_abi_scalar_type(scalar_type);
  if (*out_vm_scalar_type == IREE_VM_SCALAR_TYPE_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "VM ABI does not support Loom scalar %s",
                            loom_scalar_type_name(scalar_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_testbench_tooling_scalar_bits(
    loom_scalar_type_t scalar_type, const iree_tooling_value_t* value,
    uint64_t* out_bits) {
  *out_bits = 0;
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_I32: {
      uint32_t bits = 0;
      if (value->kind == IREE_TOOLING_VALUE_KIND_I32) {
        memcpy(&bits, &value->storage.i32, sizeof(bits));
      } else if (value->kind == IREE_TOOLING_VALUE_KIND_U32) {
        bits = value->storage.u32;
      } else if (value->kind == IREE_TOOLING_VALUE_KIND_RAW_U32) {
        bits = value->storage.u32;
      } else {
        break;
      }
      *out_bits = bits;
      return iree_ok_status();
    }
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I64:
      if (value->kind == IREE_TOOLING_VALUE_KIND_I64) {
        memcpy(out_bits, &value->storage.i64, sizeof(*out_bits));
        return iree_ok_status();
      } else if (value->kind == IREE_TOOLING_VALUE_KIND_U64) {
        *out_bits = value->storage.u64;
        return iree_ok_status();
      }
      break;
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      if (value->kind == IREE_TOOLING_VALUE_KIND_RAW_U32) {
        *out_bits = value->storage.u32;
        return iree_ok_status();
      }
      break;
    case LOOM_SCALAR_TYPE_F32:
      if (value->kind == IREE_TOOLING_VALUE_KIND_F32) {
        uint32_t bits = 0;
        memcpy(&bits, &value->storage.f32, sizeof(bits));
        *out_bits = bits;
        return iree_ok_status();
      }
      break;
    case LOOM_SCALAR_TYPE_F64:
      if (value->kind == IREE_TOOLING_VALUE_KIND_F64) {
        memcpy(out_bits, &value->storage.f64, sizeof(*out_bits));
        return iree_ok_status();
      }
      break;
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "VM testbench cannot materialize scalar boundary type %s",
          loom_scalar_type_name(scalar_type));
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "VM testbench scalar %s received tooling value kind %u",
      loom_scalar_type_name(scalar_type), (unsigned)value->kind);
}

static iree_status_t loom_vm_testbench_variant_from_value(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_testbench_value_t* value, iree_vm_variant_t* out_variant) {
  if (value_id >= module->values.count ||
      !loom_testbench_value_is_scalar(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM testbench arguments must be scalar values");
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_scalar(type)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "VM testbench argument type is not scalar");
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  iree_vm_scalar_type_t vm_scalar_type = IREE_VM_SCALAR_TYPE_NONE;
  IREE_RETURN_IF_ERROR(
      loom_vm_testbench_vm_scalar_type(scalar_type, &vm_scalar_type));
  uint64_t bits = 0;
  IREE_RETURN_IF_ERROR(loom_vm_testbench_tooling_scalar_bits(
      scalar_type, &value->scalar, &bits));
  return iree_vm_variant_from_scalar_bits(vm_scalar_type, bits, out_variant);
}

static iree_status_t loom_vm_testbench_value_from_variant(
    const loom_module_t* module, loom_value_id_t value_id,
    iree_vm_variant_t variant, loom_testbench_value_t* out_value) {
  if (value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM testbench result value is out of range");
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_scalar(type)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "VM testbench result type is not scalar");
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  iree_vm_scalar_type_t vm_scalar_type = IREE_VM_SCALAR_TYPE_NONE;
  IREE_RETURN_IF_ERROR(
      loom_vm_testbench_vm_scalar_type(scalar_type, &vm_scalar_type));
  uint64_t bits = 0;
  IREE_RETURN_IF_ERROR(
      iree_vm_scalar_bits_from_variant(variant, vm_scalar_type, &bits));

  iree_tooling_value_t tooling_value = {0};
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I1:
      tooling_value.kind = IREE_TOOLING_VALUE_KIND_I32;
      tooling_value.storage.i32 = (int32_t)(uint8_t)bits;
      break;
    case LOOM_SCALAR_TYPE_I8: {
      const uint8_t unsigned_value = (uint8_t)bits;
      int8_t signed_value = 0;
      memcpy(&signed_value, &unsigned_value, sizeof(signed_value));
      tooling_value.kind = IREE_TOOLING_VALUE_KIND_I32;
      tooling_value.storage.i32 = signed_value;
      break;
    }
    case LOOM_SCALAR_TYPE_I16: {
      const uint16_t unsigned_value = (uint16_t)bits;
      int16_t signed_value = 0;
      memcpy(&signed_value, &unsigned_value, sizeof(signed_value));
      tooling_value.kind = IREE_TOOLING_VALUE_KIND_I32;
      tooling_value.storage.i32 = signed_value;
      break;
    }
    case LOOM_SCALAR_TYPE_I32:
      tooling_value.kind = IREE_TOOLING_VALUE_KIND_I32;
      memcpy(&tooling_value.storage.i32, &bits,
             sizeof(tooling_value.storage.i32));
      break;
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I64:
      tooling_value.kind = IREE_TOOLING_VALUE_KIND_I64;
      memcpy(&tooling_value.storage.i64, &bits,
             sizeof(tooling_value.storage.i64));
      break;
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      tooling_value.kind = IREE_TOOLING_VALUE_KIND_RAW_U32;
      tooling_value.storage.u32 = (uint32_t)bits;
      break;
    case LOOM_SCALAR_TYPE_F32:
      tooling_value.kind = IREE_TOOLING_VALUE_KIND_F32;
      memcpy(&tooling_value.storage.f32, &bits,
             sizeof(tooling_value.storage.f32));
      break;
    case LOOM_SCALAR_TYPE_F64:
      tooling_value.kind = IREE_TOOLING_VALUE_KIND_F64;
      memcpy(&tooling_value.storage.f64, &bits,
             sizeof(tooling_value.storage.f64));
      break;
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "VM testbench cannot expose scalar boundary type %s",
          loom_scalar_type_name(scalar_type));
  }
  *out_value = (loom_testbench_value_t){
      .kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR,
      .scalar = tooling_value,
  };
  return iree_ok_status();
}

static iree_vm_variant_span_t loom_vm_testbench_argument_span(
    loom_vm_testbench_actual_t* actual, iree_host_size_t count) {
  return iree_vm_variant_span_from_ptr(count == 0 ? NULL : actual->arguments,
                                       count);
}

static iree_vm_variant_span_t loom_vm_testbench_result_span(
    loom_vm_testbench_actual_t* actual, iree_host_size_t count) {
  return iree_vm_variant_span_from_ptr(count == 0 ? NULL : actual->results,
                                       count);
}

static iree_status_t loom_vm_testbench_invoke_process(
    loom_vm_testbench_actual_t* actual, iree_vm_process_t* process,
    const loom_testbench_invocation_plan_t* invocation) {
  iree_string_view_t export_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_vm_testbench_resolve_function_export_name(
      actual->source_module, invocation->callee_ref, &export_name));
  iree_vm_function_t function = {0};
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, actual->module_name, export_name, &function));
  return iree_vm_invoke(
      actual->invocation, function,
      loom_vm_testbench_argument_span(actual, invocation->input_count),
      loom_vm_testbench_result_span(actual, invocation->result_count));
}

static iree_status_t loom_vm_testbench_prepare_array_arguments(
    loom_vm_testbench_actual_t* actual,
    const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_t* inputs) {
  for (iree_host_size_t i = 0; i < invocation->input_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vm_testbench_variant_from_value(
        actual->source_module, invocation->input_value_ids[i], &inputs[i],
        &actual->arguments[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_testbench_prepare_table_arguments(
    loom_vm_testbench_actual_t* actual,
    const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* table) {
  for (iree_host_size_t i = 0; i < invocation->input_count; ++i) {
    const loom_testbench_value_t* input = NULL;
    IREE_RETURN_IF_ERROR(loom_testbench_value_table_lookup_borrow(
        table, invocation->input_value_ids[i], &input));
    IREE_RETURN_IF_ERROR(loom_vm_testbench_variant_from_value(
        actual->source_module, invocation->input_value_ids[i], input,
        &actual->arguments[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_testbench_store_array_results(
    loom_vm_testbench_actual_t* actual,
    const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_value_t* out_results) {
  for (iree_host_size_t i = 0; i < invocation->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vm_testbench_value_from_variant(
        actual->source_module, invocation->result_value_ids[i],
        actual->results[i], &out_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_testbench_store_table_results(
    loom_vm_testbench_actual_t* actual,
    const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_value_table_t* table) {
  for (iree_host_size_t i = 0; i < invocation->result_count; ++i) {
    loom_testbench_value_t result = {0};
    iree_status_t status = loom_vm_testbench_value_from_variant(
        actual->source_module, invocation->result_value_ids[i],
        actual->results[i], &result);
    if (iree_status_is_ok(status)) {
      status = loom_testbench_value_table_assign_move(
          table, invocation->result_value_ids[i], &result);
    }
    loom_testbench_value_deinitialize(&result);
    IREE_RETURN_IF_ERROR(status);
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_testbench_validate_invocation(
    const loom_vm_testbench_actual_t* actual,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t workload_count, iree_host_size_t input_count,
    iree_host_size_t result_count) {
  if (workload_count != 0 || input_count != invocation->input_count ||
      result_count != invocation->result_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM testbench invocation arity does not match its "
                            "source plan");
  }
  if (input_count > actual->argument_capacity ||
      result_count > actual->result_capacity) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM testbench invocation exceeds prepared scratch "
                            "capacity");
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_testbench_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t workload_count, const loom_testbench_value_t* workloads,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_host_size_t result_count, loom_testbench_value_t* out_results) {
  (void)workloads;
  loom_vm_testbench_actual_t* actual = (loom_vm_testbench_actual_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_vm_testbench_validate_invocation(
      actual, invocation, workload_count, input_count, result_count));

  iree_vm_process_t* process = NULL;
  iree_status_t status = iree_vm_process_create(
      actual->program, actual->invocation, iree_vm_variant_span_empty(),
      actual->host_allocator, &process);
  if (iree_status_is_ok(status)) {
    status =
        loom_vm_testbench_prepare_array_arguments(actual, invocation, inputs);
  }
  if (iree_status_is_ok(status)) {
    status = loom_vm_testbench_invoke_process(actual, process, invocation);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_vm_testbench_store_array_results(actual, invocation, out_results);
  }
  iree_vm_variant_span_reset(
      loom_vm_testbench_argument_span(actual, invocation->input_count));
  iree_vm_variant_span_reset(
      loom_vm_testbench_result_span(actual, invocation->result_count));
  iree_vm_process_release(process);
  return status;
}

static iree_status_t loom_vm_testbench_invoke_sequence(
    void* user_data, iree_host_size_t sample_ordinal,
    iree_host_size_t invocation_count,
    const loom_testbench_prepared_invocation_t* invocations,
    loom_testbench_value_table_t* table) {
  (void)sample_ordinal;
  loom_vm_testbench_actual_t* actual = (loom_vm_testbench_actual_t*)user_data;
  iree_vm_process_t* process = NULL;
  iree_status_t status = iree_vm_process_create(
      actual->program, actual->invocation, iree_vm_variant_span_empty(),
      actual->host_allocator, &process);
  for (iree_host_size_t invocation_index = 0;
       iree_status_is_ok(status) && invocation_index < invocation_count;
       ++invocation_index) {
    const loom_testbench_invocation_plan_t* invocation =
        invocations[invocation_index].plan;
    iree_host_size_t argument_reset_count = 0;
    iree_host_size_t result_reset_count = 0;
    status = loom_vm_testbench_validate_invocation(actual, invocation, 0,
                                                   invocation->input_count,
                                                   invocation->result_count);
    if (iree_status_is_ok(status)) {
      argument_reset_count = invocation->input_count;
      result_reset_count = invocation->result_count;
      status =
          loom_vm_testbench_prepare_table_arguments(actual, invocation, table);
    }
    if (iree_status_is_ok(status)) {
      status = loom_vm_testbench_invoke_process(actual, process, invocation);
    }
    if (iree_status_is_ok(status)) {
      status = loom_vm_testbench_store_table_results(actual, invocation, table);
    }
    iree_vm_variant_span_reset(
        loom_vm_testbench_argument_span(actual, argument_reset_count));
    iree_vm_variant_span_reset(
        loom_vm_testbench_result_span(actual, result_reset_count));
  }
  iree_vm_process_release(process);
  return status;
}

iree_status_t loom_vm_testbench_actual_initialize(
    const loom_vm_testbench_actual_options_t* options,
    loom_vm_testbench_actual_t* out_actual) {
  if (out_actual == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_actual is required");
  }
  *out_actual = (loom_vm_testbench_actual_t){0};
  if (options == NULL || options->session == NULL ||
      options->target_environment == NULL || options->run_module == NULL ||
      options->run_module->module == NULL || options->module_plan == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM testbench actual requires complete options");
  }

  loom_vm_testbench_actual_options_t normalized_options = *options;
  if (iree_allocator_is_null(normalized_options.host_allocator)) {
    normalized_options.host_allocator = iree_allocator_system();
  }
  loom_vm_testbench_actual_t actual = {
      .source_module = normalized_options.run_module->module,
      .module_name =
          loom_vm_testbench_module_name(normalized_options.run_module->module),
      .host_allocator = normalized_options.host_allocator,
  };
  iree_status_t status = loom_vm_testbench_compile_program(
      &normalized_options, &actual.program, &actual.argument_capacity,
      &actual.result_capacity);
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_invocation_allocate(LOOM_VM_TESTBENCH_INVOCATION_STORAGE_SIZE,
                                    actual.host_allocator, &actual.invocation);
  }
  iree_host_size_t variant_capacity = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_add(actual.argument_capacity,
                                  actual.result_capacity, &variant_capacity)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM testbench variant capacity overflow");
  }
  if (iree_status_is_ok(status) && variant_capacity != 0) {
    status = iree_allocator_malloc_array(
        actual.host_allocator, variant_capacity,
        sizeof(*actual.variant_storage), (void**)&actual.variant_storage);
    if (iree_status_is_ok(status)) {
      memset(actual.variant_storage, 0,
             variant_capacity * sizeof(*actual.variant_storage));
      actual.arguments = actual.variant_storage;
      actual.results = actual.variant_storage + actual.argument_capacity;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_actual = actual;
  } else {
    loom_vm_testbench_actual_deinitialize(&actual);
  }
  return status;
}

void loom_vm_testbench_actual_deinitialize(loom_vm_testbench_actual_t* actual) {
  if (actual == NULL) {
    return;
  }
  if (actual->variant_storage != NULL) {
    iree_vm_variant_span_reset(
        loom_vm_testbench_argument_span(actual, actual->argument_capacity));
    iree_vm_variant_span_reset(
        loom_vm_testbench_result_span(actual, actual->result_capacity));
  }
  iree_allocator_free(actual->host_allocator, actual->variant_storage);
  iree_vm_invocation_free(actual->invocation);
  iree_vm_program_release(actual->program);
  *actual = (loom_vm_testbench_actual_t){0};
}

loom_testbench_invocation_provider_t loom_vm_testbench_actual_provider(
    loom_vm_testbench_actual_t* actual) {
  return (loom_testbench_invocation_provider_t){
      .invoke = loom_vm_testbench_invoke,
      .invoke_sequence = loom_vm_testbench_invoke_sequence,
      .user_data = actual,
  };
}
