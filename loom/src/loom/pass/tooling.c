// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/tooling.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/builder.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/program.h"
#include "loom/pass/registry.h"

static iree_status_t loom_pass_tool_verify_options(
    const loom_pass_tool_run_options_t* options) {
  return loom_pass_environment_verify(&options->environment);
}

static void loom_pass_tool_accumulate_result(
    loom_pass_run_result_t* result,
    const loom_pass_run_result_t* invocation_result) {
  result->error_count += invocation_result->error_count;
  result->warning_count += invocation_result->warning_count;
  result->remark_count += invocation_result->remark_count;
}

static iree_status_t loom_pass_tool_run_program(
    loom_module_t* module, const loom_pass_program_t* program,
    const loom_pass_tool_run_options_t* options,
    loom_pass_run_result_t* out_result) {
  loom_pass_interpreter_options_t interpreter_options = {
      .block_pool = options->block_pool,
      .predicate_provider = options->predicate_provider,
      .diagnostic_emitter = options->diagnostic_emitter,
      .environment = options->environment,
      .report = options->report,
      .trace = options->trace,
  };
  if (program->root_kind == LOOM_PASS_MODULE) {
    return loom_pass_interpreter_run_module(program, module,
                                            &interpreter_options, out_result);
  }
  if (program->root_kind != LOOM_PASS_FUNCTION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported pass pipeline root kind %d",
                            (int)program->root_kind);
  }

  iree_arena_allocator_t snapshot_arena;
  iree_arena_initialize(options->block_pool, &snapshot_arena);
  uint16_t* symbol_ids = NULL;
  iree_status_t status = iree_arena_allocate_array(
      &snapshot_arena, module->symbols.count > 0 ? module->symbols.count : 1,
      sizeof(*symbol_ids), (void**)&symbol_ids);
  iree_host_size_t symbol_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      loom_symbol_t* symbol = &module->symbols.entries[i];
      if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
        continue;
      }
      loom_func_like_t function =
          loom_func_like_cast(module, symbol->defining_op);
      if (!loom_func_like_body(function)) {
        continue;
      }
      symbol_ids[symbol_count++] = (uint16_t)i;
    }
  }

  for (iree_host_size_t i = 0; i < symbol_count && iree_status_is_ok(status) &&
                               out_result->error_count == 0;
       ++i) {
    uint16_t symbol_id = symbol_ids[i];
    if (symbol_id >= module->symbols.count) {
      continue;
    }
    loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_body(function)) {
      continue;
    }
    loom_pass_run_result_t invocation_result = {0};
    status = loom_pass_interpreter_run_function(
        program, module, function, &interpreter_options, &invocation_result);
    loom_pass_tool_accumulate_result(out_result, &invocation_result);
  }

  iree_arena_deinitialize(&snapshot_arena);
  return status;
}

iree_status_t loom_pass_tool_run_pipeline_op(
    loom_module_t* module, const loom_op_t* pipeline_op,
    const loom_pass_tool_run_options_t* options,
    loom_pass_run_result_t* out_result) {
  return loom_pass_tool_run_pipeline_module_op(module, module, pipeline_op,
                                               options, out_result);
}

iree_status_t loom_pass_tool_run_pipeline_module_op(
    loom_module_t* module, loom_module_t* pipeline_module,
    const loom_op_t* pipeline_op, const loom_pass_tool_run_options_t* options,
    loom_pass_run_result_t* out_result) {
  *out_result = (loom_pass_run_result_t){0};
  IREE_RETURN_IF_ERROR(loom_pass_tool_verify_options(options));

  loom_pass_program_compile_options_t compile_options = {
      .registry = options->registry,
      .environment = options->environment,
      .predicate_provider = options->predicate_provider,
  };
  loom_pass_program_t program = {0};
  iree_status_t status = loom_pass_program_compile_pipeline(
      pipeline_module, pipeline_op, &compile_options, options->block_pool,
      &program);
  if (iree_status_is_ok(status)) {
    status = loom_pass_tool_run_program(module, &program, options, out_result);
  }
  loom_pass_program_deinitialize(&program);
  return status;
}

static iree_string_view_t loom_pass_tool_trim_symbol_sigils(
    iree_string_view_t symbol) {
  symbol = iree_string_view_trim(symbol);
  while (symbol.size > 0 && symbol.data[0] == '@') {
    symbol = iree_string_view_substr(symbol, 1, IREE_STRING_VIEW_NPOS);
  }
  return symbol;
}

iree_status_t loom_pass_tool_run_pipeline_symbol(
    loom_module_t* module, iree_string_view_t pipeline_symbol,
    const loom_pass_tool_run_options_t* options,
    loom_pass_run_result_t* out_result) {
  *out_result = (loom_pass_run_result_t){0};
  IREE_RETURN_IF_ERROR(loom_pass_tool_verify_options(options));

  iree_string_view_t name = loom_pass_tool_trim_symbol_sigils(pipeline_symbol);
  if (iree_string_view_is_empty(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass pipeline symbol name is required");
  }
  loom_string_id_t name_id = loom_module_lookup_string(module, name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "pass pipeline @%.*s was not found", (int)name.size,
                            name.data);
  }
  uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "pass pipeline @%.*s was not found", (int)name.size,
                            name.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  if (!symbol->defining_op || !loom_pass_pipeline_isa(symbol->defining_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol @%.*s does not define a pass.pipeline",
                            (int)name.size, name.data);
  }
  return loom_pass_tool_run_pipeline_op(module, symbol->defining_op, options,
                                        out_result);
}

static const loom_pass_option_schema_t* loom_pass_tool_find_option_schema(
    const loom_pass_descriptor_t* descriptor, iree_string_view_t option_name) {
  for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
    const loom_pass_option_schema_t* schema = &descriptor->option_schema[i];
    if (iree_string_view_equal(schema->name, option_name)) {
      return schema;
    }
  }
  return NULL;
}

typedef struct loom_pass_tool_option_build_context_t {
  // Module that owns option key and string value IDs.
  loom_module_t* module;
  // Descriptor whose textual options are being translated.
  const loom_pass_descriptor_t* descriptor;
  // Mutable option entries, bounded by descriptor schema count.
  loom_named_attr_t* attrs;
  // Number of populated attrs entries.
  iree_host_size_t attr_count;
} loom_pass_tool_option_build_context_t;

static bool loom_pass_tool_has_option_attr(
    const loom_pass_tool_option_build_context_t* context,
    iree_string_view_t option_name) {
  for (iree_host_size_t i = 0; i < context->attr_count; ++i) {
    iree_string_view_t existing =
        context->module->strings.entries[context->attrs[i].name_id];
    if (iree_string_view_equal(existing, option_name)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_pass_tool_parse_option_attr(
    void* user_data, iree_string_view_t option_name,
    iree_string_view_t option_value) {
  loom_pass_tool_option_build_context_t* context =
      (loom_pass_tool_option_build_context_t*)user_data;
  const loom_pass_descriptor_t* descriptor = context->descriptor;
  const loom_pass_option_schema_t* schema =
      loom_pass_tool_find_option_schema(descriptor, option_name);
  if (!schema) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown option '%.*s' for pass '%.*s'",
                            (int)option_name.size, option_name.data,
                            (int)descriptor->key.size, descriptor->key.data);
  }
  if (loom_pass_tool_has_option_attr(context, option_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate option '%.*s' for pass '%.*s'",
                            (int)option_name.size, option_name.data,
                            (int)descriptor->key.size, descriptor->key.data);
  }
  if (context->attr_count >= descriptor->option_schema_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "too many options for pass '%.*s'",
                            (int)descriptor->key.size, descriptor->key.data);
  }

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(context->module, option_name, &name_id));

  loom_attribute_t value_attr = {0};
  switch (schema->kind) {
    case LOOM_PASS_OPTION_SCHEMA_STRING:
    case LOOM_PASS_OPTION_SCHEMA_ENUM: {
      loom_string_id_t value_id = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_module_intern_string(context->module, option_value, &value_id));
      value_attr = loom_attr_string(value_id);
      break;
    }
    case LOOM_PASS_OPTION_SCHEMA_UINT32: {
      uint32_t value = 0;
      IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
          descriptor->key, schema->name, option_value, &value));
      if (value < schema->minimum_uint32 || value > schema->maximum_uint32) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pass '%.*s' option '%.*s' must be in range %" PRIu32 "..%" PRIu32,
            (int)descriptor->key.size, descriptor->key.data,
            (int)schema->name.size, schema->name.data, schema->minimum_uint32,
            schema->maximum_uint32);
      }
      value_attr = loom_attr_i64(value);
      break;
    }
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' has unknown schema kind %d",
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data, (int)schema->kind);
  }

  context->attrs[context->attr_count++] = (loom_named_attr_t){
      .name_id = name_id,
      .value = value_attr,
  };
  return iree_ok_status();
}

static iree_status_t loom_pass_tool_build_option_attrs(
    loom_module_t* module, const loom_pass_descriptor_t* descriptor,
    iree_string_view_t options, loom_named_attr_slice_t* out_attrs) {
  *out_attrs = loom_make_named_attr_slice(NULL, 0);
  options = iree_string_view_trim(options);
  if (iree_string_view_is_empty(options)) {
    return iree_ok_status();
  }
  if (descriptor->option_schema_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass '%.*s' does not accept options, got '{%.*s}'",
                            (int)descriptor->key.size, descriptor->key.data,
                            (int)options.size, options.data);
  }

  loom_named_attr_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&module->arena, descriptor->option_schema_count,
                                sizeof(*attrs), (void**)&attrs));
  loom_pass_tool_option_build_context_t context = {
      .module = module,
      .descriptor = descriptor,
      .attrs = attrs,
  };
  IREE_RETURN_IF_ERROR(
      loom_pass_options_parse(descriptor->key, options,
                              (loom_pass_option_parse_callback_t){
                                  .fn = loom_pass_tool_parse_option_attr,
                                  .user_data = &context,
                              }));
  *out_attrs = loom_make_named_attr_slice(attrs, context.attr_count);
  return iree_ok_status();
}

static iree_status_t loom_pass_tool_build_flat_run(
    loom_builder_t* builder, const loom_pass_descriptor_t* descriptor,
    iree_string_view_t options) {
  loom_named_attr_slice_t option_attrs = {0};
  IREE_RETURN_IF_ERROR(loom_pass_tool_build_option_attrs(
      builder->module, descriptor, options, &option_attrs));
  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(builder, descriptor->key, option_attrs,
                                &run_op);
}

typedef struct loom_pass_tool_flat_function_group_t {
  // Builder receiving synthetic pass pipeline operations.
  loom_builder_t* builder;
  // Open pass.for body scope.
  loom_pass_ir_scope_t scope;
  // True when builder currently inserts into a pass.for<func> body.
  bool is_open;
} loom_pass_tool_flat_function_group_t;

static iree_status_t loom_pass_tool_flat_function_group_open(
    loom_pass_tool_flat_function_group_t* group) {
  if (group->is_open) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_pass_ir_begin_for(
      group->builder, LOOM_PASS_ANCHOR_FUNC, &group->scope));
  group->is_open = true;
  return iree_ok_status();
}

static iree_status_t loom_pass_tool_flat_function_group_close(
    loom_pass_tool_flat_function_group_t* group) {
  if (!group->is_open) {
    return iree_ok_status();
  }
  iree_status_t status = loom_pass_ir_end_scope(group->builder, &group->scope);
  group->is_open = false;
  return status;
}

static iree_status_t loom_pass_tool_flat_function_group_append_run(
    loom_pass_tool_flat_function_group_t* group,
    const loom_pass_descriptor_t* descriptor, iree_string_view_t options) {
  IREE_RETURN_IF_ERROR(loom_pass_tool_flat_function_group_open(group));
  return loom_pass_tool_build_flat_run(group->builder, descriptor, options);
}

typedef struct loom_pass_tool_flat_pipeline_build_context_t {
  // Registry used to resolve textual pass keys.
  const loom_pass_registry_t* registry;
  // Textual shallow pass list being converted into pass IR.
  iree_string_view_t pipeline;
} loom_pass_tool_flat_pipeline_build_context_t;

static iree_status_t loom_pass_tool_build_flat_pipeline_body(
    loom_builder_t* builder, void* user_data) {
  const loom_pass_tool_flat_pipeline_build_context_t* context =
      (const loom_pass_tool_flat_pipeline_build_context_t*)user_data;
  loom_pass_tool_flat_function_group_t function_group = {
      .builder = builder,
  };
  iree_string_view_t remaining = context->pipeline;
  iree_host_size_t pipeline_index = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    loom_pass_pipeline_entry_spec_t spec = {0};
    bool has_entry = false;
    status = loom_pass_pipeline_consume_entry(&remaining, &spec, &has_entry);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (!has_entry) {
      break;
    }

    const loom_pass_descriptor_t* descriptor = NULL;
    status =
        loom_pass_registry_lookup(context->registry, spec.name, &descriptor);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (!descriptor) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "unknown pass '%.*s' at pipeline entry %zu",
                           (int)spec.name.size, spec.name.data, pipeline_index);
      break;
    }
    if (!loom_pass_descriptor_is_available(descriptor)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "pass '%.*s' is unavailable: %.*s",
                                (int)descriptor->key.size, descriptor->key.data,
                                (int)descriptor->unavailable_reason.size,
                                descriptor->unavailable_reason.data);
      break;
    }
    const loom_pass_info_t* info = descriptor->info();
    if (!info) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "pass descriptor '%.*s' returned no info",
                           (int)descriptor->key.size, descriptor->key.data);
      break;
    }
    if (info->kind == LOOM_PASS_MODULE) {
      status = loom_pass_tool_flat_function_group_close(&function_group);
      if (iree_status_is_ok(status)) {
        status =
            loom_pass_tool_build_flat_run(builder, descriptor, spec.options);
      }
    } else if (info->kind == LOOM_PASS_FUNCTION) {
      status = loom_pass_tool_flat_function_group_append_run(
          &function_group, descriptor, spec.options);
    } else {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "pass '%.*s' has unknown kind %d",
          (int)descriptor->key.size, descriptor->key.data, (int)info->kind);
      break;
    }
    ++pipeline_index;
  }
  return iree_status_join(
      status, loom_pass_tool_flat_function_group_close(&function_group));
}

iree_status_t loom_pass_tool_build_flat_pipeline(
    loom_module_t* pipeline_module, iree_string_view_t pipeline,
    const loom_pass_registry_t* registry, const loom_op_t** out_pipeline_op) {
  *out_pipeline_op = NULL;
  const loom_pass_tool_flat_pipeline_build_context_t context = {
      .registry = registry,
      .pipeline = pipeline,
  };
  loom_op_t* pipeline_op = NULL;
  iree_status_t status = loom_pass_ir_build_pipeline(
      pipeline_module, IREE_SV("__command_line"), LOOM_PASS_ANCHOR_MODULE,
      loom_pass_tool_build_flat_pipeline_body, (void*)&context, &pipeline_op);
  if (iree_status_is_ok(status)) {
    *out_pipeline_op = pipeline_op;
  }
  return status;
}

iree_status_t loom_pass_tool_run_flat_pipeline(
    loom_module_t* module, iree_string_view_t pipeline,
    const loom_pass_tool_run_options_t* options,
    loom_pass_run_result_t* out_result) {
  *out_result = (loom_pass_run_result_t){0};
  IREE_RETURN_IF_ERROR(loom_pass_tool_verify_options(options));

  loom_module_t* pipeline_module = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate(
      module->context, IREE_SV("__pass_tool"), options->block_pool, NULL,
      iree_allocator_system(), &pipeline_module));
  const loom_op_t* pipeline_op = NULL;
  iree_status_t status = loom_pass_tool_build_flat_pipeline(
      pipeline_module, pipeline, options->registry, &pipeline_op);
  if (iree_status_is_ok(status)) {
    status = loom_pass_tool_run_pipeline_module_op(
        module, pipeline_module, pipeline_op, options, out_result);
  }
  loom_module_free(pipeline_module);
  return status;
}
