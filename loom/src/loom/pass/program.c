// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/program.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/pass/ops.h"

typedef struct loom_pass_program_compiler_t {
  // Source module containing pass pipeline IR.
  const loom_module_t* module;
  // Caller-supplied compile options.
  const loom_pass_program_compile_options_t* options;
  // Program being populated.
  loom_pass_program_t* program;
} loom_pass_program_compiler_t;

typedef struct loom_pass_program_compile_scope_t {
  // pass.pipeline whose body is currently being compiled.
  const loom_op_t* pipeline_op;
  // Active expansion stack of pass.call ops.
  const loom_op_t* const* call_stack;
  // Number of entries in call_stack.
  iree_host_size_t call_stack_count;
} loom_pass_program_compile_scope_t;

static const char* loom_pass_program_anchor_name(loom_pass_kind_t kind) {
  switch (kind) {
    case LOOM_PASS_MODULE:
      return "module";
    case LOOM_PASS_FUNCTION:
      return "func";
    default:
      return "unknown";
  }
}

static loom_pass_kind_t loom_pass_program_kind_from_anchor(uint8_t anchor) {
  switch (anchor) {
    case LOOM_PASS_ANCHOR_MODULE:
      return LOOM_PASS_MODULE;
    case LOOM_PASS_ANCHOR_FUNC:
      return LOOM_PASS_FUNCTION;
    default:
      return LOOM_PASS_COUNT_;
  }
}

static iree_status_t loom_pass_program_string_from_id(
    const loom_module_t* module, loom_string_id_t string_id, const char* label,
    iree_string_view_t* out_string) {
  if (!module || !out_string || string_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid %s string id", label);
  }
  *out_string = module->strings.entries[string_id];
  return iree_ok_status();
}

static iree_status_t loom_pass_program_copy_string(
    iree_arena_allocator_t* arena, iree_string_view_t source,
    iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) {
    return iree_ok_status();
  }
  char* target = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, source.size, (void**)&target));
  memcpy(target, source.data, source.size);
  *out_string = iree_make_string_view(target, source.size);
  return iree_ok_status();
}

static iree_status_t loom_pass_program_copy_call_stack(
    loom_pass_program_t* program,
    const loom_pass_program_compile_scope_t* scope,
    const loom_op_t* const** out_call_stack) {
  *out_call_stack = NULL;
  if (scope->call_stack_count == 0) {
    return iree_ok_status();
  }
  const loom_op_t** call_stack = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&program->arena, scope->call_stack_count,
                                sizeof(*call_stack), (void**)&call_stack));
  memcpy(call_stack, scope->call_stack,
         scope->call_stack_count * sizeof(*call_stack));
  *out_call_stack = call_stack;
  return iree_ok_status();
}

static iree_status_t loom_pass_program_push_call(
    loom_pass_program_t* program,
    const loom_pass_program_compile_scope_t* scope, const loom_op_t* call_op,
    loom_pass_program_compile_scope_t* out_scope) {
  iree_host_size_t new_count = scope->call_stack_count + 1;
  const loom_op_t** call_stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &program->arena, new_count, sizeof(*call_stack), (void**)&call_stack));
  if (scope->call_stack_count > 0) {
    memcpy(call_stack, scope->call_stack,
           scope->call_stack_count * sizeof(*call_stack));
  }
  call_stack[scope->call_stack_count] = call_op;
  *out_scope = *scope;
  out_scope->call_stack = call_stack;
  out_scope->call_stack_count = new_count;
  return iree_ok_status();
}

static iree_status_t loom_pass_program_emit_instruction(
    loom_pass_program_compiler_t* compiler,
    const loom_pass_program_compile_scope_t* scope, const loom_op_t* source_op,
    loom_pass_program_instruction_kind_t instruction_kind,
    loom_pass_kind_t anchor_kind, iree_host_size_t* out_index) {
  loom_pass_program_t* program = compiler->program;
  if (program->instruction_count >= program->instruction_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &program->arena, program->instruction_count,
        program->instruction_capacity == 0 ? 16
                                           : program->instruction_capacity * 2,
        sizeof(program->instructions[0]), &program->instruction_capacity,
        (void**)&program->instructions));
  }

  const loom_op_t* const* call_stack = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_program_copy_call_stack(program, scope, &call_stack));

  iree_host_size_t index = program->instruction_count++;
  program->instructions[index] = (loom_pass_program_instruction_t){
      .kind = instruction_kind,
      .anchor_kind = anchor_kind,
      .source =
          {
              .op = source_op,
              .pipeline_op = scope->pipeline_op,
              .call_stack = call_stack,
              .call_stack_count = scope->call_stack_count,
          },
  };
  *out_index = index;
  return iree_ok_status();
}

static iree_status_t loom_pass_program_copy_attr_value(
    loom_pass_program_compiler_t* compiler, loom_attribute_t source_attr,
    loom_pass_program_attr_value_t* out_value);

static iree_status_t loom_pass_program_copy_attrs(
    loom_pass_program_compiler_t* compiler,
    loom_named_attr_slice_t source_attrs,
    loom_pass_program_attr_list_t* out_attrs) {
  *out_attrs = (loom_pass_program_attr_list_t){0};
  if (source_attrs.count == 0) {
    return iree_ok_status();
  }

  loom_pass_program_attr_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&compiler->program->arena, source_attrs.count,
                                sizeof(*attrs), (void**)&attrs));
  for (iree_host_size_t i = 0; i < source_attrs.count; ++i) {
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_pass_program_string_from_id(
        compiler->module, source_attrs.entries[i].name_id,
        "pass predicate attr name", &name));
    IREE_RETURN_IF_ERROR(loom_pass_program_copy_string(
        &compiler->program->arena, name, &attrs[i].name));
    IREE_RETURN_IF_ERROR(loom_pass_program_copy_attr_value(
        compiler, source_attrs.entries[i].value, &attrs[i].value));
  }

  *out_attrs = (loom_pass_program_attr_list_t){
      .attrs = attrs,
      .attr_count = source_attrs.count,
  };
  return iree_ok_status();
}

static iree_status_t loom_pass_program_copy_attr_value(
    loom_pass_program_compiler_t* compiler, loom_attribute_t source_attr,
    loom_pass_program_attr_value_t* out_value) {
  *out_value = (loom_pass_program_attr_value_t){
      .kind = source_attr.kind,
      .count = source_attr.count,
  };
  switch (source_attr.kind) {
    case LOOM_ATTR_ABSENT:
      return iree_ok_status();
    case LOOM_ATTR_I64:
      out_value->i64_value = source_attr.i64;
      return iree_ok_status();
    case LOOM_ATTR_F64:
      out_value->f64_value = source_attr.f64;
      return iree_ok_status();
    case LOOM_ATTR_STRING: {
      iree_string_view_t source_string = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_pass_program_string_from_id(
          compiler->module, source_attr.string_id, "pass predicate attr value",
          &source_string));
      return loom_pass_program_copy_string(
          &compiler->program->arena, source_string, &out_value->string_value);
    }
    case LOOM_ATTR_BOOL:
      out_value->bool_value = source_attr.raw != 0;
      return iree_ok_status();
    case LOOM_ATTR_ENUM:
      out_value->enum_value = (uint8_t)source_attr.raw;
      return iree_ok_status();
    case LOOM_ATTR_I64_ARRAY: {
      if (source_attr.count == 0) {
        return iree_ok_status();
      }
      int64_t* values = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          &compiler->program->arena, source_attr.count, sizeof(*values),
          (void**)&values));
      memcpy(values, source_attr.i64_array,
             source_attr.count * sizeof(*values));
      out_value->i64_array = values;
      return iree_ok_status();
    }
    case LOOM_ATTR_SYMBOL:
      out_value->symbol_value = source_attr.symbol;
      return iree_ok_status();
    case LOOM_ATTR_TYPE:
      out_value->type_id = source_attr.type_id;
      return iree_ok_status();
    case LOOM_ATTR_PREDICATE_LIST: {
      if (source_attr.count == 0) {
        return iree_ok_status();
      }
      loom_predicate_t* predicates = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          &compiler->program->arena, source_attr.count, sizeof(*predicates),
          (void**)&predicates));
      memcpy(predicates, source_attr.predicate_list,
             source_attr.count * sizeof(*predicates));
      out_value->predicates = predicates;
      return iree_ok_status();
    }
    case LOOM_ATTR_DICT: {
      loom_pass_program_attr_list_t dictionary_attrs = {0};
      IREE_RETURN_IF_ERROR(loom_pass_program_copy_attrs(
          compiler, loom_attr_as_dict(source_attr), &dictionary_attrs));
      out_value->dictionary_attrs = dictionary_attrs.attrs;
      out_value->count = (uint16_t)dictionary_attrs.attr_count;
      return iree_ok_status();
    }
    case LOOM_ATTR_ENCODING:
      out_value->encoding_id = (uint16_t)source_attr.encoding_id;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported pass predicate attr kind %u",
                              (unsigned)source_attr.kind);
  }
}

static iree_status_t loom_pass_program_resolve_pipeline_ref(
    const loom_module_t* module, loom_symbol_ref_t ref,
    const loom_op_t** out_pipeline_op) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid pass.call callee symbol reference");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (!symbol->defining_op || !loom_pass_pipeline_isa(symbol->defining_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass.call callee does not resolve to pipeline");
  }
  *out_pipeline_op = symbol->defining_op;
  return iree_ok_status();
}

static iree_status_t loom_pass_program_compile_region(
    loom_pass_program_compiler_t* compiler,
    const loom_pass_program_compile_scope_t* scope,
    loom_pass_kind_t current_kind, const loom_region_t* region);

static iree_status_t loom_pass_program_compile_run(
    loom_pass_program_compiler_t* compiler,
    const loom_pass_program_compile_scope_t* scope, const loom_op_t* op,
    loom_pass_kind_t current_kind) {
  loom_pass_invocation_t invocation = {0};
  IREE_RETURN_IF_ERROR(loom_pass_invocation_resolve_run_op(
      compiler->module, compiler->options->registry, op, current_kind,
      &compiler->program->arena, &invocation));

  loom_pass_decoded_options_t* decoded_options = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&compiler->program->arena,
                                           sizeof(*decoded_options),
                                           (void**)&decoded_options));
  *decoded_options = invocation.decoded_options;

  iree_host_size_t instruction_index = 0;
  IREE_RETURN_IF_ERROR(loom_pass_program_emit_instruction(
      compiler, scope, op, LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE, current_kind,
      &instruction_index));
  loom_module_pass_fn_t module_run = NULL;
  loom_function_pass_fn_t function_run = NULL;
  if (invocation.info->kind == LOOM_PASS_MODULE) {
    module_run = invocation.descriptor->module_run;
  } else if (invocation.info->kind == LOOM_PASS_FUNCTION) {
    function_run = invocation.descriptor->function_run;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor has unsupported kind");
  }
  compiler->program->instructions[instruction_index].invoke =
      (loom_pass_program_invoke_t){
          .descriptor = invocation.descriptor,
          .info = invocation.info,
          .decoded_options = decoded_options,
          .module_run = module_run,
          .function_run = function_run,
      };
  return iree_ok_status();
}

static iree_status_t loom_pass_program_compile_for(
    loom_pass_program_compiler_t* compiler,
    const loom_pass_program_compile_scope_t* scope, const loom_op_t* op,
    loom_pass_kind_t current_kind) {
  uint8_t anchor = loom_pass_for_anchor(op);
  loom_pass_kind_t body_kind = loom_pass_program_kind_from_anchor(anchor);
  iree_host_size_t instruction_index = 0;
  IREE_RETURN_IF_ERROR(loom_pass_program_emit_instruction(
      compiler, scope, op, LOOM_PASS_PROGRAM_INSTRUCTION_FOR_EACH_SYMBOL,
      current_kind, &instruction_index));
  compiler->program->instructions[instruction_index].for_each_symbol =
      (loom_pass_program_for_each_symbol_t){
          .symbol_kind = body_kind,
          .snapshot_kind =
              LOOM_PASS_PROGRAM_SYMBOL_SNAPSHOT_FUNCTIONS_BY_SYMBOL_ID,
          .body_start = compiler->program->instruction_count,
          .body_end = compiler->program->instruction_count,
      };
  IREE_RETURN_IF_ERROR(loom_pass_program_compile_region(
      compiler, scope, body_kind, loom_pass_for_body(op)));
  compiler->program->instructions[instruction_index].for_each_symbol.body_end =
      compiler->program->instruction_count;
  return iree_ok_status();
}

static iree_status_t loom_pass_program_compile_where(
    loom_pass_program_compiler_t* compiler,
    const loom_pass_program_compile_scope_t* scope, const loom_op_t* op,
    loom_pass_kind_t current_kind) {
  iree_string_view_t predicate = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_pass_program_string_from_id(
      compiler->module, loom_pass_where_predicate(op), "pass.where predicate",
      &predicate));
  iree_host_size_t instruction_index = 0;
  IREE_RETURN_IF_ERROR(loom_pass_program_emit_instruction(
      compiler, scope, op, LOOM_PASS_PROGRAM_INSTRUCTION_WHERE, current_kind,
      &instruction_index));
  loom_pass_program_where_t where = {0};
  IREE_RETURN_IF_ERROR(loom_pass_program_copy_string(
      &compiler->program->arena, predicate, &where.predicate));
  IREE_RETURN_IF_ERROR(loom_pass_program_copy_attrs(
      compiler, loom_pass_where_attrs(op), &where.attrs));
  where.body_start = compiler->program->instruction_count;
  where.body_end = compiler->program->instruction_count;
  compiler->program->instructions[instruction_index].where = where;
  IREE_RETURN_IF_ERROR(loom_pass_program_compile_region(
      compiler, scope, current_kind, loom_pass_where_body(op)));
  compiler->program->instructions[instruction_index].where.body_end =
      compiler->program->instruction_count;
  return iree_ok_status();
}

static iree_status_t loom_pass_program_compile_repeat(
    loom_pass_program_compiler_t* compiler,
    const loom_pass_program_compile_scope_t* scope, const loom_op_t* op,
    loom_pass_kind_t current_kind) {
  iree_host_size_t instruction_index = 0;
  IREE_RETURN_IF_ERROR(loom_pass_program_emit_instruction(
      compiler, scope, op, LOOM_PASS_PROGRAM_INSTRUCTION_REPEAT, current_kind,
      &instruction_index));
  compiler->program->instructions[instruction_index].repeat =
      (loom_pass_program_repeat_t){
          .mode = loom_pass_repeat_mode(op),
          .count = loom_pass_repeat_count(op),
          .max_iterations = loom_pass_repeat_max_iterations(op),
          .body_start = compiler->program->instruction_count,
          .body_end = compiler->program->instruction_count,
      };
  IREE_RETURN_IF_ERROR(loom_pass_program_compile_region(
      compiler, scope, current_kind, loom_pass_repeat_body(op)));
  compiler->program->instructions[instruction_index].repeat.body_end =
      compiler->program->instruction_count;
  return iree_ok_status();
}

static iree_status_t loom_pass_program_compile_call(
    loom_pass_program_compiler_t* compiler,
    const loom_pass_program_compile_scope_t* scope, const loom_op_t* op,
    loom_pass_kind_t current_kind) {
  const loom_op_t* callee_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_program_resolve_pipeline_ref(
      compiler->module, loom_pass_call_callee(op), &callee_op));
  loom_pass_kind_t callee_kind =
      loom_pass_program_kind_from_anchor(loom_pass_pipeline_anchor(callee_op));
  if (callee_kind != current_kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass.call callee anchor %s does not match current %s anchor",
        loom_pass_program_anchor_name(callee_kind),
        loom_pass_program_anchor_name(current_kind));
  }

  loom_pass_program_compile_scope_t callee_scope = {0};
  IREE_RETURN_IF_ERROR(
      loom_pass_program_push_call(compiler->program, scope, op, &callee_scope));
  callee_scope.pipeline_op = callee_op;
  return loom_pass_program_compile_region(compiler, &callee_scope, current_kind,
                                          loom_pass_pipeline_body(callee_op));
}

static iree_status_t loom_pass_program_compile_message(
    loom_pass_program_compiler_t* compiler,
    const loom_pass_program_compile_scope_t* scope, const loom_op_t* op,
    loom_pass_kind_t current_kind,
    loom_pass_program_instruction_kind_t instruction_kind,
    loom_string_id_t message_id) {
  iree_string_view_t message = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_pass_program_string_from_id(
      compiler->module, message_id, "pass message", &message));
  iree_host_size_t instruction_index = 0;
  IREE_RETURN_IF_ERROR(loom_pass_program_emit_instruction(
      compiler, scope, op, instruction_kind, current_kind, &instruction_index));
  return loom_pass_program_copy_string(
      &compiler->program->arena, message,
      &compiler->program->instructions[instruction_index].message.message);
}

static iree_status_t loom_pass_program_compile_region(
    loom_pass_program_compiler_t* compiler,
    const loom_pass_program_compile_scope_t* scope,
    loom_pass_kind_t current_kind, const loom_region_t* region) {
  const loom_block_t* entry_block = loom_region_const_entry_block(region);
  const loom_op_t* last_op = entry_block->last_op;
  const loom_op_t* op = NULL;
  loom_block_for_each_op(entry_block, op) {
    if (op == last_op) {
      break;
    }
    switch (op->kind) {
      case LOOM_OP_PASS_FOR: {
        IREE_RETURN_IF_ERROR(
            loom_pass_program_compile_for(compiler, scope, op, current_kind));
        break;
      }
      case LOOM_OP_PASS_WHERE: {
        IREE_RETURN_IF_ERROR(
            loom_pass_program_compile_where(compiler, scope, op, current_kind));
        break;
      }
      case LOOM_OP_PASS_REPEAT: {
        IREE_RETURN_IF_ERROR(loom_pass_program_compile_repeat(
            compiler, scope, op, current_kind));
        break;
      }
      case LOOM_OP_PASS_CALL: {
        IREE_RETURN_IF_ERROR(
            loom_pass_program_compile_call(compiler, scope, op, current_kind));
        break;
      }
      case LOOM_OP_PASS_RUN: {
        IREE_RETURN_IF_ERROR(
            loom_pass_program_compile_run(compiler, scope, op, current_kind));
        break;
      }
      case LOOM_OP_PASS_FAIL: {
        IREE_RETURN_IF_ERROR(loom_pass_program_compile_message(
            compiler, scope, op, current_kind,
            LOOM_PASS_PROGRAM_INSTRUCTION_FAIL, loom_pass_fail_message(op)));
        break;
      }
      case LOOM_OP_PASS_HALT: {
        IREE_RETURN_IF_ERROR(loom_pass_program_compile_message(
            compiler, scope, op, current_kind,
            LOOM_PASS_PROGRAM_INSTRUCTION_HALT, loom_pass_halt_message(op)));
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unexpected op in verified pass program");
    }
  }
  return iree_ok_status();
}

iree_status_t loom_pass_program_compile_pipeline(
    const loom_module_t* module, const loom_op_t* pipeline_op,
    const loom_pass_program_compile_options_t* options,
    iree_arena_block_pool_t* block_pool, loom_pass_program_t* out_program) {
  if (!loom_pass_pipeline_isa(pipeline_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected pass.pipeline op");
  }

  memset(out_program, 0, sizeof(*out_program));
  out_program->source_module = module;
  out_program->pipeline_op = pipeline_op;
  out_program->root_kind = loom_pass_program_kind_from_anchor(
      loom_pass_pipeline_anchor(pipeline_op));
  out_program->block_pool = block_pool;
  iree_arena_initialize(block_pool, &out_program->arena);

  iree_arena_allocator_t verify_arena;
  iree_arena_initialize(block_pool, &verify_arena);
  loom_pass_verify_options_t verify_options = {
      .registry = options->registry,
      .environment = options->environment,
      .predicate_provider = options->predicate_provider,
  };
  iree_status_t status = loom_pass_verify_pipeline_op(
      module, pipeline_op, &verify_options, &verify_arena);
  iree_arena_deinitialize(&verify_arena);

  loom_pass_program_compiler_t compiler = {
      .module = module,
      .options = options,
      .program = out_program,
  };
  loom_pass_program_compile_scope_t root_scope = {
      .pipeline_op = pipeline_op,
  };
  if (iree_status_is_ok(status)) {
    status = loom_pass_program_compile_region(
        &compiler, &root_scope, out_program->root_kind,
        loom_pass_pipeline_body(pipeline_op));
  }
  if (!iree_status_is_ok(status)) {
    loom_pass_program_deinitialize(out_program);
    return status;
  }
  return iree_ok_status();
}

void loom_pass_program_deinitialize(loom_pass_program_t* program) {
  if (!program) {
    return;
  }
  if (program->block_pool) {
    iree_arena_deinitialize(&program->arena);
  }
  memset(program, 0, sizeof(*program));
}
