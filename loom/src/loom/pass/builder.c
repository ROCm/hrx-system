// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/builder.h"

#include "loom/ir/context.h"
#include "loom/ir/module.h"

static iree_status_t loom_pass_ir_enter_scope(loom_builder_t* builder,
                                              loom_op_t* owner_op,
                                              loom_region_t* region,
                                              loom_pass_ir_scope_t* out_scope) {
  *out_scope = (loom_pass_ir_scope_t){
      .owner_op = owner_op,
      .saved_insertion_point =
          loom_builder_enter_region(builder, owner_op, region),
  };
  return iree_ok_status();
}

static iree_status_t loom_pass_ir_build_scope(
    loom_builder_t* builder, loom_op_t* owner_op, loom_region_t* region,
    loom_pass_ir_body_build_fn_t build_body, void* user_data) {
  loom_pass_ir_scope_t scope = {0};
  IREE_RETURN_IF_ERROR(
      loom_pass_ir_enter_scope(builder, owner_op, region, &scope));
  iree_status_t status =
      build_body ? build_body(builder, user_data) : iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = loom_pass_ir_end_scope(builder, &scope);
  } else {
    loom_pass_ir_abandon_scope(builder, &scope);
  }
  return status;
}

iree_status_t loom_pass_ir_build_pipeline(
    loom_module_t* module, iree_string_view_t name, loom_pass_anchor_t anchor,
    loom_pass_ir_body_build_fn_t build_body, void* user_data,
    loom_op_t** out_pipeline_op) {
  *out_pipeline_op = NULL;
  if (!module || !module->context ||
      loom_context_resolve_op(module->context, LOOM_OP_PASS_PIPELINE) == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "pass IR construction requires a context with the pass dialect");
  }
  if (iree_string_view_is_empty(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass pipeline name is required");
  }

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  if (loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "pass pipeline symbol '@%.*s' already exists",
                            (int)name.size, name.data);
  }
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* pipeline_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_pipeline_build(
      &builder, anchor,
      (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id},
      LOOM_LOCATION_UNKNOWN, &pipeline_op));

  iree_status_t status = loom_pass_ir_build_scope(
      &builder, pipeline_op, loom_pass_pipeline_body(pipeline_op), build_body,
      user_data);
  if (iree_status_is_ok(status)) {
    *out_pipeline_op = pipeline_op;
  }
  return status;
}

iree_status_t loom_pass_ir_build_run(loom_builder_t* builder,
                                     iree_string_view_t key,
                                     loom_named_attr_slice_t options,
                                     loom_op_t** out_run_op) {
  *out_run_op = NULL;
  loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, key, &key_id));
  return loom_pass_run_build(builder, key_id, options, LOOM_LOCATION_UNKNOWN,
                             out_run_op);
}

iree_status_t loom_pass_ir_begin_for(loom_builder_t* builder,
                                     loom_pass_anchor_t anchor,
                                     loom_pass_ir_scope_t* out_scope) {
  *out_scope = (loom_pass_ir_scope_t){0};
  loom_op_t* for_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_for_build(builder, anchor, LOOM_LOCATION_UNKNOWN, &for_op));
  return loom_pass_ir_enter_scope(builder, for_op, loom_pass_for_body(for_op),
                                  out_scope);
}

iree_status_t loom_pass_ir_build_for(loom_builder_t* builder,
                                     loom_pass_anchor_t anchor,
                                     loom_pass_ir_body_build_fn_t build_body,
                                     void* user_data, loom_op_t** out_for_op) {
  *out_for_op = NULL;
  loom_op_t* for_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_for_build(builder, anchor, LOOM_LOCATION_UNKNOWN, &for_op));
  iree_status_t status = loom_pass_ir_build_scope(
      builder, for_op, loom_pass_for_body(for_op), build_body, user_data);
  if (iree_status_is_ok(status)) {
    *out_for_op = for_op;
  }
  return status;
}

iree_status_t loom_pass_ir_begin_where(loom_builder_t* builder,
                                       iree_string_view_t predicate,
                                       loom_named_attr_slice_t attrs,
                                       loom_pass_ir_scope_t* out_scope) {
  *out_scope = (loom_pass_ir_scope_t){0};
  loom_string_id_t predicate_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, predicate, &predicate_id));
  loom_op_t* where_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_where_build(builder, predicate_id, attrs,
                                             LOOM_LOCATION_UNKNOWN, &where_op));
  return loom_pass_ir_enter_scope(builder, where_op,
                                  loom_pass_where_body(where_op), out_scope);
}

iree_status_t loom_pass_ir_build_where(loom_builder_t* builder,
                                       iree_string_view_t predicate,
                                       loom_named_attr_slice_t attrs,
                                       loom_pass_ir_body_build_fn_t build_body,
                                       void* user_data,
                                       loom_op_t** out_where_op) {
  *out_where_op = NULL;
  loom_string_id_t predicate_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, predicate, &predicate_id));
  loom_op_t* where_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_where_build(builder, predicate_id, attrs,
                                             LOOM_LOCATION_UNKNOWN, &where_op));
  iree_status_t status = loom_pass_ir_build_scope(
      builder, where_op, loom_pass_where_body(where_op), build_body, user_data);
  if (iree_status_is_ok(status)) {
    *out_where_op = where_op;
  }
  return status;
}

iree_status_t loom_pass_ir_begin_repeat(
    loom_builder_t* builder, loom_pass_repeat_build_flags_t build_flags,
    loom_pass_repeat_mode_t mode, int64_t count, int64_t max_iterations,
    loom_pass_ir_scope_t* out_scope) {
  *out_scope = (loom_pass_ir_scope_t){0};
  loom_op_t* repeat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_repeat_build(builder, build_flags, mode, count, max_iterations,
                             LOOM_LOCATION_UNKNOWN, &repeat_op));
  return loom_pass_ir_enter_scope(builder, repeat_op,
                                  loom_pass_repeat_body(repeat_op), out_scope);
}

iree_status_t loom_pass_ir_build_repeat(
    loom_builder_t* builder, loom_pass_repeat_build_flags_t build_flags,
    loom_pass_repeat_mode_t mode, int64_t count, int64_t max_iterations,
    loom_pass_ir_body_build_fn_t build_body, void* user_data,
    loom_op_t** out_repeat_op) {
  *out_repeat_op = NULL;
  loom_op_t* repeat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_repeat_build(builder, build_flags, mode, count, max_iterations,
                             LOOM_LOCATION_UNKNOWN, &repeat_op));
  iree_status_t status = loom_pass_ir_build_scope(
      builder, repeat_op, loom_pass_repeat_body(repeat_op), build_body,
      user_data);
  if (iree_status_is_ok(status)) {
    *out_repeat_op = repeat_op;
  }
  return status;
}

iree_status_t loom_pass_ir_build_call(loom_builder_t* builder,
                                      loom_symbol_ref_t callee,
                                      loom_op_t** out_call_op) {
  *out_call_op = NULL;
  return loom_pass_call_build(builder, callee, LOOM_LOCATION_UNKNOWN,
                              out_call_op);
}

iree_status_t loom_pass_ir_build_yield(loom_builder_t* builder) {
  loom_op_t* yield_op = NULL;
  return loom_pass_yield_build(builder, LOOM_LOCATION_UNKNOWN, &yield_op);
}

iree_status_t loom_pass_ir_end_scope(loom_builder_t* builder,
                                     const loom_pass_ir_scope_t* scope) {
  iree_status_t status = loom_pass_ir_build_yield(builder);
  loom_builder_restore(builder, scope->saved_insertion_point);
  return status;
}

void loom_pass_ir_abandon_scope(loom_builder_t* builder,
                                const loom_pass_ir_scope_t* scope) {
  loom_builder_restore(builder, scope->saved_insertion_point);
}
