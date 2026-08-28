// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/template_expansion_pipeline.h"

typedef struct loom_template_expansion_pipeline_context_t {
  // Cleanup body exposing facts after one expansion iteration changes IR.
  loom_pass_ir_body_build_fn_t cleanup_body;
  // Opaque user data forwarded to |cleanup_body|.
  void* cleanup_user_data;
} loom_template_expansion_pipeline_context_t;

static iree_status_t loom_template_expansion_pipeline_build_selection(
    loom_builder_t* builder, iree_string_view_t mode) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(builder, IREE_SV("mode"), &name_id));
  loom_string_id_t value_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, mode, &value_id));
  const loom_named_attr_t option = {
      .name_id = name_id,
      .value = loom_attr_string(value_id),
  };
  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(builder, LOOM_PASS_RUN_BUILD_FLAG_HAS_OPTIONS,
                                IREE_SV("select-templates"),
                                loom_make_named_attr_slice(&option, 1),
                                &run_op);
}

static iree_status_t loom_template_expansion_pipeline_build_iteration(
    loom_builder_t* builder, void* user_data) {
  const loom_template_expansion_pipeline_context_t* context =
      (const loom_template_expansion_pipeline_context_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_template_expansion_pipeline_build_selection(
      builder, IREE_SV("early")));
  loom_op_t* run_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_ir_build_run(builder, 0, IREE_SV("inline-callables"),
                             loom_named_attr_slice_empty(), &run_op));
  loom_op_t* if_changed_op = NULL;
  return loom_pass_ir_build_if_changed(builder, context->cleanup_body,
                                       context->cleanup_user_data,
                                       &if_changed_op);
}

iree_status_t loom_template_expansion_pipeline_build(
    loom_builder_t* builder, loom_pass_ir_body_build_fn_t cleanup_body,
    void* cleanup_user_data) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(cleanup_body);
  const loom_template_expansion_pipeline_context_t context = {
      .cleanup_body = cleanup_body,
      .cleanup_user_data = cleanup_user_data,
  };
  loom_op_t* repeat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_ir_build_repeat(
      builder, LOOM_PASS_REPEAT_BUILD_FLAG_HAS_MAX_ITERATIONS,
      LOOM_PASS_REPEAT_MODE_UNTIL_CONVERGED, 0, 64,
      loom_template_expansion_pipeline_build_iteration, (void*)&context,
      &repeat_op));
  return loom_template_expansion_pipeline_build_selection(builder,
                                                          IREE_SV("final"));
}
