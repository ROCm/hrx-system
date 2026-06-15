// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/pipeline.h"

#include "loom/codegen/low/pipeline/pipeline.h"
#include "loom/pass/builder.h"

typedef struct loom_target_pipeline_build_context_t {
  // Target providers linked into the current compile driver.
  const loom_target_environment_t* target_environment;
  // Pass capabilities required by produced pass IR.
  loom_pass_environment_t pass_environment;
  // Caller-selected options for the pipeline.
  const loom_target_pipeline_options_t* options;
  // True when source-low output is a required-asm artifact surface.
  bool source_low_artifact_preparation;
} loom_target_pipeline_build_context_t;

typedef struct loom_target_pipeline_function_body_t {
  // Function-anchor body builder guarded by the target predicate.
  loom_pass_ir_body_build_fn_t build_body;
  // Opaque user data forwarded to |build_body|.
  void* user_data;
} loom_target_pipeline_function_body_t;

static iree_status_t loom_target_pipeline_resolve_control_flow_lowering(
    const loom_target_pipeline_options_t* options,
    loom_target_control_flow_lowering_t* out_lowering) {
  if (options == NULL) {
    *out_lowering = LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG;
    return iree_ok_status();
  }
  switch (options->control_flow_lowering) {
    case LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG:
    case LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW:
      *out_lowering = options->control_flow_lowering;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown target control-flow lowering mode %d",
                              (int)options->control_flow_lowering);
  }
}

static iree_status_t loom_target_pipeline_build_run(loom_builder_t* builder,
                                                    iree_string_view_t key) {
  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(builder, key, loom_named_attr_slice_empty(),
                                &run_op);
}

static iree_status_t loom_target_pipeline_build_string_attr(
    loom_builder_t* builder, iree_string_view_t name, iree_string_view_t value,
    loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, name, &name_id));
  loom_string_id_t value_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, value, &value_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_string(value_id),
  };
  return iree_ok_status();
}

static iree_status_t loom_target_pipeline_build_run_with_string_option(
    loom_builder_t* builder, iree_string_view_t key, iree_string_view_t name,
    iree_string_view_t value) {
  loom_named_attr_t option_attr = {0};
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_string_attr(
      builder, name, value, &option_attr));
  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(
      builder, key, loom_make_named_attr_slice(&option_attr, 1), &run_op);
}

static iree_status_t loom_target_pipeline_build_target_legalize(
    loom_builder_t* builder, iree_string_view_t mode) {
  return loom_target_pipeline_build_run_with_string_option(
      builder, IREE_SV("target-legalize"), IREE_SV("mode"), mode);
}

static iree_status_t loom_target_pipeline_build_authoring_expansion(
    loom_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run_with_string_option(
      builder, IREE_SV("select-templates"), IREE_SV("mode"), IREE_SV("final")));
  return loom_target_pipeline_build_run(builder, IREE_SV("inline-callables"));
}

static iree_status_t loom_target_pipeline_build_target_function_body(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_function_body_t* body =
      (const loom_target_pipeline_function_body_t*)user_data;
  loom_op_t* where_op = NULL;
  return loom_pass_ir_build_where(builder, IREE_SV("target"),
                                  loom_named_attr_slice_empty(),
                                  body->build_body, body->user_data, &where_op);
}

static iree_status_t loom_target_pipeline_build_for_target_functions(
    loom_builder_t* builder, loom_pass_ir_body_build_fn_t build_body,
    void* user_data, loom_op_t** out_for_op) {
  const loom_target_pipeline_function_body_t body = {
      .build_body = build_body,
      .user_data = user_data,
  };
  return loom_pass_ir_build_for(builder, LOOM_PASS_ANCHOR_FUNC,
                                loom_target_pipeline_build_target_function_body,
                                (void*)&body, out_for_op);
}

static iree_status_t loom_target_pipeline_build_source_to_low(
    loom_builder_t* builder, const loom_target_pipeline_options_t* options) {
  loom_named_attr_t option_attrs[3] = {0};
  loom_named_attr_slice_t option_slice = loom_named_attr_slice_empty();
  iree_host_size_t option_count = 0;
  loom_target_control_flow_lowering_t control_flow_lowering =
      LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_resolve_control_flow_lowering(
      options, &control_flow_lowering));
  if (control_flow_lowering ==
      LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW) {
    IREE_RETURN_IF_ERROR(loom_target_pipeline_build_string_attr(
        builder, IREE_SV("control-flow"), IREE_SV("structured-low"),
        &option_attrs[option_count++]));
  }
  if (options != NULL && options->source_to_low_max_errors != 0) {
    loom_string_id_t option_name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        builder->module, IREE_SV("max-errors"), &option_name_id));
    option_attrs[option_count++] = (loom_named_attr_t){
        .name_id = option_name_id,
        .value = loom_attr_i64(options->source_to_low_max_errors),
    };
  }
  const loom_target_low_legality_diagnostic_flags_t diagnostic_flags =
      options != NULL ? options->source_to_low_legality_diagnostic_flags : 0;
  if (diagnostic_flags != 0) {
    iree_string_view_t diagnostics_value = iree_string_view_empty();
    switch (diagnostic_flags) {
      case LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS:
        diagnostics_value = IREE_SV("memory");
        break;
      case LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_ALL:
        diagnostics_value = IREE_SV("all");
        break;
      default:
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "unknown source-to-low legality diagnostic flags 0x%08X",
            diagnostic_flags);
    }
    IREE_RETURN_IF_ERROR(loom_target_pipeline_build_string_attr(
        builder, IREE_SV("diagnostics"), diagnostics_value,
        &option_attrs[option_count++]));
  }
  if (option_count != 0) {
    option_slice = loom_make_named_attr_slice(option_attrs, option_count);
  }

  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(builder, IREE_SV("source-to-low"), option_slice,
                                &run_op);
}

static iree_status_t loom_target_pipeline_contribute_phase(
    loom_builder_t* builder,
    const loom_target_pipeline_build_context_t* context,
    loom_target_pipeline_phase_t phase) {
  return loom_target_environment_contribute_pipeline(
      context->target_environment, phase, context->pass_environment, builder);
}

static iree_status_t loom_target_pipeline_build_cleanup(
    loom_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("canonicalize")));
  return loom_target_pipeline_build_run(builder, IREE_SV("cse"));
}

static iree_status_t
loom_target_pipeline_build_source_normalization_before_legalize(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_build_context_t* context =
      (const loom_target_pipeline_build_context_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_contribute_phase(
      builder, context, LOOM_TARGET_PIPELINE_PHASE_SOURCE_NORMALIZATION));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("legalize-math")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("normalize-kernel-resources")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("promote-private-fragments")));
  return loom_target_pipeline_build_cleanup(builder);
}

static iree_status_t
loom_target_pipeline_build_source_safe_normalization_after_legalize(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("vector-gather-to-scalar")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("linearize-view-accesses")));
  return loom_target_pipeline_build_cleanup(builder);
}

static iree_status_t
loom_target_pipeline_build_cfg_source_normalization_after_legalize(
    loom_builder_t* builder, void* user_data) {
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_source_safe_normalization_after_legalize(
          builder, user_data));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("unroll-scf-for")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup(builder));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("scf-to-cfg")));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("cfg-simplify")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup(builder));
  return loom_target_pipeline_build_run(builder, IREE_SV("branch-sink"));
}

static iree_status_t loom_target_pipeline_build_low_cleanup_body(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_build_context_t* context =
      (const loom_target_pipeline_build_context_t*)user_data;
  loom_target_control_flow_lowering_t control_flow_lowering =
      LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_resolve_control_flow_lowering(
      context->options, &control_flow_lowering));
  if (control_flow_lowering == LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG) {
    IREE_RETURN_IF_ERROR(
        loom_target_pipeline_build_run(builder, IREE_SV("cfg-simplify")));
  }
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup(builder));
  return loom_target_pipeline_build_run(builder, IREE_SV("low-dce"));
}

static iree_status_t loom_target_pipeline_build_source_low_artifact_preparation(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_build_context_t* context =
      (const loom_target_pipeline_build_context_t*)user_data;
  return loom_target_pipeline_contribute_phase(
      builder, context,
      LOOM_TARGET_PIPELINE_PHASE_SOURCE_LOW_ARTIFACT_PREPARATION);
}

static iree_status_t loom_target_pipeline_build_low_preparation(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_build_context_t* context =
      (const loom_target_pipeline_build_context_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_contribute_phase(
      builder, context, LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MATERIALIZATION));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup(builder));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_contribute_phase(
      builder, context, LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_PREPARATION));
  return loom_low_pipeline_build_packetization_preparation(builder);
}

static iree_status_t loom_target_pipeline_build_source_low_body(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_build_context_t* context =
      (const loom_target_pipeline_build_context_t*)user_data;
  loom_target_control_flow_lowering_t control_flow_lowering =
      LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_resolve_control_flow_lowering(
      context->options, &control_flow_lowering));
  loom_op_t* for_op = NULL;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_for_target_functions(
      builder, loom_target_pipeline_build_source_normalization_before_legalize,
      user_data, &for_op));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_authoring_expansion(builder));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_target_legalize(builder, IREE_SV("eager")));
  loom_pass_ir_body_build_fn_t source_finish_body =
      control_flow_lowering == LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW
          ? loom_target_pipeline_build_source_safe_normalization_after_legalize
          : loom_target_pipeline_build_cfg_source_normalization_after_legalize;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_for_target_functions(
      builder, source_finish_body, user_data, &for_op));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_target_legalize(builder, IREE_SV("eager")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_contribute_phase(
      builder, context, LOOM_TARGET_PIPELINE_PHASE_SOURCE_TO_LOW));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_source_to_low(builder, context->options));
  if (context->source_low_artifact_preparation) {
    IREE_RETURN_IF_ERROR(loom_target_pipeline_build_for_target_functions(
        builder, loom_target_pipeline_build_low_cleanup_body, user_data,
        &for_op));
    IREE_RETURN_IF_ERROR(loom_target_pipeline_build_for_target_functions(
        builder, loom_target_pipeline_build_source_low_artifact_preparation,
        user_data, &for_op));
  }
  return loom_target_pipeline_build_for_target_functions(
      builder, loom_target_pipeline_build_low_cleanup_body, user_data, &for_op);
}

static iree_status_t
loom_target_pipeline_build_source_low_diagnostic_artifacts_body(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_build_context_t* context =
      (const loom_target_pipeline_build_context_t*)user_data;
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_source_to_low(builder, context->options));
  loom_op_t* for_op = NULL;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_for_target_functions(
      builder, loom_target_pipeline_build_source_low_artifact_preparation,
      user_data, &for_op));
  return loom_target_pipeline_build_for_target_functions(
      builder, loom_target_pipeline_build_low_cleanup_body, user_data, &for_op);
}

static iree_status_t loom_target_pipeline_build_prepared_low_body(
    loom_builder_t* builder, void* user_data) {
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_source_low_body(builder, user_data));
  loom_op_t* for_op = NULL;
  return loom_target_pipeline_build_for_target_functions(
      builder, loom_target_pipeline_build_low_preparation, user_data, &for_op);
}

iree_status_t loom_target_pipeline_build_to_source_low(
    loom_module_t* pipeline_module, iree_string_view_t name,
    const loom_target_pipeline_options_t* options,
    const loom_target_environment_t* target_environment,
    loom_pass_environment_t pass_environment, loom_op_t** out_pipeline_op) {
  IREE_ASSERT_ARGUMENT(pipeline_module);
  IREE_ASSERT_ARGUMENT(target_environment);
  IREE_ASSERT_ARGUMENT(out_pipeline_op);
  *out_pipeline_op = NULL;

  const loom_target_pipeline_build_context_t context = {
      .target_environment = target_environment,
      .pass_environment = pass_environment,
      .options = options,
  };
  return loom_pass_ir_build_pipeline(pipeline_module, name,
                                     LOOM_PASS_ANCHOR_MODULE,
                                     loom_target_pipeline_build_source_low_body,
                                     (void*)&context, out_pipeline_op);
}

iree_status_t loom_target_pipeline_build_to_source_low_diagnostic_artifacts(
    loom_module_t* pipeline_module, iree_string_view_t name,
    const loom_target_pipeline_options_t* options,
    const loom_target_environment_t* target_environment,
    loom_pass_environment_t pass_environment, loom_op_t** out_pipeline_op) {
  IREE_ASSERT_ARGUMENT(pipeline_module);
  IREE_ASSERT_ARGUMENT(target_environment);
  IREE_ASSERT_ARGUMENT(out_pipeline_op);
  *out_pipeline_op = NULL;

  const loom_target_pipeline_build_context_t context = {
      .target_environment = target_environment,
      .pass_environment = pass_environment,
      .options = options,
  };
  return loom_pass_ir_build_pipeline(
      pipeline_module, name, LOOM_PASS_ANCHOR_MODULE,
      loom_target_pipeline_build_source_low_diagnostic_artifacts_body,
      (void*)&context, out_pipeline_op);
}

iree_status_t loom_target_pipeline_build_to_source_low_artifacts(
    loom_module_t* pipeline_module, iree_string_view_t name,
    const loom_target_pipeline_options_t* options,
    const loom_target_environment_t* target_environment,
    loom_pass_environment_t pass_environment, loom_op_t** out_pipeline_op) {
  IREE_ASSERT_ARGUMENT(pipeline_module);
  IREE_ASSERT_ARGUMENT(target_environment);
  IREE_ASSERT_ARGUMENT(out_pipeline_op);
  *out_pipeline_op = NULL;

  const loom_target_pipeline_build_context_t context = {
      .target_environment = target_environment,
      .pass_environment = pass_environment,
      .options = options,
      .source_low_artifact_preparation = true,
  };
  return loom_pass_ir_build_pipeline(pipeline_module, name,
                                     LOOM_PASS_ANCHOR_MODULE,
                                     loom_target_pipeline_build_source_low_body,
                                     (void*)&context, out_pipeline_op);
}

iree_status_t loom_target_pipeline_build_to_prepared_low(
    loom_module_t* pipeline_module, iree_string_view_t name,
    const loom_target_pipeline_options_t* options,
    const loom_target_environment_t* target_environment,
    loom_pass_environment_t pass_environment, loom_op_t** out_pipeline_op) {
  IREE_ASSERT_ARGUMENT(pipeline_module);
  IREE_ASSERT_ARGUMENT(target_environment);
  IREE_ASSERT_ARGUMENT(out_pipeline_op);
  *out_pipeline_op = NULL;

  const loom_target_pipeline_build_context_t context = {
      .target_environment = target_environment,
      .pass_environment = pass_environment,
      .options = options,
  };
  return loom_pass_ir_build_pipeline(
      pipeline_module, name, LOOM_PASS_ANCHOR_MODULE,
      loom_target_pipeline_build_prepared_low_body, (void*)&context,
      out_pipeline_op);
}
