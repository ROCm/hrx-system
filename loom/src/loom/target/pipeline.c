// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/pipeline.h"

#include "loom/codegen/low/pipeline/pipeline.h"
#include "loom/pass/builder.h"
#include "loom/sanitizer/options.h"
#include "loom/transforms/symbol/template_expansion_pipeline.h"

typedef struct loom_target_pipeline_build_context_t {
  // Target providers linked into the current compile driver.
  const loom_target_environment_t* target_environment;
  // Pass capabilities required by produced pass IR.
  loom_pass_environment_t pass_environment;
  // Caller-selected options for the pipeline.
  const loom_target_pipeline_options_t* options;
  // True when source-low output is a required-asm artifact surface.
  bool source_low_artifact_preparation;
  // Sanitizer options resolved for the pipeline being built.
  loom_sanitizer_options_t sanitizer_options;
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
  return loom_pass_ir_build_run(builder, 0, key, loom_named_attr_slice_empty(),
                                &run_op);
}

static bool loom_target_pipeline_sanitizer_enabled(
    const loom_target_pipeline_build_context_t* context) {
  return loom_sanitizer_options_is_enabled(&context->sanitizer_options);
}

static bool loom_target_pipeline_sanitizer_has_checks(
    const loom_target_pipeline_build_context_t* context,
    loom_sanitizer_checks_t checks) {
  return iree_any_bit_set(context->sanitizer_options.checks, checks);
}

static bool loom_target_pipeline_source_to_low_has_memory_diagnostics(
    const loom_target_pipeline_build_context_t* context) {
  if (!context->options) {
    return false;
  }
  return iree_any_bit_set(
      context->options->source_to_low_legality_diagnostic_flags,
      LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS);
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
      builder, LOOM_PASS_RUN_BUILD_FLAG_HAS_OPTIONS, key,
      loom_make_named_attr_slice(&option_attr, 1), &run_op);
}

static iree_status_t loom_target_pipeline_build_target_legalize(
    loom_builder_t* builder, iree_string_view_t mode) {
  return loom_target_pipeline_build_run_with_string_option(
      builder, IREE_SV("target-legalize"), IREE_SV("mode"), mode);
}

static iree_status_t loom_target_pipeline_build_sanitizer_assertion_selection(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_build_context_t* context =
      (const loom_target_pipeline_build_context_t*)user_data;
  iree_string_view_t checks_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_sanitizer_checks_format(
      context->sanitizer_options.checks, &checks_value));
  return loom_target_pipeline_build_run_with_string_option(
      builder, IREE_SV("sanitizer-insert-assertions"), IREE_SV("checks"),
      checks_value);
}

static iree_status_t loom_target_pipeline_build_sanitizer_race_observations(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_build_context_t* context =
      (const loom_target_pipeline_build_context_t*)user_data;
  iree_string_view_t checks_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_sanitizer_checks_format(
      context->sanitizer_options.checks, &checks_value));
  return loom_target_pipeline_build_run_with_string_option(
      builder, IREE_SV("sanitizer-insert-race-observations"), IREE_SV("checks"),
      checks_value);
}

static iree_status_t loom_target_pipeline_build_vector_memory_footprint(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  return loom_target_pipeline_build_run(builder,
                                        IREE_SV("vector-memory-footprint"));
}

static iree_status_t loom_target_pipeline_build_target_function_body(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_function_body_t* body =
      (const loom_target_pipeline_function_body_t*)user_data;
  loom_op_t* where_op = NULL;
  return loom_pass_ir_build_where(builder, 0, IREE_SV("target"),
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

static iree_status_t loom_target_pipeline_build_canonicalize_body(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  return loom_target_pipeline_build_run(builder, IREE_SV("canonicalize"));
}

static iree_status_t
loom_target_pipeline_build_cleanup_expanded_target_functions(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* for_op = NULL;
  return loom_target_pipeline_build_for_target_functions(
      builder, loom_target_pipeline_build_canonicalize_body, NULL, &for_op);
}

static iree_status_t loom_target_pipeline_build_source_to_low(
    loom_builder_t* builder, const loom_target_pipeline_options_t* options) {
  loom_named_attr_t option_attrs[4] = {0};
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
      case LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_OPERAND_FORM:
        diagnostics_value = IREE_SV("operand-forms");
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
  const loom_sanitizer_reporting_mode_t sanitizer_reporting_mode =
      options != NULL ? options->sanitizer.reporting_mode
                      : LOOM_SANITIZER_REPORTING_MODE_DEFAULT;
  if (sanitizer_reporting_mode != LOOM_SANITIZER_REPORTING_MODE_DEFAULT) {
    iree_string_view_t reporting_value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_sanitizer_reporting_mode_format(
        sanitizer_reporting_mode, &reporting_value));
    IREE_RETURN_IF_ERROR(loom_target_pipeline_build_string_attr(
        builder, IREE_SV("sanitizer-reporting"), reporting_value,
        &option_attrs[option_count++]));
  }
  if (option_count != 0) {
    option_slice = loom_make_named_attr_slice(option_attrs, option_count);
  }

  loom_op_t* run_op = NULL;
  const loom_pass_run_build_flags_t build_flags =
      option_count != 0 ? LOOM_PASS_RUN_BUILD_FLAG_HAS_OPTIONS : 0;
  return loom_pass_ir_build_run(builder, build_flags, IREE_SV("source-to-low"),
                                option_slice, &run_op);
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

static iree_status_t loom_target_pipeline_build_cleanup_body(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  return loom_target_pipeline_build_cleanup(builder);
}

static iree_status_t loom_target_pipeline_build_cleanup_if_changed(
    loom_builder_t* builder) {
  loom_op_t* if_changed_op = NULL;
  return loom_pass_ir_build_if_changed(
      builder, loom_target_pipeline_build_cleanup_body, NULL, &if_changed_op);
}

static iree_status_t loom_target_pipeline_build_cleanup_target_functions(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* for_op = NULL;
  return loom_target_pipeline_build_for_target_functions(
      builder, loom_target_pipeline_build_cleanup_body, NULL, &for_op);
}

static iree_status_t
loom_target_pipeline_build_source_normalization_before_authoring_expansion(
    loom_builder_t* builder, void* user_data) {
  const loom_target_pipeline_build_context_t* context =
      (const loom_target_pipeline_build_context_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_contribute_phase(
      builder, context, LOOM_TARGET_PIPELINE_PHASE_SOURCE_NORMALIZATION));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("normalize-kernel-resources")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("promote-private-fragments")));
  return loom_target_pipeline_build_cleanup(builder);
}

static iree_status_t
loom_target_pipeline_build_math_legalization_after_authoring_expansion(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  return loom_target_pipeline_build_run(builder, IREE_SV("legalize-math"));
}

static iree_status_t
loom_target_pipeline_build_source_safe_normalization_after_legalize(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("vector-memory-to-scalar")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("linearize-view-accesses")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup(builder));

  // CSE may replace a branch predicate and leave assumptions on a newly
  // unreachable edge. Recanonicalize only when CSE changed the function so
  // dead edges and their contradictory assumptions are removed before target
  // verification.
  loom_op_t* if_changed_op = NULL;
  return loom_pass_ir_build_if_changed(
      builder, loom_target_pipeline_build_canonicalize_body, NULL,
      &if_changed_op);
}

static iree_status_t
loom_target_pipeline_build_cfg_source_finalization_after_legalize(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("unroll-scf-for")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup_if_changed(builder));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("sroa-vector-banks")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup_if_changed(builder));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("sink-single-use-reads")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup_if_changed(builder));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("promote-private-fragments")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_cleanup_if_changed(builder));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("scf-to-cfg")));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("cfg-simplify")));
  // This is the final source canonicalization boundary before lowering. Run it
  // even when CFG simplification made no change because canonicalization may
  // have independent pending work, such as propagating callee purity.
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
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("low-decompose-cfg-tuples")));
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
  if (loom_target_pipeline_sanitizer_has_checks(
          context, LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_VALUE |
                       LOOM_SANITIZER_CHECK_OPERATION)) {
    IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
        builder, IREE_SV("sanitizer-materialize-assertions")));
  }
  IREE_RETURN_IF_ERROR(loom_target_pipeline_contribute_phase(
      builder, context, LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_PREPARATION));
  return loom_low_pipeline_build_packetization_preparation(builder);
}

static iree_status_t loom_target_pipeline_build_expanded_source_body(
    loom_builder_t* builder, void* user_data) {
  loom_op_t* for_op = NULL;
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_for_target_functions(
      builder,
      loom_target_pipeline_build_source_normalization_before_authoring_expansion,
      user_data, &for_op));
  return loom_template_expansion_pipeline_build(
      builder, loom_target_pipeline_build_cleanup_expanded_target_functions,
      NULL);
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
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_expanded_source_body(builder, user_data));
  // Authoring expansion has selected every resolvable provider and exposed
  // its retained callees. Specialize the complete semantic call graph before
  // any target-aware function pass observes those callees.
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_run(
      builder, IREE_SV("specialize-target-callgraph")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_for_target_functions(
      builder,
      loom_target_pipeline_build_math_legalization_after_authoring_expansion,
      user_data, &for_op));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_target_legalize(builder, IREE_SV("eager")));
  IREE_RETURN_IF_ERROR(loom_target_pipeline_build_for_target_functions(
      builder,
      loom_target_pipeline_build_source_safe_normalization_after_legalize,
      user_data, &for_op));
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_target_legalize(builder, IREE_SV("eager")));
  // Module legalization may mutate any target function. Normalize the full
  // target set only when it did so before function-local CFG finalization.
  loom_op_t* if_changed_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_ir_build_if_changed(
      builder, loom_target_pipeline_build_cleanup_target_functions, NULL,
      &if_changed_op));
  if (control_flow_lowering == LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG) {
    IREE_RETURN_IF_ERROR(loom_target_pipeline_build_for_target_functions(
        builder,
        loom_target_pipeline_build_cfg_source_finalization_after_legalize,
        user_data, &for_op));
  }
  if (loom_target_pipeline_sanitizer_has_checks(
          context, LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_VALUE |
                       LOOM_SANITIZER_CHECK_OPERATION)) {
    IREE_RETURN_IF_ERROR(loom_target_pipeline_build_for_target_functions(
        builder, loom_target_pipeline_build_sanitizer_assertion_selection,
        user_data, &for_op));
  }
  if (loom_target_pipeline_sanitizer_has_checks(context,
                                                LOOM_SANITIZER_CHECK_RACE)) {
    IREE_RETURN_IF_ERROR(loom_target_pipeline_build_for_target_functions(
        builder, loom_target_pipeline_build_sanitizer_race_observations,
        user_data, &for_op));
  }
  if (loom_target_pipeline_source_to_low_has_memory_diagnostics(context)) {
    IREE_RETURN_IF_ERROR(loom_target_pipeline_build_for_target_functions(
        builder, loom_target_pipeline_build_vector_memory_footprint, user_data,
        &for_op));
  }
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
  loom_op_t* for_op = NULL;
  if (loom_target_pipeline_source_to_low_has_memory_diagnostics(context)) {
    IREE_RETURN_IF_ERROR(loom_target_pipeline_build_for_target_functions(
        builder, loom_target_pipeline_build_vector_memory_footprint, user_data,
        &for_op));
  }
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_source_to_low(builder, context->options));
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
  // Required-inline boundaries such as low.invoke disappear during
  // source-to-Low lowering. Prune their now-unreachable private definitions
  // before target-Low function passes can mistake them for artifact entries.
  IREE_RETURN_IF_ERROR(
      loom_target_pipeline_build_run(builder, IREE_SV("symbol-dce")));
  loom_op_t* for_op = NULL;
  return loom_target_pipeline_build_for_target_functions(
      builder, loom_target_pipeline_build_low_preparation, user_data, &for_op);
}

iree_status_t loom_target_pipeline_build_to_expanded_source(
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
      loom_target_pipeline_build_expanded_source_body, (void*)&context,
      out_pipeline_op);
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

  const loom_sanitizer_options_t sanitizer_options =
      options ? options->sanitizer : (loom_sanitizer_options_t){0};
  IREE_RETURN_IF_ERROR(loom_sanitizer_options_validate(&sanitizer_options));
  const loom_target_pipeline_build_context_t context = {
      .target_environment = target_environment,
      .pass_environment = pass_environment,
      .options = options,
      .sanitizer_options = sanitizer_options,
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

  const loom_sanitizer_options_t sanitizer_options =
      options ? options->sanitizer : (loom_sanitizer_options_t){0};
  IREE_RETURN_IF_ERROR(loom_sanitizer_options_validate(&sanitizer_options));
  const loom_target_pipeline_build_context_t context = {
      .target_environment = target_environment,
      .pass_environment = pass_environment,
      .options = options,
      .source_low_artifact_preparation = true,
      .sanitizer_options = sanitizer_options,
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

  const loom_sanitizer_options_t sanitizer_options =
      options ? options->sanitizer : (loom_sanitizer_options_t){0};
  IREE_RETURN_IF_ERROR(loom_sanitizer_options_validate(&sanitizer_options));
  const loom_target_pipeline_build_context_t context = {
      .target_environment = target_environment,
      .pass_environment = pass_environment,
      .options = options,
      .sanitizer_options = sanitizer_options,
  };
  return loom_pass_ir_build_pipeline(
      pipeline_module, name, LOOM_PASS_ANCHOR_MODULE,
      loom_target_pipeline_build_prepared_low_body, (void*)&context,
      out_pipeline_op);
}
