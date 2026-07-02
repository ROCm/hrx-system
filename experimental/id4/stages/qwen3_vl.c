// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/qwen3_vl.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/pipeline/program_stage.h"
#include "iree/base/internal/arena.h"

#define ID4_QWEN3_VL_STAGE_ALIGNMENT 16
#define ID4_QWEN3_VL_STAGE_PROGRAM_BLOCK_SIZE (64 * 1024)

typedef struct id4_qwen3_vl_stage_t {
  // Base stage; must be the first field.
  id4_pipeline_stage_t base;
  // Allocator used for stage-owned metadata.
  iree_allocator_t host_allocator;
  // Kernel cache used for Loom compilation and HAL executable preparation.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Parameter provider scope containing Qwen3-VL weights.
  iree_string_view_t parameter_scope;
  // Static model configuration owned by the stage.
  id4_qwen3_vl_model_config_t model;
  // Selected layer ordinal storage owned by the stage.
  uint32_t* selected_layer_ordinals;
  // True after load has completed.
  bool is_loaded;
} id4_qwen3_vl_stage_t;

static id4_qwen3_vl_stage_t* id4_qwen3_vl_stage_cast(
    id4_pipeline_stage_t* base_stage) {
  return (id4_qwen3_vl_stage_t*)base_stage;
}

static const id4_qwen3_vl_stage_t* id4_qwen3_vl_stage_const_cast(
    const id4_pipeline_stage_t* base_stage) {
  return (const id4_qwen3_vl_stage_t*)base_stage;
}

static iree_status_t id4_qwen3_vl_stage_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_qwen3_vl_stage_validate_model_config(
    const id4_qwen3_vl_model_config_t* model) {
  if (model->layer_count == 0 || model->vocab_size == 0 ||
      model->hidden_size == 0 || model->intermediate_size == 0 ||
      model->attention_head_count == 0 || model->key_value_head_count == 0 ||
      model->head_size == 0 || model->max_token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL model dimensions must be nonzero");
  }
  if (model->selected_layer_count == 0 || !model->selected_layer_ordinals) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL selected layer ordinals are required");
  }
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_stage_validate_create_options(
    const id4_qwen3_vl_stage_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Qwen3-VL stage create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Qwen3-VL stage create extension structures are not supported");
  }
  if (!options->services.device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage device group is required");
  }
  if (!options->kernel_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage kernel cache is required");
  }
  return id4_qwen3_vl_stage_validate_model_config(&options->model);
}

static iree_status_t id4_qwen3_vl_stage_copy_model_config(
    const id4_qwen3_vl_model_config_t* source, id4_qwen3_vl_stage_t* stage) {
  stage->model = *source;
  stage->model.selected_layer_ordinals = NULL;
  stage->selected_layer_ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      stage->host_allocator, source->selected_layer_count,
      sizeof(stage->selected_layer_ordinals[0]),
      (void**)&stage->selected_layer_ordinals));
  memcpy(
      stage->selected_layer_ordinals, source->selected_layer_ordinals,
      source->selected_layer_count * sizeof(stage->selected_layer_ordinals[0]));
  stage->model.selected_layer_ordinals = stage->selected_layer_ordinals;
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_stage_copy_parameter_scope(
    iree_string_view_t source, id4_qwen3_vl_stage_t* stage) {
  stage->parameter_scope = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(stage->host_allocator, source.size,
                                             (void**)&storage));
  memcpy(storage, source.data, source.size);
  stage->parameter_scope = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static id4_pipeline_diagnostic_event_t id4_qwen3_vl_stage_lifecycle_event(
    iree_string_view_t key, iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event = {
      // Lifecycle event emitted by the concrete Qwen3-VL stage.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE,
      // Stable stage name used across Qwen3-VL diagnostics.
      .stage_name = IREE_SV(ID4_QWEN3_VL_STAGE_NAME),
      // Stable lifecycle key.
      .key = key,
      // Short lifecycle summary.
      .message = message,
  };
  return event;
}

static iree_status_t id4_qwen3_vl_stage_emit_lifecycle(
    id4_pipeline_diagnostics_sink_t* diagnostics_sink, iree_string_view_t key,
    iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event =
      id4_qwen3_vl_stage_lifecycle_event(key, message);
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static iree_status_t id4_qwen3_vl_stage_author_program(
    const id4_qwen3_vl_stage_t* stage, id4_qwen3_vl_request_config_t request,
    id4_qwen3_vl_weight_execution_strategy_t weight_execution_strategy,
    iree_string_view_list_t diagnostic_tap_names,
    iree_allocator_t host_allocator, id4_pipeline_program_t** out_program) {
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = NULL;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(ID4_QWEN3_VL_STAGE_PROGRAM_BLOCK_SIZE,
                                   host_allocator, &block_pool);

  id4_pipeline_program_builder_t* builder = NULL;
  id4_pipeline_program_builder_create_options_t builder_options;
  memset(&builder_options, 0, sizeof(builder_options));
  builder_options.structure_size = sizeof(builder_options);
  builder_options.program_name = IREE_SV(ID4_QWEN3_VL_STAGE_NAME);
  builder_options.block_pool = &block_pool;
  iree_status_t status = id4_pipeline_program_builder_create(
      &builder_options, host_allocator, &builder);
  if (iree_status_is_ok(status)) {
    id4_qwen3_vl_program_options_t program_options;
    memset(&program_options, 0, sizeof(program_options));
    program_options.structure_size = sizeof(program_options);
    program_options.parameter_scope = stage->parameter_scope;
    program_options.model = stage->model;
    program_options.request = request;
    program_options.host_allocator = host_allocator;
    program_options.weight_execution_strategy = weight_execution_strategy;
    program_options.diagnostic_tap_names = diagnostic_tap_names;
    status = id4_qwen3_vl_program_author_forward(&program_options, builder);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_pipeline_program_builder_seal(builder, host_allocator, out_program);
  }
  id4_pipeline_program_builder_destroy(builder);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static iree_status_t id4_qwen3_vl_stage_parse_plan_extension(
    const id4_qwen3_vl_stage_t* stage,
    const id4_pipeline_stage_plan_options_t* options,
    const id4_qwen3_vl_stage_plan_options_t** out_qwen_options) {
  *out_qwen_options = NULL;
  if (!options->next) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage plan options are required");
  }
  const id4_qwen3_vl_stage_plan_options_t* qwen_options =
      (const id4_qwen3_vl_stage_plan_options_t*)options->next;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_validate_options_size(
      qwen_options->structure_size, sizeof(*qwen_options),
      IREE_SV("Qwen3-VL stage plan")));
  if (qwen_options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Qwen3-VL stage plan extension structures are not supported");
  }
  if (qwen_options->request.token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL plan token count must be nonzero");
  }
  if (!qwen_options->request.token_ids) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL plan token ids are required");
  }
  for (uint32_t i = 0; i < qwen_options->request.token_count; ++i) {
    if (qwen_options->request.token_ids[i] < 0 ||
        (uint32_t)qwen_options->request.token_ids[i] >=
            stage->model.vocab_size) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Qwen3-VL plan token id %" PRIi32 " at position %" PRIu32
          " exceeds vocabulary size %" PRIu32,
          qwen_options->request.token_ids[i], i, stage->model.vocab_size);
    }
  }
  switch (qwen_options->weight_execution_strategy) {
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Qwen3-VL plan weight execution strategy %" PRIu32 " is invalid",
          (uint32_t)qwen_options->weight_execution_strategy);
  }
  *out_qwen_options = qwen_options;
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_stage_create_program_plan(
    id4_qwen3_vl_stage_t* stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_program_t* program, id4_pipeline_plan_t** out_plan) {
  id4_pipeline_program_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = IREE_SV(ID4_QWEN3_VL_STAGE_NAME);
  plan_options.stage_options = options;
  plan_options.program = program;
  plan_options.device_group = stage->base.services.device_group;
  plan_options.parameter_scope = stage->parameter_scope;
  plan_options.alignment = ID4_QWEN3_VL_STAGE_ALIGNMENT;
  return id4_pipeline_program_stage_create_plan(
      &plan_options, stage->host_allocator, out_plan);
}

static iree_status_t id4_qwen3_vl_stage_load(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_load_options_t* options) {
  id4_qwen3_vl_stage_t* stage = id4_qwen3_vl_stage_cast(base_stage);
  stage->is_loaded = true;
  return id4_qwen3_vl_stage_emit_lifecycle(
      options->diagnostics_sink, IREE_SV("stage.load"),
      IREE_SV("loaded Qwen3-VL forward stage"));
}

static iree_status_t id4_qwen3_vl_stage_plan(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_qwen3_vl_stage_t* stage = id4_qwen3_vl_stage_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen3-VL stage must be loaded before planning");
  }
  const id4_qwen3_vl_stage_plan_options_t* qwen_options = NULL;
  IREE_RETURN_IF_ERROR(
      id4_qwen3_vl_stage_parse_plan_extension(stage, options, &qwen_options));

  id4_pipeline_program_t* program = NULL;
  iree_status_t status = id4_qwen3_vl_stage_author_program(
      stage, qwen_options->request, qwen_options->weight_execution_strategy,
      options->diagnostic_tap_names, stage->host_allocator, &program);
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_create_program_plan(stage, options, program,
                                                    out_plan);
  }
  id4_pipeline_program_release(program);
  return status;
}

static iree_status_t id4_qwen3_vl_stage_prepare(
    id4_pipeline_stage_t* base_stage, const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options,
    id4_pipeline_bundle_t** out_bundle) {
  id4_qwen3_vl_stage_t* stage = id4_qwen3_vl_stage_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen3-VL stage must be loaded before preparation");
  }

  id4_pipeline_program_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.stage_name = IREE_SV(ID4_QWEN3_VL_STAGE_NAME);
  prepare_options.stage_options = options;
  prepare_options.plan = plan;
  prepare_options.device_group = stage->base.services.device_group;
  prepare_options.kernel_cache = stage->kernel_cache;
  prepare_options.executable_cache = stage->base.services.executable_cache;
  return id4_pipeline_program_stage_prepare(&prepare_options,
                                            stage->host_allocator, out_bundle);
}

static iree_status_t id4_qwen3_vl_stage_issue(
    id4_pipeline_stage_t* base_stage, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  (void)base_stage;
  return id4_pipeline_program_stage_issue(IREE_SV(ID4_QWEN3_VL_STAGE_NAME),
                                          bundle, options);
}

static void id4_qwen3_vl_stage_destroy(id4_pipeline_stage_t* base_stage) {
  id4_qwen3_vl_stage_t* stage = id4_qwen3_vl_stage_cast(base_stage);
  iree_allocator_t host_allocator = stage->host_allocator;
  id4_pipeline_kernel_cache_release(stage->kernel_cache);
  iree_allocator_free(host_allocator, stage->selected_layer_ordinals);
  iree_allocator_free(host_allocator, (void*)stage->parameter_scope.data);
  id4_pipeline_stage_deinitialize(base_stage);
  iree_allocator_free(host_allocator, stage);
}

static const id4_pipeline_stage_vtable_t id4_qwen3_vl_stage_vtable = {
    // Destroys the concrete Qwen3-VL stage.
    id4_qwen3_vl_stage_destroy,
    // Loads Qwen3-VL immutable state.
    id4_qwen3_vl_stage_load,
    // Builds a Qwen3-VL forward plan.
    id4_qwen3_vl_stage_plan,
    // Prepares a Qwen3-VL forward bundle.
    id4_qwen3_vl_stage_prepare,
    // Issues a Qwen3-VL forward bundle.
    id4_qwen3_vl_stage_issue,
};

iree_status_t id4_qwen3_vl_stage_create(
    const id4_qwen3_vl_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = NULL;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_validate_create_options(options));

  id4_qwen3_vl_stage_t* stage = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*stage), (void**)&stage);
  if (iree_status_is_ok(status)) {
    memset(stage, 0, sizeof(*stage));
    stage->host_allocator = host_allocator;
    status = id4_pipeline_stage_initialize(&id4_qwen3_vl_stage_vtable,
                                           &options->services, &stage->base);
  }
  if (iree_status_is_ok(status)) {
    stage->kernel_cache = options->kernel_cache;
    id4_pipeline_kernel_cache_retain(stage->kernel_cache);
    status = id4_qwen3_vl_stage_copy_parameter_scope(options->parameter_scope,
                                                     stage);
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_copy_model_config(&options->model, stage);
  }
  if (iree_status_is_ok(status)) {
    *out_stage = &stage->base;
  } else if (stage) {
    id4_qwen3_vl_stage_destroy(&stage->base);
  }
  return status;
}
