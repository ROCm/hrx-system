// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/pipeline/program_stage.h"
#include "iree/base/internal/arena.h"

#define ID4_IDEOGRAM4_DIT_STAGE_ALIGNMENT 16
#define ID4_IDEOGRAM4_DIT_STAGE_PARAMETER_BINDING_SLOT 0
#define ID4_IDEOGRAM4_DIT_STAGE_BOUNDARY_BINDING_SLOT_BASE 1
#define ID4_IDEOGRAM4_DIT_STAGE_PROGRAM_BLOCK_SIZE (32 * 1024)
#define ID4_IDEOGRAM4_DIT_STAGE_LLM_HIDDEN_STATE_LAYER_COUNT 13

typedef struct id4_ideogram4_dit_stage_t {
  // Base stage; must be the first field.
  id4_pipeline_stage_t base;
  // Allocator used for stage-owned metadata.
  iree_allocator_t host_allocator;
  // Kernel cache used for Loom compilation and HAL executable preparation.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Parameter provider scope containing Ideogram4 DiT weights.
  iree_string_view_t parameter_scope;
  // Static model configuration owned by the stage.
  id4_ideogram4_dit_model_config_t model;
  // True after load has completed.
  bool is_loaded;
} id4_ideogram4_dit_stage_t;

static id4_ideogram4_dit_stage_t* id4_ideogram4_dit_stage_cast(
    id4_pipeline_stage_t* base_stage) {
  return (id4_ideogram4_dit_stage_t*)base_stage;
}

static iree_status_t id4_ideogram4_dit_stage_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_ideogram4_dit_stage_validate_model_config(
    const id4_ideogram4_dit_model_config_t* model) {
  if (model->layer_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT layer count must be nonzero");
  }
  if (model->input_channel_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT input channel count must be "
                            "nonzero");
  }
  if ((model->input_channel_count % 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT input channel count must be a "
                            "multiple of 4");
  }
  if (model->hidden_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT hidden size must be nonzero");
  }
  if ((model->hidden_size % 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT hidden size must be a multiple of "
                            "4");
  }
  if (model->intermediate_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT intermediate size must be nonzero");
  }
  if (model->attention_head_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT attention head count must be "
                            "nonzero");
  }
  if ((model->hidden_size % model->attention_head_count) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT hidden size must divide evenly by "
                            "attention head count");
  }
  if (((model->hidden_size / model->attention_head_count) % 2) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT attention head size must be even");
  }
  if (model->adaln_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT AdaLN size must be nonzero");
  }
  if (model->llm_feature_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT LLM feature count must be "
                            "nonzero");
  }
  if ((model->llm_feature_count % 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT LLM feature count must be a "
                            "multiple of 4");
  }
  if ((model->llm_feature_count %
       ID4_IDEOGRAM4_DIT_STAGE_LLM_HIDDEN_STATE_LAYER_COUNT) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT LLM feature count must divide "
                            "evenly by Qwen hidden-state layer count");
  }
  if (model->image_indicator_count != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT image indicator count must be 2");
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_stage_validate_request(
    const id4_ideogram4_dit_model_config_t* model,
    id4_ideogram4_dit_request_config_t request) {
  if (request.latent_shape.rank != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT latent shape rank must be 4");
  }
  if (request.latent_shape.dims[0] == 0 || request.latent_shape.dims[1] == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT latent spatial dimensions must be "
                            "nonzero");
  }
  if (request.latent_shape.dims[2] != model->input_channel_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT latent channel count %" PRIu64
                            " does not match model channel count %" PRIu32,
                            request.latent_shape.dims[2],
                            model->input_channel_count);
  }
  if (request.latent_shape.dims[3] != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT latent batch dimension must be 1");
  }
  switch (request.conditioning_mode) {
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED:
      if (request.text_token_count != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram4 DiT unconditioned requests must not provide text "
            "tokens");
      }
      break;
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED:
      if (request.text_token_count == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram4 DiT conditioned requests require text tokens");
      }
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram4 DiT conditioning mode %" PRIu32
                              " is invalid",
                              (uint32_t)request.conditioning_mode);
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_stage_validate_create_options(
    const id4_ideogram4_dit_stage_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT stage create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_stage_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram4 DiT stage create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram4 DiT stage create extension structures are not supported");
  }
  if (!options->services.device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT stage device group is required");
  }
  return id4_ideogram4_dit_stage_validate_model_config(&options->model);
}

static iree_status_t id4_ideogram4_dit_stage_copy_parameter_scope(
    iree_string_view_t source, id4_ideogram4_dit_stage_t* stage) {
  stage->parameter_scope = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(stage->host_allocator, source.size,
                                             (void**)&storage));
  memcpy(storage, source.data, source.size);
  stage->parameter_scope = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static id4_pipeline_diagnostic_event_t id4_ideogram4_dit_stage_lifecycle_event(
    iree_string_view_t key, iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event = {
      // Lifecycle event emitted by the concrete Ideogram4 DiT stage.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE,
      // Stable stage name used across Ideogram4 DiT diagnostics.
      .stage_name = IREE_SV(ID4_IDEOGRAM4_DIT_STAGE_NAME),
      // Stable lifecycle key.
      .key = key,
      // Short lifecycle summary.
      .message = message,
  };
  return event;
}

static iree_status_t id4_ideogram4_dit_stage_emit_lifecycle(
    id4_pipeline_diagnostics_sink_t* diagnostics_sink, iree_string_view_t key,
    iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event =
      id4_ideogram4_dit_stage_lifecycle_event(key, message);
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static iree_status_t id4_ideogram4_dit_stage_author_program(
    const id4_ideogram4_dit_stage_t* stage,
    id4_ideogram4_dit_request_config_t request, iree_allocator_t host_allocator,
    id4_pipeline_program_t** out_program) {
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = NULL;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(ID4_IDEOGRAM4_DIT_STAGE_PROGRAM_BLOCK_SIZE,
                                   host_allocator, &block_pool);

  id4_pipeline_program_builder_t* builder = NULL;
  id4_pipeline_program_builder_create_options_t builder_options;
  memset(&builder_options, 0, sizeof(builder_options));
  builder_options.structure_size = sizeof(builder_options);
  builder_options.program_name = IREE_SV(ID4_IDEOGRAM4_DIT_STAGE_NAME);
  builder_options.block_pool = &block_pool;
  iree_status_t status = id4_pipeline_program_builder_create(
      &builder_options, host_allocator, &builder);
  if (iree_status_is_ok(status)) {
    id4_ideogram4_dit_program_options_t program_options;
    memset(&program_options, 0, sizeof(program_options));
    program_options.structure_size = sizeof(program_options);
    program_options.model = stage->model;
    program_options.request = request;
    status =
        id4_ideogram4_dit_program_author_forward(&program_options, builder);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_pipeline_program_builder_seal(builder, host_allocator, out_program);
  }
  id4_pipeline_program_builder_destroy(builder);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static iree_status_t id4_ideogram4_dit_stage_parse_plan_extension(
    const id4_ideogram4_dit_stage_t* stage,
    const id4_pipeline_stage_plan_options_t* options,
    const id4_ideogram4_dit_stage_plan_options_t** out_dit_options) {
  *out_dit_options = NULL;
  if (!options->next) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 DiT stage plan options are required");
  }
  const id4_ideogram4_dit_stage_plan_options_t* dit_options =
      (const id4_ideogram4_dit_stage_plan_options_t*)options->next;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_stage_validate_options_size(
      dit_options->structure_size, sizeof(*dit_options),
      IREE_SV("Ideogram4 DiT stage plan")));
  if (dit_options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram4 DiT stage plan extension structures are not supported");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_stage_validate_request(
      &stage->model, dit_options->request));
  *out_dit_options = dit_options;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_dit_stage_create_program_plan(
    id4_ideogram4_dit_stage_t* stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_program_t* program, id4_pipeline_plan_t** out_plan) {
  id4_pipeline_program_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = IREE_SV(ID4_IDEOGRAM4_DIT_STAGE_NAME);
  plan_options.stage_options = options;
  plan_options.program = program;
  plan_options.device_group = stage->base.services.device_group;
  plan_options.parameter_scope = stage->parameter_scope;
  plan_options.parameter_slab_binding_slot =
      ID4_IDEOGRAM4_DIT_STAGE_PARAMETER_BINDING_SLOT;
  plan_options.boundary_binding_slot_base =
      ID4_IDEOGRAM4_DIT_STAGE_BOUNDARY_BINDING_SLOT_BASE;
  plan_options.alignment = ID4_IDEOGRAM4_DIT_STAGE_ALIGNMENT;
  return id4_pipeline_program_stage_create_plan(
      &plan_options, stage->host_allocator, out_plan);
}

static iree_status_t id4_ideogram4_dit_stage_load(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_load_options_t* options) {
  id4_ideogram4_dit_stage_t* stage = id4_ideogram4_dit_stage_cast(base_stage);
  stage->is_loaded = true;
  return id4_ideogram4_dit_stage_emit_lifecycle(
      options->diagnostics_sink, IREE_SV("stage.load"),
      IREE_SV("loaded Ideogram4 DiT forward stage"));
}

static iree_status_t id4_ideogram4_dit_stage_plan(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_ideogram4_dit_stage_t* stage = id4_ideogram4_dit_stage_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram4 DiT stage must be loaded before planning");
  }
  const id4_ideogram4_dit_stage_plan_options_t* dit_options = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_stage_parse_plan_extension(
      stage, options, &dit_options));

  id4_pipeline_program_t* program = NULL;
  iree_status_t status = id4_ideogram4_dit_stage_author_program(
      stage, dit_options->request, stage->host_allocator, &program);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_dit_stage_create_program_plan(stage, options,
                                                         program, out_plan);
  }
  id4_pipeline_program_release(program);
  return status;
}

static iree_status_t id4_ideogram4_dit_stage_prepare(
    id4_pipeline_stage_t* base_stage, const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options,
    id4_pipeline_bundle_t** out_bundle) {
  id4_ideogram4_dit_stage_t* stage = id4_ideogram4_dit_stage_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram4 DiT stage must be loaded before preparation");
  }

  id4_pipeline_program_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.stage_name = IREE_SV(ID4_IDEOGRAM4_DIT_STAGE_NAME);
  prepare_options.stage_options = options;
  prepare_options.plan = plan;
  prepare_options.device_group = stage->base.services.device_group;
  prepare_options.kernel_cache = stage->kernel_cache;
  prepare_options.executable_cache = stage->base.services.executable_cache;
  return id4_pipeline_program_stage_prepare(&prepare_options,
                                            stage->host_allocator, out_bundle);
}

static iree_status_t id4_ideogram4_dit_stage_issue(
    id4_pipeline_stage_t* base_stage, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  (void)base_stage;
  return id4_pipeline_program_stage_issue(IREE_SV(ID4_IDEOGRAM4_DIT_STAGE_NAME),
                                          bundle, options);
}

static void id4_ideogram4_dit_stage_destroy(id4_pipeline_stage_t* base_stage) {
  id4_ideogram4_dit_stage_t* stage = id4_ideogram4_dit_stage_cast(base_stage);
  iree_allocator_t host_allocator = stage->host_allocator;
  id4_pipeline_kernel_cache_release(stage->kernel_cache);
  iree_allocator_free(host_allocator, (void*)stage->parameter_scope.data);
  id4_pipeline_stage_deinitialize(base_stage);
  iree_allocator_free(host_allocator, stage);
}

static const id4_pipeline_stage_vtable_t id4_ideogram4_dit_stage_vtable = {
    // Destroys the concrete Ideogram4 DiT stage.
    id4_ideogram4_dit_stage_destroy,
    // Loads Ideogram4 DiT immutable state.
    id4_ideogram4_dit_stage_load,
    // Builds an Ideogram4 DiT forward plan.
    id4_ideogram4_dit_stage_plan,
    // Prepares an Ideogram4 DiT forward bundle.
    id4_ideogram4_dit_stage_prepare,
    // Issues an Ideogram4 DiT forward bundle.
    id4_ideogram4_dit_stage_issue,
};

iree_status_t id4_ideogram4_dit_stage_create(
    const id4_ideogram4_dit_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = NULL;
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_dit_stage_validate_create_options(options));

  id4_ideogram4_dit_stage_t* stage = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*stage), (void**)&stage);
  if (iree_status_is_ok(status)) {
    memset(stage, 0, sizeof(*stage));
    stage->host_allocator = host_allocator;
    stage->model = options->model;
    status = id4_pipeline_stage_initialize(&id4_ideogram4_dit_stage_vtable,
                                           &options->services, &stage->base);
  }
  if (iree_status_is_ok(status)) {
    stage->kernel_cache = options->kernel_cache;
    id4_pipeline_kernel_cache_retain(stage->kernel_cache);
    status = id4_ideogram4_dit_stage_copy_parameter_scope(
        options->parameter_scope, stage);
  }
  if (iree_status_is_ok(status)) {
    *out_stage = &stage->base;
  } else if (stage) {
    id4_ideogram4_dit_stage_destroy(&stage->base);
  }
  return status;
}
