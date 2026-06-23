// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_decode.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/pipeline/program_stage.h"
#include "iree/base/internal/arena.h"

#define ID4_IDEOGRAM4_DECODE_STAGE_ALIGNMENT 16
#define ID4_IDEOGRAM4_DECODE_STAGE_PROGRAM_BLOCK_SIZE (16 * 1024)

typedef struct id4_ideogram4_decode_stage_t {
  // Base stage; must be the first field.
  id4_pipeline_stage_t base;
  // Allocator used for stage-owned metadata.
  iree_allocator_t host_allocator;
  // Kernel cache used for Loom compilation and HAL executable preparation.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Parameter provider scope containing VAE weights.
  iree_string_view_t parameter_scope;
  // Static decode model configuration owned by the stage.
  id4_ideogram4_decode_model_config_t model;
  // True after load has completed.
  bool is_loaded;
} id4_ideogram4_decode_stage_t;

static id4_ideogram4_decode_stage_t* id4_ideogram4_decode_stage_cast(
    id4_pipeline_stage_t* base_stage) {
  return (id4_ideogram4_decode_stage_t*)base_stage;
}

static iree_status_t id4_ideogram4_decode_stage_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_ideogram4_decode_stage_validate_create_options(
    const id4_ideogram4_decode_stage_create_options_t* options) {
  if (!options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram4 decode stage create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_decode_stage_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram4 decode stage create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram4 decode stage create extension structures are not supported");
  }
  if (!options->services.device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 decode stage device group is required");
  }
  return id4_ideogram4_decode_program_validate_diffusion_latent_shape(
      options->model, id4_pipeline_program_make_shape_rank4(
                          1, 1, options->model.vae.latent_channel_count, 1));
}

static iree_status_t id4_ideogram4_decode_stage_copy_parameter_scope(
    iree_string_view_t source, id4_ideogram4_decode_stage_t* stage) {
  stage->parameter_scope = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(stage->host_allocator, source.size,
                                  sizeof(storage[0]), (void**)&storage));
  memcpy(storage, source.data, source.size);
  stage->parameter_scope = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static id4_pipeline_diagnostic_event_t
id4_ideogram4_decode_stage_lifecycle_event(iree_string_view_t key,
                                           iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event = {
      // Lifecycle event emitted by the concrete Ideogram4 decode stage.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE,
      // Stable stage name used across Ideogram4 decode diagnostics.
      .stage_name = IREE_SV(ID4_IDEOGRAM4_DECODE_STAGE_NAME),
      // Stable lifecycle key.
      .key = key,
      // Short lifecycle summary.
      .message = message,
  };
  return event;
}

static iree_status_t id4_ideogram4_decode_stage_emit_lifecycle(
    id4_pipeline_diagnostics_sink_t* diagnostics_sink, iree_string_view_t key,
    iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event =
      id4_ideogram4_decode_stage_lifecycle_event(key, message);
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static iree_status_t id4_ideogram4_decode_stage_author_program(
    const id4_ideogram4_decode_stage_t* stage,
    id4_ideogram4_decode_request_config_t request,
    iree_allocator_t host_allocator, id4_pipeline_program_t** out_program) {
  if (!out_program) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 decode program output is required");
  }
  *out_program = NULL;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(
      ID4_IDEOGRAM4_DECODE_STAGE_PROGRAM_BLOCK_SIZE, host_allocator,
      &block_pool);

  id4_pipeline_program_builder_t* builder = NULL;
  id4_pipeline_program_builder_create_options_t builder_options;
  memset(&builder_options, 0, sizeof(builder_options));
  builder_options.structure_size = sizeof(builder_options);
  builder_options.program_name = IREE_SV(ID4_IDEOGRAM4_DECODE_STAGE_NAME);
  builder_options.block_pool = &block_pool;
  iree_status_t status = id4_pipeline_program_builder_create(
      &builder_options, host_allocator, &builder);
  if (iree_status_is_ok(status)) {
    id4_ideogram4_decode_program_options_t program_options;
    memset(&program_options, 0, sizeof(program_options));
    program_options.structure_size = sizeof(program_options);
    program_options.model = stage->model;
    program_options.request = request;
    status =
        id4_ideogram4_decode_program_author_decode(&program_options, builder);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_pipeline_program_builder_seal(builder, host_allocator, out_program);
  }
  id4_pipeline_program_builder_destroy(builder);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static iree_status_t id4_ideogram4_decode_stage_parse_plan_extension(
    const id4_pipeline_stage_plan_options_t* options,
    const id4_ideogram4_decode_stage_plan_options_t** out_decode_options) {
  *out_decode_options = NULL;
  if (!options->next) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 decode stage plan options are required");
  }
  const id4_ideogram4_decode_stage_plan_options_t* decode_options =
      (const id4_ideogram4_decode_stage_plan_options_t*)options->next;
  IREE_RETURN_IF_ERROR(id4_ideogram4_decode_stage_validate_options_size(
      decode_options->structure_size, sizeof(*decode_options),
      IREE_SV("Ideogram4 decode stage plan")));
  if (decode_options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram4 decode stage plan extension structures are not supported");
  }
  *out_decode_options = decode_options;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_decode_stage_create_program_plan(
    id4_ideogram4_decode_stage_t* stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_program_t* program, id4_pipeline_plan_t** out_plan) {
  id4_pipeline_program_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = IREE_SV(ID4_IDEOGRAM4_DECODE_STAGE_NAME);
  plan_options.stage_options = options;
  plan_options.program = program;
  plan_options.device_group = stage->base.services.device_group;
  plan_options.parameter_scope = stage->parameter_scope;
  plan_options.alignment = ID4_IDEOGRAM4_DECODE_STAGE_ALIGNMENT;
  return id4_pipeline_program_stage_create_plan(
      &plan_options, stage->host_allocator, out_plan);
}

static iree_status_t id4_ideogram4_decode_stage_load(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_load_options_t* options) {
  id4_ideogram4_decode_stage_t* stage =
      id4_ideogram4_decode_stage_cast(base_stage);
  stage->is_loaded = true;
  return id4_ideogram4_decode_stage_emit_lifecycle(
      options->diagnostics_sink, IREE_SV("stage.load"),
      IREE_SV("loaded Ideogram4 decode stage"));
}

static iree_status_t id4_ideogram4_decode_stage_plan(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_ideogram4_decode_stage_t* stage =
      id4_ideogram4_decode_stage_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram4 decode stage must be loaded before planning");
  }
  const id4_ideogram4_decode_stage_plan_options_t* decode_options = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_decode_stage_parse_plan_extension(
      options, &decode_options));

  id4_pipeline_program_t* program = NULL;
  iree_status_t status = id4_ideogram4_decode_stage_author_program(
      stage, decode_options->request, stage->host_allocator, &program);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_decode_stage_create_program_plan(stage, options,
                                                            program, out_plan);
  }
  id4_pipeline_program_release(program);
  return status;
}

static iree_status_t id4_ideogram4_decode_stage_prepare(
    id4_pipeline_stage_t* base_stage, const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options,
    id4_pipeline_bundle_t** out_bundle) {
  id4_ideogram4_decode_stage_t* stage =
      id4_ideogram4_decode_stage_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram4 decode stage must be loaded before preparation");
  }

  id4_pipeline_program_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.stage_name = IREE_SV(ID4_IDEOGRAM4_DECODE_STAGE_NAME);
  prepare_options.stage_options = options;
  prepare_options.plan = plan;
  prepare_options.device_group = stage->base.services.device_group;
  prepare_options.kernel_cache = stage->kernel_cache;
  prepare_options.executable_cache = stage->base.services.executable_cache;
  return id4_pipeline_program_stage_prepare(&prepare_options,
                                            stage->host_allocator, out_bundle);
}

static iree_status_t id4_ideogram4_decode_stage_issue(
    id4_pipeline_stage_t* base_stage, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  (void)base_stage;
  return id4_pipeline_program_stage_issue(
      IREE_SV(ID4_IDEOGRAM4_DECODE_STAGE_NAME), bundle, options);
}

static void id4_ideogram4_decode_stage_destroy(
    id4_pipeline_stage_t* base_stage) {
  id4_ideogram4_decode_stage_t* stage =
      id4_ideogram4_decode_stage_cast(base_stage);
  iree_allocator_t host_allocator = stage->host_allocator;
  id4_pipeline_kernel_cache_release(stage->kernel_cache);
  iree_allocator_free(host_allocator, (void*)stage->parameter_scope.data);
  id4_pipeline_stage_deinitialize(base_stage);
  iree_allocator_free(host_allocator, stage);
}

static const id4_pipeline_stage_vtable_t id4_ideogram4_decode_stage_vtable = {
    // Destroys the concrete Ideogram4 decode stage.
    id4_ideogram4_decode_stage_destroy,
    // Loads Ideogram4 decode immutable state.
    id4_ideogram4_decode_stage_load,
    // Builds an Ideogram4 decode plan.
    id4_ideogram4_decode_stage_plan,
    // Prepares an Ideogram4 decode bundle.
    id4_ideogram4_decode_stage_prepare,
    // Issues an Ideogram4 decode bundle.
    id4_ideogram4_decode_stage_issue,
};

iree_status_t id4_ideogram4_decode_stage_create(
    const id4_ideogram4_decode_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage) {
  if (!out_stage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram4 decode stage output is required");
  }
  *out_stage = NULL;
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_decode_stage_validate_create_options(options));

  id4_ideogram4_decode_stage_t* stage = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*stage), (void**)&stage);
  if (iree_status_is_ok(status)) {
    memset(stage, 0, sizeof(*stage));
    stage->host_allocator = host_allocator;
    status = id4_pipeline_stage_initialize(&id4_ideogram4_decode_stage_vtable,
                                           &options->services, &stage->base);
  }
  if (iree_status_is_ok(status)) {
    stage->kernel_cache = options->kernel_cache;
    id4_pipeline_kernel_cache_retain(stage->kernel_cache);
    stage->model = options->model;
    status = id4_ideogram4_decode_stage_copy_parameter_scope(
        options->parameter_scope, stage);
  }
  if (iree_status_is_ok(status)) {
    *out_stage = &stage->base;
  } else if (stage) {
    id4_ideogram4_decode_stage_destroy(&stage->base);
  }
  return status;
}
