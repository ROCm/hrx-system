// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/sampler.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/pipeline/program_stage.h"
#include "iree/base/internal/arena.h"

#define ID4_SAMPLER_DENOISE_STAGE_ALIGNMENT 16
#define ID4_SAMPLER_DENOISE_STAGE_PROGRAM_BLOCK_SIZE (16 * 1024)

typedef struct id4_sampler_denoise_stage_t {
  // Base stage; must be the first field.
  id4_pipeline_stage_t base;
  // Allocator used for stage-owned metadata.
  iree_allocator_t host_allocator;
  // Kernel cache used for Loom compilation and HAL executable preparation.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // True after load has completed.
  bool is_loaded;
} id4_sampler_denoise_stage_t;

static id4_sampler_denoise_stage_t* id4_sampler_denoise_stage_cast(
    id4_pipeline_stage_t* base_stage) {
  return (id4_sampler_denoise_stage_t*)base_stage;
}

static iree_status_t id4_sampler_denoise_stage_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_sampler_denoise_stage_validate_create_options(
    const id4_sampler_denoise_stage_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sampler denoise stage create options are "
                            "required");
  }
  IREE_RETURN_IF_ERROR(id4_sampler_denoise_stage_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("sampler denoise stage create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "sampler denoise stage create extension structures are not supported");
  }
  if (!options->services.device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sampler denoise stage device group is required");
  }
  return iree_ok_status();
}

static id4_pipeline_diagnostic_event_t
id4_sampler_denoise_stage_lifecycle_event(iree_string_view_t key,
                                          iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event = {
      // Lifecycle event emitted by the concrete sampler stage.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE,
      // Stable stage name used across sampler diagnostics.
      .stage_name = IREE_SV(ID4_SAMPLER_DENOISE_STAGE_NAME),
      // Stable lifecycle key.
      .key = key,
      // Short lifecycle summary.
      .message = message,
  };
  return event;
}

static iree_status_t id4_sampler_denoise_stage_emit_lifecycle(
    id4_pipeline_diagnostics_sink_t* diagnostics_sink, iree_string_view_t key,
    iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event =
      id4_sampler_denoise_stage_lifecycle_event(key, message);
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static iree_status_t id4_sampler_denoise_stage_author_program(
    id4_sampler_denoise_request_config_t request,
    iree_allocator_t host_allocator, id4_pipeline_program_t** out_program) {
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = NULL;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(ID4_SAMPLER_DENOISE_STAGE_PROGRAM_BLOCK_SIZE,
                                   host_allocator, &block_pool);

  id4_pipeline_program_builder_t* builder = NULL;
  id4_pipeline_program_builder_create_options_t builder_options;
  memset(&builder_options, 0, sizeof(builder_options));
  builder_options.structure_size = sizeof(builder_options);
  builder_options.program_name = IREE_SV(ID4_SAMPLER_DENOISE_STAGE_NAME);
  builder_options.block_pool = &block_pool;
  iree_status_t status = id4_pipeline_program_builder_create(
      &builder_options, host_allocator, &builder);
  if (iree_status_is_ok(status)) {
    id4_sampler_program_options_t program_options;
    memset(&program_options, 0, sizeof(program_options));
    program_options.structure_size = sizeof(program_options);
    program_options.request = request;
    status = id4_sampler_program_author_denoise_step(&program_options, builder);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_pipeline_program_builder_seal(builder, host_allocator, out_program);
  }
  id4_pipeline_program_builder_destroy(builder);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static iree_status_t id4_sampler_denoise_stage_parse_plan_extension(
    const id4_pipeline_stage_plan_options_t* options,
    const id4_sampler_denoise_stage_plan_options_t** out_sampler_options) {
  *out_sampler_options = NULL;
  if (!options->next) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sampler denoise stage plan options are required");
  }
  const id4_sampler_denoise_stage_plan_options_t* sampler_options =
      (const id4_sampler_denoise_stage_plan_options_t*)options->next;
  IREE_RETURN_IF_ERROR(id4_sampler_denoise_stage_validate_options_size(
      sampler_options->structure_size, sizeof(*sampler_options),
      IREE_SV("sampler denoise stage plan")));
  if (sampler_options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "sampler denoise stage plan extension structures are not supported");
  }
  uint64_t element_count = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_shape_element_count(
      sampler_options->request.latent_shape, &element_count));
  if (sampler_options->request.latent_shape.rank == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sampler latent shape rank must be nonzero");
  }
  if (element_count > ID4_SAMPLER_DENOISE_MAX_ELEMENT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "sampler latent element count %" PRIu64 " exceeds max count %u",
        element_count, ID4_SAMPLER_DENOISE_MAX_ELEMENT_COUNT);
  }
  *out_sampler_options = sampler_options;
  return iree_ok_status();
}

static iree_status_t id4_sampler_denoise_stage_create_program_plan(
    id4_sampler_denoise_stage_t* stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_program_t* program, id4_pipeline_plan_t** out_plan) {
  id4_pipeline_program_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = IREE_SV(ID4_SAMPLER_DENOISE_STAGE_NAME);
  plan_options.stage_options = options;
  plan_options.program = program;
  plan_options.device_group = stage->base.services.device_group;
  plan_options.parameter_scope = iree_string_view_empty();
  plan_options.alignment = ID4_SAMPLER_DENOISE_STAGE_ALIGNMENT;
  return id4_pipeline_program_stage_create_plan(
      &plan_options, stage->host_allocator, out_plan);
}

static iree_status_t id4_sampler_denoise_stage_load(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_load_options_t* options) {
  id4_sampler_denoise_stage_t* stage =
      id4_sampler_denoise_stage_cast(base_stage);
  stage->is_loaded = true;
  return id4_sampler_denoise_stage_emit_lifecycle(
      options->diagnostics_sink, IREE_SV("stage.load"),
      IREE_SV("loaded sampler denoise stage"));
}

static iree_status_t id4_sampler_denoise_stage_plan(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_sampler_denoise_stage_t* stage =
      id4_sampler_denoise_stage_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "sampler denoise stage must be loaded before planning");
  }
  const id4_sampler_denoise_stage_plan_options_t* sampler_options = NULL;
  IREE_RETURN_IF_ERROR(id4_sampler_denoise_stage_parse_plan_extension(
      options, &sampler_options));

  id4_pipeline_program_t* program = NULL;
  iree_status_t status = id4_sampler_denoise_stage_author_program(
      sampler_options->request, stage->host_allocator, &program);
  if (iree_status_is_ok(status)) {
    status = id4_sampler_denoise_stage_create_program_plan(stage, options,
                                                           program, out_plan);
  }
  id4_pipeline_program_release(program);
  return status;
}

static iree_status_t id4_sampler_denoise_stage_prepare(
    id4_pipeline_stage_t* base_stage, const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options,
    id4_pipeline_bundle_t** out_bundle) {
  id4_sampler_denoise_stage_t* stage =
      id4_sampler_denoise_stage_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "sampler denoise stage must be loaded before preparation");
  }

  id4_pipeline_program_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.stage_name = IREE_SV(ID4_SAMPLER_DENOISE_STAGE_NAME);
  prepare_options.stage_options = options;
  prepare_options.plan = plan;
  prepare_options.device_group = stage->base.services.device_group;
  prepare_options.kernel_cache = stage->kernel_cache;
  prepare_options.executable_cache = stage->base.services.executable_cache;
  return id4_pipeline_program_stage_prepare(&prepare_options,
                                            stage->host_allocator, out_bundle);
}

static iree_status_t id4_sampler_denoise_stage_issue(
    id4_pipeline_stage_t* base_stage, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  (void)base_stage;
  return id4_pipeline_program_stage_issue(
      IREE_SV(ID4_SAMPLER_DENOISE_STAGE_NAME), bundle, options);
}

static void id4_sampler_denoise_stage_destroy(
    id4_pipeline_stage_t* base_stage) {
  id4_sampler_denoise_stage_t* stage =
      id4_sampler_denoise_stage_cast(base_stage);
  iree_allocator_t host_allocator = stage->host_allocator;
  id4_pipeline_kernel_cache_release(stage->kernel_cache);
  id4_pipeline_stage_deinitialize(base_stage);
  iree_allocator_free(host_allocator, stage);
}

static const id4_pipeline_stage_vtable_t id4_sampler_denoise_stage_vtable = {
    // Destroys the concrete sampler denoise stage.
    id4_sampler_denoise_stage_destroy,
    // Loads sampler denoise immutable state.
    id4_sampler_denoise_stage_load,
    // Builds a sampler denoise-step plan.
    id4_sampler_denoise_stage_plan,
    // Prepares a sampler denoise-step bundle.
    id4_sampler_denoise_stage_prepare,
    // Issues a sampler denoise-step bundle.
    id4_sampler_denoise_stage_issue,
};

iree_status_t id4_sampler_denoise_stage_create(
    const id4_sampler_denoise_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = NULL;
  IREE_RETURN_IF_ERROR(
      id4_sampler_denoise_stage_validate_create_options(options));

  id4_sampler_denoise_stage_t* stage = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*stage), (void**)&stage);
  if (iree_status_is_ok(status)) {
    memset(stage, 0, sizeof(*stage));
    stage->host_allocator = host_allocator;
    status = id4_pipeline_stage_initialize(&id4_sampler_denoise_stage_vtable,
                                           &options->services, &stage->base);
  }
  if (iree_status_is_ok(status)) {
    stage->kernel_cache = options->kernel_cache;
    id4_pipeline_kernel_cache_retain(stage->kernel_cache);
    *out_stage = &stage->base;
  } else if (stage) {
    id4_sampler_denoise_stage_destroy(&stage->base);
  }
  return status;
}
