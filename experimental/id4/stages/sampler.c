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

#define ID4_SAMPLER_STAGE_ALIGNMENT 16
#define ID4_SAMPLER_STAGE_PROGRAM_BLOCK_SIZE (16 * 1024)

typedef enum id4_sampler_stage_kind_e {
  // Device-side initial latent noise generation.
  ID4_SAMPLER_STAGE_KIND_NOISE = 0,
  // Device-side classifier-free guidance and Euler denoise step.
  ID4_SAMPLER_STAGE_KIND_DENOISE = 1,
} id4_sampler_stage_kind_t;

typedef struct id4_sampler_stage_descriptor_t {
  // Concrete sampler stage semantic kind.
  id4_sampler_stage_kind_t kind;
  // Stable stage name used in plans and diagnostics.
  iree_string_view_t stage_name;
  // Human-readable create-options label for diagnostics.
  iree_string_view_t create_options_name;
  // Human-readable plan-options label for diagnostics.
  iree_string_view_t plan_options_name;
  // Lifecycle load message emitted by this stage.
  iree_string_view_t loaded_message;
  // Maximum flattened latent element count accepted by the stage kernel.
  uint64_t max_element_count;
} id4_sampler_stage_descriptor_t;

typedef struct id4_sampler_stage_t {
  // Base stage; must be the first field.
  id4_pipeline_stage_t base;
  // Descriptor selecting the concrete sampler program family.
  const id4_sampler_stage_descriptor_t* descriptor;
  // Allocator used for stage-owned metadata.
  iree_allocator_t host_allocator;
  // Kernel cache used for Loom compilation and HAL executable preparation.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // True after load has completed.
  bool is_loaded;
} id4_sampler_stage_t;

static const id4_sampler_stage_descriptor_t id4_sampler_noise_stage_descriptor =
    {
        .kind = ID4_SAMPLER_STAGE_KIND_NOISE,
        .stage_name = IREE_SVL(ID4_SAMPLER_NOISE_STAGE_NAME),
        .create_options_name = IREE_SVL("sampler noise stage create"),
        .plan_options_name = IREE_SVL("sampler noise stage plan"),
        .loaded_message = IREE_SVL("loaded sampler noise stage"),
        .max_element_count = ID4_SAMPLER_NOISE_MAX_ELEMENT_COUNT,
};

static const id4_sampler_stage_descriptor_t
    id4_sampler_denoise_stage_descriptor = {
        .kind = ID4_SAMPLER_STAGE_KIND_DENOISE,
        .stage_name = IREE_SVL(ID4_SAMPLER_DENOISE_STAGE_NAME),
        .create_options_name = IREE_SVL("sampler denoise stage create"),
        .plan_options_name = IREE_SVL("sampler denoise stage plan"),
        .loaded_message = IREE_SVL("loaded sampler denoise stage"),
        .max_element_count = ID4_SAMPLER_DENOISE_MAX_ELEMENT_COUNT,
};

static id4_sampler_stage_t* id4_sampler_stage_cast(
    id4_pipeline_stage_t* base_stage) {
  return (id4_sampler_stage_t*)base_stage;
}

static iree_status_t id4_sampler_stage_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_sampler_stage_validate_create_values(
    const id4_sampler_stage_descriptor_t* descriptor,
    id4_pipeline_stage_services_t services) {
  if (!services.device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s device group is required",
                            (int)descriptor->create_options_name.size,
                            descriptor->create_options_name.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_sampler_stage_parse_noise_plan_options(
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_program_shape_t* out_latent_shape) {
  const id4_sampler_noise_stage_plan_options_t* sampler_options =
      (const id4_sampler_noise_stage_plan_options_t*)options->next;
  IREE_RETURN_IF_ERROR(id4_sampler_stage_validate_options_size(
      sampler_options->structure_size, sizeof(*sampler_options),
      IREE_SV("sampler noise stage plan")));
  if (sampler_options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "sampler noise stage plan extension structures are not supported");
  }
  *out_latent_shape = sampler_options->request.latent_shape;
  return iree_ok_status();
}

static iree_status_t id4_sampler_stage_parse_denoise_plan_options(
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_program_shape_t* out_latent_shape) {
  const id4_sampler_denoise_stage_plan_options_t* sampler_options =
      (const id4_sampler_denoise_stage_plan_options_t*)options->next;
  IREE_RETURN_IF_ERROR(id4_sampler_stage_validate_options_size(
      sampler_options->structure_size, sizeof(*sampler_options),
      IREE_SV("sampler denoise stage plan")));
  if (sampler_options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "sampler denoise stage plan extension structures are not supported");
  }
  *out_latent_shape = sampler_options->request.latent_shape;
  return iree_ok_status();
}

static iree_status_t id4_sampler_stage_parse_plan_extension(
    const id4_sampler_stage_t* stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_program_shape_t* out_latent_shape) {
  if (!options || !options->next) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s options are required",
                            (int)stage->descriptor->plan_options_name.size,
                            stage->descriptor->plan_options_name.data);
  }
  switch (stage->descriptor->kind) {
    case ID4_SAMPLER_STAGE_KIND_NOISE:
      return id4_sampler_stage_parse_noise_plan_options(options,
                                                        out_latent_shape);
    case ID4_SAMPLER_STAGE_KIND_DENOISE:
      return id4_sampler_stage_parse_denoise_plan_options(options,
                                                          out_latent_shape);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "sampler stage kind %" PRIu32 " is invalid",
                          (uint32_t)stage->descriptor->kind);
}

static iree_status_t id4_sampler_stage_validate_latent_shape(
    const id4_sampler_stage_t* stage,
    id4_pipeline_program_shape_t latent_shape) {
  uint64_t element_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_shape_element_count(latent_shape, &element_count));
  if (latent_shape.rank == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s latent shape rank must be nonzero",
                            (int)stage->descriptor->stage_name.size,
                            stage->descriptor->stage_name.data);
  }
  if (element_count > stage->descriptor->max_element_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%.*s latent element count %" PRIu64
                            " exceeds max count %" PRIu64,
                            (int)stage->descriptor->stage_name.size,
                            stage->descriptor->stage_name.data, element_count,
                            stage->descriptor->max_element_count);
  }
  return iree_ok_status();
}

static iree_status_t id4_sampler_stage_calculate_noise_generator_thread_count(
    const id4_sampler_stage_t* stage, iree_host_size_t device_index,
    id4_pipeline_program_shape_t latent_shape,
    uint64_t* out_generator_thread_count) {
  *out_generator_thread_count = 0;
  if (device_index >=
      iree_hal_device_group_device_count(stage->base.services.device_group)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sampler noise device index %" PRIhsz
                            " is outside the device group",
                            device_index);
  }
  uint64_t element_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_shape_element_count(latent_shape, &element_count));
  const uint64_t rounded_element_count = iree_align_uint64(element_count, 256);
  if (rounded_element_count == 256) {
    *out_generator_thread_count = rounded_element_count;
    return iree_ok_status();
  }
  iree_hal_device_t* device = iree_hal_device_group_device_at(
      stage->base.services.device_group, device_index);
  const iree_hal_device_dispatch_spec_t* dispatch_spec =
      iree_hal_device_spec_dispatch(iree_hal_device_spec(device));
  if (!dispatch_spec || dispatch_spec->execution.unit_count == 0 ||
      dispatch_spec->execution.maximum_resident_invocation_count < 256) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "sampler noise requires device execution-unit and resident-invocation "
        "limits");
  }
  const uint64_t resident_workgroups_per_unit =
      dispatch_spec->execution.maximum_resident_invocation_count / 256;
  const uint64_t resident_thread_count =
      (uint64_t)dispatch_spec->execution.unit_count *
      resident_workgroups_per_unit * 256;
  *out_generator_thread_count =
      iree_min(rounded_element_count, resident_thread_count);
  return iree_ok_status();
}

static id4_pipeline_diagnostic_event_t id4_sampler_stage_lifecycle_event(
    const id4_sampler_stage_t* stage, iree_string_view_t key,
    iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event = {
      // Lifecycle event emitted by the concrete sampler stage.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE,
      // Stable stage name used across sampler diagnostics.
      .stage_name = stage->descriptor->stage_name,
      // Stable lifecycle key.
      .key = key,
      // Short lifecycle summary.
      .message = message,
  };
  return event;
}

static iree_status_t id4_sampler_stage_emit_lifecycle(
    const id4_sampler_stage_t* stage,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink, iree_string_view_t key,
    iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event =
      id4_sampler_stage_lifecycle_event(stage, key, message);
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static iree_status_t id4_sampler_stage_author_noise_program(
    id4_pipeline_program_shape_t latent_shape, uint64_t generator_thread_count,
    id4_pipeline_program_builder_t* builder) {
  id4_sampler_noise_program_options_t program_options;
  memset(&program_options, 0, sizeof(program_options));
  program_options.structure_size = sizeof(program_options);
  program_options.request.latent_shape = latent_shape;
  program_options.request.generator_thread_count = generator_thread_count;
  return id4_sampler_program_author_noise(&program_options, builder);
}

static iree_status_t id4_sampler_stage_author_denoise_program(
    id4_pipeline_program_shape_t latent_shape,
    id4_pipeline_program_builder_t* builder) {
  id4_sampler_denoise_program_options_t program_options;
  memset(&program_options, 0, sizeof(program_options));
  program_options.structure_size = sizeof(program_options);
  program_options.request.latent_shape = latent_shape;
  return id4_sampler_program_author_denoise_step(&program_options, builder);
}

static iree_status_t id4_sampler_stage_author_program(
    const id4_sampler_stage_t* stage, id4_pipeline_program_shape_t latent_shape,
    uint64_t generator_thread_count, iree_allocator_t host_allocator,
    id4_pipeline_program_t** out_program) {
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = NULL;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(ID4_SAMPLER_STAGE_PROGRAM_BLOCK_SIZE,
                                   host_allocator, &block_pool);

  id4_pipeline_program_builder_t* builder = NULL;
  id4_pipeline_program_builder_create_options_t builder_options;
  memset(&builder_options, 0, sizeof(builder_options));
  builder_options.structure_size = sizeof(builder_options);
  builder_options.program_name = stage->descriptor->stage_name;
  builder_options.block_pool = &block_pool;
  iree_status_t status = id4_pipeline_program_builder_create(
      &builder_options, host_allocator, &builder);
  if (iree_status_is_ok(status)) {
    switch (stage->descriptor->kind) {
      case ID4_SAMPLER_STAGE_KIND_NOISE:
        status = id4_sampler_stage_author_noise_program(
            latent_shape, generator_thread_count, builder);
        break;
      case ID4_SAMPLER_STAGE_KIND_DENOISE:
        status =
            id4_sampler_stage_author_denoise_program(latent_shape, builder);
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "sampler stage kind %" PRIu32 " is invalid",
                                  (uint32_t)stage->descriptor->kind);
        break;
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_pipeline_program_builder_seal(builder, host_allocator, out_program);
  }
  id4_pipeline_program_builder_destroy(builder);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static iree_status_t id4_sampler_stage_create_program_plan(
    id4_sampler_stage_t* stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_program_t* program, id4_pipeline_plan_t** out_plan) {
  id4_pipeline_program_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.stage_name = stage->descriptor->stage_name;
  plan_options.stage_options = options;
  plan_options.program = program;
  plan_options.device_group = stage->base.services.device_group;
  plan_options.parameter_scope = iree_string_view_empty();
  plan_options.alignment = ID4_SAMPLER_STAGE_ALIGNMENT;
  return id4_pipeline_program_stage_create_plan(
      &plan_options, stage->host_allocator, out_plan);
}

static iree_status_t id4_sampler_stage_load(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_load_options_t* options) {
  id4_sampler_stage_t* stage = id4_sampler_stage_cast(base_stage);
  stage->is_loaded = true;
  return id4_sampler_stage_emit_lifecycle(stage, options->diagnostics_sink,
                                          IREE_SV("stage.load"),
                                          stage->descriptor->loaded_message);
}

static iree_status_t id4_sampler_stage_plan(
    id4_pipeline_stage_t* base_stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_sampler_stage_t* stage = id4_sampler_stage_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%.*s stage must be loaded before planning",
                            (int)stage->descriptor->stage_name.size,
                            stage->descriptor->stage_name.data);
  }

  id4_pipeline_program_shape_t latent_shape =
      id4_pipeline_program_make_shape_rank0();
  IREE_RETURN_IF_ERROR(
      id4_sampler_stage_parse_plan_extension(stage, options, &latent_shape));
  IREE_RETURN_IF_ERROR(
      id4_sampler_stage_validate_latent_shape(stage, latent_shape));

  uint64_t generator_thread_count = 0;
  if (stage->descriptor->kind == ID4_SAMPLER_STAGE_KIND_NOISE) {
    IREE_RETURN_IF_ERROR(
        id4_sampler_stage_calculate_noise_generator_thread_count(
            stage, options->device_index, latent_shape,
            &generator_thread_count));
  }

  id4_pipeline_program_t* program = NULL;
  iree_status_t status = id4_sampler_stage_author_program(
      stage, latent_shape, generator_thread_count, stage->host_allocator,
      &program);
  if (iree_status_is_ok(status)) {
    status = id4_sampler_stage_create_program_plan(stage, options, program,
                                                   out_plan);
  }
  id4_pipeline_program_release(program);
  return status;
}

static iree_status_t id4_sampler_stage_prepare(
    id4_pipeline_stage_t* base_stage, const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options,
    id4_pipeline_bundle_t** out_bundle) {
  id4_sampler_stage_t* stage = id4_sampler_stage_cast(base_stage);
  if (!stage->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%.*s stage must be loaded before preparation",
                            (int)stage->descriptor->stage_name.size,
                            stage->descriptor->stage_name.data);
  }

  id4_pipeline_program_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.stage_name = stage->descriptor->stage_name;
  prepare_options.stage_options = options;
  prepare_options.plan = plan;
  prepare_options.device_group = stage->base.services.device_group;
  prepare_options.kernel_cache = stage->kernel_cache;
  prepare_options.executable_cache = stage->base.services.executable_cache;
  return id4_pipeline_program_stage_prepare(&prepare_options,
                                            stage->host_allocator, out_bundle);
}

static iree_status_t id4_sampler_stage_issue(
    id4_pipeline_stage_t* base_stage, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  id4_sampler_stage_t* stage = id4_sampler_stage_cast(base_stage);
  return id4_pipeline_program_stage_issue(stage->descriptor->stage_name, bundle,
                                          options);
}

static void id4_sampler_stage_destroy(id4_pipeline_stage_t* base_stage) {
  id4_sampler_stage_t* stage = id4_sampler_stage_cast(base_stage);
  iree_allocator_t host_allocator = stage->host_allocator;
  id4_pipeline_kernel_cache_release(stage->kernel_cache);
  id4_pipeline_stage_deinitialize(base_stage);
  iree_allocator_free(host_allocator, stage);
}

static const id4_pipeline_stage_vtable_t id4_sampler_stage_vtable = {
    // Destroys the concrete sampler stage.
    id4_sampler_stage_destroy,
    // Loads sampler immutable state.
    id4_sampler_stage_load,
    // Builds a sampler plan.
    id4_sampler_stage_plan,
    // Prepares a sampler bundle.
    id4_sampler_stage_prepare,
    // Issues a sampler bundle.
    id4_sampler_stage_issue,
};

static iree_status_t id4_sampler_stage_create(
    const id4_sampler_stage_descriptor_t* descriptor,
    id4_pipeline_stage_services_t services,
    id4_pipeline_kernel_cache_t* kernel_cache, iree_allocator_t host_allocator,
    id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = NULL;
  IREE_RETURN_IF_ERROR(
      id4_sampler_stage_validate_create_values(descriptor, services));

  id4_sampler_stage_t* stage = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*stage), (void**)&stage);
  if (iree_status_is_ok(status)) {
    memset(stage, 0, sizeof(*stage));
    stage->descriptor = descriptor;
    stage->host_allocator = host_allocator;
    status = id4_pipeline_stage_initialize(&id4_sampler_stage_vtable, &services,
                                           &stage->base);
  }
  if (iree_status_is_ok(status)) {
    stage->kernel_cache = kernel_cache;
    id4_pipeline_kernel_cache_retain(stage->kernel_cache);
    *out_stage = &stage->base;
  } else if (stage) {
    id4_sampler_stage_destroy(&stage->base);
  }
  return status;
}

iree_status_t id4_sampler_denoise_stage_create(
    const id4_sampler_denoise_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sampler denoise stage create options are "
                            "required");
  }
  IREE_RETURN_IF_ERROR(id4_sampler_stage_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("sampler denoise stage create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "sampler denoise stage create extension structures are not supported");
  }
  return id4_sampler_stage_create(&id4_sampler_denoise_stage_descriptor,
                                  options->services, options->kernel_cache,
                                  host_allocator, out_stage);
}

iree_status_t id4_sampler_noise_stage_create(
    const id4_sampler_noise_stage_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_stage_t** out_stage) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sampler noise stage create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_sampler_stage_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("sampler noise stage create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "sampler noise stage create extension structures are not supported");
  }
  return id4_sampler_stage_create(&id4_sampler_noise_stage_descriptor,
                                  options->services, options->kernel_cache,
                                  host_allocator, out_stage);
}
