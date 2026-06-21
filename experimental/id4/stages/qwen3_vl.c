// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/qwen3_vl.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/parameter_slab.h"
#include "experimental/id4/pipeline/program.h"
#include "experimental/id4/pipeline/program_plan.h"
#include "experimental/id4/pipeline/program_prepare.h"
#include "iree/base/internal/arena.h"

#define ID4_QWEN3_VL_STAGE_ALIGNMENT 16
#define ID4_QWEN3_VL_STAGE_PARAMETER_BINDING_SLOT 0
#define ID4_QWEN3_VL_STAGE_BOUNDARY_BINDING_SLOT_BASE 1
#define ID4_QWEN3_VL_STAGE_PROGRAM_BLOCK_SIZE (64 * 1024)

typedef struct id4_qwen3_vl_stage_t {
  // Base stage; must be the first field.
  id4_pipeline_stage_t base;
  // Allocator used for stage-owned metadata.
  iree_allocator_t host_allocator;
  // Kernel cache used for Loom compilation and HAL executable preparation.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // Static model configuration owned by the stage.
  id4_qwen3_vl_model_config_t model;
  // Selected layer ordinal storage owned by the stage.
  uint32_t* selected_layer_ordinals;
  // True after load has completed.
  bool is_loaded;
} id4_qwen3_vl_stage_t;

typedef struct id4_qwen3_vl_stage_bundle_payload_t {
  // Prepared semantic program retained by the bundle payload.
  id4_pipeline_program_prepared_t* prepared_program;
} id4_qwen3_vl_stage_bundle_payload_t;

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
      model->head_size == 0) {
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
    program_options.model = stage->model;
    program_options.request = request;
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
  *out_qwen_options = qwen_options;
  return iree_ok_status();
}

static iree_host_size_t id4_qwen3_vl_stage_count_boundary_tensors(
    const id4_pipeline_program_t* program) {
  iree_host_size_t boundary_count = 0;
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (op && op->kind == ID4_PIPELINE_PROGRAM_OP_KIND_IMPORT) {
      ++boundary_count;
    }
  }
  return boundary_count;
}

static iree_host_size_t id4_qwen3_vl_stage_count_diagnostic_taps(
    const id4_pipeline_program_t* program) {
  iree_host_size_t diagnostic_tap_count = 0;
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (op && op->kind == ID4_PIPELINE_PROGRAM_OP_KIND_TAP) {
      ++diagnostic_tap_count;
    }
  }
  return diagnostic_tap_count;
}

static iree_status_t id4_qwen3_vl_stage_make_binding_layout(
    const id4_pipeline_program_t* program,
    id4_pipeline_stage_plan_flags_t plan_flags,
    uint32_t* out_diagnostic_tap_binding_slot_base,
    uint32_t* out_local_binding_slot, iree_host_size_t* out_binding_capacity) {
  *out_diagnostic_tap_binding_slot_base = 0;
  *out_local_binding_slot = 0;
  *out_binding_capacity = 0;
  const iree_host_size_t boundary_count =
      id4_qwen3_vl_stage_count_boundary_tensors(program);
  const bool captures_diagnostic_taps = iree_all_bits_set(
      plan_flags, ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS);
  const iree_host_size_t diagnostic_tap_count =
      captures_diagnostic_taps
          ? id4_qwen3_vl_stage_count_diagnostic_taps(program)
          : 0;
  if (boundary_count > UINT32_MAX ||
      ID4_QWEN3_VL_STAGE_BOUNDARY_BINDING_SLOT_BASE >
          UINT32_MAX - (uint32_t)boundary_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL boundary binding slot overflow");
  }
  const uint32_t diagnostic_tap_binding_slot_base =
      ID4_QWEN3_VL_STAGE_BOUNDARY_BINDING_SLOT_BASE + (uint32_t)boundary_count;
  if (diagnostic_tap_count > UINT32_MAX ||
      diagnostic_tap_binding_slot_base >
          UINT32_MAX - (uint32_t)diagnostic_tap_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL diagnostic tap binding slot overflow");
  }
  const uint32_t local_binding_slot =
      diagnostic_tap_binding_slot_base + (uint32_t)diagnostic_tap_count;
  if (local_binding_slot == ID4_QWEN3_VL_STAGE_PARAMETER_BINDING_SLOT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL local binding slot collides with the "
                            "parameter binding slot");
  }
  iree_host_size_t binding_capacity = 0;
  if (!iree_host_size_checked_add(local_binding_slot, 1, &binding_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL binding capacity overflow");
  }
  *out_diagnostic_tap_binding_slot_base = diagnostic_tap_binding_slot_base;
  *out_local_binding_slot = local_binding_slot;
  *out_binding_capacity = binding_capacity;
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_stage_create_program_plan(
    id4_qwen3_vl_stage_t* stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_program_t* program, id4_pipeline_plan_t** out_plan) {
  uint32_t local_binding_slot = 0;
  uint32_t diagnostic_tap_binding_slot_base = 0;
  iree_host_size_t binding_capacity = 0;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_make_binding_layout(
      program, options->flags, &diagnostic_tap_binding_slot_base,
      &local_binding_slot, &binding_capacity));

  id4_pipeline_device_placement_t placement;
  memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = options->device_index;
  placement.queue_affinity = options->queue_affinity;

  iree_hal_buffer_params_t parameter_params =
      id4_pipeline_parameter_slab_device_local_params(
          options->queue_affinity,
          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
              IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
          ID4_QWEN3_VL_STAGE_ALIGNMENT);

  iree_hal_buffer_params_t local_params;
  memset(&local_params, 0, sizeof(local_params));
  local_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  local_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  local_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  if (iree_all_bits_set(options->flags,
                        ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS)) {
    local_params.usage |= IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
  }
  local_params.queue_affinity = options->queue_affinity;
  local_params.min_alignment = ID4_QWEN3_VL_STAGE_ALIGNMENT;

  id4_pipeline_program_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.flags =
      iree_all_bits_set(options->flags,
                        ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS)
          ? ID4_PIPELINE_PROGRAM_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS
          : 0;
  plan_options.program = program;
  plan_options.device_group = stage->base.services.device_group;
  plan_options.placement_count = 1;
  plan_options.placements = &placement;
  plan_options.parameter_scope = iree_string_view_empty();
  plan_options.parameter_slab_placement_id = 0;
  plan_options.parameter_slab_binding_slot =
      ID4_QWEN3_VL_STAGE_PARAMETER_BINDING_SLOT;
  plan_options.parameter_slab_target_params = parameter_params;
  plan_options.parameter_slab_alignment = ID4_QWEN3_VL_STAGE_ALIGNMENT;
  plan_options.parameter_request_alignment = ID4_QWEN3_VL_STAGE_ALIGNMENT;
  plan_options.kernel_placement_id = 0;
  plan_options.region_placement_id = 0;
  plan_options.region_local_slab_params = local_params;
  plan_options.region_local_slab_alignment = ID4_QWEN3_VL_STAGE_ALIGNMENT;
  plan_options.region_local_tensor_alignment = ID4_QWEN3_VL_STAGE_ALIGNMENT;
  plan_options.region_binding_capacity = binding_capacity;
  plan_options.region_local_binding_slot = local_binding_slot;
  plan_options.region_boundary_binding_slot_base =
      ID4_QWEN3_VL_STAGE_BOUNDARY_BINDING_SLOT_BASE;
  plan_options.diagnostic_tap_binding_slot_base =
      diagnostic_tap_binding_slot_base;
  plan_options.diagnostics_sink = options->diagnostics_sink;
  return id4_pipeline_program_create_plan(&plan_options, stage->host_allocator,
                                          out_plan);
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
      id4_qwen3_vl_stage_parse_plan_extension(options, &qwen_options));

  id4_pipeline_program_t* program = NULL;
  iree_status_t status = id4_qwen3_vl_stage_author_program(
      stage, qwen_options->request, stage->host_allocator, &program);
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_create_program_plan(stage, options, program,
                                                    out_plan);
  }
  id4_pipeline_program_release(program);
  return status;
}

static iree_status_t id4_qwen3_vl_stage_request_from_plan(
    const id4_pipeline_plan_t* plan,
    id4_qwen3_vl_request_config_t* out_request) {
  memset(out_request, 0, sizeof(*out_request));
  const iree_string_view_t token_ids_name =
      id4_qwen3_vl_program_token_ids_boundary_name();
  const id4_pipeline_boundary_tensor_plan_t* token_ids = NULL;
  const iree_host_size_t boundary_tensor_count =
      id4_pipeline_plan_boundary_tensor_count(plan);
  for (iree_host_size_t i = 0; i < boundary_tensor_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary_tensor =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (boundary_tensor &&
        iree_string_view_equal(boundary_tensor->layout.name, token_ids_name)) {
      token_ids = boundary_tensor;
      break;
    }
  }
  if (!token_ids) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL plan has no %.*s boundary tensor",
                            (int)token_ids_name.size, token_ids_name.data);
  }
  if (token_ids->layout.shape.rank != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL token-id boundary rank %u does not match expected rank 1",
        token_ids->layout.shape.rank);
  }
  if (token_ids->layout.shape.dims[0] > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL token count exceeds uint32 range");
  }
  out_request->token_count = (uint32_t)token_ids->layout.shape.dims[0];
  return iree_ok_status();
}

static iree_status_t id4_qwen3_vl_stage_join_prepare_readiness(
    iree_hal_semaphore_list_t signal_semaphore_list) {
  if (signal_semaphore_list.count == 0) return iree_ok_status();
  return iree_hal_semaphore_list_wait(signal_semaphore_list,
                                      iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static void id4_qwen3_vl_stage_bundle_payload_destroy(
    id4_pipeline_bundle_t* bundle, void* payload) {
  (void)bundle;
  id4_qwen3_vl_stage_bundle_payload_t* qwen_payload =
      (id4_qwen3_vl_stage_bundle_payload_t*)payload;
  id4_pipeline_program_prepared_release(qwen_payload->prepared_program);
  memset(qwen_payload, 0, sizeof(*qwen_payload));
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
  if (!iree_string_view_equal(id4_pipeline_plan_stage_name(plan),
                              IREE_SV(ID4_QWEN3_VL_STAGE_NAME))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage prepare requires a Qwen3-VL plan");
  }
  if (id4_pipeline_plan_device_group(plan) !=
      stage->base.services.device_group) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL stage prepare requires a plan from the stage device group");
  }
  if (id4_pipeline_plan_parameter_slab_count(plan) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL stage plan has no parameter slabs");
  }
  if (!options->parameter_provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL parameter provider is required");
  }
  if (!options->kernel_library) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL kernel library is required");
  }
  if (!stage->kernel_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL kernel cache is required");
  }
  if (!stage->base.services.executable_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen3-VL HAL executable cache is required");
  }

  id4_qwen3_vl_request_config_t request;
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_request_from_plan(plan, &request));

  id4_pipeline_program_t* program = NULL;
  id4_pipeline_parameter_slab_set_t* parameter_slabs = NULL;
  id4_pipeline_program_prepared_t* prepared_program = NULL;
  id4_pipeline_bundle_t* bundle = NULL;
  bool parameter_load_submitted = false;

  iree_status_t status = id4_qwen3_vl_stage_author_program(
      stage, request, stage->host_allocator, &program);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_plan_load_parameter_slabs(
        plan, options->parameter_provider, options->wait_semaphore_list,
        options->signal_semaphore_list, options->diagnostics_sink,
        stage->host_allocator, &parameter_slabs);
  }
  parameter_load_submitted = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    id4_pipeline_program_prepare_options_t prepare_options;
    memset(&prepare_options, 0, sizeof(prepare_options));
    prepare_options.structure_size = sizeof(prepare_options);
    prepare_options.program = program;
    prepare_options.plan = plan;
    prepare_options.kernel_cache = stage->kernel_cache;
    prepare_options.kernel_library = options->kernel_library;
    prepare_options.executable_cache = stage->base.services.executable_cache;
    prepare_options.executable_caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    prepare_options.diagnostic_artifact_flags =
        ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_MODULE_TEXT |
        ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_COMPILE_REPORT_JSON |
        ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_EMIT_MANIFEST_JSON;
    prepare_options.local_slab_alloca_flags = IREE_HAL_ALLOCA_FLAG_NONE;
    prepare_options.local_slab_dealloca_flags = IREE_HAL_DEALLOCA_FLAG_NONE;
    prepare_options.diagnostics_sink = options->diagnostics_sink;
    status = id4_pipeline_program_prepare(
        &prepare_options, stage->host_allocator, &prepared_program);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_bundle_create_options_t create_options;
    memset(&create_options, 0, sizeof(create_options));
    create_options.structure_size = sizeof(create_options);
    create_options.plan = plan;
    create_options.parameter_slabs = parameter_slabs;
    create_options.readiness_semaphore_list = options->signal_semaphore_list;
    create_options.payload_size = sizeof(id4_qwen3_vl_stage_bundle_payload_t);
    create_options.payload_alignment =
        iree_alignof(id4_qwen3_vl_stage_bundle_payload_t);
    create_options.payload_destroy = id4_qwen3_vl_stage_bundle_payload_destroy;
    status = id4_pipeline_bundle_create(&create_options, stage->host_allocator,
                                        &bundle);
  }
  if (iree_status_is_ok(status)) {
    id4_qwen3_vl_stage_bundle_payload_t* payload =
        (id4_qwen3_vl_stage_bundle_payload_t*)id4_pipeline_bundle_payload(
            bundle);
    payload->prepared_program = prepared_program;
    prepared_program = NULL;
  }
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_stage_emit_lifecycle(
        options->diagnostics_sink, IREE_SV("stage.prepare"),
        IREE_SV("prepared Qwen3-VL forward bundle"));
  }
  if (iree_status_is_ok(status)) {
    *out_bundle = bundle;
  } else {
    if (parameter_load_submitted) {
      status =
          iree_status_join(status, id4_qwen3_vl_stage_join_prepare_readiness(
                                       options->signal_semaphore_list));
    }
    id4_pipeline_bundle_release(bundle);
    id4_pipeline_program_prepared_release(prepared_program);
  }
  id4_pipeline_parameter_slab_set_release(parameter_slabs);
  id4_pipeline_program_release(program);
  return status;
}

static iree_status_t id4_qwen3_vl_stage_issue(
    id4_pipeline_stage_t* base_stage, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  (void)base_stage;
  id4_qwen3_vl_stage_bundle_payload_t* payload =
      (id4_qwen3_vl_stage_bundle_payload_t*)id4_pipeline_bundle_payload(bundle);
  if (!payload || !payload->prepared_program) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen3-VL bundle payload is not prepared");
  }
  IREE_RETURN_IF_ERROR(id4_qwen3_vl_stage_emit_lifecycle(
      options->diagnostics_sink, IREE_SV("stage.issue"),
      IREE_SV("issued Qwen3-VL forward bundle")));
  return id4_pipeline_program_prepared_issue(payload->prepared_program, bundle,
                                             options);
}

static void id4_qwen3_vl_stage_destroy(id4_pipeline_stage_t* base_stage) {
  id4_qwen3_vl_stage_t* stage = id4_qwen3_vl_stage_cast(base_stage);
  iree_allocator_t host_allocator = stage->host_allocator;
  id4_pipeline_kernel_cache_release(stage->kernel_cache);
  iree_allocator_free(host_allocator, stage->selected_layer_ordinals);
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
    status = id4_qwen3_vl_stage_copy_model_config(&options->model, stage);
  }
  if (iree_status_is_ok(status)) {
    *out_stage = &stage->base;
  } else if (stage) {
    id4_qwen3_vl_stage_destroy(&stage->base);
  }
  return status;
}
