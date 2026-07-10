// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/program_stage.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "experimental/id4/pipeline/parameter_layout.h"
#include "experimental/id4/pipeline/parameter_slab.h"
#include "experimental/id4/pipeline/program_plan.h"
#include "experimental/id4/pipeline/program_prepare.h"

typedef struct id4_pipeline_program_stage_counts_t {
  // Number of external tensor import operations in the program.
  iree_host_size_t import_count;
  // Number of parameter operations in the program.
  iree_host_size_t parameter_count;
  // Number of program-owned constant operations in the program.
  iree_host_size_t constant_count;
  // Number of diagnostic tap operations in the program.
  iree_host_size_t tap_count;
} id4_pipeline_program_stage_counts_t;

typedef struct id4_pipeline_program_stage_binding_layout_t {
  // Binding-table slot reserved for the packed parameter slab when present.
  uint32_t parameter_slab_binding_slot;
  // Binding-table slot reserved for the packed constant slab when present.
  uint32_t constant_slab_binding_slot;
  // First binding-table slot assigned to external boundary tensors.
  uint32_t boundary_binding_slot_base;
  // First binding-table slot assigned to diagnostic tap tensors.
  uint32_t diagnostic_tap_binding_slot_base;
  // Binding-table slot reserved for the executable region local slab.
  uint32_t local_binding_slot;
  // Binding-table slot reserved for the plan-shared transient slab.
  uint32_t shared_binding_slot;
  // Exact issue-time binding-table capacity for the executable region.
  iree_host_size_t binding_capacity;
} id4_pipeline_program_stage_binding_layout_t;

typedef struct id4_pipeline_program_stage_bundle_payload_t {
  // Prepared semantic program retained by the bundle payload.
  id4_pipeline_program_prepared_t* prepared_program;
} id4_pipeline_program_stage_bundle_payload_t;

static iree_status_t id4_pipeline_program_stage_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_pipeline_program_stage_validate_stage_name(
    iree_string_view_t stage_name) {
  if (!iree_string_view_is_empty(stage_name)) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "program stage name is required");
}

static iree_status_t id4_pipeline_program_stage_validate_plan_options(
    const id4_pipeline_program_stage_plan_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program stage plan options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_stage_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("program stage plan")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program stage plan extension structures are not supported");
  }
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_stage_validate_stage_name(options->stage_name));
  if (!options->stage_options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program stage plan stage options are required");
  }
  if (!options->program) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program stage plan program is required");
  }
  if (!options->device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program stage plan device group is required");
  }
  if (options->alignment == 0 ||
      !iree_device_size_is_power_of_two(options->alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program stage plan alignment must be a nonzero power of two");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_program_stage_validate_prepare_options(
    const id4_pipeline_program_stage_prepare_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program stage prepare options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_stage_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("program stage prepare")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "program stage prepare extension structures are not supported");
  }
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_stage_validate_stage_name(options->stage_name));
  if (!options->stage_options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program stage prepare stage options are required");
  }
  if (!options->plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program stage prepare plan is required");
  }
  if (!id4_pipeline_plan_source_program(options->plan)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program stage prepare requires a program-backed plan");
  }
  if (!options->device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program stage prepare device group is required");
  }
  if (!options->kernel_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program stage prepare kernel cache is required");
  }
  if (!options->stage_options->kernel_library) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program stage prepare kernel library is required");
  }
  if (!options->executable_cache) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program stage prepare HAL executable cache is required");
  }
  return iree_ok_status();
}

static id4_pipeline_program_stage_counts_t id4_pipeline_program_stage_count_ops(
    const id4_pipeline_program_t* program) {
  id4_pipeline_program_stage_counts_t counts;
  memset(&counts, 0, sizeof(counts));
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* op =
        id4_pipeline_program_operation_at(program, i);
    if (!op) continue;
    switch (op->kind) {
      case ID4_PIPELINE_PROGRAM_OP_KIND_IMPORT:
        ++counts.import_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER:
        ++counts.parameter_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_CONSTANT:
        ++counts.constant_count;
        break;
      case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
        ++counts.tap_count;
        break;
      default:
        break;
    }
  }
  return counts;
}

static bool id4_pipeline_program_stage_tap_name_requested(
    const id4_pipeline_stage_plan_options_t* options, iree_string_view_t name) {
  if (!iree_all_bits_set(
          options->flags,
          ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < options->diagnostic_tap_names.count; ++i) {
    if (iree_string_view_equal(options->diagnostic_tap_names.values[i], name)) {
      return true;
    }
  }
  return false;
}

static bool id4_pipeline_program_stage_operation_uses_tensor(
    const id4_pipeline_stage_plan_options_t* options,
    const id4_pipeline_program_op_t* op, id4_pipeline_program_tensor_t tensor) {
  if (!op) return false;
  switch (op->kind) {
    case ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM:
      for (iree_host_size_t i = 0; i < op->payload.dispatch_loom.binding_count;
           ++i) {
        if (op->payload.dispatch_loom.bindings[i].tensor.ordinal ==
            tensor.ordinal) {
          return true;
        }
      }
      return false;
    case ID4_PIPELINE_PROGRAM_OP_KIND_TAP:
      return id4_pipeline_program_stage_tap_name_requested(
                 options, op->payload.tap.name) &&
             op->payload.tap.tensor.ordinal == tensor.ordinal;
    case ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT:
      return op->payload.export_value.tensor.ordinal == tensor.ordinal;
    default:
      return false;
  }
}

static bool id4_pipeline_program_stage_needs_shared_transient_slot(
    const id4_pipeline_program_stage_plan_options_t* options) {
  const bool region_per_dispatch =
      iree_all_bits_set(options->stage_options->flags,
                        ID4_PIPELINE_STAGE_PLAN_FLAG_REGION_PER_DISPATCH);
  const iree_host_size_t operation_count =
      id4_pipeline_program_operation_count(options->program);
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const id4_pipeline_program_op_t* acquire_op =
        id4_pipeline_program_operation_at(options->program, i);
    if (!acquire_op ||
        acquire_op->kind != ID4_PIPELINE_PROGRAM_OP_KIND_ACQUIRE) {
      continue;
    }
    bool passed_region_cut = false;
    for (iree_host_size_t j = i + 1; j < operation_count; ++j) {
      const id4_pipeline_program_op_t* op =
          id4_pipeline_program_operation_at(options->program, j);
      if (!op) continue;
      if (op->kind == ID4_PIPELINE_PROGRAM_OP_KIND_REGION_CUT) {
        passed_region_cut = true;
        continue;
      }
      if (passed_region_cut &&
          id4_pipeline_program_stage_operation_uses_tensor(
              options->stage_options, op, acquire_op->payload.acquire.tensor)) {
        return true;
      }
      if (region_per_dispatch &&
          op->kind == ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM) {
        passed_region_cut = true;
      }
    }
  }
  return false;
}

static iree_status_t id4_pipeline_program_stage_make_binding_layout(
    const id4_pipeline_program_stage_plan_options_t* options,
    id4_pipeline_program_stage_counts_t counts,
    id4_pipeline_program_stage_binding_layout_t* out_layout) {
  memset(out_layout, 0, sizeof(*out_layout));
  const bool captures_diagnostic_taps =
      iree_all_bits_set(options->stage_options->flags,
                        ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS);
  const iree_host_size_t diagnostic_tap_count =
      captures_diagnostic_taps
          ? options->stage_options->diagnostic_tap_names.count
          : 0;

  uint32_t next_binding_slot = 0;
  if (counts.parameter_count != 0) {
    out_layout->parameter_slab_binding_slot = next_binding_slot++;
  }
  if (counts.constant_count != 0) {
    out_layout->constant_slab_binding_slot = next_binding_slot++;
  }
  out_layout->boundary_binding_slot_base = next_binding_slot;
  if (counts.import_count > UINT32_MAX ||
      next_binding_slot > UINT32_MAX - (uint32_t)counts.import_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program stage boundary binding slot overflow");
  }
  next_binding_slot += (uint32_t)counts.import_count;
  const bool needs_shared_transient_slot =
      id4_pipeline_program_stage_needs_shared_transient_slot(options);
  uint32_t shared_binding_slot = next_binding_slot;
  if (needs_shared_transient_slot) {
    if (next_binding_slot == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "program stage shared binding slot overflow");
    }
    shared_binding_slot = next_binding_slot++;
  }
  if (next_binding_slot == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "program stage local binding slot overflow");
  }
  const uint32_t local_binding_slot = next_binding_slot++;
  const uint32_t diagnostic_tap_binding_slot_base = next_binding_slot;
  if (diagnostic_tap_count > UINT32_MAX ||
      diagnostic_tap_binding_slot_base >
          UINT32_MAX - (uint32_t)diagnostic_tap_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "program stage diagnostic tap binding slot overflow");
  }
  next_binding_slot += (uint32_t)diagnostic_tap_count;
  const iree_host_size_t binding_capacity = next_binding_slot;

  out_layout->diagnostic_tap_binding_slot_base =
      diagnostic_tap_binding_slot_base;
  out_layout->local_binding_slot = local_binding_slot;
  out_layout->shared_binding_slot = shared_binding_slot;
  out_layout->binding_capacity = binding_capacity;
  return iree_ok_status();
}

static id4_pipeline_diagnostic_event_t
id4_pipeline_program_stage_lifecycle_event(iree_string_view_t stage_name,
                                           iree_string_view_t key,
                                           iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event = {
      // Lifecycle event emitted by the program-backed stage adapter.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE,
      // Stable stage name supplied by the concrete stage.
      .stage_name = stage_name,
      // Stable lifecycle key.
      .key = key,
      // Short lifecycle summary.
      .message = message,
  };
  return event;
}

static iree_status_t id4_pipeline_program_stage_emit_lifecycle(
    iree_string_view_t stage_name,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink, iree_string_view_t key,
    iree_string_view_t message) {
  id4_pipeline_diagnostic_event_t event =
      id4_pipeline_program_stage_lifecycle_event(stage_name, key, message);
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

typedef struct id4_pipeline_program_stage_parameter_policy_name_t {
  // Parameter source kind selected by the policy.
  id4_pipeline_stage_parameter_source_kind_t kind;
  // Parameter residency selected by the policy.
  id4_pipeline_stage_parameter_residency_t residency;
  // Stable diagnostic name for the complete policy.
  iree_string_view_t name;
} id4_pipeline_program_stage_parameter_policy_name_t;

static iree_string_view_t id4_pipeline_program_stage_parameter_policy_name(
    const id4_pipeline_stage_parameter_source_t* source) {
  static const id4_pipeline_program_stage_parameter_policy_name_t names[] = {
      {ID4_PIPELINE_STAGE_PARAMETER_SOURCE_KIND_NONE,
       ID4_PIPELINE_STAGE_PARAMETER_RESIDENCY_NONE, IREE_SVL("none")},
      {ID4_PIPELINE_STAGE_PARAMETER_SOURCE_KIND_CHECKPOINT,
       ID4_PIPELINE_STAGE_PARAMETER_RESIDENCY_RESIDENT,
       IREE_SVL("checkpoint/resident")},
      {ID4_PIPELINE_STAGE_PARAMETER_SOURCE_KIND_CHECKPOINT,
       ID4_PIPELINE_STAGE_PARAMETER_RESIDENCY_STREAMING,
       IREE_SVL("checkpoint/streaming")},
      {ID4_PIPELINE_STAGE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT,
       ID4_PIPELINE_STAGE_PARAMETER_RESIDENCY_RESIDENT,
       IREE_SVL("execution_layout/resident")},
      {ID4_PIPELINE_STAGE_PARAMETER_SOURCE_KIND_RESIDENT,
       ID4_PIPELINE_STAGE_PARAMETER_RESIDENCY_RESIDENT, IREE_SVL("resident")},
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(names); ++i) {
    if (names[i].kind == source->kind &&
        names[i].residency == source->residency) {
      return names[i].name;
    }
  }
  return IREE_SV("invalid");
}

iree_status_t id4_pipeline_program_stage_create_plan(
    const id4_pipeline_program_stage_plan_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_stage_validate_plan_options(options));

  const id4_pipeline_program_stage_counts_t counts =
      id4_pipeline_program_stage_count_ops(options->program);
  id4_pipeline_program_stage_binding_layout_t binding_layout;
  IREE_RETURN_IF_ERROR(id4_pipeline_program_stage_make_binding_layout(
      options, counts, &binding_layout));

  id4_pipeline_device_placement_t placement;
  memset(&placement, 0, sizeof(placement));
  placement.role = IREE_SV("default");
  placement.device_index = options->stage_options->device_index;
  placement.queue_affinity = options->stage_options->queue_affinity;

  iree_hal_buffer_params_t parameter_params =
      id4_pipeline_parameter_slab_device_local_params(
          options->stage_options->queue_affinity,
          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
              IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
              IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
          options->alignment);

  iree_hal_buffer_params_t local_params;
  memset(&local_params, 0, sizeof(local_params));
  local_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  local_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  local_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  if (iree_all_bits_set(options->stage_options->flags,
                        ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS)) {
    local_params.usage |= IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
  }
  local_params.queue_affinity = options->stage_options->queue_affinity;
  local_params.min_alignment = options->alignment;

  id4_pipeline_program_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.flags =
      iree_all_bits_set(options->stage_options->flags,
                        ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS)
          ? ID4_PIPELINE_PROGRAM_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS
          : 0;
  if (iree_all_bits_set(options->stage_options->flags,
                        ID4_PIPELINE_STAGE_PLAN_FLAG_REGION_PER_DISPATCH)) {
    plan_options.flags |= ID4_PIPELINE_PROGRAM_PLAN_FLAG_REGION_PER_DISPATCH;
  }
  plan_options.stage_name = options->stage_name;
  plan_options.program = options->program;
  plan_options.device_group = options->device_group;
  plan_options.placement_count = 1;
  plan_options.placements = &placement;
  plan_options.parameter_scope = options->parameter_scope;
  plan_options.parameter_slab_placement_id = 0;
  plan_options.parameter_slab_binding_slot =
      binding_layout.parameter_slab_binding_slot;
  plan_options.parameter_slab_target_params = parameter_params;
  plan_options.parameter_slab_alignment = options->alignment;
  plan_options.parameter_request_alignment = options->alignment;
  plan_options.constant_slab_placement_id = 0;
  plan_options.constant_slab_binding_slot =
      binding_layout.constant_slab_binding_slot;
  plan_options.constant_slab_target_params = parameter_params;
  plan_options.constant_slab_alignment = options->alignment;
  plan_options.constant_request_alignment = options->alignment;
  plan_options.kernel_placement_id = 0;
  plan_options.region_placement_id = 0;
  plan_options.region_local_slab_params = local_params;
  plan_options.region_local_slab_alignment = options->alignment;
  plan_options.region_local_tensor_alignment = options->alignment;
  plan_options.region_binding_capacity = binding_layout.binding_capacity;
  plan_options.region_local_binding_slot = binding_layout.local_binding_slot;
  plan_options.region_shared_binding_slot = binding_layout.shared_binding_slot;
  plan_options.region_boundary_binding_slot_base =
      binding_layout.boundary_binding_slot_base;
  plan_options.diagnostic_tap_names =
      options->stage_options->diagnostic_tap_names;
  plan_options.diagnostic_tap_binding_slot_base =
      binding_layout.diagnostic_tap_binding_slot_base;
  plan_options.diagnostics_sink = options->stage_options->diagnostics_sink;
  return id4_pipeline_program_create_plan(&plan_options, host_allocator,
                                          out_plan);
}

static iree_status_t id4_pipeline_program_stage_join_prepare_readiness(
    iree_hal_semaphore_list_t signal_semaphore_list) {
  if (signal_semaphore_list.count == 0) return iree_ok_status();
  return iree_hal_semaphore_list_wait(signal_semaphore_list,
                                      iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static void id4_pipeline_program_stage_bundle_payload_destroy(
    id4_pipeline_bundle_t* bundle, void* payload) {
  (void)bundle;
  id4_pipeline_program_stage_bundle_payload_t* program_payload =
      (id4_pipeline_program_stage_bundle_payload_t*)payload;
  id4_pipeline_program_prepared_release(program_payload->prepared_program);
  memset(program_payload, 0, sizeof(*program_payload));
}

static iree_status_t id4_pipeline_program_stage_validate_prepare_plan(
    const id4_pipeline_program_stage_prepare_options_t* options) {
  if (!iree_string_view_equal(id4_pipeline_plan_stage_name(options->plan),
                              options->stage_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program stage prepare requires a matching stage plan");
  }
  if (id4_pipeline_plan_device_group(options->plan) != options->device_group) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program stage prepare requires a plan from the stage device group");
  }
  const iree_host_size_t parameter_slab_count =
      id4_pipeline_plan_parameter_slab_count(options->plan);
  const id4_pipeline_stage_parameter_source_t* parameter_source =
      &options->stage_options->parameter_source;
  if (parameter_slab_count == 0 &&
      parameter_source->kind != ID4_PIPELINE_STAGE_PARAMETER_SOURCE_KIND_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program stage parameter source requires planned parameter slabs");
  }
  if (parameter_slab_count != 0 &&
      parameter_source->kind == ID4_PIPELINE_STAGE_PARAMETER_SOURCE_KIND_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "program stage parameter source is required by the plan");
  }
  if (parameter_source->kind ==
      ID4_PIPELINE_STAGE_PARAMETER_SOURCE_KIND_RESIDENT) {
    if (id4_pipeline_parameter_slab_set_has_deferred_load_context(
            parameter_source->storage.resident.slabs)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter slab reuse requires resident parameter slabs");
    }
    IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_parameter_slabs(
        options->plan, parameter_source->storage.resident.slabs));
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_program_stage_prepare(
    const id4_pipeline_program_stage_prepare_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_stage_validate_prepare_options(options));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_stage_validate_prepare_plan(options));
  IREE_RETURN_IF_ERROR(id4_pipeline_program_stage_emit_lifecycle(
      options->stage_name, options->stage_options->diagnostics_sink,
      IREE_SV("stage.prepare.parameter_policy"),
      id4_pipeline_program_stage_parameter_policy_name(
          &options->stage_options->parameter_source)));

  id4_pipeline_parameter_slab_set_t* parameter_slabs = NULL;
  id4_pipeline_program_prepared_t* prepared_program = NULL;
  id4_pipeline_bundle_t* bundle = NULL;
  bool parameter_load_submitted = false;
  const id4_pipeline_stage_parameter_source_t* parameter_source =
      &options->stage_options->parameter_source;
  const bool streams_parameters =
      parameter_source->kind ==
          ID4_PIPELINE_STAGE_PARAMETER_SOURCE_KIND_CHECKPOINT &&
      parameter_source->residency ==
          ID4_PIPELINE_STAGE_PARAMETER_RESIDENCY_STREAMING;
  iree_status_t status = iree_ok_status();
  if (parameter_source->kind ==
      ID4_PIPELINE_STAGE_PARAMETER_SOURCE_KIND_RESIDENT) {
    parameter_slabs = parameter_source->storage.resident.slabs;
    id4_pipeline_parameter_slab_set_retain(parameter_slabs);
  } else if (parameter_source->kind ==
             ID4_PIPELINE_STAGE_PARAMETER_SOURCE_KIND_CHECKPOINT) {
    id4_pipeline_parameter_slab_set_load_options_t load_options;
    memset(&load_options, 0, sizeof(load_options));
    load_options.structure_size = sizeof(load_options);
    load_options.encoder_staging_chunk_byte_capacity =
        ID4_PIPELINE_PARAMETER_ENCODER_DEFAULT_STAGING_CHUNK_BYTE_CAPACITY;
    load_options.encoder_staging_memory_type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    load_options.provider = parameter_source->storage.checkpoint.provider;
    load_options.kernel_library = options->stage_options->kernel_library;
    load_options.kernel_cache = options->kernel_cache;
    load_options.executable_cache = options->executable_cache;
    load_options.command_buffer_mode =
        options->stage_options->command_buffer_mode;
    load_options.diagnostic_artifact_flags =
        options->stage_options->kernel_diagnostic_artifact_flags;
    load_options.wait_semaphore_list =
        options->stage_options->wait_semaphore_list;
    load_options.signal_semaphore_list =
        options->stage_options->signal_semaphore_list;
    load_options.diagnostics_sink = options->stage_options->diagnostics_sink;
    if (streams_parameters) {
      status = id4_pipeline_plan_prepare_parameter_load_context(
          options->plan, &load_options, host_allocator, &parameter_slabs);
    } else {
      status = id4_pipeline_plan_load_parameter_slabs(
          options->plan, &load_options, host_allocator, &parameter_slabs);
      parameter_load_submitted = iree_status_is_ok(status);
    }
  } else if (parameter_source->kind ==
             ID4_PIPELINE_STAGE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT) {
    id4_pipeline_parameter_layout_load_options_t load_options;
    memset(&load_options, 0, sizeof(load_options));
    load_options.structure_size = sizeof(load_options);
    load_options.index = parameter_source->storage.execution_layout.index;
    load_options.provider = parameter_source->storage.execution_layout.provider;
    load_options.scope = parameter_source->storage.execution_layout.scope;
    load_options.wait_semaphore_list =
        options->stage_options->wait_semaphore_list;
    load_options.signal_semaphore_list =
        options->stage_options->signal_semaphore_list;
    load_options.diagnostics_sink = options->stage_options->diagnostics_sink;
    status = id4_pipeline_parameter_layout_load(
        options->plan, &load_options, host_allocator, &parameter_slabs);
    parameter_load_submitted = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_program_prepare_options_t prepare_options;
    memset(&prepare_options, 0, sizeof(prepare_options));
    prepare_options.structure_size = sizeof(prepare_options);
    if (streams_parameters) {
      prepare_options.flags =
          ID4_PIPELINE_PROGRAM_PREPARE_FLAG_COMPACT_PARAMETER_WINDOWS;
    }
    prepare_options.program = id4_pipeline_plan_source_program(options->plan);
    prepare_options.plan = options->plan;
    prepare_options.kernel_cache = options->kernel_cache;
    prepare_options.kernel_library = options->stage_options->kernel_library;
    prepare_options.executable_cache = options->executable_cache;
    prepare_options.executable_caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_NONE;
    prepare_options.command_buffer_mode =
        options->stage_options->command_buffer_mode;
    prepare_options.diagnostic_artifact_flags =
        options->stage_options->kernel_diagnostic_artifact_flags;
    prepare_options.local_slab_alloca_flags = IREE_HAL_ALLOCA_FLAG_NONE;
    prepare_options.local_slab_dealloca_flags = IREE_HAL_DEALLOCA_FLAG_NONE;
    prepare_options.diagnostics_sink = options->stage_options->diagnostics_sink;
    status = id4_pipeline_program_prepare(&prepare_options, host_allocator,
                                          &prepared_program);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_bundle_create_options_t create_options;
    memset(&create_options, 0, sizeof(create_options));
    create_options.structure_size = sizeof(create_options);
    create_options.plan = options->plan;
    create_options.parameter_slabs = parameter_slabs;
    create_options.readiness_semaphore_list =
        parameter_load_submitted ? options->stage_options->signal_semaphore_list
                                 : iree_hal_semaphore_list_empty();
    create_options.payload_size =
        sizeof(id4_pipeline_program_stage_bundle_payload_t);
    create_options.payload_alignment =
        iree_alignof(id4_pipeline_program_stage_bundle_payload_t);
    create_options.payload_destroy =
        id4_pipeline_program_stage_bundle_payload_destroy;
    status =
        id4_pipeline_bundle_create(&create_options, host_allocator, &bundle);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_program_stage_bundle_payload_t* payload =
        (id4_pipeline_program_stage_bundle_payload_t*)
            id4_pipeline_bundle_payload(bundle);
    payload->prepared_program = prepared_program;
    prepared_program = NULL;
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_program_stage_emit_lifecycle(
        options->stage_name, options->stage_options->diagnostics_sink,
        IREE_SV("stage.prepare"), IREE_SV("prepared program bundle"));
  }
  if (iree_status_is_ok(status)) {
    *out_bundle = bundle;
  } else {
    if (parameter_load_submitted) {
      status = iree_status_join(
          status, id4_pipeline_program_stage_join_prepare_readiness(
                      options->stage_options->signal_semaphore_list));
    }
    id4_pipeline_bundle_release(bundle);
    id4_pipeline_program_prepared_release(prepared_program);
  }
  id4_pipeline_parameter_slab_set_release(parameter_slabs);
  return status;
}

iree_status_t id4_pipeline_program_stage_issue(
    iree_string_view_t stage_name, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  IREE_RETURN_IF_ERROR(
      id4_pipeline_program_stage_validate_stage_name(stage_name));
  if (!bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program stage bundle is required");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program stage issue options are required");
  }
  id4_pipeline_program_stage_bundle_payload_t* payload =
      (id4_pipeline_program_stage_bundle_payload_t*)id4_pipeline_bundle_payload(
          bundle);
  if (!payload || !payload->prepared_program) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "program stage bundle payload is not prepared");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_program_stage_emit_lifecycle(
      stage_name, options->diagnostics_sink, IREE_SV("stage.issue"),
      IREE_SV("issued program bundle")));
  return id4_pipeline_program_prepared_issue(payload->prepared_program, bundle,
                                             options);
}
