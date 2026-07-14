// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_residency.h"

#include <inttypes.h>
#include <string.h>

#include "experimental/id4/pipeline/json.h"
#include "experimental/id4/pipeline/program.h"
#include "iree/base/internal/arena.h"

typedef struct id4_pipeline_parameter_residency_segment_node_t {
  // Next segment in source-program order.
  struct id4_pipeline_parameter_residency_segment_node_t* next;
  // Segment metadata baked into the final residency plan.
  id4_pipeline_parameter_residency_segment_t segment;
  // Mutable compact window ownership transferred while baking.
  id4_pipeline_parameter_window_t* window;
} id4_pipeline_parameter_residency_segment_node_t;

typedef struct id4_pipeline_parameter_residency_build_state_t {
  // Source plan borrowed for the planner call.
  const id4_pipeline_plan_t* plan;
  // Source semantic program borrowed from plan.
  const id4_pipeline_program_t* program;
  // Maximum compact target allocation permitted per segment.
  iree_device_size_t maximum_target_byte_length;
  // Encoder staging capacity used for exact segment diagnostics.
  iree_device_size_t encoder_staging_chunk_byte_capacity;
  // Source representation used to populate compact windows.
  id4_pipeline_parameter_window_source_kind_t source_kind;
  // Plan parameter count and bit-vector length.
  iree_host_size_t parameter_tensor_count;
  // Plan parameter index by source-program tensor ordinal.
  const iree_host_size_t* parameter_indices_by_program_tensor;
  // Parameters selected by at least one finalized residency segment.
  uint8_t* selected_parameter_bits;
  // Arena retaining append-only build metadata.
  iree_arena_allocator_t* arena;
  // Host allocator used for compact windows and the baked result.
  iree_allocator_t host_allocator;
  // First append-only segment node.
  id4_pipeline_parameter_residency_segment_node_t* segment_head;
  // Last append-only segment node.
  id4_pipeline_parameter_residency_segment_node_t* segment_tail;
  // Number of append-only segment nodes.
  iree_host_size_t segment_count;
  // Number of parameter tensor ordinals across all segments.
  iree_host_size_t segment_parameter_tensor_count;
  // Aggregate statistics accumulated while finalizing segments.
  id4_pipeline_parameter_residency_statistics_t statistics;
} id4_pipeline_parameter_residency_build_state_t;

typedef struct id4_pipeline_parameter_residency_current_segment_t {
  // True after at least one source-program operation enters the segment.
  bool has_operations;
  // First source-program operation in the segment.
  iree_host_size_t source_operation_offset;
  // Number of source-program operations in the segment.
  iree_host_size_t source_operation_count;
  // Number of Loom dispatches in the segment.
  iree_host_size_t dispatch_count;
  // Number of authored barriers in the segment.
  iree_host_size_t barrier_count;
  // Selected plan parameter tensor bits.
  uint8_t* parameter_bits;
  // Exact compact window for the current parameter bit set.
  id4_pipeline_parameter_window_t* window;
} id4_pipeline_parameter_residency_current_segment_t;

struct id4_pipeline_parameter_residency_plan_t {
  // Reference count for shared residency-plan ownership.
  iree_atomic_ref_count_t ref_count;
  // Host allocator used for the packed plan allocation.
  iree_allocator_t host_allocator;
  // Source plan retained for segment tensor names and later lowering.
  id4_pipeline_plan_t* plan;
  // Source representation used to populate compact parameter windows.
  id4_pipeline_parameter_window_source_kind_t source_kind;
  // Encoder staging chunk capacity used by checkpoint window schedules.
  iree_device_size_t encoder_staging_chunk_byte_capacity;
  // Aggregate fixed-plan statistics.
  id4_pipeline_parameter_residency_statistics_t statistics;
  // Ordered segments packed after this header.
  id4_pipeline_parameter_residency_segment_t* segments;
};

static iree_status_t id4_pipeline_parameter_residency_add_host_size(
    iree_host_size_t addend, iree_string_view_t field_name,
    iree_host_size_t* inout_value) {
  if (!iree_host_size_checked_add(*inout_value, addend, inout_value)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter residency %.*s count overflows",
                            (int)field_name.size, field_name.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_residency_add_device_size(
    iree_device_size_t addend, iree_string_view_t field_name,
    iree_device_size_t* inout_value) {
  if (!iree_device_size_checked_add(*inout_value, addend, inout_value)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter residency %.*s byte length overflows",
                            (int)field_name.size, field_name.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_residency_validate_options(
    const id4_pipeline_parameter_residency_plan_create_options_t* options) {
  if (!options || options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter residency options are invalid");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "parameter residency extension structures are not supported");
  }
  if (!options->plan || options->maximum_target_byte_length == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter residency requires a plan and target budget");
  }
  if (options->source_kind !=
          ID4_PIPELINE_PARAMETER_WINDOW_SOURCE_KIND_CHECKPOINT &&
      options->source_kind !=
          ID4_PIPELINE_PARAMETER_WINDOW_SOURCE_KIND_EXECUTION_LAYOUT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter residency source representation is invalid");
  }
  if (options->source_kind ==
          ID4_PIPELINE_PARAMETER_WINDOW_SOURCE_KIND_CHECKPOINT &&
      options->encoder_staging_chunk_byte_capacity == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "checkpoint parameter residency requires an encoder staging "
        "capacity");
  }
  if (!id4_pipeline_plan_source_program(options->plan)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter residency requires a program-backed source plan");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_residency_build_parameter_index_map(
    const id4_pipeline_plan_t* plan, const id4_pipeline_program_t* program,
    iree_arena_allocator_t* arena, iree_host_size_t** out_parameter_indices,
    iree_host_size_t* out_program_tensor_count) {
  *out_parameter_indices = NULL;
  *out_program_tensor_count = id4_pipeline_program_tensor_count(program);
  if (*out_program_tensor_count == 0) return iree_ok_status();
  iree_host_size_t* parameter_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, *out_program_tensor_count, sizeof(parameter_indices[0]),
      (void**)&parameter_indices));
  for (iree_host_size_t i = 0; i < *out_program_tensor_count; ++i) {
    parameter_indices[i] = IREE_HOST_SIZE_MAX;
  }
  const iree_host_size_t parameter_tensor_count =
      id4_pipeline_plan_parameter_tensor_count(plan);
  for (iree_host_size_t parameter_index = 0;
       parameter_index < parameter_tensor_count; ++parameter_index) {
    const id4_pipeline_parameter_tensor_plan_t* parameter =
        id4_pipeline_plan_parameter_tensor_at(plan, parameter_index);
    if (!parameter ||
        parameter->program_tensor_ordinal >= *out_program_tensor_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "planned parameter %" PRIhsz
                              " program tensor ordinal is invalid",
                              parameter_index);
    }
    if (parameter_indices[parameter->program_tensor_ordinal] !=
        IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "program tensor %" PRIu32
                              " has multiple parameter plans",
                              parameter->program_tensor_ordinal);
    }
    parameter_indices[parameter->program_tensor_ordinal] = parameter_index;
  }
  *out_parameter_indices = parameter_indices;
  return iree_ok_status();
}

static iree_host_size_t id4_pipeline_parameter_residency_count_bits(
    iree_host_size_t bit_count, const uint8_t* bits) {
  iree_host_size_t selected_count = 0;
  for (iree_host_size_t i = 0; i < bit_count; ++i) {
    if (bits[i]) ++selected_count;
  }
  return selected_count;
}

static void id4_pipeline_parameter_residency_clear_bits(
    iree_host_size_t bit_count, uint8_t* bits) {
  if (bit_count == 0) return;
  memset(bits, 0, bit_count * sizeof(bits[0]));
}

static void id4_pipeline_parameter_residency_copy_bits(
    iree_host_size_t bit_count, const uint8_t* source, uint8_t* target) {
  if (bit_count == 0) return;
  memcpy(target, source, bit_count * sizeof(target[0]));
}

static iree_host_size_t id4_pipeline_parameter_residency_collect_ordinals(
    const id4_pipeline_plan_t* plan, iree_host_size_t parameter_tensor_count,
    const uint8_t* parameter_bits, uint32_t* out_ordinals) {
  iree_host_size_t ordinal_count = 0;
  for (iree_host_size_t parameter_index = 0;
       parameter_index < parameter_tensor_count; ++parameter_index) {
    if (!parameter_bits[parameter_index]) continue;
    const id4_pipeline_parameter_tensor_plan_t* parameter =
        id4_pipeline_plan_parameter_tensor_at(plan, parameter_index);
    out_ordinals[ordinal_count++] = parameter->program_tensor_ordinal;
  }
  return ordinal_count;
}

static iree_status_t id4_pipeline_parameter_residency_create_window(
    id4_pipeline_parameter_residency_build_state_t* state,
    const uint8_t* parameter_bits, uint32_t* ordinal_scratch,
    id4_pipeline_parameter_window_t** out_window,
    iree_device_size_t* out_target_byte_length) {
  *out_window = NULL;
  *out_target_byte_length = 0;
  const iree_host_size_t parameter_tensor_count =
      id4_pipeline_parameter_residency_collect_ordinals(
          state->plan, state->parameter_tensor_count, parameter_bits,
          ordinal_scratch);
  id4_pipeline_parameter_window_create_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.plan = state->plan;
  options.parameter_tensor_count = parameter_tensor_count;
  options.parameter_tensor_ordinals =
      parameter_tensor_count == 0 ? NULL : ordinal_scratch;
  id4_pipeline_parameter_window_t* window = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_window_create(
      &options, state->host_allocator, &window));
  iree_status_t status = iree_ok_status();
  const iree_host_size_t slab_count =
      id4_pipeline_parameter_window_slab_count(window);
  for (iree_host_size_t slab_index = 0;
       slab_index < slab_count && iree_status_is_ok(status); ++slab_index) {
    const id4_pipeline_parameter_window_slab_t* slab =
        id4_pipeline_parameter_window_slab_at(window, slab_index);
    status = id4_pipeline_parameter_residency_add_device_size(
        slab->byte_length, IREE_SV("segment target"), out_target_byte_length);
  }
  if (iree_status_is_ok(status)) {
    *out_window = window;
  } else {
    id4_pipeline_parameter_window_release(window);
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_residency_update_statistics(
    id4_pipeline_parameter_residency_build_state_t* state,
    const id4_pipeline_parameter_window_resource_statistics_t* resources) {
  state->statistics.peak_segment_target_byte_length =
      iree_max(state->statistics.peak_segment_target_byte_length,
               resources->target_byte_length);
  state->statistics.peak_encoder_staging_byte_length =
      iree_max(state->statistics.peak_encoder_staging_byte_length,
               resources->encoder_staging_byte_length);
  iree_device_size_t segment_live_byte_length = resources->target_byte_length;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_residency_add_device_size(
      resources->encoder_staging_byte_length, IREE_SV("segment live"),
      &segment_live_byte_length));
  state->statistics.peak_segment_live_byte_length =
      iree_max(state->statistics.peak_segment_live_byte_length,
               segment_live_byte_length);
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_residency_add_device_size(
      resources->target_byte_length, IREE_SV("total segment target"),
      &state->statistics.total_target_byte_length));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_residency_add_device_size(
      resources->source_transfer_byte_length, IREE_SV("total source transfer"),
      &state->statistics.total_source_transfer_byte_length));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_residency_add_host_size(
      resources->load_group_count, IREE_SV("total load group"),
      &state->statistics.total_load_group_count));
  return id4_pipeline_parameter_residency_add_host_size(
      resources->encode_load_step_count, IREE_SV("total encoded load step"),
      &state->statistics.total_encode_load_step_count);
}

static iree_status_t id4_pipeline_parameter_residency_finalize_segment(
    id4_pipeline_parameter_residency_build_state_t* state,
    uint32_t semantic_region_id,
    id4_pipeline_parameter_residency_current_segment_t* current) {
  if (!current->has_operations) return iree_ok_status();
  if (!current->window) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter residency segment has no compact window");
  }

  const iree_host_size_t parameter_tensor_count =
      id4_pipeline_parameter_residency_count_bits(state->parameter_tensor_count,
                                                  current->parameter_bits);
  uint32_t* parameter_tensor_ordinals = NULL;
  if (parameter_tensor_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->arena, parameter_tensor_count,
                                  sizeof(parameter_tensor_ordinals[0]),
                                  (void**)&parameter_tensor_ordinals));
    const iree_host_size_t populated_count =
        id4_pipeline_parameter_residency_collect_ordinals(
            state->plan, state->parameter_tensor_count, current->parameter_bits,
            parameter_tensor_ordinals);
    if (populated_count != parameter_tensor_count) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "parameter residency tensor ordinal count changed while finalizing");
    }
  }

  id4_pipeline_parameter_window_resource_statistics_t resources;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_window_query_resource_statistics(
      state->plan, current->window, state->source_kind,
      state->encoder_staging_chunk_byte_capacity, state->host_allocator,
      &resources));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_residency_update_statistics(state, &resources));
  for (iree_host_size_t i = 0; i < state->parameter_tensor_count; ++i) {
    state->selected_parameter_bits[i] |= current->parameter_bits[i];
  }

  id4_pipeline_parameter_residency_segment_node_t* node = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(state->arena, sizeof(*node), (void**)&node));
  memset(node, 0, sizeof(*node));
  node->segment.semantic_region_id = semantic_region_id;
  node->segment.source_operation_offset = current->source_operation_offset;
  node->segment.source_operation_count = current->source_operation_count;
  node->segment.dispatch_count = current->dispatch_count;
  node->segment.barrier_count = current->barrier_count;
  node->segment.parameter_tensor_count = parameter_tensor_count;
  node->segment.parameter_tensor_ordinals = parameter_tensor_ordinals;
  node->segment.window = current->window;
  node->segment.resource_statistics = resources;
  node->window = current->window;
  current->window = NULL;
  if (state->segment_tail) {
    state->segment_tail->next = node;
  } else {
    state->segment_head = node;
  }
  state->segment_tail = node;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_residency_add_host_size(
      1, IREE_SV("segment"), &state->segment_count));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_residency_add_host_size(
      parameter_tensor_count, IREE_SV("segment parameter tensor"),
      &state->segment_parameter_tensor_count));
  state->statistics.segment_count = state->segment_count;
  return iree_ok_status();
}

static void id4_pipeline_parameter_residency_reset_current_segment(
    iree_host_size_t parameter_tensor_count,
    id4_pipeline_parameter_residency_current_segment_t* current) {
  current->has_operations = false;
  current->source_operation_offset = 0;
  current->source_operation_count = 0;
  current->dispatch_count = 0;
  current->barrier_count = 0;
  id4_pipeline_parameter_residency_clear_bits(parameter_tensor_count,
                                              current->parameter_bits);
  id4_pipeline_parameter_window_release(current->window);
  current->window = NULL;
}

static const id4_pipeline_parameter_tensor_plan_t*
id4_pipeline_parameter_residency_largest_parameter(
    const id4_pipeline_plan_t* plan, iree_host_size_t parameter_tensor_count,
    const uint8_t* parameter_bits) {
  const id4_pipeline_parameter_tensor_plan_t* largest = NULL;
  for (iree_host_size_t parameter_index = 0;
       parameter_index < parameter_tensor_count; ++parameter_index) {
    if (!parameter_bits[parameter_index]) continue;
    const id4_pipeline_parameter_tensor_plan_t* parameter =
        id4_pipeline_plan_parameter_tensor_at(plan, parameter_index);
    if (!largest ||
        parameter->layout.byte_length > largest->layout.byte_length) {
      largest = parameter;
    }
  }
  return largest;
}

static iree_status_t id4_pipeline_parameter_residency_accept_epoch(
    id4_pipeline_parameter_residency_build_state_t* state,
    uint32_t semantic_region_id, iree_host_size_t epoch_operation_offset,
    iree_host_size_t epoch_operation_count,
    iree_host_size_t epoch_dispatch_count, iree_host_size_t epoch_barrier_count,
    const uint8_t* epoch_parameter_bits, uint8_t* candidate_parameter_bits,
    uint32_t* ordinal_scratch,
    id4_pipeline_parameter_residency_current_segment_t* current) {
  id4_pipeline_parameter_residency_copy_bits(state->parameter_tensor_count,
                                             current->parameter_bits,
                                             candidate_parameter_bits);
  for (iree_host_size_t i = 0; i < state->parameter_tensor_count; ++i) {
    candidate_parameter_bits[i] |= epoch_parameter_bits[i];
  }

  id4_pipeline_parameter_window_t* candidate_window = NULL;
  iree_device_size_t candidate_target_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_residency_create_window(
      state, candidate_parameter_bits, ordinal_scratch, &candidate_window,
      &candidate_target_byte_length));
  if (candidate_target_byte_length > state->maximum_target_byte_length &&
      current->has_operations) {
    id4_pipeline_parameter_window_release(candidate_window);
    candidate_window = NULL;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_residency_finalize_segment(
        state, semantic_region_id, current));
    id4_pipeline_parameter_residency_reset_current_segment(
        state->parameter_tensor_count, current);
    id4_pipeline_parameter_residency_copy_bits(state->parameter_tensor_count,
                                               epoch_parameter_bits,
                                               candidate_parameter_bits);
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_residency_create_window(
        state, candidate_parameter_bits, ordinal_scratch, &candidate_window,
        &candidate_target_byte_length));
  }
  if (candidate_target_byte_length > state->maximum_target_byte_length) {
    const id4_pipeline_parameter_tensor_plan_t* largest =
        id4_pipeline_parameter_residency_largest_parameter(
            state->plan, state->parameter_tensor_count, epoch_parameter_bits);
    id4_pipeline_parameter_window_release(candidate_window);
    if (largest) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "parameter residency epoch [%" PRIhsz ", %" PRIhsz
          ") requires %" PRIu64 " target bytes, exceeds budget %" PRIu64
          "; largest parameter '%.*s' requires %" PRIu64 " bytes",
          epoch_operation_offset,
          epoch_operation_offset + epoch_operation_count,
          (uint64_t)candidate_target_byte_length,
          (uint64_t)state->maximum_target_byte_length,
          (int)largest->layout.name.size, largest->layout.name.data,
          (uint64_t)largest->layout.byte_length);
    }
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "parameter residency epoch [%" PRIhsz ", %" PRIhsz ") requires %" PRIu64
        " target bytes, exceeds budget %" PRIu64,
        epoch_operation_offset, epoch_operation_offset + epoch_operation_count,
        (uint64_t)candidate_target_byte_length,
        (uint64_t)state->maximum_target_byte_length);
  }

  if (!current->has_operations) {
    current->has_operations = true;
    current->source_operation_offset = epoch_operation_offset;
  }
  iree_host_size_t source_operation_count = current->source_operation_count;
  iree_host_size_t dispatch_count = current->dispatch_count;
  iree_host_size_t barrier_count = current->barrier_count;
  iree_status_t status = id4_pipeline_parameter_residency_add_host_size(
      epoch_operation_count, IREE_SV("segment operation"),
      &source_operation_count);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_residency_add_host_size(
        epoch_dispatch_count, IREE_SV("segment dispatch"), &dispatch_count);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_residency_add_host_size(
        epoch_barrier_count, IREE_SV("segment barrier"), &barrier_count);
  }
  if (!iree_status_is_ok(status)) {
    id4_pipeline_parameter_window_release(candidate_window);
    return status;
  }
  current->source_operation_count = source_operation_count;
  current->dispatch_count = dispatch_count;
  current->barrier_count = barrier_count;
  id4_pipeline_parameter_residency_copy_bits(state->parameter_tensor_count,
                                             candidate_parameter_bits,
                                             current->parameter_bits);
  id4_pipeline_parameter_window_release(current->window);
  current->window = candidate_window;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_residency_collect_dispatch_reads(
    const id4_pipeline_parameter_residency_build_state_t* state,
    iree_host_size_t program_tensor_count,
    const id4_pipeline_program_dispatch_loom_op_t* dispatch,
    uint8_t* epoch_parameter_bits) {
  for (iree_host_size_t binding_index = 0;
       binding_index < dispatch->binding_count; ++binding_index) {
    const id4_pipeline_program_dispatch_binding_t* binding =
        &dispatch->bindings[binding_index];
    if (!iree_any_bit_set(binding->access,
                          ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ)) {
      continue;
    }
    if (binding->tensor.ordinal >= program_tensor_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "dispatch '%.*s' binding %" PRIhsz
                              " tensor ordinal %" PRIu32 " is invalid",
                              (int)dispatch->name.size, dispatch->name.data,
                              binding_index, binding->tensor.ordinal);
    }
    const id4_pipeline_program_tensor_record_t* tensor =
        id4_pipeline_program_tensor_at(state->program, binding->tensor.ordinal);
    if (!tensor || tensor->storage_root_ordinal >= program_tensor_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "dispatch '%.*s' binding %" PRIhsz " storage root is invalid",
          (int)dispatch->name.size, dispatch->name.data, binding_index);
    }
    const iree_host_size_t parameter_index =
        state
            ->parameter_indices_by_program_tensor[tensor->storage_root_ordinal];
    if (parameter_index == IREE_HOST_SIZE_MAX) continue;
    if (parameter_index >= state->parameter_tensor_count) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "dispatch parameter index is outside the plan");
    }
    epoch_parameter_bits[parameter_index] = 1;
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_residency_plan_region(
    id4_pipeline_parameter_residency_build_state_t* state,
    iree_host_size_t program_tensor_count, uint32_t semantic_region_id,
    const id4_pipeline_region_plan_t* region, uint8_t* epoch_parameter_bits,
    uint8_t* candidate_parameter_bits, uint32_t* ordinal_scratch,
    id4_pipeline_parameter_residency_current_segment_t* current) {
  iree_host_size_t region_operation_limit = 0;
  if (!iree_host_size_checked_add(region->source_operation_offset,
                                  region->source_operation_count,
                                  &region_operation_limit)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "region %" PRIu32 " operation range overflows",
                            semantic_region_id);
  }
  iree_host_size_t epoch_operation_offset = region->source_operation_offset;
  iree_host_size_t epoch_dispatch_count = 0;
  iree_host_size_t epoch_barrier_count = 0;
  id4_pipeline_parameter_residency_clear_bits(state->parameter_tensor_count,
                                              epoch_parameter_bits);
  for (iree_host_size_t operation_index = region->source_operation_offset;
       operation_index < region_operation_limit; ++operation_index) {
    const id4_pipeline_program_op_t* operation =
        id4_pipeline_program_operation_at(state->program, operation_index);
    if (!operation) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "region %" PRIu32 " source operation %" PRIhsz
                              " is missing",
                              semantic_region_id, operation_index);
    }
    if (operation->kind == ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM) {
      IREE_RETURN_IF_ERROR(
          id4_pipeline_parameter_residency_collect_dispatch_reads(
              state, program_tensor_count, &operation->payload.dispatch_loom,
              epoch_parameter_bits));
      ++epoch_dispatch_count;
    }
    if (operation->kind != ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER) continue;
    ++epoch_barrier_count;
    const iree_host_size_t epoch_operation_count =
        operation_index + 1 - epoch_operation_offset;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_residency_accept_epoch(
        state, semantic_region_id, epoch_operation_offset,
        epoch_operation_count, epoch_dispatch_count, epoch_barrier_count,
        epoch_parameter_bits, candidate_parameter_bits, ordinal_scratch,
        current));
    epoch_operation_offset = operation_index + 1;
    epoch_dispatch_count = 0;
    epoch_barrier_count = 0;
    id4_pipeline_parameter_residency_clear_bits(state->parameter_tensor_count,
                                                epoch_parameter_bits);
  }
  if (epoch_operation_offset < region_operation_limit) {
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_residency_accept_epoch(
        state, semantic_region_id, epoch_operation_offset,
        region_operation_limit - epoch_operation_offset, epoch_dispatch_count,
        epoch_barrier_count, epoch_parameter_bits, candidate_parameter_bits,
        ordinal_scratch, current));
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_residency_finalize_segment(
      state, semantic_region_id, current));
  id4_pipeline_parameter_residency_reset_current_segment(
      state->parameter_tensor_count, current);
  return iree_ok_status();
}

static void id4_pipeline_parameter_residency_release_build_windows(
    id4_pipeline_parameter_residency_build_state_t* state,
    id4_pipeline_parameter_residency_current_segment_t* current) {
  id4_pipeline_parameter_window_release(current->window);
  current->window = NULL;
  for (id4_pipeline_parameter_residency_segment_node_t* node =
           state->segment_head;
       node; node = node->next) {
    id4_pipeline_parameter_window_release(node->window);
    node->window = NULL;
  }
}

static iree_status_t
id4_pipeline_parameter_residency_finalize_duplication_statistics(
    id4_pipeline_parameter_residency_build_state_t* state,
    uint32_t* ordinal_scratch) {
  id4_pipeline_parameter_window_t* unique_window = NULL;
  iree_device_size_t unique_target_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_residency_create_window(
      state, state->selected_parameter_bits, ordinal_scratch, &unique_window,
      &unique_target_byte_length));
  id4_pipeline_parameter_window_resource_statistics_t unique_resources;
  iree_status_t status =
      id4_pipeline_parameter_window_query_resource_statistics(
          state->plan, unique_window, state->source_kind,
          state->encoder_staging_chunk_byte_capacity, state->host_allocator,
          &unique_resources);
  if (iree_status_is_ok(status)) {
    state->statistics.unique_target_byte_length = unique_target_byte_length;
    state->statistics.unique_source_transfer_byte_length =
        unique_resources.source_transfer_byte_length;
    if (state->statistics.total_target_byte_length <
            unique_target_byte_length ||
        state->statistics.total_source_transfer_byte_length <
            unique_resources.source_transfer_byte_length) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "parameter residency segment traffic is smaller than its union");
    }
  }
  if (iree_status_is_ok(status)) {
    state->statistics.duplicated_target_byte_length =
        state->statistics.total_target_byte_length - unique_target_byte_length;
    state->statistics.duplicated_source_transfer_byte_length =
        state->statistics.total_source_transfer_byte_length -
        unique_resources.source_transfer_byte_length;
  }
  id4_pipeline_parameter_window_release(unique_window);
  return status;
}

static iree_status_t id4_pipeline_parameter_residency_bake(
    id4_pipeline_parameter_residency_build_state_t* state,
    id4_pipeline_parameter_residency_plan_t** out_residency_plan) {
  *out_residency_plan = NULL;
  iree_host_size_t segments_offset = 0;
  iree_host_size_t parameter_ordinals_offset = 0;
  iree_host_size_t allocation_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(id4_pipeline_parameter_residency_plan_t), &allocation_size,
      IREE_STRUCT_FIELD(state->segment_count,
                        id4_pipeline_parameter_residency_segment_t,
                        &segments_offset),
      IREE_STRUCT_FIELD(state->segment_parameter_tensor_count, uint32_t,
                        &parameter_ordinals_offset)));
  id4_pipeline_parameter_residency_plan_t* residency_plan = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      state->host_allocator, allocation_size, (void**)&residency_plan));
  memset(residency_plan, 0, allocation_size);
  iree_atomic_ref_count_init(&residency_plan->ref_count);
  residency_plan->host_allocator = state->host_allocator;
  residency_plan->plan = (id4_pipeline_plan_t*)state->plan;
  id4_pipeline_plan_retain(residency_plan->plan);
  residency_plan->source_kind = state->source_kind;
  residency_plan->encoder_staging_chunk_byte_capacity =
      state->encoder_staging_chunk_byte_capacity;
  residency_plan->statistics = state->statistics;
  uint8_t* allocation_bytes = (uint8_t*)residency_plan;
  residency_plan->segments =
      (id4_pipeline_parameter_residency_segment_t*)(allocation_bytes +
                                                    segments_offset);
  uint32_t* parameter_ordinal_cursor =
      (uint32_t*)(allocation_bytes + parameter_ordinals_offset);

  iree_host_size_t segment_index = 0;
  for (id4_pipeline_parameter_residency_segment_node_t* node =
           state->segment_head;
       node; node = node->next, ++segment_index) {
    id4_pipeline_parameter_residency_segment_t* segment =
        &residency_plan->segments[segment_index];
    *segment = node->segment;
    if (segment->parameter_tensor_count != 0) {
      memcpy(parameter_ordinal_cursor, node->segment.parameter_tensor_ordinals,
             segment->parameter_tensor_count *
                 sizeof(parameter_ordinal_cursor[0]));
      segment->parameter_tensor_ordinals = parameter_ordinal_cursor;
      parameter_ordinal_cursor += segment->parameter_tensor_count;
    } else {
      segment->parameter_tensor_ordinals = NULL;
    }
    segment->window = node->window;
    node->window = NULL;
  }
  if (segment_index != state->segment_count ||
      parameter_ordinal_cursor !=
          (uint32_t*)(allocation_bytes + parameter_ordinals_offset) +
              state->segment_parameter_tensor_count) {
    id4_pipeline_parameter_residency_plan_release(residency_plan);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "parameter residency packed count mismatch");
  }

  *out_residency_plan = residency_plan;
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_residency_plan_create(
    const id4_pipeline_parameter_residency_plan_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_residency_plan_t** out_residency_plan) {
  IREE_ASSERT_ARGUMENT(out_residency_plan);
  *out_residency_plan = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_residency_validate_options(options));

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*total_block_size=*/4096, host_allocator,
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  id4_pipeline_parameter_residency_build_state_t state;
  memset(&state, 0, sizeof(state));
  state.plan = options->plan;
  state.program = id4_pipeline_plan_source_program(options->plan);
  state.maximum_target_byte_length = options->maximum_target_byte_length;
  state.encoder_staging_chunk_byte_capacity =
      options->encoder_staging_chunk_byte_capacity;
  state.source_kind = options->source_kind;
  state.parameter_tensor_count =
      id4_pipeline_plan_parameter_tensor_count(options->plan);
  state.arena = &arena;
  state.host_allocator = host_allocator;
  state.statistics.semantic_region_count =
      id4_pipeline_plan_region_count(options->plan);
  state.statistics.maximum_target_byte_length =
      options->maximum_target_byte_length;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t slab_index = 0;
       slab_index < id4_pipeline_plan_parameter_slab_count(options->plan) &&
       iree_status_is_ok(status);
       ++slab_index) {
    const id4_pipeline_parameter_slab_plan_t* slab =
        id4_pipeline_plan_parameter_slab_at(options->plan, slab_index);
    status = id4_pipeline_parameter_residency_add_device_size(
        slab->byte_length, IREE_SV("resident target"),
        &state.statistics.resident_target_byte_length);
  }

  iree_host_size_t* parameter_indices_by_program_tensor = NULL;
  iree_host_size_t program_tensor_count = 0;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_residency_build_parameter_index_map(
        state.plan, state.program, &arena, &parameter_indices_by_program_tensor,
        &program_tensor_count);
  }
  state.parameter_indices_by_program_tensor =
      parameter_indices_by_program_tensor;

  uint8_t* parameter_bit_storage = NULL;
  uint32_t* ordinal_scratch = NULL;
  if (iree_status_is_ok(status) && state.parameter_tensor_count != 0) {
    status = iree_arena_allocate_array(&arena, /*element_count=*/4,
                                       state.parameter_tensor_count,
                                       (void**)&parameter_bit_storage);
  }
  if (iree_status_is_ok(status) && state.parameter_tensor_count != 0) {
    status = iree_arena_allocate_array(&arena, state.parameter_tensor_count,
                                       sizeof(ordinal_scratch[0]),
                                       (void**)&ordinal_scratch);
  }
  uint8_t* current_parameter_bits = parameter_bit_storage;
  uint8_t* epoch_parameter_bits =
      parameter_bit_storage
          ? parameter_bit_storage + state.parameter_tensor_count
          : NULL;
  uint8_t* candidate_parameter_bits =
      epoch_parameter_bits ? epoch_parameter_bits + state.parameter_tensor_count
                           : NULL;
  state.selected_parameter_bits =
      candidate_parameter_bits
          ? candidate_parameter_bits + state.parameter_tensor_count
          : NULL;
  id4_pipeline_parameter_residency_current_segment_t current;
  memset(&current, 0, sizeof(current));
  current.parameter_bits = current_parameter_bits;
  id4_pipeline_parameter_residency_clear_bits(state.parameter_tensor_count,
                                              current.parameter_bits);
  id4_pipeline_parameter_residency_clear_bits(state.parameter_tensor_count,
                                              state.selected_parameter_bits);

  const iree_host_size_t region_count =
      id4_pipeline_plan_region_count(options->plan);
  for (iree_host_size_t region_index = 0;
       region_index < region_count && iree_status_is_ok(status);
       ++region_index) {
    if (region_index > UINT32_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "semantic region index exceeds uint32_t");
      break;
    }
    const id4_pipeline_region_plan_t* region =
        id4_pipeline_plan_region_at(options->plan, region_index);
    if (!region) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "semantic region %" PRIhsz " is missing",
                                region_index);
      break;
    }
    status = id4_pipeline_parameter_residency_plan_region(
        &state, program_tensor_count, (uint32_t)region_index, region,
        epoch_parameter_bits, candidate_parameter_bits, ordinal_scratch,
        &current);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_residency_finalize_duplication_statistics(
        &state, ordinal_scratch);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_residency_bake(&state, out_residency_plan);
  }
  id4_pipeline_parameter_residency_release_build_windows(&state, &current);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

void id4_pipeline_parameter_residency_plan_retain(
    id4_pipeline_parameter_residency_plan_t* residency_plan) {
  if (!residency_plan) return;
  iree_atomic_ref_count_inc(&residency_plan->ref_count);
}

static void id4_pipeline_parameter_residency_plan_destroy(
    id4_pipeline_parameter_residency_plan_t* residency_plan) {
  const iree_host_size_t segment_count =
      residency_plan->statistics.segment_count;
  for (iree_host_size_t i = 0; i < segment_count; ++i) {
    id4_pipeline_parameter_window_release(
        (id4_pipeline_parameter_window_t*)residency_plan->segments[i].window);
  }
  id4_pipeline_plan_release(residency_plan->plan);
  iree_allocator_free(residency_plan->host_allocator, residency_plan);
}

void id4_pipeline_parameter_residency_plan_release(
    id4_pipeline_parameter_residency_plan_t* residency_plan) {
  if (residency_plan &&
      iree_atomic_ref_count_dec(&residency_plan->ref_count) == 1) {
    id4_pipeline_parameter_residency_plan_destroy(residency_plan);
  }
}

const id4_pipeline_plan_t* id4_pipeline_parameter_residency_plan_source_plan(
    const id4_pipeline_parameter_residency_plan_t* residency_plan) {
  return residency_plan ? residency_plan->plan : NULL;
}

id4_pipeline_parameter_residency_statistics_t
id4_pipeline_parameter_residency_plan_statistics(
    const id4_pipeline_parameter_residency_plan_t* residency_plan) {
  id4_pipeline_parameter_residency_statistics_t statistics;
  memset(&statistics, 0, sizeof(statistics));
  return residency_plan ? residency_plan->statistics : statistics;
}

iree_host_size_t id4_pipeline_parameter_residency_plan_segment_count(
    const id4_pipeline_parameter_residency_plan_t* residency_plan) {
  return residency_plan ? residency_plan->statistics.segment_count : 0;
}

const id4_pipeline_parameter_residency_segment_t*
id4_pipeline_parameter_residency_plan_segment_at(
    const id4_pipeline_parameter_residency_plan_t* residency_plan,
    iree_host_size_t index) {
  if (!residency_plan || index >= residency_plan->statistics.segment_count) {
    return NULL;
  }
  return &residency_plan->segments[index];
}

iree_status_t id4_pipeline_parameter_residency_plan_query_live_statistics(
    const id4_pipeline_parameter_residency_plan_t* residency_plan,
    iree_host_size_t parameter_load_prefetch_segment_distance,
    id4_pipeline_parameter_window_statistics_t* out_statistics) {
  if (!residency_plan || !out_statistics) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter residency plan and live statistics output are required");
  }
  iree_host_size_t concurrent_window_count = 0;
  if (!iree_host_size_checked_add(parameter_load_prefetch_segment_distance, 1,
                                  &concurrent_window_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter residency prefetch segment distance overflows");
  }

  const iree_host_size_t segment_count =
      residency_plan->statistics.segment_count;
  const id4_pipeline_parameter_window_t** windows = NULL;
  iree_status_t status = iree_ok_status();
  if (segment_count != 0) {
    status = iree_allocator_malloc_array(residency_plan->host_allocator,
                                         segment_count, sizeof(windows[0]),
                                         (void**)&windows);
  }
  for (iree_host_size_t i = 0; i < segment_count && iree_status_is_ok(status);
       ++i) {
    windows[i] = residency_plan->segments[i].window;
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_parameter_window_sequence_statistics_options_t options;
    memset(&options, 0, sizeof(options));
    options.structure_size = sizeof(options);
    options.plan = residency_plan->plan;
    options.source_kind = residency_plan->source_kind;
    options.window_count = segment_count;
    options.windows = windows;
    options.concurrent_window_count = concurrent_window_count;
    options.encoder_staging_chunk_byte_capacity =
        residency_plan->encoder_staging_chunk_byte_capacity;
    status = id4_pipeline_parameter_window_query_sequence_statistics(
        &options, residency_plan->host_allocator, out_statistics);
  }
  iree_allocator_free(residency_plan->host_allocator, windows);
  return status;
}

static const id4_pipeline_parameter_tensor_plan_t*
id4_pipeline_parameter_residency_find_parameter(
    const id4_pipeline_plan_t* plan, uint32_t program_tensor_ordinal) {
  const iree_host_size_t parameter_tensor_count =
      id4_pipeline_plan_parameter_tensor_count(plan);
  for (iree_host_size_t i = 0; i < parameter_tensor_count; ++i) {
    const id4_pipeline_parameter_tensor_plan_t* parameter =
        id4_pipeline_plan_parameter_tensor_at(plan, i);
    if (parameter->program_tensor_ordinal == program_tensor_ordinal) {
      return parameter;
    }
  }
  return NULL;
}

iree_status_t id4_pipeline_parameter_residency_plan_format_json(
    const id4_pipeline_parameter_residency_plan_t* residency_plan,
    iree_string_builder_t* builder) {
  if (!residency_plan || !builder) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "residency plan and JSON builder are required");
  }
  const id4_pipeline_parameter_residency_statistics_t statistics =
      residency_plan->statistics;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "{\"semantic_region_count\":%" PRIhsz ",\"segment_count\":%" PRIhsz
      ",\"maximum_target_byte_length\":%" PRIu64
      ",\"resident_target_byte_length\":%" PRIu64
      ",\"peak_segment_target_byte_length\":%" PRIu64
      ",\"peak_encoder_staging_byte_length\":%" PRIu64
      ",\"peak_segment_live_byte_length\":%" PRIu64
      ",\"unique_target_byte_length\":%" PRIu64
      ",\"unique_source_transfer_byte_length\":%" PRIu64
      ",\"total_target_byte_length\":%" PRIu64
      ",\"total_source_transfer_byte_length\":%" PRIu64
      ",\"duplicated_target_byte_length\":%" PRIu64
      ",\"duplicated_source_transfer_byte_length\":%" PRIu64
      ",\"total_load_group_count\":%" PRIhsz
      ",\"total_encode_load_step_count\":%" PRIhsz ",\"segments\":[",
      statistics.semantic_region_count, statistics.segment_count,
      (uint64_t)statistics.maximum_target_byte_length,
      (uint64_t)statistics.resident_target_byte_length,
      (uint64_t)statistics.peak_segment_target_byte_length,
      (uint64_t)statistics.peak_encoder_staging_byte_length,
      (uint64_t)statistics.peak_segment_live_byte_length,
      (uint64_t)statistics.unique_target_byte_length,
      (uint64_t)statistics.unique_source_transfer_byte_length,
      (uint64_t)statistics.total_target_byte_length,
      (uint64_t)statistics.total_source_transfer_byte_length,
      (uint64_t)statistics.duplicated_target_byte_length,
      (uint64_t)statistics.duplicated_source_transfer_byte_length,
      statistics.total_load_group_count,
      statistics.total_encode_load_step_count));
  for (iree_host_size_t segment_index = 0;
       segment_index < statistics.segment_count; ++segment_index) {
    if (segment_index != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const id4_pipeline_parameter_residency_segment_t* segment =
        &residency_plan->segments[segment_index];
    const id4_pipeline_region_plan_t* region = id4_pipeline_plan_region_at(
        residency_plan->plan, segment->semantic_region_id);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "{\"semantic_region_id\":%" PRIu32 ",\"semantic_region_name\":",
        segment->semantic_region_id));
    IREE_RETURN_IF_ERROR(id4_pipeline_json_append_string(
        builder, region ? region->name : iree_string_view_empty()));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"source_operation_offset\":%" PRIhsz
        ",\"source_operation_count\":%" PRIhsz ",\"dispatch_count\":%" PRIhsz
        ",\"barrier_count\":%" PRIhsz ",\"target_byte_length\":%" PRIu64
        ",\"encoder_staging_byte_length\":%" PRIu64
        ",\"source_transfer_byte_length\":%" PRIu64
        ",\"load_group_count\":%" PRIhsz ",\"encode_load_step_count\":%" PRIhsz
        ",\"parameters\":[",
        segment->source_operation_offset, segment->source_operation_count,
        segment->dispatch_count, segment->barrier_count,
        (uint64_t)segment->resource_statistics.target_byte_length,
        (uint64_t)segment->resource_statistics.encoder_staging_byte_length,
        (uint64_t)segment->resource_statistics.source_transfer_byte_length,
        segment->resource_statistics.load_group_count,
        segment->resource_statistics.encode_load_step_count));
    for (iree_host_size_t parameter_index = 0;
         parameter_index < segment->parameter_tensor_count; ++parameter_index) {
      if (parameter_index != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
      }
      const uint32_t program_tensor_ordinal =
          segment->parameter_tensor_ordinals[parameter_index];
      const id4_pipeline_parameter_tensor_plan_t* parameter =
          id4_pipeline_parameter_residency_find_parameter(
              residency_plan->plan, program_tensor_ordinal);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "{\"program_tensor_ordinal\":%" PRIu32 ",\"name\":",
          program_tensor_ordinal));
      IREE_RETURN_IF_ERROR(id4_pipeline_json_append_string(
          builder,
          parameter ? parameter->layout.name : iree_string_view_empty()));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ",\"byte_length\":%" PRIu64 "}",
          parameter ? (uint64_t)parameter->layout.byte_length : 0));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "]}"));
  }
  return iree_string_builder_append_cstring(builder, "]}");
}
