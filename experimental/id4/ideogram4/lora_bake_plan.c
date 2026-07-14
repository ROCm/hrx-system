// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/lora_bake_plan.h"

#include <string.h>

#include "experimental/id4/pipeline/program.h"
#include "iree/base/internal/atomics.h"

enum {
  ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE = 16,
  ID4_IDEOGRAM4_LORA_BAKE_WORKING_ALIGNMENT = 16,
};

struct id4_ideogram4_lora_bake_plan_t {
  // Reference count for shared bake-plan ownership.
  iree_atomic_ref_count_t ref_count;
  // Host allocator used for object storage.
  iree_allocator_t host_allocator;
  // Exact-base conditioned-DiT plan retained for slab metadata.
  const id4_pipeline_plan_t* base_plan;
  // Ordered immutable LoRA topology retained for source parameters.
  id4_ideogram4_lora_topology_t* topology;
  // Index of the independently replaceable patchable parameter slab.
  iree_host_size_t patchable_slab_index;
  // Byte length of the independently replaceable patchable parameter slab.
  iree_device_size_t patchable_slab_byte_length;
  // Maximum device working bytes used by any target window.
  iree_device_size_t working_set_high_water_mark;
  // Total BF16 source adapter bytes consumed by all planned targets.
  iree_device_size_t adapter_byte_length;
  // Number of entries in |targets|.
  iree_host_size_t target_count;
  // Planned target records stored inline with this object.
  id4_ideogram4_lora_bake_target_t* targets;
};

static iree_status_t id4_ideogram4_lora_bake_plan_validate_options(
    const id4_ideogram4_lora_bake_plan_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA bake plan options are required");
  }
  if (options->structure_size != sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA bake plan options size is invalid");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "LoRA bake plan extension structures are not supported");
  }
  if (!options->base_plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA bake base plan is required");
  }
  if (!id4_pipeline_plan_source_program(options->base_plan)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA baking requires a program-backed conditioned-DiT plan");
  }
  if (!options->topology ||
      id4_ideogram4_lora_topology_adapter_count(options->topology) == 0 ||
      id4_ideogram4_lora_topology_target_count(options->topology) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA baking requires a nonempty topology");
  }
  if (options->working_set_byte_capacity == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA bake working-set byte capacity must be nonzero");
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_lora_bake_plan_validate_base_program(
    const id4_pipeline_plan_t* base_plan) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(base_plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(base_plan, i);
    if (boundary && iree_string_view_equal(boundary->layout.name,
                                           IREE_SV("lora.strengths"))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LoRA baking requires an exact-base plan without dynamic adapter "
          "execution");
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_lora_bake_plan_find_patchable_slab(
    const id4_pipeline_plan_t* base_plan, iree_host_size_t* out_slab_index,
    const id4_pipeline_parameter_slab_plan_t** out_slab) {
  *out_slab_index = IREE_HOST_SIZE_MAX;
  *out_slab = NULL;
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_parameter_slab_count(base_plan); ++i) {
    const id4_pipeline_parameter_slab_plan_t* slab =
        id4_pipeline_plan_parameter_slab_at(base_plan, i);
    if (!slab ||
        !iree_string_view_equal(slab->domain, IREE_SV("lora_patchable"))) {
      continue;
    }
    if (*out_slab) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LoRA bake base plan contains multiple patchable parameter slabs");
    }
    *out_slab_index = i;
    *out_slab = slab;
  }
  if (!*out_slab) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "LoRA bake base plan has no `lora_patchable` parameter slab");
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_lora_bake_plan_find_parameter(
    const id4_pipeline_plan_t* base_plan, iree_string_view_t key,
    const id4_pipeline_parameter_tensor_plan_t** out_tensor,
    const id4_pipeline_program_parameter_op_t** out_parameter) {
  *out_tensor = NULL;
  *out_parameter = NULL;
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_parameter_tensor_count(base_plan); ++i) {
    const id4_pipeline_parameter_tensor_plan_t* tensor =
        id4_pipeline_plan_parameter_tensor_at(base_plan, i);
    if (!tensor || !iree_string_view_equal(tensor->layout.name, key)) continue;
    if (*out_tensor) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LoRA bake parameter `%.*s` appears more than once in the base plan",
          (int)key.size, key.data);
    }
    *out_tensor = tensor;
  }
  if (!*out_tensor) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "LoRA bake parameter `%.*s` is not in the base plan", (int)key.size,
        key.data);
  }

  const id4_pipeline_program_t* program =
      id4_pipeline_plan_source_program(base_plan);
  const id4_pipeline_program_tensor_record_t* tensor_record =
      id4_pipeline_program_tensor_at(program,
                                     (*out_tensor)->program_tensor_ordinal);
  if (!tensor_record) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "LoRA bake parameter `%.*s` has no source program tensor",
        (int)key.size, key.data);
  }
  const id4_pipeline_program_op_t* operation =
      id4_pipeline_program_operation_at(
          program, tensor_record->producer_operation_ordinal);
  if (!operation || operation->kind != ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER ||
      operation->payload.parameter.tensor.ordinal !=
          (*out_tensor)->program_tensor_ordinal) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "LoRA bake parameter `%.*s` has no source parameter operation",
        (int)key.size, key.data);
  }
  *out_parameter = &operation->payload.parameter;
  return iree_ok_status();
}

static bool id4_ideogram4_lora_bake_shape_matches(
    id4_pipeline_tensor_shape_t shape, uint32_t rank, uint64_t dim0,
    uint64_t dim1) {
  return shape.rank == rank && shape.dims[0] == dim0 &&
         (rank == 1 || shape.dims[1] == dim1);
}

static iree_status_t id4_ideogram4_lora_bake_validate_parameter_range(
    const id4_pipeline_parameter_tensor_plan_t* tensor,
    iree_host_size_t patchable_slab_index,
    iree_device_size_t patchable_slab_byte_length,
    iree_string_view_t target_key,
    id4_ideogram4_lora_bake_parameter_range_t* out_range) {
  if (tensor->parameter_slab_index != patchable_slab_index) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA bake parameter `%.*s` is outside the patchable slab",
        (int)target_key.size, target_key.data);
  }
  iree_device_size_t end = 0;
  if (!iree_device_size_checked_add(tensor->offset, tensor->layout.byte_length,
                                    &end) ||
      end > patchable_slab_byte_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "LoRA bake parameter `%.*s` exceeds the patchable slab",
        (int)target_key.size, target_key.data);
  }
  *out_range = (id4_ideogram4_lora_bake_parameter_range_t){
      .offset = tensor->offset,
      .length = tensor->layout.byte_length,
  };
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_lora_bake_pack_working_range(
    iree_device_size_t byte_length,
    id4_ideogram4_lora_bake_parameter_range_t* out_range,
    iree_device_size_t* inout_total) {
  iree_device_size_t aligned_offset = 0;
  if (!iree_device_size_checked_align(*inout_total,
                                      ID4_IDEOGRAM4_LORA_BAKE_WORKING_ALIGNMENT,
                                      &aligned_offset) ||
      !iree_device_size_checked_add(aligned_offset, byte_length, inout_total)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LoRA bake working-set size overflows");
  }
  *out_range = (id4_ideogram4_lora_bake_parameter_range_t){
      .offset = aligned_offset,
      .length = byte_length,
  };
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_lora_bake_calculate_working_set(
    iree_host_size_t adapter_count,
    const id4_ideogram4_dit_lora_target_t* source_target,
    uint32_t maximum_segment_rank, uint32_t output_row_count,
    id4_ideogram4_lora_bake_working_set_t* out_working_set) {
  memset(out_working_set, 0, sizeof(*out_working_set));
  iree_device_size_t down_source_element_count = 0;
  iree_device_size_t down_element_count = 0;
  iree_device_size_t up_element_count = 0;
  iree_device_size_t effective_weight_element_count = 0;
  iree_device_size_t down_source_byte_length = 0;
  iree_device_size_t down_byte_length = 0;
  iree_device_size_t up_byte_length = 0;
  iree_device_size_t effective_weight_byte_length = 0;
  iree_device_size_t strength_byte_length = 0;
  if (!iree_device_size_checked_mul(maximum_segment_rank,
                                    source_target->input_size,
                                    &down_source_element_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LoRA bake source Down size overflows");
  }
  for (iree_host_size_t i = 0; i < source_target->segment_count; ++i) {
    iree_device_size_t padded_rank = 0;
    iree_device_size_t segment_element_count = 0;
    if (!iree_device_size_checked_align(source_target->segments[i].rank,
                                        ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE,
                                        &padded_rank) ||
        !iree_device_size_checked_mul(padded_rank, source_target->input_size,
                                      &segment_element_count) ||
        !iree_device_size_checked_add(down_element_count, segment_element_count,
                                      &down_element_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "LoRA bake packed Down size overflows");
    }
  }
  if (!iree_device_size_checked_mul(down_source_element_count, sizeof(uint16_t),
                                    &down_source_byte_length) ||
      !iree_device_size_checked_mul(down_element_count, sizeof(uint16_t),
                                    &down_byte_length) ||
      !iree_device_size_checked_mul(output_row_count, maximum_segment_rank,
                                    &up_element_count) ||
      !iree_device_size_checked_mul(output_row_count, source_target->input_size,
                                    &effective_weight_element_count) ||
      !iree_device_size_checked_mul(up_element_count, sizeof(uint16_t),
                                    &up_byte_length) ||
      !iree_device_size_checked_mul(effective_weight_element_count,
                                    sizeof(float),
                                    &effective_weight_byte_length) ||
      !iree_device_size_checked_mul(adapter_count, sizeof(float),
                                    &strength_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LoRA bake target working-set size overflows");
  }

  iree_device_size_t total = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_pack_working_range(
      strength_byte_length, &out_working_set->strengths, &total));
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_pack_working_range(
      down_source_byte_length, &out_working_set->down_source, &total));
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_pack_working_range(
      down_byte_length, &out_working_set->down, &total));
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_pack_working_range(
      up_byte_length, &out_working_set->up, &total));
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_pack_working_range(
      effective_weight_byte_length, &out_working_set->effective_weight,
      &total));
  out_working_set->byte_length = total;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_lora_bake_plan_target(
    const id4_ideogram4_lora_bake_plan_create_options_t* options,
    iree_host_size_t patchable_slab_index,
    iree_device_size_t patchable_slab_byte_length,
    const id4_ideogram4_dit_lora_target_t* source_target,
    id4_ideogram4_lora_bake_target_t* out_target,
    iree_device_size_t* out_adapter_byte_length) {
  memset(out_target, 0, sizeof(*out_target));
  *out_adapter_byte_length = 0;
  if (source_target->input_size % ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE != 0 ||
      source_target->output_size % ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA bake target `%.*s` dimensions must be multiples of %u",
        (int)source_target->base_parameter_key.size,
        source_target->base_parameter_key.data,
        ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE);
  }

  const id4_pipeline_parameter_tensor_plan_t* weight_tensor = NULL;
  const id4_pipeline_program_parameter_op_t* weight_parameter = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_plan_find_parameter(
      options->base_plan, source_target->base_parameter_key, &weight_tensor,
      &weight_parameter));
  if (weight_tensor->layout.dtype != ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3 ||
      !id4_ideogram4_lora_bake_shape_matches(weight_tensor->layout.shape, 2,
                                             source_target->output_size,
                                             source_target->input_size) ||
      weight_parameter->encoding !=
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA bake target `%.*s` is not a compact FP8 RHS weight",
        (int)source_target->base_parameter_key.size,
        source_target->base_parameter_key.data);
  }
  id4_ideogram4_lora_bake_parameter_range_t weight_range;
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_validate_parameter_range(
      weight_tensor, patchable_slab_index, patchable_slab_byte_length,
      source_target->base_parameter_key, &weight_range));

  char scale_key_buffer[ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY];
  iree_string_view_t scale_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_program_format_parameter_scale_key(
      source_target->base_parameter_key, scale_key_buffer,
      IREE_ARRAYSIZE(scale_key_buffer), &scale_key));
  const id4_pipeline_parameter_tensor_plan_t* scale_tensor = NULL;
  const id4_pipeline_program_parameter_op_t* scale_parameter = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_plan_find_parameter(
      options->base_plan, scale_key, &scale_tensor, &scale_parameter));
  if (scale_tensor->layout.dtype != ID4_PIPELINE_TENSOR_DTYPE_F32 ||
      !id4_ideogram4_lora_bake_shape_matches(scale_tensor->layout.shape, 1,
                                             source_target->output_size, 0) ||
      scale_parameter->encoding !=
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA bake target `%.*s` has no direct F32 output-row scale",
        (int)source_target->base_parameter_key.size,
        source_target->base_parameter_key.data);
  }
  id4_ideogram4_lora_bake_parameter_range_t scale_range;
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_validate_parameter_range(
      scale_tensor, patchable_slab_index, patchable_slab_byte_length, scale_key,
      &scale_range));

  uint32_t maximum_segment_rank = 0;
  for (iree_host_size_t i = 0; i < source_target->segment_count; ++i) {
    maximum_segment_rank =
        iree_max(maximum_segment_rank, source_target->segments[i].rank);
  }
  if (maximum_segment_rank == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA bake target `%.*s` has no nonempty adapter segments",
        (int)source_target->base_parameter_key.size,
        source_target->base_parameter_key.data);
  }

  uint32_t selected_tile_count = 0;
  uint32_t minimum_tile_count = 1;
  uint32_t maximum_tile_count =
      source_target->output_size / ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE;
  while (minimum_tile_count <= maximum_tile_count) {
    const uint32_t candidate_tile_count =
        minimum_tile_count + (maximum_tile_count - minimum_tile_count) / 2;
    const uint32_t candidate_row_count =
        candidate_tile_count * ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE;
    id4_ideogram4_lora_bake_working_set_t candidate_working_set;
    IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_calculate_working_set(
        id4_ideogram4_lora_topology_adapter_count(options->topology),
        source_target, maximum_segment_rank, candidate_row_count,
        &candidate_working_set));
    if (candidate_working_set.byte_length <=
        options->working_set_byte_capacity) {
      selected_tile_count = candidate_tile_count;
      minimum_tile_count = candidate_tile_count + 1;
    } else {
      maximum_tile_count = candidate_tile_count - 1;
    }
  }
  if (selected_tile_count == 0) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LoRA bake target `%.*s` needs more than %" PRIu64
                            " working bytes for one %u-row tile",
                            (int)source_target->base_parameter_key.size,
                            source_target->base_parameter_key.data,
                            options->working_set_byte_capacity,
                            ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE);
  }
  const uint32_t selected_row_count =
      selected_tile_count * ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE;
  id4_ideogram4_lora_bake_working_set_t working_set;
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_calculate_working_set(
      id4_ideogram4_lora_topology_adapter_count(options->topology),
      source_target, maximum_segment_rank, selected_row_count, &working_set));

  iree_device_size_t full_down_element_count = 0;
  iree_device_size_t full_up_element_count = 0;
  iree_device_size_t full_down_byte_length = 0;
  iree_device_size_t full_up_byte_length = 0;
  if (!iree_device_size_checked_mul(source_target->input_size,
                                    source_target->total_rank,
                                    &full_down_element_count) ||
      !iree_device_size_checked_mul(full_down_element_count, sizeof(uint16_t),
                                    &full_down_byte_length) ||
      !iree_device_size_checked_mul(source_target->output_size,
                                    source_target->total_rank,
                                    &full_up_element_count) ||
      !iree_device_size_checked_mul(full_up_element_count, sizeof(uint16_t),
                                    &full_up_byte_length) ||
      !iree_device_size_checked_add(full_down_byte_length, full_up_byte_length,
                                    out_adapter_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LoRA bake adapter byte length overflows");
  }

  *out_target = (id4_ideogram4_lora_bake_target_t){
      .base_parameter_key = source_target->base_parameter_key,
      .weight_range = weight_range,
      .scale_range = scale_range,
      .input_size = source_target->input_size,
      .output_size = source_target->output_size,
      .total_rank = source_target->total_rank,
      .maximum_segment_rank = maximum_segment_rank,
      .output_rows_per_window = selected_row_count,
      .window_count = (source_target->output_size + selected_row_count - 1) /
                      selected_row_count,
      .working_set = working_set,
  };
  return iree_ok_status();
}

iree_status_t id4_ideogram4_lora_bake_plan_create(
    const id4_ideogram4_lora_bake_plan_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_ideogram4_lora_bake_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_plan_validate_options(options));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_lora_bake_plan_validate_base_program(options->base_plan));

  iree_host_size_t patchable_slab_index = IREE_HOST_SIZE_MAX;
  const id4_pipeline_parameter_slab_plan_t* patchable_slab = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_plan_find_patchable_slab(
      options->base_plan, &patchable_slab_index, &patchable_slab));

  const iree_host_size_t target_count =
      id4_ideogram4_lora_topology_target_count(options->topology);
  iree_host_size_t targets_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(id4_ideogram4_lora_bake_plan_t), &total_size,
      IREE_STRUCT_FIELD(target_count, id4_ideogram4_lora_bake_target_t,
                        &targets_offset)));

  id4_ideogram4_lora_bake_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&plan));
  memset(plan, 0, total_size);
  iree_atomic_ref_count_init(&plan->ref_count);
  plan->host_allocator = host_allocator;
  plan->base_plan = options->base_plan;
  id4_pipeline_plan_retain((id4_pipeline_plan_t*)plan->base_plan);
  plan->topology = options->topology;
  id4_ideogram4_lora_topology_retain(plan->topology);
  plan->patchable_slab_index = patchable_slab_index;
  plan->patchable_slab_byte_length = patchable_slab->byte_length;
  plan->target_count = target_count;
  plan->targets =
      (id4_ideogram4_lora_bake_target_t*)((uint8_t*)plan + targets_offset);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < target_count && iree_status_is_ok(status);
       ++i) {
    const id4_ideogram4_dit_lora_target_t* source_target =
        id4_ideogram4_lora_topology_target_at(options->topology, i);
    iree_device_size_t adapter_byte_length = 0;
    status = id4_ideogram4_lora_bake_plan_target(
        options, patchable_slab_index, patchable_slab->byte_length,
        source_target, &plan->targets[i], &adapter_byte_length);
    if (iree_status_is_ok(status) &&
        !iree_device_size_checked_add(plan->adapter_byte_length,
                                      adapter_byte_length,
                                      &plan->adapter_byte_length)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "LoRA bake adapter byte total overflows");
    }
    if (iree_status_is_ok(status)) {
      plan->working_set_high_water_mark =
          iree_max(plan->working_set_high_water_mark,
                   plan->targets[i].working_set.byte_length);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_plan = plan;
  } else {
    id4_ideogram4_lora_bake_plan_release(plan);
  }
  return status;
}

void id4_ideogram4_lora_bake_plan_retain(id4_ideogram4_lora_bake_plan_t* plan) {
  if (!plan) return;
  iree_atomic_ref_count_inc(&plan->ref_count);
}

void id4_ideogram4_lora_bake_plan_release(
    id4_ideogram4_lora_bake_plan_t* plan) {
  if (!plan) return;
  if (iree_atomic_ref_count_dec(&plan->ref_count) != 1) return;
  iree_allocator_t host_allocator = plan->host_allocator;
  id4_ideogram4_lora_topology_release(plan->topology);
  id4_pipeline_plan_release((id4_pipeline_plan_t*)plan->base_plan);
  iree_allocator_free(host_allocator, plan);
}

const id4_pipeline_plan_t* id4_ideogram4_lora_bake_plan_base_plan(
    const id4_ideogram4_lora_bake_plan_t* plan) {
  return plan ? plan->base_plan : NULL;
}

id4_ideogram4_lora_topology_t* id4_ideogram4_lora_bake_plan_topology(
    const id4_ideogram4_lora_bake_plan_t* plan) {
  return plan ? plan->topology : NULL;
}

iree_host_size_t id4_ideogram4_lora_bake_plan_patchable_slab_index(
    const id4_ideogram4_lora_bake_plan_t* plan) {
  return plan ? plan->patchable_slab_index : IREE_HOST_SIZE_MAX;
}

iree_device_size_t id4_ideogram4_lora_bake_plan_patchable_slab_byte_length(
    const id4_ideogram4_lora_bake_plan_t* plan) {
  return plan ? plan->patchable_slab_byte_length : 0;
}

iree_device_size_t id4_ideogram4_lora_bake_plan_working_set_high_water_mark(
    const id4_ideogram4_lora_bake_plan_t* plan) {
  return plan ? plan->working_set_high_water_mark : 0;
}

iree_device_size_t id4_ideogram4_lora_bake_plan_adapter_byte_length(
    const id4_ideogram4_lora_bake_plan_t* plan) {
  return plan ? plan->adapter_byte_length : 0;
}

iree_host_size_t id4_ideogram4_lora_bake_plan_target_count(
    const id4_ideogram4_lora_bake_plan_t* plan) {
  return plan ? plan->target_count : 0;
}

const id4_ideogram4_lora_bake_target_t* id4_ideogram4_lora_bake_plan_target_at(
    const id4_ideogram4_lora_bake_plan_t* plan, iree_host_size_t index) {
  if (!plan || index >= plan->target_count) return NULL;
  return &plan->targets[index];
}
