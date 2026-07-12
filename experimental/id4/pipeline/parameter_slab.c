// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_slab.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

enum {
  ID4_PIPELINE_PARAMETER_ENCODER_CONFIG_VALUE_CAPACITY = 24,
  // A staging chunk owns one logical target encode. The staging window still
  // pipelines chunks through reusable slots, while each provider gather,
  // command buffer, and readiness edge remains parameter-local.
  ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_STEP_CAPACITY = 1,
  ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT = 2,
  ID4_PIPELINE_PARAMETER_ENCODER_MAX_SOURCE_COUNT = 2,
  ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY =
      ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_STEP_CAPACITY *
      ID4_PIPELINE_PARAMETER_ENCODER_MAX_SOURCE_COUNT,
  ID4_PIPELINE_PARAMETER_ENCODER_WAVE_CHUNK_CAPACITY =
      ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT,
  ID4_PIPELINE_PARAMETER_ENCODER_WAVE_SOURCE_CAPACITY =
      ID4_PIPELINE_PARAMETER_ENCODER_WAVE_CHUNK_CAPACITY *
      ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY,
  ID4_PIPELINE_PARAMETER_ENCODER_CLEANUP_WAIT_CAPACITY =
      ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT *
      ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY,
  ID4_PIPELINE_PARAMETER_ENCODER_LINEAR_CONFIG_COUNT = 2,
};

typedef uint32_t id4_pipeline_parameter_slab_prepare_flags_t;

enum {
  ID4_PIPELINE_PARAMETER_SLAB_PREPARE_FLAG_ALLOCATE_RESIDENT_BUFFERS = 1u << 0,
};

static iree_status_t id4_pipeline_parameter_add_device_size(
    iree_device_size_t value, iree_device_size_t* inout_total,
    const char* name) {
  iree_device_size_t result = 0;
  if (!iree_device_size_checked_add(*inout_total, value, &result)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter %s byte length overflows", name);
  }
  *inout_total = result;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_add_host_size(
    iree_host_size_t value, iree_host_size_t* inout_total, const char* name) {
  iree_host_size_t result = 0;
  if (!iree_host_size_checked_add(*inout_total, value, &result)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter %s count overflows", name);
  }
  *inout_total = result;
  return iree_ok_status();
}

static iree_string_view_t id4_pipeline_parameter_load_group_kind_name(
    id4_pipeline_parameter_load_group_kind_t kind) {
  switch (kind) {
    case ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_GATHER:
      return IREE_SV("gather");
    case ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_ENCODE:
      return IREE_SV("encode");
    default:
      return IREE_SV("unknown");
  }
}

struct id4_pipeline_parameter_slab_set_t {
  // Reference count for shared slab set ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for slab set storage.
  iree_allocator_t host_allocator;
  // Number of loaded slab buffers.
  iree_host_size_t count;
  // Loaded slab buffers retained by this set.
  iree_hal_buffer_t** buffers;
  // Number of copied slab load descriptors.
  iree_host_size_t load_count;
  // Copied slab load descriptors used by deferred group submissions.
  id4_pipeline_parameter_slab_load_t* loads;
  // Slab plans owned by copied load descriptors.
  id4_pipeline_parameter_slab_plan_t* slab_plans;
  // Request tables owned by copied load descriptors.
  id4_pipeline_parameter_request_table_t* request_tables;
  // Number of retained parameter load readiness groups.
  iree_host_size_t load_group_count;
  // Retained parameter load group descriptors.
  id4_pipeline_parameter_load_group_t* load_groups;
  // Number of copied parameter load-step descriptors.
  iree_host_size_t load_step_count;
  // Copied parameter load-step descriptors used for failure diagnostics.
  id4_pipeline_parameter_load_step_t* load_steps;
  // Readiness semaphores retained in parameter load group order.
  iree_hal_semaphore_t** load_group_semaphores;
  // Per-load-group submission state owned by this set.
  bool* load_group_submitted;
  // Provider retained for deferred direct and encoded source gathers.
  iree_io_parameter_provider_t* provider;
  // Kernel library retained for deferred encoder JITs.
  id4_pipeline_kernel_library_t* kernel_library;
  // Loom kernel cache retained for deferred encoder JITs.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // HAL executable cache retained for deferred encoder JITs.
  iree_hal_executable_cache_t* executable_cache;
  // HAL command-buffer mode used by deferred encoder dispatches.
  iree_hal_command_buffer_mode_t command_buffer_mode;
  // HAL memory type used for deferred encoder source staging allocations.
  iree_hal_memory_type_t encoder_staging_memory_type;
  // Maximum source bytes staged in one deferred encoder chunk.
  iree_device_size_t encoder_staging_chunk_byte_capacity;
  // Diagnostic artifact classes requested for deferred encoder JITs.
  id4_pipeline_kernel_diagnostic_artifact_flags_t diagnostic_artifact_flags;
  // Number of retained prepare wait edges.
  iree_host_size_t wait_count;
  // Retained prepare wait semaphores used by deferred load submissions.
  iree_hal_semaphore_t** wait_semaphores;
  // Payload values paired with retained prepare wait semaphores.
  uint64_t* wait_payload_values;
  // Last submitted encoded load group readiness semaphore.
  iree_hal_semaphore_t* encode_tail_semaphore;
  // Payload value signaled by the last submitted encoded load group.
  uint64_t encode_tail_payload_value;
};

static iree_status_t id4_pipeline_parameter_string_clone(
    iree_string_view_t source, iree_allocator_t host_allocator,
    iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, source.size, (void**)&storage));
  memcpy(storage, source.data, source.size);
  *out_string = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static void id4_pipeline_parameter_string_release(
    iree_string_view_t string, iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, (void*)string.data);
}

static void id4_pipeline_parameter_slab_plan_deinitialize(
    id4_pipeline_parameter_slab_plan_t* slab, iree_allocator_t host_allocator) {
  (void)host_allocator;
  memset(slab, 0, sizeof(*slab));
}

static iree_status_t id4_pipeline_parameter_slab_plan_clone(
    const id4_pipeline_parameter_slab_plan_t* source,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_plan_t* target) {
  memset(target, 0, sizeof(*target));
  if (!source) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab load is missing its slab plan");
  }
  (void)host_allocator;
  *target = *source;
  return iree_ok_status();
}

static void id4_pipeline_parameter_request_table_deinitialize(
    id4_pipeline_parameter_request_table_t* table,
    iree_allocator_t host_allocator) {
  id4_pipeline_parameter_request_t* requests =
      (id4_pipeline_parameter_request_t*)table->values;
  for (iree_host_size_t i = 0; i < table->count; ++i) {
    id4_pipeline_parameter_string_release(requests[i].key, host_allocator);
  }
  iree_allocator_free(host_allocator, requests);
  memset(table, 0, sizeof(*table));
}

static iree_status_t id4_pipeline_parameter_request_table_clone(
    const id4_pipeline_parameter_request_table_t* source,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_request_table_t* target) {
  memset(target, 0, sizeof(*target));
  if (!source) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab load is missing its request table");
  }
  target->count = source->count;
  iree_status_t status = iree_ok_status();
  id4_pipeline_parameter_request_t* requests = NULL;
  if (target->count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, target->count, sizeof(requests[0]), (void**)&requests);
    if (iree_status_is_ok(status)) {
      memset(requests, 0, target->count * sizeof(requests[0]));
    }
  }
  for (iree_host_size_t i = 0; i < target->count && iree_status_is_ok(status);
       ++i) {
    requests[i].span = source->values[i].span;
    status = id4_pipeline_parameter_string_clone(
        source->values[i].key, host_allocator, &requests[i].key);
  }
  if (iree_status_is_ok(status)) {
    target->values = requests;
  } else {
    if (requests) {
      for (iree_host_size_t i = 0; i < target->count; ++i) {
        id4_pipeline_parameter_string_release(requests[i].key, host_allocator);
      }
    }
    iree_allocator_free(host_allocator, requests);
    memset(target, 0, sizeof(*target));
  }
  return status;
}

iree_status_t id4_pipeline_parameter_slab_pack_span(
    iree_device_size_t byte_length, iree_device_size_t alignment,
    iree_device_size_t* io_slab_byte_length,
    iree_io_parameter_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(io_slab_byte_length);
  IREE_ASSERT_ARGUMENT(out_span);
  if (byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter span byte length must be nonzero");
  }
  if (alignment != 0 && !iree_device_size_is_power_of_two(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter span alignment must be a power of two");
  }
  const iree_device_size_t effective_alignment = alignment == 0 ? 1 : alignment;
  iree_device_size_t buffer_offset = 0;
  if (!iree_device_size_checked_align(*io_slab_byte_length, effective_alignment,
                                      &buffer_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter span alignment overflow");
  }
  iree_device_size_t slab_byte_length = 0;
  if (!iree_device_size_checked_add(buffer_offset, byte_length,
                                    &slab_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter slab byte length overflow");
  }
  *io_slab_byte_length = slab_byte_length;
  *out_span = id4_pipeline_parameter_span(/*parameter_offset=*/0, buffer_offset,
                                          byte_length);
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_slab_validate(
    const id4_pipeline_parameter_slab_plan_t* slab,
    iree_host_size_t placement_count) {
  IREE_ASSERT_ARGUMENT(slab);
  if (slab->placement_id >= placement_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter slab placement %u outside placement count %" PRIhsz,
        slab->placement_id, placement_count);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_request_table_validate(
    const id4_pipeline_parameter_slab_plan_t* slab,
    const id4_pipeline_parameter_request_table_t* request_table) {
  if (!request_table) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter request table is required");
  }
  if (request_table->count != 0 && !request_table->values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter request table values are required");
  }
  for (iree_host_size_t i = 0; i < request_table->count; ++i) {
    const id4_pipeline_parameter_request_t* request = &request_table->values[i];
    if (iree_string_view_is_empty(request->key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter request %" PRIhsz " has no key", i);
    }
    if (request->span.length == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter request %" PRIhsz " has zero byte length", i);
    }
    if (request->span.buffer_offset > slab->byte_length ||
        request->span.length >
            slab->byte_length - request->span.buffer_offset) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "parameter request %" PRIhsz " exceeds slab byte length", i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_load_step_validate_header(
    const id4_pipeline_parameter_load_step_t* step) {
  IREE_ASSERT_ARGUMENT(step);
  if (iree_string_view_is_empty(step->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter load step has no name");
  }
  if (step->request_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter load step '%.*s' request count must be nonzero",
        (int)step->name.size, step->name.data);
  }
  switch (step->kind) {
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER:
      if (step->source_count != 0 || step->sources) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter gather load step '%.*s' must not have encoded sources",
            (int)step->name.size, step->name.data);
      }
      break;
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16:
      if (step->request_indices) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter encode load step '%.*s' must not have request indices",
            (int)step->name.size, step->name.data);
      }
      if (step->request_count != 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter encode load step '%.*s' must target one request",
            (int)step->name.size, step->name.data);
      }
      if (step->source_count != 2 || !step->sources) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter encode load step '%.*s' must have weight and scale "
            "sources",
            (int)step->name.size, step->name.data);
      }
      break;
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_BF16_LINEAR_RHS_TILE:
      if (step->request_indices) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter BF16 RHS tile encode load step '%.*s' must not have "
            "request indices",
            (int)step->name.size, step->name.data);
      }
      if (step->request_count != 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter BF16 RHS tile encode load step '%.*s' must target one "
            "request",
            (int)step->name.size, step->name.data);
      }
      if (step->source_count != 1 || !step->sources) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter BF16 RHS tile encode load step '%.*s' must have one "
            "source",
            (int)step->name.size, step->name.data);
      }
      break;
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE:
      if (step->request_indices) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter FP8 RHS tile encode load step '%.*s' must not have "
            "request indices",
            (int)step->name.size, step->name.data);
      }
      if (step->request_count != 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter FP8 RHS tile encode load step '%.*s' must target one "
            "request",
            (int)step->name.size, step->name.data);
      }
      if (step->source_count != 2 || !step->sources) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter FP8 RHS tile encode load step '%.*s' must have weight "
            "and scale sources",
            (int)step->name.size, step->name.data);
      }
      break;
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_LINEAR_RHS_TILE:
      if (step->request_indices) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter FP8 RHS tile encode load step '%.*s' must not have "
            "request indices",
            (int)step->name.size, step->name.data);
      }
      if (step->request_count != 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter FP8 RHS tile encode load step '%.*s' must target one "
            "request",
            (int)step->name.size, step->name.data);
      }
      if (step->source_count != 1 || !step->sources) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter FP8 RHS tile encode load step '%.*s' must have one "
            "source",
            (int)step->name.size, step->name.data);
      }
      break;
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE:
      if (step->request_indices) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter block-scaled FP8 RHS tile encode load step '%.*s' must "
            "not have request indices",
            (int)step->name.size, step->name.data);
      }
      if (step->request_count != 1) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter block-scaled FP8 RHS tile encode load step '%.*s' must "
            "target one request",
            (int)step->name.size, step->name.data);
      }
      if (step->source_count != 2 || !step->sources) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameter block-scaled FP8 RHS tile encode load step '%.*s' must "
            "have weight and scale sources",
            (int)step->name.size, step->name.data);
      }
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter load step '%.*s' has unknown kind %u",
                              (int)step->name.size, step->name.data,
                              (uint32_t)step->kind);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_load_source_validate(
    const id4_pipeline_parameter_load_source_t* source,
    iree_string_view_t step_name, iree_host_size_t source_index) {
  if (iree_string_view_is_empty(source->key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter load step '%.*s' source %" PRIhsz
                            " key is required",
                            (int)step_name.size, step_name.data, source_index);
  }
  if (source->shape.rank > ID4_PIPELINE_TENSOR_MAX_RANK) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter load step '%.*s' source %" PRIhsz
                            " rank %u exceeds max %u",
                            (int)step_name.size, step_name.data, source_index,
                            source->shape.rank, ID4_PIPELINE_TENSOR_MAX_RANK);
  }
  uint64_t element_count = 1;
  for (uint32_t i = 0; i < source->shape.rank; ++i) {
    if (source->shape.dims[i] == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter load step '%.*s' source %" PRIhsz " dimension %u is zero",
          (int)step_name.size, step_name.data, source_index, i);
    }
    if (element_count > UINT64_MAX / source->shape.dims[i]) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter load step '%.*s' source %" PRIhsz
                              " element count overflows",
                              (int)step_name.size, step_name.data,
                              source_index);
    }
    element_count *= source->shape.dims[i];
  }
  const iree_device_size_t dtype_byte_length =
      id4_pipeline_tensor_dtype_byte_length(source->dtype);
  if (dtype_byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter load step '%.*s' source %" PRIhsz
                            " dtype %u is invalid",
                            (int)step_name.size, step_name.data, source_index,
                            (uint32_t)source->dtype);
  }
  if (element_count > IREE_DEVICE_SIZE_MAX / dtype_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter load step '%.*s' source %" PRIhsz
                            " byte length overflows",
                            (int)step_name.size, step_name.data, source_index);
  }
  const iree_device_size_t byte_length =
      (iree_device_size_t)element_count * dtype_byte_length;
  if (source->byte_length != byte_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter load step '%.*s' source %" PRIhsz " byte length %" PRIu64
        " does not match dense length %" PRIu64,
        (int)step_name.size, step_name.data, source_index,
        (uint64_t)source->byte_length, (uint64_t)byte_length);
  }
  return iree_ok_status();
}

static bool id4_pipeline_parameter_load_step_is_encode(
    const id4_pipeline_parameter_load_step_t* step) {
  switch (step->kind) {
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16:
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_BF16_LINEAR_RHS_TILE:
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE:
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_LINEAR_RHS_TILE:
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE:
      return true;
    default:
      return false;
  }
}

static iree_status_t
id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_step_validate(
    const id4_pipeline_parameter_load_step_t* step) {
  const id4_pipeline_parameter_load_source_t* weight_source = &step->sources[0];
  const id4_pipeline_parameter_load_source_t* scale_source = &step->sources[1];
  if (weight_source->dtype != ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter encode load step '%.*s' weight source must be f8e4m3",
        (int)step->name.size, step->name.data);
  }
  if (scale_source->dtype != ID4_PIPELINE_TENSOR_DTYPE_F32 ||
      scale_source->shape.rank != 1 || weight_source->shape.rank == 0 ||
      scale_source->shape.dims[0] != weight_source->shape.dims[0]) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter encode load step '%.*s' scale source must be f32[output]",
        (int)step->name.size, step->name.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_validate_linear_rhs_tile_shape(
    const id4_pipeline_parameter_load_step_t* step,
    const id4_pipeline_parameter_load_source_t* weight_source,
    iree_string_view_t encoding_name) {
  if (weight_source->shape.rank != 2 ||
      (weight_source->shape.dims[0] % 16) != 0 ||
      (weight_source->shape.dims[1] % 16) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter %.*s load step '%.*s' weight source must be rank 2 with "
        "both dimensions divisible by 16",
        (int)encoding_name.size, encoding_name.data, (int)step->name.size,
        step->name.data);
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_parameter_encode_bf16_linear_rhs_tile_step_validate(
    const id4_pipeline_parameter_load_step_t* step) {
  const id4_pipeline_parameter_load_source_t* weight_source = &step->sources[0];
  if (weight_source->dtype != ID4_PIPELINE_TENSOR_DTYPE_BF16) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter BF16 RHS tile encode load step '%.*s' weight source must "
        "be bf16",
        (int)step->name.size, step->name.data);
  }
  return id4_pipeline_parameter_validate_linear_rhs_tile_shape(
      step, weight_source, IREE_SV("BF16 RHS tile encode"));
}

static iree_status_t
id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_linear_rhs_tile_step_validate(
    const id4_pipeline_parameter_load_step_t* step) {
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_step_validate(
          step));
  return id4_pipeline_parameter_validate_linear_rhs_tile_shape(
      step, &step->sources[0], IREE_SV("FP8 RHS tile encode"));
}

static iree_status_t
id4_pipeline_parameter_encode_fp8_e4m3_block_scaled_to_bf16_linear_rhs_tile_step_validate(
    const id4_pipeline_parameter_load_step_t* step) {
  const id4_pipeline_parameter_load_source_t* weight_source = &step->sources[0];
  const id4_pipeline_parameter_load_source_t* scale_source = &step->sources[1];
  if (weight_source->dtype != ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter block-scaled FP8 RHS tile encode load step '%.*s' weight "
        "source must be f8e4m3",
        (int)step->name.size, step->name.data);
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_validate_linear_rhs_tile_shape(
      step, weight_source, IREE_SV("block-scaled FP8 RHS tile encode")));
  if ((weight_source->shape.dims[0] % 128) != 0 ||
      (weight_source->shape.dims[1] % 128) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter block-scaled FP8 RHS tile encode load step '%.*s' weight "
        "source dimensions must both be divisible by 128",
        (int)step->name.size, step->name.data);
  }
  const uint64_t scale_output_count = weight_source->shape.dims[0] / 128;
  const uint64_t scale_input_count = weight_source->shape.dims[1] / 128;
  if (scale_source->dtype != ID4_PIPELINE_TENSOR_DTYPE_F32 ||
      scale_source->shape.rank != 2 ||
      scale_source->shape.dims[0] != scale_output_count ||
      scale_source->shape.dims[1] != scale_input_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter block-scaled FP8 RHS tile encode load step '%.*s' scale "
        "source must be f32[output/128, input/128]",
        (int)step->name.size, step->name.data);
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_parameter_encode_fp8_e4m3_linear_rhs_tile_step_validate(
    const id4_pipeline_parameter_load_step_t* step) {
  const id4_pipeline_parameter_load_source_t* weight_source = &step->sources[0];
  if (weight_source->dtype != ID4_PIPELINE_TENSOR_DTYPE_F8_E4M3) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter FP8 RHS tile encode load step '%.*s' weight source must be "
        "f8e4m3",
        (int)step->name.size, step->name.data);
  }
  return id4_pipeline_parameter_validate_linear_rhs_tile_shape(
      step, weight_source, IREE_SV("FP8 RHS tile encode"));
}

static iree_status_t id4_pipeline_parameter_load_step_validate_sources(
    const id4_pipeline_parameter_load_step_t* step) {
  for (iree_host_size_t i = 0; i < step->source_count; ++i) {
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_source_validate(
        &step->sources[i], step->name, i));
  }
  switch (step->kind) {
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER:
      return iree_ok_status();
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16:
      return id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_step_validate(
          step);
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_BF16_LINEAR_RHS_TILE:
      return id4_pipeline_parameter_encode_bf16_linear_rhs_tile_step_validate(
          step);
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE:
      return id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_linear_rhs_tile_step_validate(
          step);
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_LINEAR_RHS_TILE:
      return id4_pipeline_parameter_encode_fp8_e4m3_linear_rhs_tile_step_validate(
          step);
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE:
      return id4_pipeline_parameter_encode_fp8_e4m3_block_scaled_to_bf16_linear_rhs_tile_step_validate(
          step);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter load step '%.*s' has unknown kind %u",
                              (int)step->name.size, step->name.data,
                              (uint32_t)step->kind);
  }
}

static iree_status_t id4_pipeline_parameter_load_step_validate_request_range(
    const id4_pipeline_parameter_load_step_t* step,
    iree_host_size_t target_request_count) {
  if (step->request_indices) {
    for (iree_host_size_t i = 0; i < step->request_count; ++i) {
      if (step->request_indices[i] >= target_request_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "parameter load step '%.*s' request index %" PRIhsz
            " exceeds target slab request count %" PRIhsz,
            (int)step->name.size, step->name.data, step->request_indices[i],
            target_request_count);
      }
    }
    return iree_ok_status();
  }
  if (step->request_offset > target_request_count ||
      step->request_count > target_request_count - step->request_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter load step '%.*s' request range [%" PRIhsz ", %" PRIhsz
        ") exceeds target slab request count %" PRIhsz,
        (int)step->name.size, step->name.data, step->request_offset,
        step->request_offset + step->request_count, target_request_count);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_load_step_validate_target_request(
    const id4_pipeline_parameter_load_step_t* step,
    const id4_pipeline_parameter_request_table_t* request_table) {
  if (!id4_pipeline_parameter_load_step_is_encode(step))
    return iree_ok_status();
  const id4_pipeline_parameter_request_t* request =
      &request_table->values[step->request_offset];
  const id4_pipeline_parameter_load_source_t* weight_source = &step->sources[0];
  if (weight_source->shape.rank != 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter encode load step '%.*s' weight source must be rank 2",
        (int)step->name.size, step->name.data);
  }
  iree_device_size_t expected_target_byte_length = 0;
  switch (step->kind) {
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16:
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE:
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE:
      if (weight_source->byte_length >
          IREE_DEVICE_SIZE_MAX / sizeof(uint16_t)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "parameter encode load step '%.*s' target byte length overflows",
            (int)step->name.size, step->name.data);
      }
      expected_target_byte_length =
          weight_source->byte_length * sizeof(uint16_t);
      break;
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_BF16_LINEAR_RHS_TILE:
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_LINEAR_RHS_TILE:
      expected_target_byte_length = weight_source->byte_length;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter load step '%.*s' has unknown kind %u",
                              (int)step->name.size, step->name.data,
                              (uint32_t)step->kind);
  }
  if (request->span.parameter_offset != 0 ||
      request->span.length != expected_target_byte_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter encode load step '%.*s' target request must cover "
        "execution storage",
        (int)step->name.size, step->name.data);
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_load_step_validate(
    const id4_pipeline_parameter_load_step_t* step, iree_host_size_t slab_count,
    const id4_pipeline_parameter_slab_plan_t* slabs,
    const id4_pipeline_parameter_request_table_t* request_tables) {
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_step_validate_header(step));
  if (step->target_slab_index >= slab_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter load step '%.*s' target slab index %" PRIhsz
        " outside slab count %" PRIhsz,
        (int)step->name.size, step->name.data, step->target_slab_index,
        slab_count);
  }
  if (!slabs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter load step slab array is required");
  }
  const id4_pipeline_parameter_slab_plan_t* slab =
      &slabs[step->target_slab_index];
  if (!request_tables) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter load step request table array is required");
  }
  const id4_pipeline_parameter_request_table_t* request_table =
      &request_tables[step->target_slab_index];
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_request_table_validate(slab, request_table));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_step_validate_sources(step));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_step_validate_request_range(
      step, request_table->count));
  return id4_pipeline_parameter_load_step_validate_target_request(
      step, request_table);
}

iree_status_t id4_pipeline_parameter_slab_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(user_data);
  IREE_ASSERT_ARGUMENT(out_key);
  IREE_ASSERT_ARGUMENT(out_span);
  id4_pipeline_parameter_slab_enumerator_state_t* state =
      (id4_pipeline_parameter_slab_enumerator_state_t*)user_data;
  if (!state->request_table || i >= state->request_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter request index %" PRIhsz
                            " is outside enumerated request count %" PRIhsz,
                            i, state->request_count);
  }
  const iree_host_size_t request_index = state->request_indices
                                             ? state->request_indices[i]
                                             : state->request_offset + i;
  if (request_index >= state->request_table->count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter request index %" PRIhsz
                            " is outside slab request count %" PRIhsz,
                            request_index, state->request_table->count);
  }
  const id4_pipeline_parameter_request_t* request =
      &state->request_table->values[request_index];
  *out_key = request->key;
  *out_span = request->span;
  return iree_ok_status();
}

iree_io_parameter_enumerator_t id4_pipeline_parameter_slab_enumerator(
    id4_pipeline_parameter_slab_enumerator_state_t* state) {
  iree_io_parameter_enumerator_t enumerator = {
      // Callback used by IREE parameter provider gather/load APIs.
      .fn = id4_pipeline_parameter_slab_enumerate,
      // Enumerator state holding the planned slab request array.
      .user_data = state,
  };
  return enumerator;
}

static void id4_pipeline_parameter_slab_set_destroy(
    id4_pipeline_parameter_slab_set_t* slab_set) {
  iree_allocator_t host_allocator = slab_set->host_allocator;
  if (slab_set->buffers) {
    for (iree_host_size_t i = 0; i < slab_set->count; ++i) {
      iree_hal_buffer_release(slab_set->buffers[i]);
    }
  }
  if (slab_set->loads) {
    for (iree_host_size_t i = 0; i < slab_set->load_count; ++i) {
      iree_hal_device_release(slab_set->loads[i].device);
    }
  }
  if (slab_set->slab_plans) {
    for (iree_host_size_t i = 0; i < slab_set->load_count; ++i) {
      id4_pipeline_parameter_slab_plan_deinitialize(&slab_set->slab_plans[i],
                                                    host_allocator);
    }
  }
  if (slab_set->request_tables) {
    for (iree_host_size_t i = 0; i < slab_set->load_count; ++i) {
      id4_pipeline_parameter_request_table_deinitialize(
          &slab_set->request_tables[i], host_allocator);
    }
  }
  if (slab_set->load_group_semaphores) {
    for (iree_host_size_t i = 0; i < slab_set->load_group_count; ++i) {
      iree_hal_semaphore_release(slab_set->load_group_semaphores[i]);
    }
  }
  for (iree_host_size_t i = 0; i < slab_set->wait_count; ++i) {
    iree_hal_semaphore_release(slab_set->wait_semaphores[i]);
  }
  iree_hal_executable_cache_release(slab_set->executable_cache);
  id4_pipeline_kernel_cache_release(slab_set->kernel_cache);
  id4_pipeline_kernel_library_release(slab_set->kernel_library);
  iree_io_parameter_provider_release(slab_set->provider);
  iree_allocator_free(host_allocator, slab_set->wait_payload_values);
  iree_allocator_free(host_allocator, slab_set->wait_semaphores);
  iree_allocator_free(host_allocator, slab_set->load_group_submitted);
  iree_allocator_free(host_allocator, slab_set->load_group_semaphores);
  iree_allocator_free(host_allocator, slab_set->load_steps);
  iree_allocator_free(host_allocator, slab_set->load_groups);
  iree_allocator_free(host_allocator, slab_set->request_tables);
  iree_allocator_free(host_allocator, slab_set->slab_plans);
  iree_allocator_free(host_allocator, slab_set->loads);
  iree_allocator_free(host_allocator, slab_set->buffers);
  iree_allocator_free(host_allocator, slab_set);
}

static iree_status_t id4_pipeline_parameter_slab_set_create_empty(
    iree_host_size_t count, iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set) {
  IREE_ASSERT_ARGUMENT(out_slab_set);
  *out_slab_set = NULL;

  id4_pipeline_parameter_slab_set_t* slab_set = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*slab_set), (void**)&slab_set);
  if (iree_status_is_ok(status)) {
    memset(slab_set, 0, sizeof(*slab_set));
    iree_atomic_ref_count_init(&slab_set->ref_count);
    slab_set->host_allocator = host_allocator;
    slab_set->count = count;
  }
  if (iree_status_is_ok(status) && count != 0) {
    status = iree_allocator_malloc_array(host_allocator, count,
                                         sizeof(slab_set->buffers[0]),
                                         (void**)&slab_set->buffers);
    if (iree_status_is_ok(status)) {
      memset(slab_set->buffers, 0, count * sizeof(slab_set->buffers[0]));
    }
  }
  if (iree_status_is_ok(status)) {
    *out_slab_set = slab_set;
  } else if (slab_set) {
    id4_pipeline_parameter_slab_set_destroy(slab_set);
  }
  return status;
}

static bool id4_pipeline_parameter_load_steps_require_encoder(
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps) {
  if (!load_steps) return false;
  for (iree_host_size_t i = 0; i < load_step_count; ++i) {
    if (id4_pipeline_parameter_load_step_is_encode(&load_steps[i])) {
      return true;
    }
  }
  return false;
}

static iree_status_t id4_pipeline_parameter_slab_set_copy_loads(
    id4_pipeline_parameter_slab_set_t* slab_set, iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads) {
  if (load_count == 0) return iree_ok_status();
  iree_status_t status = iree_allocator_malloc_array(
      slab_set->host_allocator, load_count, sizeof(slab_set->loads[0]),
      (void**)&slab_set->loads);
  if (!iree_status_is_ok(status)) return status;
  memset(slab_set->loads, 0, load_count * sizeof(slab_set->loads[0]));
  status = iree_allocator_malloc_array(slab_set->host_allocator, load_count,
                                       sizeof(slab_set->slab_plans[0]),
                                       (void**)&slab_set->slab_plans);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(slab_set->host_allocator, slab_set->loads);
    slab_set->loads = NULL;
    return status;
  }
  memset(slab_set->slab_plans, 0, load_count * sizeof(slab_set->slab_plans[0]));
  status = iree_allocator_malloc_array(slab_set->host_allocator, load_count,
                                       sizeof(slab_set->request_tables[0]),
                                       (void**)&slab_set->request_tables);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(slab_set->host_allocator, slab_set->slab_plans);
    iree_allocator_free(slab_set->host_allocator, slab_set->loads);
    slab_set->slab_plans = NULL;
    slab_set->loads = NULL;
    return status;
  }
  memset(slab_set->request_tables, 0,
         load_count * sizeof(slab_set->request_tables[0]));

  iree_host_size_t retained_device_count = 0;
  for (iree_host_size_t i = 0; i < load_count && iree_status_is_ok(status);
       ++i) {
    slab_set->loads[i] = loads[i];
    status = id4_pipeline_parameter_slab_plan_clone(
        loads[i].slab, slab_set->host_allocator, &slab_set->slab_plans[i]);
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_parameter_request_table_clone(
          loads[i].request_table, slab_set->host_allocator,
          &slab_set->request_tables[i]);
    }
    if (iree_status_is_ok(status)) {
      slab_set->loads[i].slab = &slab_set->slab_plans[i];
      slab_set->loads[i].request_table = &slab_set->request_tables[i];
      iree_hal_device_retain(slab_set->loads[i].device);
      ++retained_device_count;
    }
  }
  if (iree_status_is_ok(status)) {
    slab_set->load_count = load_count;
  } else {
    for (iree_host_size_t i = 0; i < retained_device_count; ++i) {
      iree_hal_device_release(slab_set->loads[i].device);
      slab_set->loads[i].device = NULL;
    }
    for (iree_host_size_t i = 0; i < load_count; ++i) {
      id4_pipeline_parameter_slab_plan_deinitialize(&slab_set->slab_plans[i],
                                                    slab_set->host_allocator);
      id4_pipeline_parameter_request_table_deinitialize(
          &slab_set->request_tables[i], slab_set->host_allocator);
    }
    iree_allocator_free(slab_set->host_allocator, slab_set->request_tables);
    iree_allocator_free(slab_set->host_allocator, slab_set->slab_plans);
    iree_allocator_free(slab_set->host_allocator, slab_set->loads);
    slab_set->slab_plans = NULL;
    slab_set->request_tables = NULL;
    slab_set->loads = NULL;
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_slab_set_copy_load_steps(
    id4_pipeline_parameter_slab_set_t* slab_set,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps) {
  if (load_step_count == 0) return iree_ok_status();
  iree_status_t status = iree_allocator_malloc_array(
      slab_set->host_allocator, load_step_count,
      sizeof(slab_set->load_steps[0]), (void**)&slab_set->load_steps);
  if (!iree_status_is_ok(status)) return status;
  memcpy(slab_set->load_steps, load_steps,
         load_step_count * sizeof(slab_set->load_steps[0]));
  slab_set->load_step_count = load_step_count;
  return iree_ok_status();
}

static void id4_pipeline_parameter_slab_set_capture_encoder_diagnostics(
    id4_pipeline_parameter_slab_set_t* slab_set,
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps) {
  if (!id4_pipeline_parameter_load_steps_require_encoder(load_step_count,
                                                         load_steps)) {
    return;
  }
  slab_set->encoder_staging_chunk_byte_capacity =
      options->encoder_staging_chunk_byte_capacity;
  slab_set->encoder_staging_memory_type = options->encoder_staging_memory_type;
}

static iree_status_t id4_pipeline_parameter_slab_set_copy_waits(
    id4_pipeline_parameter_slab_set_t* slab_set,
    iree_hal_semaphore_list_t wait_semaphore_list) {
  if (wait_semaphore_list.count == 0) return iree_ok_status();
  iree_status_t status = iree_allocator_malloc_array(
      slab_set->host_allocator, wait_semaphore_list.count,
      sizeof(slab_set->wait_semaphores[0]), (void**)&slab_set->wait_semaphores);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        slab_set->host_allocator, wait_semaphore_list.count,
        sizeof(slab_set->wait_payload_values[0]),
        (void**)&slab_set->wait_payload_values);
  }
  iree_host_size_t retained_count = 0;
  for (iree_host_size_t i = 0;
       i < wait_semaphore_list.count && iree_status_is_ok(status); ++i) {
    slab_set->wait_semaphores[i] = wait_semaphore_list.semaphores[i];
    iree_hal_semaphore_retain(slab_set->wait_semaphores[i]);
    slab_set->wait_payload_values[i] = wait_semaphore_list.payload_values[i];
    ++retained_count;
  }
  if (iree_status_is_ok(status)) {
    slab_set->wait_count = wait_semaphore_list.count;
  } else {
    for (iree_host_size_t i = 0; i < retained_count; ++i) {
      iree_hal_semaphore_release(slab_set->wait_semaphores[i]);
      slab_set->wait_semaphores[i] = NULL;
    }
    iree_allocator_free(slab_set->host_allocator,
                        slab_set->wait_payload_values);
    iree_allocator_free(slab_set->host_allocator, slab_set->wait_semaphores);
    slab_set->wait_payload_values = NULL;
    slab_set->wait_semaphores = NULL;
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_slab_set_retain_load_context(
    id4_pipeline_parameter_slab_set_t* slab_set,
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps) {
  slab_set->provider = options->provider;
  iree_io_parameter_provider_retain(slab_set->provider);
  if (id4_pipeline_parameter_load_steps_require_encoder(load_step_count,
                                                        load_steps)) {
    slab_set->kernel_library = options->kernel_library;
    id4_pipeline_kernel_library_retain(slab_set->kernel_library);
    slab_set->kernel_cache = options->kernel_cache;
    id4_pipeline_kernel_cache_retain(slab_set->kernel_cache);
    slab_set->executable_cache = options->executable_cache;
    iree_hal_executable_cache_retain(slab_set->executable_cache);
    slab_set->command_buffer_mode = options->command_buffer_mode;
    slab_set->encoder_staging_memory_type =
        options->encoder_staging_memory_type;
    slab_set->encoder_staging_chunk_byte_capacity =
        options->encoder_staging_chunk_byte_capacity;
    slab_set->diagnostic_artifact_flags = options->diagnostic_artifact_flags;
  }
  return id4_pipeline_parameter_slab_set_copy_waits(
      slab_set, options->wait_semaphore_list);
}

static iree_status_t id4_pipeline_parameter_slab_allocate_buffer(
    const id4_pipeline_parameter_slab_load_t* load,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(load);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  if (!load->slab) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab load has no slab plan");
  }
  if (!load->device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab load has no device");
  }
  return iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(load->device), load->slab->target_params,
      load->slab->byte_length, out_buffer);
}

static iree_status_t id4_pipeline_parameter_slab_validate_semaphore_list(
    iree_hal_semaphore_list_t semaphore_list, iree_string_view_t list_name) {
  if (semaphore_list.count == 0) return iree_ok_status();
  if (!semaphore_list.semaphores) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s semaphore array is required",
                            (int)list_name.size, list_name.data);
  }
  if (!semaphore_list.payload_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s payload value array is required",
                            (int)list_name.size, list_name.data);
  }
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    if (!semaphore_list.semaphores[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s semaphore %" PRIhsz " is NULL",
                              (int)list_name.size, list_name.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_slab_set_load_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_pipeline_parameter_slab_set_load_validate_options(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab load options are required");
  }
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_slab_set_load_validate_options_size(
          options->structure_size, sizeof(*options),
          IREE_SV("parameter slab load")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "parameter slab load extension structures are not supported");
  }
  if (!options->provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab load provider is required");
  }
  const bool requires_encoder =
      id4_pipeline_parameter_load_steps_require_encoder(load_step_count,
                                                        load_steps);
  if (requires_encoder && !options->kernel_library) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slab encoded loading requires a kernel library");
  }
  if (requires_encoder && !options->kernel_cache) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slab encoded loading requires a Loom kernel cache");
  }
  if (requires_encoder && !options->executable_cache) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slab encoded loading requires a HAL executable cache");
  }
  if (requires_encoder && options->encoder_staging_chunk_byte_capacity == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slab encoded loading requires a nonzero staging chunk byte "
        "capacity");
  }
  if (requires_encoder &&
      options->encoder_staging_memory_type == IREE_HAL_MEMORY_TYPE_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slab encoded loading requires a staging memory type");
  }
  if (requires_encoder &&
      !iree_all_bits_set(options->encoder_staging_memory_type,
                         IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slab encoded loading requires device-visible staging");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("parameter slab wait")));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_validate_semaphore_list(
      options->signal_semaphore_list, IREE_SV("parameter slab signal")));
  return id4_pipeline_diagnostics_validate_sink(options->diagnostics_sink,
                                                IREE_SV("parameter slab load"));
}

static iree_status_t id4_pipeline_parameter_load_step_validate_against_loads(
    const id4_pipeline_parameter_load_step_t* step, iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads) {
  IREE_ASSERT_ARGUMENT(loads);
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_step_validate_header(step));
  if (step->target_slab_index >= load_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter load step '%.*s' target slab index %" PRIhsz
        " outside load count %" PRIhsz,
        (int)step->name.size, step->name.data, step->target_slab_index,
        load_count);
  }
  const id4_pipeline_parameter_slab_load_t* load =
      &loads[step->target_slab_index];
  if (!load->slab) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter load step '%.*s' target slab has no plan",
        (int)step->name.size, step->name.data);
  }
  if (!load->request_table) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter load step '%.*s' target slab has no request table",
        (int)step->name.size, step->name.data);
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_step_validate_sources(step));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_step_validate_request_range(
      step, load->request_table->count));
  return id4_pipeline_parameter_load_step_validate_target_request(
      step, load->request_table);
}

static iree_string_view_t id4_pipeline_parameter_load_step_primary_scope(
    const id4_pipeline_parameter_load_step_t* step) {
  if (step->kind == ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER) {
    return step->source_scope;
  }
  if (step->source_count != 0 && step->sources) {
    return step->sources[0].source_scope;
  }
  return iree_string_view_empty();
}

static id4_pipeline_parameter_slab_diagnostic_t
id4_pipeline_parameter_slab_make_diagnostic(
    const id4_pipeline_parameter_slab_load_t* load, iree_string_view_t scope,
    iree_host_size_t request_index) {
  const id4_pipeline_parameter_slab_plan_t* slab = load->slab;
  const id4_pipeline_parameter_request_table_t* request_table =
      load->request_table;
  id4_pipeline_parameter_slab_diagnostic_t diagnostic = {
      // Plan-local slab index.
      .slab_index = load->slab_index,
      // Request index or IREE_HOST_SIZE_MAX for slab-level events.
      .request_index = request_index,
      // Parameter scope associated with this event.
      .scope = scope,
      // Parameter key populated below for request-level events.
      .parameter_key = iree_string_view_empty(),
      // Source parameter byte offset populated below for request events.
      .parameter_offset = 0,
      // Target slab byte offset populated below for request events.
      .buffer_offset = 0,
      // Byte length populated below for request events.
      .length = 0,
      // Plan-local placement identifier.
      .placement_id = slab->placement_id,
      // Device index within the plan device group.
      .device_index = load->device_index,
      // Queue affinity used by loading work.
      .queue_affinity = load->queue_affinity,
      // HAL memory type requested for the slab buffer.
      .memory_type = slab->target_params.type,
      // HAL memory access requested for the slab buffer.
      .memory_access = slab->target_params.access,
      // HAL buffer usage requested for the slab buffer.
      .buffer_usage = slab->target_params.usage,
      // Total slab byte length.
      .slab_byte_length = slab->byte_length,
      // Required slab base alignment.
      .slab_alignment = slab->alignment,
      // Number of requests in the slab.
      .request_count = request_table ? request_table->count : 0,
  };
  if (request_table && request_index < request_table->count) {
    const id4_pipeline_parameter_request_t* request =
        &request_table->values[request_index];
    diagnostic.parameter_key = request->key;
    diagnostic.parameter_offset = request->span.parameter_offset;
    diagnostic.buffer_offset = request->span.buffer_offset;
    diagnostic.length = request->span.length;
  }
  return diagnostic;
}

static iree_status_t id4_pipeline_parameter_slab_emit_diagnostic(
    const id4_pipeline_parameter_slab_load_t* load,
    iree_string_view_t stage_name, iree_string_view_t key,
    iree_string_view_t message, iree_string_view_t scope,
    iree_host_size_t request_index,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_pipeline_parameter_slab_diagnostic_t parameter_slab =
      id4_pipeline_parameter_slab_make_diagnostic(load, scope, request_index);
  id4_pipeline_diagnostic_event_t event = {
      // Event kind for parameter slab diagnostics.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PARAMETER_SLAB,
      // Stage name associated with the load.
      .stage_name = stage_name,
      // Stable parameter slab event key.
      .key = key,
      // Short event summary.
      .message = message,
      // Structured slab payload.
      .parameter_slab = &parameter_slab,
  };
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

typedef struct id4_pipeline_parameter_encode_chunk_source_t {
  // Source tensor gathered into staging storage.
  const id4_pipeline_parameter_load_source_t* source;
  // Staging buffer offset receiving the source tensor.
  iree_device_size_t buffer_offset;
} id4_pipeline_parameter_encode_chunk_source_t;

typedef struct id4_pipeline_parameter_encode_source_batch_t {
  // Provider source scope shared by every source in this batch.
  iree_string_view_t source_scope;
  // First source ordinal in the grouped chunk source table.
  iree_host_size_t source_offset;
  // Number of grouped chunk sources in this batch.
  iree_host_size_t source_count;
} id4_pipeline_parameter_encode_source_batch_t;

typedef struct id4_pipeline_parameter_load_source_enumerator_state_t {
  // Number of source tensors enumerated by this gather.
  iree_host_size_t source_count;
  // Source tensors plus staging offsets enumerated by this gather.
  const id4_pipeline_parameter_encode_chunk_source_t* sources;
} id4_pipeline_parameter_load_source_enumerator_state_t;

static iree_status_t id4_pipeline_parameter_load_source_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  id4_pipeline_parameter_load_source_enumerator_state_t* state =
      (id4_pipeline_parameter_load_source_enumerator_state_t*)user_data;
  if (i >= state->source_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter load source index %" PRIhsz " out of range", i);
  }
  const id4_pipeline_parameter_encode_chunk_source_t* chunk_source =
      &state->sources[i];
  *out_key = chunk_source->source->key;
  *out_span = id4_pipeline_parameter_span(
      /*parameter_offset=*/0, chunk_source->buffer_offset,
      chunk_source->source->byte_length);
  return iree_ok_status();
}

static iree_io_parameter_enumerator_t
id4_pipeline_parameter_load_source_enumerator(
    id4_pipeline_parameter_load_source_enumerator_state_t* state) {
  iree_io_parameter_enumerator_t enumerator = {
      // Callback used by IREE parameter provider gather APIs.
      .fn = id4_pipeline_parameter_load_source_enumerate,
      // Enumerator state holding source descriptors and staging offsets.
      .user_data = state,
  };
  return enumerator;
}

typedef struct id4_pipeline_parameter_encoder_config_t {
  // Number of config bindings used by this encoder specialization.
  iree_host_size_t count;
  // Fixed-capacity config binding storage.
  id4_pipeline_kernel_config_binding_t
      bindings[ID4_PIPELINE_PARAMETER_ENCODER_LINEAR_CONFIG_COUNT];
  // Fixed-capacity string storage backing binding values.
  char value_storage[ID4_PIPELINE_PARAMETER_ENCODER_LINEAR_CONFIG_COUNT]
                    [ID4_PIPELINE_PARAMETER_ENCODER_CONFIG_VALUE_CAPACITY];
} id4_pipeline_parameter_encoder_config_t;

static iree_status_t id4_pipeline_parameter_format_u64(
    uint64_t value, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_string) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_string);
  int length = snprintf(buffer, buffer_capacity, "%" PRIu64, value);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "failed to format parameter encoder config value");
  }
  *out_string = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_encoder_make_linear_config(
    const id4_pipeline_parameter_load_step_t* step,
    iree_string_view_t output_size_key, iree_string_view_t input_size_key,
    id4_pipeline_parameter_encoder_config_t* out_config) {
  memset(out_config, 0, sizeof(*out_config));
  const id4_pipeline_parameter_load_source_t* weight_source = &step->sources[0];
  iree_string_view_t output_size = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_format_u64(
      weight_source->shape.dims[0], out_config->value_storage[0],
      IREE_ARRAYSIZE(out_config->value_storage[0]), &output_size));
  out_config->bindings[0] =
      id4_pipeline_make_kernel_config_binding(output_size_key, output_size);
  iree_string_view_t input_size = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_format_u64(
      weight_source->shape.dims[1], out_config->value_storage[1],
      IREE_ARRAYSIZE(out_config->value_storage[1]), &input_size));
  out_config->bindings[1] =
      id4_pipeline_make_kernel_config_binding(input_size_key, input_size);
  out_config->count = ID4_PIPELINE_PARAMETER_ENCODER_LINEAR_CONFIG_COUNT;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_prepare_encoder(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* load,
    const id4_pipeline_parameter_load_step_t* step,
    id4_pipeline_kernel_executable_t** out_executable,
    iree_hal_executable_function_t* out_function) {
  *out_executable = NULL;
  *out_function = iree_hal_executable_function_invalid();

  iree_string_view_t module_path = iree_string_view_empty();
  iree_string_view_t function_name = iree_string_view_empty();
  iree_string_view_t output_size_key = iree_string_view_empty();
  iree_string_view_t input_size_key = iree_string_view_empty();
  switch (step->kind) {
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16:
      module_path = IREE_SV("parameter/fp8_e4m3_scaled_to_bf16");
      function_name = IREE_SV("id4_parameter_fp8_e4m3_scaled_to_bf16");
      output_size_key =
          IREE_SV("id4.parameter.fp8_e4m3_scaled_to_bf16.output_size");
      input_size_key =
          IREE_SV("id4.parameter.fp8_e4m3_scaled_to_bf16.input_size");
      break;
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_BF16_LINEAR_RHS_TILE:
      module_path = IREE_SV("parameter/bf16_linear_rhs_tile");
      function_name = IREE_SV("id4_parameter_bf16_linear_rhs_tile");
      output_size_key =
          IREE_SV("id4.parameter.bf16_linear_rhs_tile.output_size");
      input_size_key = IREE_SV("id4.parameter.bf16_linear_rhs_tile.input_size");
      break;
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE:
      module_path =
          IREE_SV("parameter/fp8_e4m3_scaled_to_bf16_linear_rhs_tile");
      function_name =
          IREE_SV("id4_parameter_fp8_e4m3_scaled_to_bf16_linear_rhs_tile");
      output_size_key = IREE_SV(
          "id4.parameter.fp8_e4m3_scaled_to_bf16_linear_rhs_tile.output_size");
      input_size_key = IREE_SV(
          "id4.parameter.fp8_e4m3_scaled_to_bf16_linear_rhs_tile.input_size");
      break;
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_LINEAR_RHS_TILE:
      module_path = IREE_SV("parameter/fp8_e4m3_linear_rhs_tile");
      function_name = IREE_SV("id4_parameter_fp8_e4m3_linear_rhs_tile");
      output_size_key =
          IREE_SV("id4.parameter.fp8_e4m3_linear_rhs_tile.output_size");
      input_size_key =
          IREE_SV("id4.parameter.fp8_e4m3_linear_rhs_tile.input_size");
      break;
    case ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE:
      module_path =
          IREE_SV("parameter/fp8_e4m3_block_scaled_to_bf16_linear_rhs_tile");
      function_name = IREE_SV(
          "id4_parameter_fp8_e4m3_block_scaled_to_bf16_linear_rhs_tile");
      output_size_key = IREE_SV(
          "id4.parameter.fp8_e4m3_block_scaled_to_bf16_linear_rhs_tile.output_"
          "size");
      input_size_key = IREE_SV(
          "id4.parameter.fp8_e4m3_block_scaled_to_bf16_linear_rhs_tile.input_"
          "size");
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter load step '%.*s' has unknown kind %u",
                              (int)step->name.size, step->name.data,
                              (uint32_t)step->kind);
  }

  const id4_pipeline_kernel_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_library_lookup(
      options->kernel_library, module_path, &module));
  id4_pipeline_parameter_encoder_config_t config;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_encoder_make_linear_config(
      step, output_size_key, input_size_key, &config));

  id4_pipeline_kernel_cache_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.executable_cache = options->executable_cache;
  prepare_options.queue_affinity = load->queue_affinity;
  prepare_options.caching_mode = IREE_HAL_EXECUTABLE_CACHING_MODE_NONE;
  prepare_options.source_identifier = module->source_identifier;
  prepare_options.source_contents = module->source_contents;
  prepare_options.module_path = module->module_path;
  prepare_options.function_name = function_name;
  prepare_options.config_binding_count = config.count;
  prepare_options.config_bindings = config.bindings;
  prepare_options.diagnostic_artifact_flags =
      options->diagnostic_artifact_flags;
  prepare_options.diagnostics_sink = options->diagnostics_sink;

  id4_pipeline_kernel_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_cache_prepare_executable(
      options->kernel_cache, &prepare_options, &executable));
  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  iree_status_t status = iree_hal_executable_lookup_function_by_name(
      id4_pipeline_kernel_executable_hal_executable(executable),
      prepare_options.function_name, &function);
  if (iree_status_is_ok(status)) {
    *out_executable = executable;
    *out_function = function;
    executable = NULL;
  }
  id4_pipeline_kernel_executable_release(executable);
  return status;
}

typedef struct id4_pipeline_parameter_prepared_encoder_t {
  // Load-step operation encoded by this specialization.
  id4_pipeline_parameter_load_step_kind_t kind;
  // Logical output row count used by the encoder config.
  uint64_t output_size;
  // Logical input column count used by the encoder config.
  uint64_t input_size;
  // Retained executable wrapper prepared through the Loom kernel cache.
  id4_pipeline_kernel_executable_t* executable;
  // HAL function resolved from executable.
  iree_hal_executable_function_t function;
} id4_pipeline_parameter_prepared_encoder_t;

static void id4_pipeline_parameter_prepared_encoder_initialize(
    id4_pipeline_parameter_prepared_encoder_t* prepared_encoder) {
  memset(prepared_encoder, 0, sizeof(*prepared_encoder));
  prepared_encoder->function = iree_hal_executable_function_invalid();
}

static void id4_pipeline_parameter_prepared_encoder_deinitialize(
    id4_pipeline_parameter_prepared_encoder_t* prepared_encoder) {
  id4_pipeline_kernel_executable_release(prepared_encoder->executable);
  id4_pipeline_parameter_prepared_encoder_initialize(prepared_encoder);
}

static uint64_t id4_pipeline_parameter_encode_step_output_size(
    const id4_pipeline_parameter_load_step_t* step) {
  return step->sources[0].shape.dims[0];
}

static uint64_t id4_pipeline_parameter_encode_step_input_size(
    const id4_pipeline_parameter_load_step_t* step) {
  return step->sources[0].shape.dims[1];
}

static bool id4_pipeline_parameter_prepared_encoder_matches(
    const id4_pipeline_parameter_prepared_encoder_t* prepared_encoder,
    const id4_pipeline_parameter_load_step_t* step) {
  return prepared_encoder->kind == step->kind &&
         prepared_encoder->output_size ==
             id4_pipeline_parameter_encode_step_output_size(step) &&
         prepared_encoder->input_size ==
             id4_pipeline_parameter_encode_step_input_size(step);
}

static iree_status_t id4_pipeline_parameter_encode_run_resolve_encoder(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* load,
    const id4_pipeline_parameter_load_step_t* step,
    id4_pipeline_parameter_prepared_encoder_t* prepared_encoders,
    iree_host_size_t prepared_encoder_capacity,
    iree_host_size_t* inout_prepared_encoder_count,
    id4_pipeline_parameter_prepared_encoder_t** out_prepared_encoder) {
  *out_prepared_encoder = NULL;
  for (iree_host_size_t i = 0; i < *inout_prepared_encoder_count; ++i) {
    if (id4_pipeline_parameter_prepared_encoder_matches(&prepared_encoders[i],
                                                        step)) {
      *out_prepared_encoder = &prepared_encoders[i];
      return iree_ok_status();
    }
  }
  if (*inout_prepared_encoder_count == prepared_encoder_capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "parameter encoder specialization cache capacity exhausted");
  }

  id4_pipeline_parameter_prepared_encoder_t* prepared_encoder =
      &prepared_encoders[(*inout_prepared_encoder_count)++];
  id4_pipeline_parameter_prepared_encoder_initialize(prepared_encoder);
  prepared_encoder->kind = step->kind;
  prepared_encoder->output_size =
      id4_pipeline_parameter_encode_step_output_size(step);
  prepared_encoder->input_size =
      id4_pipeline_parameter_encode_step_input_size(step);
  iree_status_t status = id4_pipeline_parameter_prepare_encoder(
      options, load, step, &prepared_encoder->executable,
      &prepared_encoder->function);
  if (!iree_status_is_ok(status)) {
    --(*inout_prepared_encoder_count);
    id4_pipeline_parameter_prepared_encoder_deinitialize(prepared_encoder);
    return status;
  }
  *out_prepared_encoder = prepared_encoder;
  return iree_ok_status();
}

static void id4_pipeline_parameter_release_prepared_encoders(
    iree_host_size_t prepared_encoder_count,
    id4_pipeline_parameter_prepared_encoder_t* prepared_encoders,
    iree_allocator_t host_allocator) {
  for (iree_host_size_t i = 0; i < prepared_encoder_count; ++i) {
    id4_pipeline_parameter_prepared_encoder_deinitialize(&prepared_encoders[i]);
  }
  iree_allocator_free(host_allocator, prepared_encoders);
}

static iree_hal_semaphore_list_t id4_pipeline_parameter_one_semaphore_list(
    iree_hal_semaphore_t** semaphore, uint64_t* payload_value) {
  return (iree_hal_semaphore_list_t){
      // One semaphore in the list.
      .count = 1,
      // Semaphore pointer list.
      .semaphores = semaphore,
      // Required or published payload value.
      .payload_values = payload_value,
  };
}

static iree_status_t id4_pipeline_parameter_create_semaphore(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_semaphore_t** out_semaphore) {
  return iree_hal_semaphore_create(device, queue_affinity, 0,
                                   IREE_HAL_SEMAPHORE_FLAG_NONE, out_semaphore);
}

static iree_hal_buffer_usage_t
id4_pipeline_parameter_encoder_staging_buffer_usage(
    iree_hal_memory_type_t memory_type) {
  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                                  IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  }
  return usage;
}

typedef struct id4_pipeline_parameter_encode_step_staging_layout_t {
  // Staging offsets where each encoded source tensor starts.
  iree_device_size_t
      source_offsets[ID4_PIPELINE_PARAMETER_ENCODER_MAX_SOURCE_COUNT];
  // Staging byte length consumed through this packed step.
  iree_device_size_t end_offset;
} id4_pipeline_parameter_encode_step_staging_layout_t;

static iree_status_t id4_pipeline_parameter_encode_step_staging_layout(
    const id4_pipeline_parameter_load_step_t* step,
    iree_device_size_t base_offset,
    id4_pipeline_parameter_encode_step_staging_layout_t* out_layout) {
  memset(out_layout, 0, sizeof(*out_layout));
  iree_device_size_t end_offset = base_offset;
  for (iree_host_size_t i = 0; i < step->source_count; ++i) {
    iree_device_size_t source_offset = 0;
    if (!iree_device_size_checked_align(end_offset, 16, &source_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "parameter load step '%.*s' staging source %" PRIhsz
          " offset overflows",
          (int)step->name.size, step->name.data, i);
    }
    if (!iree_device_size_checked_add(
            source_offset, step->sources[i].byte_length, &end_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "parameter load step '%.*s' staging source %" PRIhsz
          " byte length overflows",
          (int)step->name.size, step->name.data, i);
    }
    out_layout->source_offsets[i] = source_offset;
  }
  out_layout->end_offset = end_offset;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_encode_plan_chunk(
    iree_host_size_t step_count,
    const id4_pipeline_parameter_load_step_t* steps,
    iree_device_size_t chunk_byte_capacity,
    id4_pipeline_parameter_encode_step_staging_layout_t* out_layouts,
    iree_host_size_t* out_chunk_step_count,
    iree_device_size_t* out_chunk_byte_length) {
  *out_chunk_step_count = 0;
  *out_chunk_byte_length = 0;
  iree_device_size_t byte_length = 0;
  for (iree_host_size_t i = 0;
       i < step_count && i < ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_STEP_CAPACITY;
       ++i) {
    id4_pipeline_parameter_encode_step_staging_layout_t layout;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_encode_step_staging_layout(
        &steps[i], byte_length, &layout));
    if (layout.end_offset > chunk_byte_capacity) {
      if (i == 0) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "parameter load step '%.*s' requires %" PRIu64
            " staging bytes, exceeding chunk capacity %" PRIu64,
            (int)steps[i].name.size, steps[i].name.data,
            (uint64_t)layout.end_offset, (uint64_t)chunk_byte_capacity);
      }
      break;
    }
    if (out_layouts) out_layouts[i] = layout;
    byte_length = layout.end_offset;
    *out_chunk_step_count = i + 1;
    *out_chunk_byte_length = byte_length;
  }
  return iree_ok_status();
}

static iree_host_size_t id4_pipeline_parameter_find_encode_source_batch(
    iree_host_size_t batch_count,
    const id4_pipeline_parameter_encode_source_batch_t* batches,
    iree_string_view_t source_scope) {
  for (iree_host_size_t i = 0; i < batch_count; ++i) {
    if (iree_string_view_equal(batches[i].source_scope, source_scope)) return i;
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_status_t id4_pipeline_parameter_encode_group_chunk_sources(
    iree_host_size_t chunk_step_count,
    const id4_pipeline_parameter_load_step_t* steps,
    const id4_pipeline_parameter_encode_step_staging_layout_t* layouts,
    iree_device_size_t slot_base_offset,
    id4_pipeline_parameter_encode_chunk_source_t* out_sources,
    id4_pipeline_parameter_encode_source_batch_t* out_batches,
    iree_host_size_t* out_batch_count) {
  *out_batch_count = 0;
  for (iree_host_size_t i = 0; i < chunk_step_count; ++i) {
    const id4_pipeline_parameter_load_step_t* step = &steps[i];
    for (iree_host_size_t j = 0; j < step->source_count; ++j) {
      const id4_pipeline_parameter_load_source_t* source = &step->sources[j];
      iree_host_size_t batch_index =
          id4_pipeline_parameter_find_encode_source_batch(
              *out_batch_count, out_batches, source->source_scope);
      if (batch_index == IREE_HOST_SIZE_MAX) {
        if (*out_batch_count ==
            ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "parameter encoder chunk source batch capacity exceeded");
        }
        batch_index = (*out_batch_count)++;
        out_batches[batch_index].source_scope = source->source_scope;
        out_batches[batch_index].source_offset = 0;
        out_batches[batch_index].source_count = 0;
      }
    }
  }
  iree_host_size_t source_count = 0;
  for (iree_host_size_t batch_index = 0; batch_index < *out_batch_count;
       ++batch_index) {
    id4_pipeline_parameter_encode_source_batch_t* batch =
        &out_batches[batch_index];
    batch->source_offset = source_count;
    for (iree_host_size_t i = 0; i < chunk_step_count; ++i) {
      const id4_pipeline_parameter_load_step_t* step = &steps[i];
      for (iree_host_size_t j = 0; j < step->source_count; ++j) {
        const id4_pipeline_parameter_load_source_t* source = &step->sources[j];
        if (!iree_string_view_equal(batch->source_scope,
                                    source->source_scope)) {
          continue;
        }
        if (source_count ==
            ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "parameter encoder chunk source capacity exceeded");
        }
        out_sources[source_count].source = source;
        out_sources[source_count].buffer_offset =
            slot_base_offset + layouts[i].source_offsets[j];
        ++source_count;
        ++batch->source_count;
      }
    }
  }
  return iree_ok_status();
}

typedef struct id4_pipeline_parameter_encode_wave_chunk_t {
  // First load step represented by this chunk within the current encode run.
  iree_host_size_t step_offset;
  // Number of load steps packed into this chunk.
  iree_host_size_t chunk_step_count;
  // Number of staging bytes required by this chunk.
  iree_device_size_t chunk_byte_length;
  // Monotonic chunk ordinal used for staging slot selection.
  iree_host_size_t chunk_ordinal;
  // Staging slot ordinal used by this chunk.
  iree_host_size_t slot;
  // Base byte offset of this chunk's staging slot.
  iree_device_size_t slot_base_offset;
  // Per-step source offsets within this chunk's staging slot.
  id4_pipeline_parameter_encode_step_staging_layout_t
      layouts[ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_STEP_CAPACITY];
  // Source descriptors grouped by provider source scope.
  id4_pipeline_parameter_encode_chunk_source_t
      grouped_sources[ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY];
  // Provider gather groups for this chunk.
  id4_pipeline_parameter_encode_source_batch_t
      source_batches[ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY];
  // Number of populated source_batches entries.
  iree_host_size_t source_batch_count;
  // Slot readiness required before source writes may begin.
  iree_hal_semaphore_list_t source_wait_list;
  // Payload value signaled by all chunk-local source gather groups.
  uint64_t source_payload_value;
  // Payload values signaled by chunk-local source gather groups.
  uint64_t source_signal_payload_values
      [ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY];
  // Signal lists published by chunk-local source gather groups.
  iree_hal_semaphore_list_t
      source_signal_lists[ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY];
} id4_pipeline_parameter_encode_wave_chunk_t;

typedef struct id4_pipeline_parameter_encode_wave_source_group_t {
  // Provider source scope shared by every source in this group.
  iree_string_view_t source_scope;
  // Slot-readiness waits shared by every source in this group.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Source tensors plus staging offsets enumerated by this group.
  id4_pipeline_parameter_encode_chunk_source_t
      sources[ID4_PIPELINE_PARAMETER_ENCODER_WAVE_SOURCE_CAPACITY];
  // Number of populated source entries.
  iree_host_size_t source_count;
  // Semaphores signaled when this group has populated all sources.
  iree_hal_semaphore_t*
      signal_semaphores[ID4_PIPELINE_PARAMETER_ENCODER_WAVE_CHUNK_CAPACITY];
  // Payload values paired with signal_semaphores.
  uint64_t
      signal_payload_values[ID4_PIPELINE_PARAMETER_ENCODER_WAVE_CHUNK_CAPACITY];
  // Number of populated signal entries.
  iree_host_size_t signal_count;
} id4_pipeline_parameter_encode_wave_source_group_t;

static bool id4_pipeline_parameter_same_semaphore_list(
    iree_hal_semaphore_list_t lhs, iree_hal_semaphore_list_t rhs) {
  if (lhs.count != rhs.count) return false;
  for (iree_host_size_t i = 0; i < lhs.count; ++i) {
    if (lhs.semaphores[i] != rhs.semaphores[i]) return false;
    if (lhs.payload_values[i] != rhs.payload_values[i]) return false;
  }
  return true;
}

static iree_host_size_t id4_pipeline_parameter_find_wave_source_group(
    iree_host_size_t group_count,
    const id4_pipeline_parameter_encode_wave_source_group_t* groups,
    iree_string_view_t source_scope, iree_hal_semaphore_list_t wait_list) {
  for (iree_host_size_t i = 0; i < group_count; ++i) {
    if (!iree_string_view_equal(groups[i].source_scope, source_scope)) {
      continue;
    }
    if (!id4_pipeline_parameter_same_semaphore_list(
            groups[i].wait_semaphore_list, wait_list)) {
      continue;
    }
    return i;
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_status_t id4_pipeline_parameter_append_wave_source_group(
    const id4_pipeline_parameter_encode_wave_chunk_t* chunk,
    iree_host_size_t batch_index,
    id4_pipeline_parameter_encode_wave_source_group_t* group) {
  const id4_pipeline_parameter_encode_source_batch_t* batch =
      &chunk->source_batches[batch_index];
  if (batch->source_count >
      ID4_PIPELINE_PARAMETER_ENCODER_WAVE_SOURCE_CAPACITY -
          group->source_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter encoder wave source group capacity exceeded");
  }
  if (group->signal_count ==
      ID4_PIPELINE_PARAMETER_ENCODER_WAVE_CHUNK_CAPACITY) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter encoder wave source signal capacity exceeded");
  }
  iree_hal_semaphore_list_t signal_list =
      chunk->source_signal_lists[batch_index];
  if (signal_list.count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter encoder chunk source batch must have one signal semaphore");
  }
  for (iree_host_size_t i = 0; i < batch->source_count; ++i) {
    group->sources[group->source_count++] =
        chunk->grouped_sources[batch->source_offset + i];
  }
  group->signal_semaphores[group->signal_count] = signal_list.semaphores[0];
  group->signal_payload_values[group->signal_count] =
      signal_list.payload_values[0];
  ++group->signal_count;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_build_source_gather_wave(
    iree_host_size_t wave_chunk_count,
    const id4_pipeline_parameter_encode_wave_chunk_t* chunks,
    id4_pipeline_parameter_encode_wave_source_group_t* groups,
    iree_host_size_t* out_group_count) {
  *out_group_count = 0;
  for (iree_host_size_t chunk_index = 0; chunk_index < wave_chunk_count;
       ++chunk_index) {
    const id4_pipeline_parameter_encode_wave_chunk_t* chunk =
        &chunks[chunk_index];
    for (iree_host_size_t batch_index = 0;
         batch_index < chunk->source_batch_count; ++batch_index) {
      const id4_pipeline_parameter_encode_source_batch_t* batch =
          &chunk->source_batches[batch_index];
      iree_host_size_t group_index =
          id4_pipeline_parameter_find_wave_source_group(
              *out_group_count, groups, batch->source_scope,
              chunk->source_wait_list);
      if (group_index == IREE_HOST_SIZE_MAX) {
        if (*out_group_count ==
            ID4_PIPELINE_PARAMETER_ENCODER_WAVE_SOURCE_CAPACITY) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "parameter encoder source gather wave capacity exceeded");
        }
        group_index = (*out_group_count)++;
        groups[group_index] =
            (id4_pipeline_parameter_encode_wave_source_group_t){
                // Provider source scope for this merged group.
                .source_scope = batch->source_scope,
                // Shared slot-readiness waits for this merged group.
                .wait_semaphore_list = chunk->source_wait_list,
            };
      }
      IREE_RETURN_IF_ERROR(id4_pipeline_parameter_append_wave_source_group(
          chunk, batch_index, &groups[group_index]));
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_submit_source_gather_wave(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* load,
    iree_host_size_t wave_chunk_count,
    const id4_pipeline_parameter_encode_wave_chunk_t* chunks,
    iree_hal_buffer_t* staging_buffer) {
  id4_pipeline_parameter_encode_wave_source_group_t
      source_groups[ID4_PIPELINE_PARAMETER_ENCODER_WAVE_SOURCE_CAPACITY] = {0};
  iree_host_size_t source_group_count = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_build_source_gather_wave(
      wave_chunk_count, chunks, source_groups, &source_group_count));

  id4_pipeline_parameter_load_source_enumerator_state_t
      enumerator_states[ID4_PIPELINE_PARAMETER_ENCODER_WAVE_SOURCE_CAPACITY] = {
          0};
  iree_io_parameter_gather_t
      gathers[ID4_PIPELINE_PARAMETER_ENCODER_WAVE_SOURCE_CAPACITY] = {0};
  iree_host_size_t gather_count = 0;
  for (iree_host_size_t group_index = 0; group_index < source_group_count;
       ++group_index) {
    id4_pipeline_parameter_encode_wave_source_group_t* group =
        &source_groups[group_index];
    if (!iree_io_parameter_provider_query_support(options->provider,
                                                  group->source_scope)) {
      return iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "parameter provider does not support source scope '%.*s'",
          (int)group->source_scope.size, group->source_scope.data);
    }
    enumerator_states[gather_count] =
        (id4_pipeline_parameter_load_source_enumerator_state_t){
            // Number of source descriptors gathered by this request.
            .source_count = group->source_count,
            // Source descriptors gathered by this request.
            .sources = group->sources,
        };
    iree_hal_semaphore_list_t signal_semaphore_list = {
        // Number of chunk-local readiness semaphores signaled by this group.
        .count = group->signal_count,
        // Chunk-local readiness semaphores signaled by this group.
        .semaphores = group->signal_semaphores,
        // Payload values paired with signal_semaphores.
        .payload_values = group->signal_payload_values,
    };
    gathers[gather_count] = (iree_io_parameter_gather_t){
        // Source scope for this provider gather group.
        .source_scope = group->source_scope,
        // Staging buffer populated by the source gather.
        .target_buffer = staging_buffer,
        // Number of source descriptors in this gather group.
        .count = group->source_count,
        // Source descriptor enumerator for this gather group.
        .enumerator = id4_pipeline_parameter_load_source_enumerator(
            &enumerator_states[gather_count]),
        // Slot readiness required before source writes may begin.
        .wait_semaphore_list = group->wait_semaphore_list,
        // Chunk-local readiness signaled after these source writes complete.
        .signal_semaphore_list = signal_semaphore_list,
    };
    ++gather_count;
  }
  return iree_io_parameter_provider_gather_batch(
      options->provider, load->device, load->queue_affinity, gather_count,
      gathers);
}

iree_status_t id4_pipeline_parameter_encode_query_statistics(
    const id4_pipeline_parameter_request_table_t* request_table,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_device_size_t staging_chunk_byte_capacity,
    id4_pipeline_parameter_encode_statistics_t* out_statistics) {
  IREE_ASSERT_ARGUMENT(out_statistics);
  memset(out_statistics, 0, sizeof(*out_statistics));
  if (load_step_count != 0 && (!request_table || !load_steps)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter encode statistics require requests and load steps");
  }
  if (load_step_count != 0 && staging_chunk_byte_capacity == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter encode statistics require nonzero staging capacity");
  }
  out_statistics->staging_slot_count =
      load_step_count == 0 ? 0
                           : ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT;
  out_statistics->encoder_dispatch_count = load_step_count;
  for (iree_host_size_t i = 0; i < load_step_count;) {
    iree_host_size_t wave_chunk_count = 0;
    bool wave_has_source_batches = false;
    while (i < load_step_count &&
           wave_chunk_count <
               ID4_PIPELINE_PARAMETER_ENCODER_WAVE_CHUNK_CAPACITY) {
      id4_pipeline_parameter_encode_step_staging_layout_t
          layouts[ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_STEP_CAPACITY];
      iree_host_size_t chunk_step_count = 0;
      iree_device_size_t chunk_byte_length = 0;
      IREE_RETURN_IF_ERROR(id4_pipeline_parameter_encode_plan_chunk(
          load_step_count - i, &load_steps[i], staging_chunk_byte_capacity,
          layouts, &chunk_step_count, &chunk_byte_length));
      if (chunk_step_count == 0) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "parameter encoder could not plan a staging chunk");
      }
      if (chunk_byte_length > out_statistics->staging_slot_byte_length) {
        out_statistics->staging_slot_byte_length = chunk_byte_length;
      }
      id4_pipeline_parameter_encode_chunk_source_t grouped_sources
          [ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY] = {0};
      id4_pipeline_parameter_encode_source_batch_t
          source_batches[ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY] =
              {0};
      iree_host_size_t source_batch_count = 0;
      IREE_RETURN_IF_ERROR(id4_pipeline_parameter_encode_group_chunk_sources(
          chunk_step_count, &load_steps[i], layouts, /*slot_base_offset=*/0,
          grouped_sources, source_batches, &source_batch_count));
      wave_has_source_batches |= source_batch_count != 0;
      IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_host_size(
          1, &out_statistics->staging_chunk_count, "staging chunk"));
      for (iree_host_size_t j = 0; j < chunk_step_count; ++j) {
        const id4_pipeline_parameter_load_step_t* step = &load_steps[i + j];
        if (step->request_count != 1 || step->request_indices ||
            step->request_offset >= request_table->count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "encoded parameter load step '%.*s' must target one valid "
              "contiguous request",
              (int)step->name.size, step->name.data);
        }
        IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_host_size(
            step->source_count, &out_statistics->logical_source_count,
            "logical source"));
        for (iree_host_size_t k = 0; k < step->source_count; ++k) {
          IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_device_size(
              step->sources[k].byte_length, &out_statistics->source_byte_length,
              "source"));
        }
        const id4_pipeline_parameter_request_t* target_request =
            &request_table->values[step->request_offset];
        IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_device_size(
            target_request->span.length, &out_statistics->target_byte_length,
            "target"));
      }
      i += chunk_step_count;
      ++wave_chunk_count;
    }
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_host_size(
        wave_has_source_batches ? 1 : 0,
        &out_statistics->source_gather_batch_count, "source gather batch"));
  }
  if (!iree_device_size_checked_mul(
          out_statistics->staging_slot_byte_length,
          out_statistics->staging_slot_count,
          &out_statistics->staging_total_byte_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter encoder staging allocation byte length overflows");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_slab_emit_encode_window_diagnostic(
    const id4_pipeline_parameter_slab_load_t* load,
    iree_string_view_t stage_name,
    id4_pipeline_parameter_load_group_context_t group_context,
    iree_host_size_t load_step_offset, iree_host_size_t load_step_count,
    iree_string_view_t first_load_step_name,
    iree_device_size_t staging_byte_length,
    const id4_pipeline_parameter_encode_statistics_t* statistics,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_pipeline_parameter_slab_diagnostic_t parameter_slab =
      id4_pipeline_parameter_slab_make_diagnostic(
          load, iree_string_view_empty(), IREE_HOST_SIZE_MAX);
  id4_pipeline_parameter_load_diagnostic_t parameter_load = {
      // Plan-local slab index populated by the loading window.
      .slab_index = load->slab_index,
      // Plan-local load group ordinal.
      .load_group_index = group_context.group_index,
      // Submission strategy used by the load group.
      .load_group_kind = id4_pipeline_parameter_load_group_kind_name(
          ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_ENCODE),
      // First planned region that consumes this group.
      .first_consumer_region_id = group_context.first_consumer_region_id,
      // Region that submitted this group.
      .submit_region_id = group_context.submit_region_id,
      // First load-step ordinal represented by the window.
      .load_step_offset = load_step_offset,
      // Number of load steps represented by the window.
      .load_step_count = load_step_count,
      // Human-readable first load-step name.
      .first_load_step_name = first_load_step_name,
      // Number of staging slots allocated for the window.
      .staging_slot_count = ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT,
      // Byte length of one staging slot.
      .staging_slot_byte_length = statistics->staging_slot_byte_length,
      // Total byte length of all staging slots.
      .staging_total_byte_length = staging_byte_length,
      // Number of chunks planned for the staging slots.
      .staging_chunk_count = statistics->staging_chunk_count,
      // Number of logical source tensors gathered into staging.
      .logical_source_count = statistics->logical_source_count,
      // Number of provider gather batches submitted.
      .source_gather_batch_count = statistics->source_gather_batch_count,
      // Total source bytes gathered into staging.
      .source_byte_length = statistics->source_byte_length,
      // Total final slab bytes populated by encoder dispatches.
      .target_byte_length = statistics->target_byte_length,
      // Number of encoder dispatches recorded.
      .encoder_dispatch_count = statistics->encoder_dispatch_count,
  };
  id4_pipeline_diagnostic_event_t event = {
      // Event kind for parameter slab diagnostics.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PARAMETER_SLAB,
      // Stage name associated with the load.
      .stage_name = stage_name,
      // Stable parameter load event key.
      .key = IREE_SV("parameter_slab.encode_window"),
      // Short event summary.
      .message = IREE_SV("encoded parameter staging window"),
      // Structured slab payload.
      .parameter_slab = &parameter_slab,
      // Structured parameter loading payload.
      .parameter_load = &parameter_load,
  };
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static iree_hal_semaphore_list_t
id4_pipeline_parameter_encode_cleanup_wait_list(
    iree_host_size_t count, iree_hal_semaphore_t** semaphores,
    uint64_t* payload_values) {
  return (iree_hal_semaphore_list_t){
      // Number of submitted operations that cleanup must wait on.
      .count = count,
      // Semaphore pointer list.
      .semaphores = semaphores,
      // Required payload values before cleanup can run.
      .payload_values = payload_values,
  };
}

typedef struct id4_pipeline_parameter_encode_staging_slot_t {
  // Semaphore signaled when the slot's current encoded dispatch completes.
  iree_hal_semaphore_t* encode_semaphore;
  // Latest payload signaled on encode_semaphore, or zero before first use.
  uint64_t encode_payload_value;
  // Semaphores signaled by parameter source gathers into this staging slot.
  iree_hal_semaphore_t*
      source_semaphores[ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY];
  // Latest payload signaled on source_semaphores entries.
  uint64_t source_payload_value;
  // Number of slot-local waits currently required before cleanup may run.
  iree_host_size_t cleanup_wait_count;
  // Slot-local semaphores that cleanup must wait on.
  iree_hal_semaphore_t* cleanup_wait_semaphores
      [ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY];
  // Slot-local payload values that cleanup must wait on.
  uint64_t cleanup_wait_payload_values
      [ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY];
} id4_pipeline_parameter_encode_staging_slot_t;

static iree_host_size_t id4_pipeline_parameter_encode_cleanup_waits_flatten(
    const id4_pipeline_parameter_encode_staging_slot_t* slots,
    iree_hal_semaphore_t** out_semaphores, uint64_t* out_payload_values) {
  iree_host_size_t count = 0;
  for (iree_host_size_t slot = 0;
       slot < ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT; ++slot) {
    for (iree_host_size_t i = 0; i < slots[slot].cleanup_wait_count; ++i) {
      out_semaphores[count] = slots[slot].cleanup_wait_semaphores[i];
      out_payload_values[count] = slots[slot].cleanup_wait_payload_values[i];
      ++count;
    }
  }
  return count;
}

static void id4_pipeline_parameter_encode_cleanup_waits_replace(
    id4_pipeline_parameter_encode_staging_slot_t* slot,
    iree_host_size_t wait_count, iree_hal_semaphore_t** wait_semaphores,
    const uint64_t* wait_payload_values) {
  slot->cleanup_wait_count = wait_count;
  for (iree_host_size_t i = 0; i < wait_count; ++i) {
    slot->cleanup_wait_semaphores[i] = wait_semaphores[i];
    slot->cleanup_wait_payload_values[i] = wait_payload_values[i];
  }
}

static iree_hal_semaphore_list_t id4_pipeline_parameter_many_semaphore_list(
    iree_host_size_t count, iree_hal_semaphore_t** semaphores,
    uint64_t* payload_values) {
  return (iree_hal_semaphore_list_t){
      // Number of semaphores in the list.
      .count = count,
      // Semaphore pointer list.
      .semaphores = semaphores,
      // Required or published payload values.
      .payload_values = payload_values,
  };
}

static iree_status_t id4_pipeline_parameter_encode_make_source_wait_list(
    iree_hal_semaphore_t* slot_wait_semaphore, uint64_t slot_wait_payload_value,
    iree_hal_semaphore_list_t base_wait_list,
    iree_hal_semaphore_t** wait_semaphores, uint64_t* wait_payload_values,
    iree_hal_semaphore_list_t* out_wait_list) {
  IREE_ASSERT_ARGUMENT(out_wait_list);
  *out_wait_list = iree_hal_semaphore_list_empty();
  if (!slot_wait_semaphore) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter encoder source wait requires slot readiness semaphore");
  }
  if (base_wait_list.count != 0 &&
      (!base_wait_list.semaphores || !base_wait_list.payload_values)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter encoder source base wait list is malformed");
  }
  iree_host_size_t wait_count = 0;
  if (!iree_host_size_checked_add(base_wait_list.count, 1, &wait_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter encoder source wait list count overflows");
  }
  wait_semaphores[0] = slot_wait_semaphore;
  wait_payload_values[0] = slot_wait_payload_value;
  for (iree_host_size_t i = 0; i < base_wait_list.count; ++i) {
    wait_semaphores[i + 1] = base_wait_list.semaphores[i];
    wait_payload_values[i + 1] = base_wait_list.payload_values[i];
  }
  *out_wait_list = id4_pipeline_parameter_many_semaphore_list(
      wait_count, wait_semaphores, wait_payload_values);
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_create_source_semaphores(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t semaphore_count, iree_hal_semaphore_t** semaphores) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < semaphore_count && iree_status_is_ok(status);
       ++i) {
    status = id4_pipeline_parameter_create_semaphore(device, queue_affinity,
                                                     &semaphores[i]);
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < semaphore_count; ++i) {
      iree_hal_semaphore_release(semaphores[i]);
      semaphores[i] = NULL;
    }
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_create_staging_slot_semaphores(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    id4_pipeline_parameter_encode_staging_slot_t* slots) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t slot = 0;
       slot < ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT &&
       iree_status_is_ok(status);
       ++slot) {
    status = id4_pipeline_parameter_create_semaphore(
        device, queue_affinity, &slots[slot].encode_semaphore);
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_parameter_create_source_semaphores(
          device, queue_affinity,
          ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY,
          slots[slot].source_semaphores);
    }
  }
  return status;
}

static void id4_pipeline_parameter_release_source_semaphores(
    iree_hal_semaphore_t** semaphores, iree_host_size_t semaphore_count) {
  for (iree_host_size_t i = 0; i < semaphore_count; ++i) {
    iree_hal_semaphore_release(semaphores[i]);
  }
}

static void id4_pipeline_parameter_release_staging_slot_semaphores(
    id4_pipeline_parameter_encode_staging_slot_t* slots) {
  for (iree_host_size_t slot = 0;
       slot < ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT; ++slot) {
    id4_pipeline_parameter_release_source_semaphores(
        slots[slot].source_semaphores,
        ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY);
    iree_hal_semaphore_release(slots[slot].encode_semaphore);
  }
}

typedef struct id4_pipeline_parameter_encode_window_t {
  // True when this target slab has at least one encoded load group.
  bool planned;
  // True when queue_alloca has been submitted for staging_buffer.
  bool alloca_submitted;
  // True when staging_slots owns reusable semaphores.
  bool staging_slots_initialized;
  // Next staging chunk ordinal used to choose a reusable staging slot.
  iree_host_size_t next_chunk_ordinal;
  // Reusable staging slot semaphore state for this target slab window.
  id4_pipeline_parameter_encode_staging_slot_t
      staging_slots[ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT];
  // Byte length required by one staging slot across all encoded groups.
  iree_device_size_t staging_slot_byte_length;
  // Byte length allocated for all staging slots.
  iree_device_size_t staging_total_byte_length;
  // Number of encoded groups planned for this target slab.
  iree_host_size_t encoded_group_count;
  // Number of staging chunks planned across all encoded groups.
  iree_host_size_t staging_chunk_count;
  // Number of provider source tensors staged across all encoded groups.
  iree_host_size_t logical_source_count;
  // Number of provider gather batches planned across all encoded groups.
  iree_host_size_t source_gather_batch_count;
  // Total provider source bytes gathered through this window.
  iree_device_size_t source_byte_length;
  // Total final slab bytes populated by encoder dispatches.
  iree_device_size_t target_byte_length;
  // Number of encoder dispatches planned across all encoded groups.
  iree_host_size_t encoder_dispatch_count;
  // Shared staging buffer allocated for this window.
  iree_hal_buffer_t* staging_buffer;
  // Semaphore signaled when queue_alloca completes.
  iree_hal_semaphore_t* staging_ready_semaphore;
  // Payload value signaled on staging_ready_semaphore.
  uint64_t staging_ready_payload_value;
  // Semaphore signaled when queue_dealloca completes.
  iree_hal_semaphore_t* cleanup_semaphore;
  // Payload value signaled on cleanup_semaphore.
  uint64_t cleanup_payload_value;
  // Number of waits protecting the currently submitted staging contents.
  iree_host_size_t cleanup_wait_count;
  // Semaphores that staging dealloca must wait on.
  iree_hal_semaphore_t* cleanup_wait_semaphores
      [ID4_PIPELINE_PARAMETER_ENCODER_CLEANUP_WAIT_CAPACITY];
  // Payload values paired with cleanup_wait_semaphores.
  uint64_t cleanup_wait_payload_values
      [ID4_PIPELINE_PARAMETER_ENCODER_CLEANUP_WAIT_CAPACITY];
} id4_pipeline_parameter_encode_window_t;

typedef struct id4_pipeline_parameter_encode_window_context_t {
  // Allocator used for context-owned arrays.
  iree_allocator_t host_allocator;
  // Borrowed slab load descriptors used by window cleanup.
  const id4_pipeline_parameter_slab_load_t* loads;
  // True when window allocation emits issue-local diagnostics.
  bool emits_issue_window_diagnostic;
  // Capacity of the context-owned prepared encoder cache.
  iree_host_size_t prepared_encoder_capacity;
  // Number of prepared encoder entries initialized in prepared_encoders.
  iree_host_size_t prepared_encoder_count;
  // Prepared encoder specializations reused across encoded load groups.
  id4_pipeline_parameter_prepared_encoder_t* prepared_encoders;
  // Number of per-target staging windows.
  iree_host_size_t window_count;
  // Per-target reusable staging windows.
  id4_pipeline_parameter_encode_window_t* windows;
  // Number of cleanup edges produced by finish.
  iree_host_size_t cleanup_count;
  // Cleanup semaphores borrowed from windows after finish.
  iree_hal_semaphore_t** cleanup_semaphores;
  // Payload values paired with cleanup_semaphores.
  uint64_t* cleanup_payload_values;
  // True once finish has been called.
  bool finished;
} id4_pipeline_parameter_encode_window_context_t;

struct id4_pipeline_parameter_slab_issue_context_t {
  // Allocator used for context storage.
  iree_allocator_t host_allocator;
  // Deferred slab set retained while issue-local loads may be submitted.
  id4_pipeline_parameter_slab_set_t* slab_set;
  // Reusable staging windows owned by this deferred issue context.
  id4_pipeline_parameter_encode_window_context_t encode_windows;
};

static iree_status_t id4_pipeline_parameter_encode_window_record_statistics(
    const id4_pipeline_parameter_slab_load_t* load,
    const id4_pipeline_parameter_load_step_t* steps, iree_host_size_t count,
    iree_device_size_t chunk_byte_capacity,
    id4_pipeline_parameter_encode_window_t* window) {
  id4_pipeline_parameter_encode_statistics_t statistics;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_encode_query_statistics(
      load->request_table, count, steps, chunk_byte_capacity, &statistics));
  window->planned = true;
  window->staging_slot_byte_length = iree_max(
      window->staging_slot_byte_length, statistics.staging_slot_byte_length);
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_host_size(
      1, &window->encoded_group_count, "encoded group"));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_host_size(
      statistics.staging_chunk_count, &window->staging_chunk_count,
      "staging chunk"));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_host_size(
      statistics.logical_source_count, &window->logical_source_count,
      "logical source"));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_host_size(
      statistics.source_gather_batch_count, &window->source_gather_batch_count,
      "source gather batch"));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_device_size(
      statistics.source_byte_length, &window->source_byte_length, "source"));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_device_size(
      statistics.target_byte_length, &window->target_byte_length, "target"));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_host_size(
      statistics.encoder_dispatch_count, &window->encoder_dispatch_count,
      "encoder dispatch"));
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_encode_window_finalize_plan(
    id4_pipeline_parameter_encode_window_t* window) {
  if (!window->planned) return iree_ok_status();
  if (!iree_device_size_checked_mul(
          window->staging_slot_byte_length,
          ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT,
          &window->staging_total_byte_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter encoder issue staging allocation byte length overflows");
  }
  return iree_ok_status();
}

static iree_hal_semaphore_list_t
id4_pipeline_parameter_encode_window_cleanup_wait_list(
    id4_pipeline_parameter_encode_window_t* window) {
  return id4_pipeline_parameter_many_semaphore_list(
      window->cleanup_wait_count, window->cleanup_wait_semaphores,
      window->cleanup_wait_payload_values);
}

static iree_status_t id4_pipeline_parameter_encode_window_set_cleanup_waits(
    id4_pipeline_parameter_encode_window_t* window,
    iree_hal_semaphore_list_t wait_list) {
  if (wait_list.count > ID4_PIPELINE_PARAMETER_ENCODER_CLEANUP_WAIT_CAPACITY) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter encoder cleanup wait count %" PRIhsz " exceeds capacity %d",
        wait_list.count, ID4_PIPELINE_PARAMETER_ENCODER_CLEANUP_WAIT_CAPACITY);
  }
  iree_hal_semaphore_list_retain(wait_list);
  iree_hal_semaphore_list_release(
      id4_pipeline_parameter_encode_window_cleanup_wait_list(window));
  window->cleanup_wait_count = wait_list.count;
  for (iree_host_size_t i = 0; i < wait_list.count; ++i) {
    window->cleanup_wait_semaphores[i] = wait_list.semaphores[i];
    window->cleanup_wait_payload_values[i] = wait_list.payload_values[i];
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_parameter_slab_emit_encode_window_allocation_diagnostic(
    const id4_pipeline_parameter_slab_load_t* load,
    iree_string_view_t stage_name,
    id4_pipeline_parameter_load_group_context_t group_context,
    iree_string_view_t key, iree_string_view_t message,
    const id4_pipeline_parameter_encode_window_t* window,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_pipeline_parameter_slab_diagnostic_t parameter_slab =
      id4_pipeline_parameter_slab_make_diagnostic(
          load, iree_string_view_empty(), IREE_HOST_SIZE_MAX);
  id4_pipeline_parameter_load_diagnostic_t parameter_load = {
      // Plan-local slab index populated through the issue window.
      .slab_index = load->slab_index,
      // First load group that allocated this issue window.
      .load_group_index = group_context.group_index,
      // Submission strategy represented by the issue window.
      .load_group_kind = id4_pipeline_parameter_load_group_kind_name(
          ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_ENCODE),
      // First planned region that consumes the allocating group.
      .first_consumer_region_id = group_context.first_consumer_region_id,
      // Region that allocated the window.
      .submit_region_id = group_context.submit_region_id,
      // Issue windows can cover non-contiguous load groups.
      .load_step_offset = IREE_HOST_SIZE_MAX,
      // Number of encoded groups sharing this window.
      .load_step_count = window->encoded_group_count,
      // Issue-window allocation is shared across non-contiguous groups.
      .first_load_step_name = iree_string_view_empty(),
      // Number of staging slots allocated by the window.
      .staging_slot_count = ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT,
      // Byte length of one staging slot.
      .staging_slot_byte_length = window->staging_slot_byte_length,
      // Total byte length of all staging slots.
      .staging_total_byte_length = window->staging_total_byte_length,
      // Number of chunks planned for this window.
      .staging_chunk_count = window->staging_chunk_count,
      // Number of logical source tensors staged by this window.
      .logical_source_count = window->logical_source_count,
      // Number of provider gather batches staged by this window.
      .source_gather_batch_count = window->source_gather_batch_count,
      // Total source bytes gathered through this window.
      .source_byte_length = window->source_byte_length,
      // Total target bytes populated through this window.
      .target_byte_length = window->target_byte_length,
      // Total encoder dispatches using this window.
      .encoder_dispatch_count = window->encoder_dispatch_count,
  };
  id4_pipeline_diagnostic_event_t event = {
      // Event kind for parameter slab diagnostics.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PARAMETER_SLAB,
      // Stage name associated with the allocated window.
      .stage_name = stage_name,
      // Stable parameter load event key.
      .key = key,
      // Short event summary.
      .message = message,
      // Structured slab payload.
      .parameter_slab = &parameter_slab,
      // Structured parameter loading payload.
      .parameter_load = &parameter_load,
  };
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static iree_status_t id4_pipeline_parameter_encode_window_allocate(
    const id4_pipeline_parameter_slab_load_t* load,
    id4_pipeline_parameter_load_group_context_t group_context,
    iree_string_view_t stage_name, iree_hal_semaphore_list_t wait_list,
    iree_hal_memory_type_t staging_memory_type,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    bool emits_issue_window_diagnostic,
    id4_pipeline_parameter_encode_window_t* window) {
  if (window->alloca_submitted) return iree_ok_status();
  if (!window->planned || window->staging_total_byte_length == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "parameter encoder issue window was not planned");
  }

  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_create_semaphore(
      load->device, load->queue_affinity, &window->staging_ready_semaphore));
  window->staging_ready_payload_value = 1;
  if (!window->staging_slots_initialized) {
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_create_staging_slot_semaphores(
        load->device, load->queue_affinity, window->staging_slots));
    window->staging_slots_initialized = true;
  }

  iree_hal_buffer_params_t staging_params = {0};
  staging_params.type = staging_memory_type;
  staging_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  staging_params.usage =
      id4_pipeline_parameter_encoder_staging_buffer_usage(staging_params.type);
  staging_params.queue_affinity = load->queue_affinity;
  staging_params.min_alignment = 16;

  iree_hal_semaphore_t* staging_ready_semaphore =
      window->staging_ready_semaphore;
  uint64_t staging_ready_payload_value = window->staging_ready_payload_value;
  iree_hal_semaphore_list_t staging_ready_signal_list =
      id4_pipeline_parameter_one_semaphore_list(&staging_ready_semaphore,
                                                &staging_ready_payload_value);
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_alloca(
      load->device, load->queue_affinity, wait_list, staging_ready_signal_list,
      /*pool=*/NULL, staging_params, window->staging_total_byte_length,
      IREE_HAL_ALLOCA_FLAG_NONE, &window->staging_buffer));
  window->alloca_submitted = true;

  iree_hal_semaphore_list_t cleanup_wait_list =
      id4_pipeline_parameter_one_semaphore_list(
          &window->staging_ready_semaphore,
          &window->staging_ready_payload_value);
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_encode_window_set_cleanup_waits(
      window, cleanup_wait_list));
  iree_string_view_t key = IREE_SV("parameter_slab.prepare_encode_window");
  iree_string_view_t message =
      IREE_SV("allocated prepare-time encoder staging window");
  if (emits_issue_window_diagnostic) {
    key = IREE_SV("parameter_slab.issue_encode_window");
    message = IREE_SV("allocated issue-local encoder staging window");
  }
  return id4_pipeline_parameter_slab_emit_encode_window_allocation_diagnostic(
      load, stage_name, group_context, key, message, window, diagnostics_sink);
}

static void id4_pipeline_parameter_encode_window_context_deinitialize(
    id4_pipeline_parameter_encode_window_context_t* context) {
  iree_allocator_t host_allocator = context->host_allocator;
  if (iree_allocator_is_null(host_allocator)) return;
  id4_pipeline_parameter_release_prepared_encoders(
      context->prepared_encoder_count, context->prepared_encoders,
      host_allocator);
  if (context->windows) {
    for (iree_host_size_t i = 0; i < context->window_count; ++i) {
      id4_pipeline_parameter_encode_window_t* window = &context->windows[i];
      iree_hal_semaphore_list_release(
          id4_pipeline_parameter_encode_window_cleanup_wait_list(window));
      id4_pipeline_parameter_release_staging_slot_semaphores(
          window->staging_slots);
      iree_hal_buffer_release(window->staging_buffer);
      iree_hal_semaphore_release(window->cleanup_semaphore);
      iree_hal_semaphore_release(window->staging_ready_semaphore);
    }
  }
  iree_allocator_free(host_allocator, context->cleanup_payload_values);
  iree_allocator_free(host_allocator, context->cleanup_semaphores);
  iree_allocator_free(host_allocator, context->windows);
  memset(context, 0, sizeof(*context));
}

static void id4_pipeline_parameter_slab_issue_context_destroy(
    id4_pipeline_parameter_slab_issue_context_t* context) {
  iree_allocator_t host_allocator = context->host_allocator;
  id4_pipeline_parameter_encode_window_context_deinitialize(
      &context->encode_windows);
  id4_pipeline_parameter_slab_set_release(context->slab_set);
  iree_allocator_free(host_allocator, context);
}

static iree_status_t id4_pipeline_parameter_slab_submit_encode_run(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    id4_pipeline_parameter_encode_window_context_t* encode_window_context,
    const id4_pipeline_parameter_slab_load_t* load, iree_host_size_t step_count,
    const id4_pipeline_parameter_load_step_t* steps,
    id4_pipeline_parameter_load_group_context_t group_context,
    iree_host_size_t load_step_offset, iree_string_view_t stage_name,
    iree_hal_buffer_t* target_buffer,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_allocator_t host_allocator) {
  id4_pipeline_parameter_encode_statistics_t statistics;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_encode_query_statistics(
      load->request_table, step_count, steps,
      options->encoder_staging_chunk_byte_capacity, &statistics));
  const iree_device_size_t staging_slot_byte_length =
      statistics.staging_slot_byte_length;
  const iree_device_size_t staging_byte_length =
      statistics.staging_total_byte_length;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_slab_emit_encode_window_diagnostic(
          load, stage_name, group_context, load_step_offset, step_count,
          steps[0].name, staging_byte_length, &statistics,
          options->diagnostics_sink));

  const bool uses_encode_window = encode_window_context != NULL;
  id4_pipeline_parameter_encode_window_t* encode_window = NULL;
  if (uses_encode_window) {
    if (load->slab_index >= encode_window_context->window_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter encoder target slab %" PRIhsz
                              " outside encode window count %" PRIhsz,
                              load->slab_index,
                              encode_window_context->window_count);
    }
    encode_window = &encode_window_context->windows[load->slab_index];
    if (staging_slot_byte_length > encode_window->staging_slot_byte_length) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "parameter encoder group requires %" PRIu64
          " staging bytes per slot, exceeding planned encode window %" PRIu64,
          (uint64_t)staging_slot_byte_length,
          (uint64_t)encode_window->staging_slot_byte_length);
    }
  }

  iree_hal_semaphore_t* run_semaphore = NULL;
  id4_pipeline_parameter_prepared_encoder_t* prepared_encoders =
      uses_encode_window ? encode_window_context->prepared_encoders : NULL;
  iree_host_size_t prepared_encoder_capacity =
      uses_encode_window ? encode_window_context->prepared_encoder_capacity
                         : step_count;
  iree_host_size_t local_prepared_encoder_count = 0;
  iree_host_size_t* prepared_encoder_count =
      uses_encode_window ? &encode_window_context->prepared_encoder_count
                         : &local_prepared_encoder_count;
  id4_pipeline_parameter_encode_staging_slot_t
      local_staging_slots[ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT] = {
          0};
  id4_pipeline_parameter_encode_staging_slot_t* slots =
      uses_encode_window ? encode_window->staging_slots : local_staging_slots;
  iree_hal_semaphore_t* cleanup_semaphore = NULL;
  iree_hal_buffer_t* staging_buffer = NULL;
  iree_status_t status = iree_ok_status();
  if (!uses_encode_window) {
    status = iree_allocator_malloc_array(host_allocator, step_count,
                                         sizeof(prepared_encoders[0]),
                                         (void**)&prepared_encoders);
  }
  if (iree_status_is_ok(status) && !uses_encode_window) {
    memset(prepared_encoders, 0, step_count * sizeof(prepared_encoders[0]));
  }
  if (iree_status_is_ok(status) && !uses_encode_window) {
    status = id4_pipeline_parameter_create_semaphore(
        load->device, load->queue_affinity, &run_semaphore);
  }
  if (iree_status_is_ok(status) && !uses_encode_window) {
    status = id4_pipeline_parameter_create_staging_slot_semaphores(
        load->device, load->queue_affinity, slots);
  }
  if (iree_status_is_ok(status) && !uses_encode_window) {
    status = id4_pipeline_parameter_create_semaphore(
        load->device, load->queue_affinity, &cleanup_semaphore);
  }

  uint64_t staging_ready_payload_value = 1;
  uint64_t cleanup_payload_value = 1;
  iree_hal_semaphore_list_t staging_ready_signal_list =
      id4_pipeline_parameter_one_semaphore_list(&run_semaphore,
                                                &staging_ready_payload_value);
  iree_hal_semaphore_list_t cleanup_signal_list =
      id4_pipeline_parameter_one_semaphore_list(&cleanup_semaphore,
                                                &cleanup_payload_value);

  iree_hal_buffer_params_t staging_params = {0};
  staging_params.type = options->encoder_staging_memory_type;
  staging_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  staging_params.usage =
      id4_pipeline_parameter_encoder_staging_buffer_usage(staging_params.type);
  staging_params.queue_affinity = load->queue_affinity;
  staging_params.min_alignment = 16;

  bool staging_alloca_submitted = false;
  iree_hal_semaphore_t* cleanup_wait_semaphores
      [ID4_PIPELINE_PARAMETER_ENCODER_CLEANUP_WAIT_CAPACITY] = {0};
  uint64_t cleanup_wait_payload_values
      [ID4_PIPELINE_PARAMETER_ENCODER_CLEANUP_WAIT_CAPACITY] = {0};
  iree_host_size_t cleanup_wait_count = 0;
  if (iree_status_is_ok(status) && uses_encode_window) {
    status = id4_pipeline_parameter_encode_window_allocate(
        load, group_context, stage_name, wait_semaphore_list,
        options->encoder_staging_memory_type, options->diagnostics_sink,
        encode_window_context->emits_issue_window_diagnostic, encode_window);
    if (iree_status_is_ok(status)) {
      staging_buffer = encode_window->staging_buffer;
      staging_alloca_submitted = encode_window->alloca_submitted;
    }
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_alloca(
        load->device, load->queue_affinity, wait_semaphore_list,
        staging_ready_signal_list,
        /*pool=*/NULL, staging_params, staging_byte_length,
        IREE_HAL_ALLOCA_FLAG_NONE, &staging_buffer);
    staging_alloca_submitted = iree_status_is_ok(status);
    if (staging_alloca_submitted) {
      cleanup_wait_semaphores[0] = run_semaphore;
      cleanup_wait_payload_values[0] = staging_ready_payload_value;
      cleanup_wait_count = 1;
    }
  }

  iree_host_size_t step_offset = 0;
  iree_host_size_t chunk_ordinal =
      uses_encode_window ? encode_window->next_chunk_ordinal : 0;
  bool run_wait_slot_used[ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT] = {
      false};
  iree_hal_semaphore_t*
      run_wait_semaphores[ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT] = {
          0};
  uint64_t run_wait_payload_values
      [ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT] = {0};
  iree_host_size_t source_wait_capacity = 0;
  if (!iree_host_size_checked_add(wait_semaphore_list.count, 1,
                                  &source_wait_capacity)) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter encoder source wait list capacity overflows");
  }
  iree_host_size_t source_wait_storage_count = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_mul(
          source_wait_capacity,
          ID4_PIPELINE_PARAMETER_ENCODER_WAVE_CHUNK_CAPACITY,
          &source_wait_storage_count)) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter encoder source wait list storage count overflows");
  }
  iree_hal_semaphore_t** source_wait_semaphores = NULL;
  uint64_t* source_wait_payload_values = NULL;
  if (iree_status_is_ok(status) && source_wait_storage_count != 0) {
    source_wait_semaphores = (iree_hal_semaphore_t**)iree_alloca(
        source_wait_storage_count * sizeof(source_wait_semaphores[0]));
    source_wait_payload_values = (uint64_t*)iree_alloca(
        source_wait_storage_count * sizeof(source_wait_payload_values[0]));
  }
  while (step_offset < step_count && iree_status_is_ok(status)) {
    id4_pipeline_parameter_encode_wave_chunk_t
        wave_chunks[ID4_PIPELINE_PARAMETER_ENCODER_WAVE_CHUNK_CAPACITY] = {0};
    iree_host_size_t wave_chunk_count = 0;
    iree_host_size_t planned_step_offset = step_offset;
    iree_host_size_t planned_chunk_ordinal = chunk_ordinal;
    while (planned_step_offset < step_count &&
           wave_chunk_count <
               ID4_PIPELINE_PARAMETER_ENCODER_WAVE_CHUNK_CAPACITY &&
           iree_status_is_ok(status)) {
      id4_pipeline_parameter_encode_wave_chunk_t* chunk =
          &wave_chunks[wave_chunk_count];
      chunk->step_offset = planned_step_offset;
      chunk->chunk_ordinal = planned_chunk_ordinal;
      status = id4_pipeline_parameter_encode_plan_chunk(
          step_count - planned_step_offset, &steps[planned_step_offset],
          options->encoder_staging_chunk_byte_capacity, chunk->layouts,
          &chunk->chunk_step_count, &chunk->chunk_byte_length);
      if (!iree_status_is_ok(status)) break;
      if (chunk->chunk_step_count == 0) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "parameter encoder could not plan a staging chunk");
        break;
      }
      if (chunk->chunk_byte_length > staging_slot_byte_length) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "parameter encoder chunk byte length exceeds staging slot");
        break;
      }

      chunk->slot = chunk->chunk_ordinal %
                    ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT;
      id4_pipeline_parameter_encode_staging_slot_t* slot_state =
          &slots[chunk->slot];
      if (!iree_device_size_checked_mul(staging_slot_byte_length, chunk->slot,
                                        &chunk->slot_base_offset)) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "parameter encoder staging slot byte offset overflows");
        break;
      }
      iree_hal_semaphore_t* slot_wait_semaphore = NULL;
      uint64_t slot_wait_payload_value = 0;
      if (slot_state->encode_payload_value == 0) {
        if (uses_encode_window) {
          slot_wait_semaphore = encode_window->staging_ready_semaphore;
          slot_wait_payload_value = encode_window->staging_ready_payload_value;
        } else {
          slot_wait_semaphore = run_semaphore;
          slot_wait_payload_value = staging_ready_payload_value;
        }
      } else {
        slot_wait_semaphore = slot_state->encode_semaphore;
        slot_wait_payload_value = slot_state->encode_payload_value;
      }
      status = id4_pipeline_parameter_encode_make_source_wait_list(
          slot_wait_semaphore, slot_wait_payload_value, wait_semaphore_list,
          source_wait_semaphores + wave_chunk_count * source_wait_capacity,
          source_wait_payload_values + wave_chunk_count * source_wait_capacity,
          &chunk->source_wait_list);
      if (!iree_status_is_ok(status)) break;
      status = id4_pipeline_parameter_encode_group_chunk_sources(
          chunk->chunk_step_count, &steps[planned_step_offset], chunk->layouts,
          chunk->slot_base_offset, chunk->grouped_sources,
          chunk->source_batches, &chunk->source_batch_count);
      if (!iree_status_is_ok(status)) break;
      chunk->source_payload_value = slot_state->source_payload_value + 1;
      for (iree_host_size_t batch_index = 0;
           batch_index < chunk->source_batch_count; ++batch_index) {
        chunk->source_signal_payload_values[batch_index] =
            chunk->source_payload_value;
        chunk->source_signal_lists[batch_index] =
            id4_pipeline_parameter_one_semaphore_list(
                &slot_state->source_semaphores[batch_index],
                &chunk->source_signal_payload_values[batch_index]);
      }

      planned_step_offset += chunk->chunk_step_count;
      ++planned_chunk_ordinal;
      ++wave_chunk_count;
    }
    if (!iree_status_is_ok(status)) break;
    if (wave_chunk_count == 0) {
      status =
          iree_make_status(IREE_STATUS_INTERNAL,
                           "parameter encoder could not plan a source wave");
      break;
    }

    status = id4_pipeline_parameter_submit_source_gather_wave(
        options, load, wave_chunk_count, wave_chunks, staging_buffer);
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t chunk_index = 0; chunk_index < wave_chunk_count;
           ++chunk_index) {
        id4_pipeline_parameter_encode_wave_chunk_t* chunk =
            &wave_chunks[chunk_index];
        id4_pipeline_parameter_encode_staging_slot_t* slot_state =
            &slots[chunk->slot];
        slot_state->cleanup_wait_count = 0;
        for (iree_host_size_t batch_index = 0;
             batch_index < chunk->source_batch_count; ++batch_index) {
          slot_state->cleanup_wait_semaphores[batch_index] =
              slot_state->source_semaphores[batch_index];
          slot_state->cleanup_wait_payload_values[batch_index] =
              chunk->source_signal_payload_values[batch_index];
          slot_state->cleanup_wait_count = batch_index + 1;
        }
        slot_state->source_payload_value = chunk->source_payload_value;
      }
      cleanup_wait_count = id4_pipeline_parameter_encode_cleanup_waits_flatten(
          slots, cleanup_wait_semaphores, cleanup_wait_payload_values);
    }

    for (iree_host_size_t chunk_index = 0;
         chunk_index < wave_chunk_count && iree_status_is_ok(status);
         ++chunk_index) {
      id4_pipeline_parameter_encode_wave_chunk_t* chunk =
          &wave_chunks[chunk_index];
      id4_pipeline_parameter_encode_staging_slot_t* slot_state =
          &slots[chunk->slot];
      iree_hal_command_buffer_t* command_buffer = NULL;
      status = iree_hal_command_buffer_create(
          load->device,
          options->command_buffer_mode | IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
          IREE_HAL_COMMAND_CATEGORY_DISPATCH, load->queue_affinity,
          /*binding_capacity=*/0, &command_buffer);
      if (iree_status_is_ok(status)) {
        status = iree_hal_command_buffer_begin(command_buffer);
      }
      for (iree_host_size_t i = 0;
           i < chunk->chunk_step_count && iree_status_is_ok(status); ++i) {
        const id4_pipeline_parameter_load_step_t* step =
            &steps[chunk->step_offset + i];
        const id4_pipeline_parameter_request_t* target_request =
            &load->request_table->values[step->request_offset];
        const id4_pipeline_parameter_load_source_t* weight_source =
            &step->sources[0];

        id4_pipeline_parameter_prepared_encoder_t* prepared_encoder = NULL;
        status = id4_pipeline_parameter_encode_run_resolve_encoder(
            options, load, step, prepared_encoders, prepared_encoder_capacity,
            prepared_encoder_count, &prepared_encoder);
        if (iree_status_is_ok(status)) {
          status = id4_pipeline_parameter_slab_emit_diagnostic(
              load, stage_name, IREE_SV("parameter_slab.encode"),
              IREE_SV("encoding parameter into resident execution layout"),
              weight_source->source_scope, step->request_offset,
              options->diagnostics_sink);
        }
        if (iree_status_is_ok(status)) {
          iree_hal_buffer_ref_t bindings[3];
          for (iree_host_size_t j = 0; j < step->source_count; ++j) {
            bindings[j] = iree_hal_make_buffer_ref(
                staging_buffer,
                chunk->slot_base_offset + chunk->layouts[i].source_offsets[j],
                step->sources[j].byte_length);
          }
          bindings[step->source_count] = iree_hal_make_buffer_ref(
              target_buffer, target_request->span.buffer_offset,
              target_request->span.length);
          iree_hal_buffer_ref_list_t binding_list = {
              // Encoded sources followed by the resident target.
              .count = step->source_count + 1,
              // Direct source and target buffer refs.
              .values = bindings,
          };
          status = iree_hal_command_buffer_dispatch(
              command_buffer,
              id4_pipeline_kernel_executable_hal_executable(
                  prepared_encoder->executable),
              prepared_encoder->function,
              id4_pipeline_kernel_executable_dispatch_config(
                  prepared_encoder->executable),
              iree_const_byte_span_empty(), binding_list,
              IREE_HAL_DISPATCH_FLAG_NONE);
        }
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_command_buffer_end(command_buffer);
      }
      uint64_t encode_payload_value = slot_state->encode_payload_value + 1;
      iree_hal_semaphore_list_t encode_signal_list =
          id4_pipeline_parameter_one_semaphore_list(
              &slot_state->encode_semaphore, &encode_payload_value);
      if (iree_status_is_ok(status)) {
        iree_hal_semaphore_list_t encode_wait_list =
            id4_pipeline_parameter_many_semaphore_list(
                chunk->source_batch_count, slot_state->source_semaphores,
                chunk->source_signal_payload_values);
        status = iree_hal_device_queue_execute(
            load->device, load->queue_affinity, encode_wait_list,
            encode_signal_list, command_buffer,
            iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE);
      }
      if (iree_status_is_ok(status)) {
        iree_hal_semaphore_t* slot_wait_semaphore =
            slot_state->encode_semaphore;
        id4_pipeline_parameter_encode_cleanup_waits_replace(
            slot_state, /*wait_count=*/1, &slot_wait_semaphore,
            &encode_payload_value);
        cleanup_wait_count =
            id4_pipeline_parameter_encode_cleanup_waits_flatten(
                slots, cleanup_wait_semaphores, cleanup_wait_payload_values);
        slot_state->encode_payload_value = encode_payload_value;
        if (!run_wait_slot_used[chunk->slot])
          run_wait_slot_used[chunk->slot] = true;
        run_wait_semaphores[chunk->slot] = slot_state->encode_semaphore;
        run_wait_payload_values[chunk->slot] = encode_payload_value;
        step_offset += chunk->chunk_step_count;
        ++chunk_ordinal;
        if (uses_encode_window)
          encode_window->next_chunk_ordinal = chunk_ordinal;
      }
      iree_hal_command_buffer_release(command_buffer);
    }
  }

  if (iree_status_is_ok(status)) {
    cleanup_wait_count = id4_pipeline_parameter_encode_cleanup_waits_flatten(
        slots, cleanup_wait_semaphores, cleanup_wait_payload_values);
    iree_hal_semaphore_t* group_wait_semaphores
        [ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT] = {0};
    uint64_t group_wait_payload_values
        [ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT] = {0};
    iree_host_size_t group_wait_count = 0;
    for (iree_host_size_t slot = 0;
         slot < ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT; ++slot) {
      if (!run_wait_slot_used[slot]) continue;
      group_wait_semaphores[group_wait_count] = run_wait_semaphores[slot];
      group_wait_payload_values[group_wait_count] =
          run_wait_payload_values[slot];
      ++group_wait_count;
    }
    if (cleanup_wait_count == 0) {
      if (uses_encode_window) {
        cleanup_wait_semaphores[0] = encode_window->staging_ready_semaphore;
        cleanup_wait_payload_values[0] =
            encode_window->staging_ready_payload_value;
      } else {
        cleanup_wait_semaphores[0] = run_semaphore;
        cleanup_wait_payload_values[0] = staging_ready_payload_value;
      }
      cleanup_wait_count = 1;
    }
    if (group_wait_count == 0) {
      if (uses_encode_window) {
        group_wait_semaphores[0] = encode_window->staging_ready_semaphore;
        group_wait_payload_values[0] =
            encode_window->staging_ready_payload_value;
      } else {
        group_wait_semaphores[0] = run_semaphore;
        group_wait_payload_values[0] = staging_ready_payload_value;
      }
      group_wait_count = 1;
    }
    iree_hal_semaphore_list_t cleanup_wait_list =
        id4_pipeline_parameter_encode_cleanup_wait_list(
            cleanup_wait_count, cleanup_wait_semaphores,
            cleanup_wait_payload_values);
    iree_hal_semaphore_list_t group_wait_list =
        id4_pipeline_parameter_encode_cleanup_wait_list(
            group_wait_count, group_wait_semaphores, group_wait_payload_values);
    if (uses_encode_window) {
      status = iree_hal_device_queue_barrier(
          load->device, load->queue_affinity, group_wait_list,
          signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
      if (iree_status_is_ok(status)) {
        status = id4_pipeline_parameter_encode_window_set_cleanup_waits(
            encode_window, cleanup_wait_list);
      }
    } else {
      status = iree_hal_device_queue_dealloca(
          load->device, load->queue_affinity, cleanup_wait_list,
          signal_semaphore_list, staging_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
    }
  } else if (staging_buffer &&
             (staging_alloca_submitted || uses_encode_window)) {
    iree_hal_semaphore_list_t cleanup_wait_list =
        id4_pipeline_parameter_encode_cleanup_wait_list(
            cleanup_wait_count, cleanup_wait_semaphores,
            cleanup_wait_payload_values);
    iree_status_t cleanup_status = iree_ok_status();
    if (uses_encode_window) {
      if (cleanup_wait_count != 0) {
        cleanup_status = id4_pipeline_parameter_encode_window_set_cleanup_waits(
            encode_window, cleanup_wait_list);
      }
    } else {
      cleanup_status = iree_hal_device_queue_dealloca(
          load->device, load->queue_affinity, cleanup_wait_list,
          cleanup_signal_list, staging_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
      if (iree_status_is_ok(cleanup_status)) {
        cleanup_status = iree_hal_semaphore_wait(
            cleanup_semaphore, cleanup_payload_value, iree_infinite_timeout(),
            IREE_ASYNC_WAIT_FLAG_NONE);
      }
    }
    status = iree_status_join(status, cleanup_status);
  }

  if (!uses_encode_window) {
    iree_hal_buffer_release(staging_buffer);
  }
  iree_hal_semaphore_release(cleanup_semaphore);
  if (!uses_encode_window) {
    id4_pipeline_parameter_release_staging_slot_semaphores(slots);
  }
  iree_hal_semaphore_release(run_semaphore);
  if (!uses_encode_window) {
    id4_pipeline_parameter_release_prepared_encoders(
        *prepared_encoder_count, prepared_encoders, host_allocator);
  }
  return status;
}

static iree_host_size_t id4_pipeline_parameter_slab_encode_run_count(
    iree_host_size_t start_index, iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps) {
  const iree_host_size_t target_slab_index =
      load_steps[start_index].target_slab_index;
  const iree_host_size_t readiness_group_key =
      load_steps[start_index].readiness_group_key;
  iree_host_size_t end_index = start_index;
  while (end_index < load_step_count &&
         id4_pipeline_parameter_load_step_is_encode(&load_steps[end_index]) &&
         load_steps[end_index].target_slab_index == target_slab_index &&
         load_steps[end_index].readiness_group_key == readiness_group_key) {
    ++end_index;
  }
  return end_index - start_index;
}

static iree_status_t id4_pipeline_parameter_slab_emit_gather_group_diagnostic(
    const id4_pipeline_parameter_slab_load_t* load,
    const id4_pipeline_parameter_load_step_t* step,
    iree_string_view_t stage_name,
    id4_pipeline_parameter_load_group_context_t group_context,
    id4_pipeline_parameter_load_group_t group,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_device_size_t byte_length = 0;
  for (iree_host_size_t i = 0; i < step->request_count; ++i) {
    const iree_host_size_t request_index = step->request_indices
                                               ? step->request_indices[i]
                                               : step->request_offset + i;
    const id4_pipeline_parameter_request_t* request =
        &load->request_table->values[request_index];
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_device_size(
        request->span.length, &byte_length, "gather group"));
  }
  id4_pipeline_parameter_slab_diagnostic_t parameter_slab =
      id4_pipeline_parameter_slab_make_diagnostic(load, step->source_scope,
                                                  IREE_HOST_SIZE_MAX);
  id4_pipeline_parameter_load_diagnostic_t parameter_load = {
      // Plan-local slab index populated by this gather group.
      .slab_index = load->slab_index,
      // Plan-local load group ordinal.
      .load_group_index = group_context.group_index,
      // Submission strategy used by the load group.
      .load_group_kind = id4_pipeline_parameter_load_group_kind_name(
          ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_GATHER),
      // First planned region that consumes this group.
      .first_consumer_region_id = group_context.first_consumer_region_id,
      // Region that submitted this group.
      .submit_region_id = group_context.submit_region_id,
      // First load-step ordinal represented by the group.
      .load_step_offset = group.step_offset,
      // Number of direct gather load steps in the group.
      .load_step_count = group.step_count,
      // Human-readable first load-step name.
      .first_load_step_name = step->name,
      // Direct gathers do not allocate staging slots.
      .staging_slot_count = 0,
      // Direct gathers do not allocate staging slot storage.
      .staging_slot_byte_length = 0,
      // Direct gathers do not allocate staging storage.
      .staging_total_byte_length = 0,
      // Direct gathers do not submit staging chunks.
      .staging_chunk_count = 0,
      // Number of provider requests in this direct gather.
      .logical_source_count = step->request_count,
      // Direct gather groups submit one provider gather batch.
      .source_gather_batch_count = 1,
      // Total source bytes gathered directly.
      .source_byte_length = byte_length,
      // Total final slab bytes populated directly.
      .target_byte_length = byte_length,
      // Direct gathers do not dispatch encoder kernels.
      .encoder_dispatch_count = 0,
  };
  id4_pipeline_diagnostic_event_t event = {
      // Event kind for parameter slab diagnostics.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PARAMETER_SLAB,
      // Stage name associated with the load.
      .stage_name = stage_name,
      // Stable parameter load event key.
      .key = IREE_SV("parameter_slab.gather_group"),
      // Short event summary.
      .message = IREE_SV("direct parameter gather group"),
      // Structured slab payload.
      .parameter_slab = &parameter_slab,
      // Structured parameter loading payload.
      .parameter_load = &parameter_load,
  };
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static iree_status_t id4_pipeline_parameter_slab_submit_gather(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* load,
    const id4_pipeline_parameter_load_step_t* step,
    iree_string_view_t stage_name,
    id4_pipeline_parameter_load_group_context_t group_context,
    id4_pipeline_parameter_load_group_t group, iree_hal_buffer_t* target_buffer,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  if (!iree_io_parameter_provider_query_support(options->provider,
                                                step->source_scope)) {
    iree_status_t status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "parameter provider does not support source scope '%.*s'",
        (int)step->source_scope.size, step->source_scope.data);
    return iree_status_join(
        status,
        id4_pipeline_parameter_slab_emit_diagnostic(
            load, stage_name, IREE_SV("parameter_slab.load.error"),
            IREE_SV("parameter provider does not support source "
                    "scope"),
            step->source_scope, IREE_HOST_SIZE_MAX, options->diagnostics_sink));
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_emit_diagnostic(
      load, stage_name, IREE_SV("parameter_slab.load"),
      IREE_SV("loading parameter slab"), step->source_scope, IREE_HOST_SIZE_MAX,
      options->diagnostics_sink));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_emit_gather_group_diagnostic(
      load, step, stage_name, group_context, group, options->diagnostics_sink));
  for (iree_host_size_t j = 0; j < step->request_count; ++j) {
    const iree_host_size_t request_index = step->request_indices
                                               ? step->request_indices[j]
                                               : step->request_offset + j;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_emit_diagnostic(
        load, stage_name, IREE_SV("parameter_slab.gather"),
        IREE_SV("gathering parameter request"), step->source_scope,
        request_index, options->diagnostics_sink));
  }
  id4_pipeline_parameter_slab_enumerator_state_t enumerator_state = {
      // Request table supplying source keys and target spans.
      .request_table = load->request_table,
      // First request ordinal loaded by this step.
      .request_offset = step->request_offset,
      // Number of requests loaded by this step.
      .request_count = step->request_count,
      // Explicit request ordinals for non-contiguous gather steps.
      .request_indices = step->request_indices,
  };
  iree_io_parameter_enumerator_t enumerator =
      id4_pipeline_parameter_slab_enumerator(&enumerator_state);
  iree_status_t status = iree_io_parameter_provider_gather(
      options->provider, load->device, load->queue_affinity,
      wait_semaphore_list, signal_semaphore_list, step->source_scope,
      target_buffer, step->request_count, enumerator);
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status,
        id4_pipeline_parameter_slab_emit_diagnostic(
            load, stage_name, IREE_SV("parameter_slab.load.error"),
            IREE_SV("parameter gather submission failed"), step->source_scope,
            IREE_HOST_SIZE_MAX, options->diagnostics_sink));
  }
  return status;
}

static id4_pipeline_parameter_load_group_t
id4_pipeline_parameter_load_group_from_steps(
    iree_host_size_t step_index, iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps) {
  id4_pipeline_parameter_load_group_t group = {
      // First load-step ordinal represented by this group.
      .step_offset = step_index,
      // One direct gather by default.
      .step_count = 1,
      // Direct gather by default.
      .kind = ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_GATHER,
      // Final slab populated by the first step.
      .target_slab_index = load_steps[step_index].target_slab_index,
  };
  if (id4_pipeline_parameter_load_step_is_encode(&load_steps[step_index])) {
    group.step_count = id4_pipeline_parameter_slab_encode_run_count(
        step_index, load_step_count, load_steps);
    group.kind = ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_ENCODE;
  }
  return group;
}

static bool id4_pipeline_parameter_load_group_is_encode(
    id4_pipeline_parameter_load_group_t group) {
  return group.kind == ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_ENCODE;
}

static iree_status_t id4_pipeline_parameter_slab_emit_load_group_submit_timing(
    const id4_pipeline_parameter_slab_load_t* load,
    iree_string_view_t stage_name,
    id4_pipeline_parameter_load_group_context_t group_context,
    id4_pipeline_parameter_load_group_t group, iree_time_t start_time_ns,
    iree_time_t end_time_ns, iree_string_view_t first_load_step_name,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_pipeline_parameter_slab_diagnostic_t parameter_slab =
      id4_pipeline_parameter_slab_make_diagnostic(
          load, iree_string_view_empty(), IREE_HOST_SIZE_MAX);
  id4_pipeline_parameter_load_diagnostic_t parameter_load = {
      // Plan-local slab index populated by the load group.
      .slab_index = load->slab_index,
      // Plan-local load group ordinal.
      .load_group_index = group_context.group_index,
      // Submission strategy used by the load group.
      .load_group_kind =
          id4_pipeline_parameter_load_group_kind_name(group.kind),
      // First planned region that consumes this group.
      .first_consumer_region_id = group_context.first_consumer_region_id,
      // Region that submitted this group.
      .submit_region_id = group_context.submit_region_id,
      // First load-step ordinal represented by the group.
      .load_step_offset = group.step_offset,
      // Number of load steps represented by the group.
      .load_step_count = group.step_count,
      // Human-readable first load-step name.
      .first_load_step_name = first_load_step_name,
      // Submit timing does not describe staging slot allocation.
      .staging_slot_count = 0,
      // Submit timing does not describe staging slot allocation.
      .staging_slot_byte_length = 0,
      // Submit timing does not describe staging allocation.
      .staging_total_byte_length = 0,
      // Submit timing does not describe staging chunks.
      .staging_chunk_count = 0,
      // Submit timing does not describe provider source count.
      .logical_source_count = 0,
      // Submit timing does not describe provider batch count.
      .source_gather_batch_count = 0,
      // Submit timing does not describe provider source bytes.
      .source_byte_length = 0,
      // Submit timing does not describe final slab bytes.
      .target_byte_length = 0,
      // Submit timing does not describe encoder dispatch count.
      .encoder_dispatch_count = 0,
  };
  id4_pipeline_timing_diagnostic_t timing = {
      // Monotonic start timestamp for the host submit span.
      .start_time_ns = start_time_ns,
      // Monotonic end timestamp for the host submit span.
      .end_time_ns = end_time_ns,
      // Host-observed elapsed submit duration.
      .duration_ns = end_time_ns - start_time_ns,
  };
  id4_pipeline_diagnostic_event_t event = {
      // Event kind for host-observed timing.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_TIMING,
      // Stage name associated with the load.
      .stage_name = stage_name,
      // Stable timing event key.
      .key = IREE_SV("parameter_slab.load_group.submit"),
      // Short event summary.
      .message = IREE_SV("submitted parameter load group"),
      // Structured slab payload.
      .parameter_slab = &parameter_slab,
      // Structured parameter loading payload.
      .parameter_load = &parameter_load,
      // Host-observed timing payload.
      .timing = &timing,
  };
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static iree_status_t id4_pipeline_parameter_slab_describe_failed_load_group(
    const id4_pipeline_parameter_slab_set_t* slab_set,
    iree_host_size_t group_index, iree_string_view_t stage_name,
    uint64_t current_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  if (!slab_set->load_groups || !slab_set->loads || !slab_set->load_steps) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter slab set has no retained load descriptors for failure "
        "diagnostics");
  }
  const id4_pipeline_parameter_load_group_t group =
      slab_set->load_groups[group_index];
  if (group.target_slab_index >= slab_set->load_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "failed parameter load group %" PRIhsz " target slab %" PRIhsz
        " is outside load count %" PRIhsz,
        group_index, group.target_slab_index, slab_set->load_count);
  }
  if (group.step_offset >= slab_set->load_step_count ||
      group.step_count > slab_set->load_step_count - group.step_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "failed parameter load group %" PRIhsz " step range [%" PRIhsz
        ", %" PRIhsz ") is outside load step count %" PRIhsz,
        group_index, group.step_offset, group.step_offset + group.step_count,
        slab_set->load_step_count);
  }

  const id4_pipeline_parameter_load_step_t* step =
      &slab_set->load_steps[group.step_offset];
  const id4_pipeline_parameter_slab_load_t* load =
      &slab_set->loads[group.target_slab_index];
  id4_pipeline_parameter_slab_diagnostic_t parameter_slab =
      id4_pipeline_parameter_slab_make_diagnostic(
          load, id4_pipeline_parameter_load_step_primary_scope(step),
          IREE_HOST_SIZE_MAX);
  id4_pipeline_parameter_load_diagnostic_t parameter_load = {
      // Plan-local slab index populated by this load group.
      .slab_index = load->slab_index,
      // Plan-local load group ordinal.
      .load_group_index = group_index,
      // Submission strategy used by the failed load group.
      .load_group_kind =
          id4_pipeline_parameter_load_group_kind_name(group.kind),
      // Failure checks run after scheduling, without region-local context.
      .first_consumer_region_id = IREE_HOST_SIZE_MAX,
      // Failure checks run after scheduling, without region-local context.
      .submit_region_id = IREE_HOST_SIZE_MAX,
      // First load-step ordinal represented by the failed group.
      .load_step_offset = group.step_offset,
      // Number of load steps represented by the failed group.
      .load_step_count = group.step_count,
      // Human-readable first load-step name.
      .first_load_step_name = step->name,
      // Populated below for encoded load groups.
      .staging_slot_count = 0,
      // Populated below for encoded load groups.
      .staging_slot_byte_length = 0,
      // Populated below for encoded load groups.
      .staging_total_byte_length = 0,
      // Populated below for encoded load groups.
      .staging_chunk_count = 0,
      // Populated below from the failed group.
      .logical_source_count = 0,
      // Populated below from the failed group.
      .source_gather_batch_count = 0,
      // Populated below from the failed group.
      .source_byte_length = 0,
      // Populated below from the failed group.
      .target_byte_length = 0,
      // Populated below for encoded load groups.
      .encoder_dispatch_count = 0,
  };
  if (id4_pipeline_parameter_load_group_is_encode(group)) {
    id4_pipeline_parameter_encode_statistics_t statistics;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_encode_query_statistics(
        load->request_table, group.step_count, step,
        slab_set->encoder_staging_chunk_byte_capacity, &statistics));
    parameter_load.staging_slot_count = statistics.staging_slot_count;
    parameter_load.staging_slot_byte_length =
        statistics.staging_slot_byte_length;
    parameter_load.staging_total_byte_length =
        statistics.staging_total_byte_length;
    parameter_load.staging_chunk_count = statistics.staging_chunk_count;
    parameter_load.logical_source_count = statistics.logical_source_count;
    parameter_load.source_gather_batch_count =
        statistics.source_gather_batch_count;
    parameter_load.source_byte_length = statistics.source_byte_length;
    parameter_load.target_byte_length = statistics.target_byte_length;
    parameter_load.encoder_dispatch_count = statistics.encoder_dispatch_count;
  } else {
    for (iree_host_size_t i = 0; i < group.step_count; ++i) {
      const id4_pipeline_parameter_load_step_t* gather_step =
          &slab_set->load_steps[group.step_offset + i];
      IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_host_size(
          gather_step->request_count, &parameter_load.logical_source_count,
          "logical source"));
      IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_host_size(
          1, &parameter_load.source_gather_batch_count, "source gather batch"));
      for (iree_host_size_t j = 0; j < gather_step->request_count; ++j) {
        const iree_host_size_t request_index =
            gather_step->request_indices ? gather_step->request_indices[j]
                                         : gather_step->request_offset + j;
        if (request_index >= load->request_table->count) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "failed parameter load group request index %" PRIhsz
              " is outside slab request count %" PRIhsz,
              request_index, load->request_table->count);
        }
        const id4_pipeline_parameter_request_t* request =
            &load->request_table->values[request_index];
        IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_device_size(
            request->span.length, &parameter_load.source_byte_length,
            "source"));
        IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_device_size(
            request->span.length, &parameter_load.target_byte_length,
            "target"));
      }
    }
  }

  id4_pipeline_diagnostic_event_t event = {
      // Event kind for parameter slab diagnostics.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PARAMETER_SLAB,
      // Stage name associated with the failed load group.
      .stage_name = stage_name,
      // Stable parameter load failure event key.
      .key = IREE_SV("parameter_slab.load_group.failure"),
      // Short event summary.
      .message = IREE_SV("parameter load group failed"),
      // Structured slab payload.
      .parameter_slab = &parameter_slab,
      // Structured parameter loading payload.
      .parameter_load = &parameter_load,
  };
  iree_status_t status =
      id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
  return iree_status_join(
      status,
      iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "parameter load group %" PRIhsz
          " failed: stage='%.*s', kind='%.*s', first_load_step='%.*s', "
          "target_slab=%" PRIhsz ", current_payload=%" PRIu64
          ", target_memory_type=0x%x, target_memory_access=0x%x, "
          "target_buffer_usage=0x%x",
          group_index, (int)stage_name.size, stage_name.data,
          (int)parameter_load.load_group_kind.size,
          parameter_load.load_group_kind.data,
          (int)parameter_load.first_load_step_name.size,
          parameter_load.first_load_step_name.data, group.target_slab_index,
          current_payload_value, parameter_slab.memory_type,
          parameter_slab.memory_access, parameter_slab.buffer_usage));
}

iree_status_t id4_pipeline_parameter_load_group_count(
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_host_size_t* out_group_count) {
  IREE_ASSERT_ARGUMENT(out_group_count);
  *out_group_count = 0;
  if (load_step_count != 0 && !load_steps) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter load step array is required");
  }
  iree_host_size_t group_count = 0;
  for (iree_host_size_t i = 0; i < load_step_count;) {
    const id4_pipeline_parameter_load_group_t group =
        id4_pipeline_parameter_load_group_from_steps(i, load_step_count,
                                                     load_steps);
    i += group.step_count;
    ++group_count;
  }
  *out_group_count = group_count;
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_load_group_at(
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_host_size_t group_index,
    id4_pipeline_parameter_load_group_t* out_group) {
  IREE_ASSERT_ARGUMENT(out_group);
  memset(out_group, 0, sizeof(*out_group));
  if (load_step_count != 0 && !load_steps) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter load step array is required");
  }
  for (iree_host_size_t step_index = 0, current_group_index = 0;
       step_index < load_step_count; ++current_group_index) {
    const id4_pipeline_parameter_load_group_t group =
        id4_pipeline_parameter_load_group_from_steps(
            step_index, load_step_count, load_steps);
    if (current_group_index == group_index) {
      *out_group = group;
      return iree_ok_status();
    }
    step_index += group.step_count;
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "parameter load group %" PRIhsz
                          " is outside load group count",
                          group_index);
}

static iree_status_t id4_pipeline_parameter_encode_window_context_initialize(
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_count, iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_device_size_t encoder_staging_chunk_byte_capacity,
    bool emits_issue_window_diagnostic, iree_allocator_t host_allocator,
    id4_pipeline_parameter_encode_window_context_t* context) {
  memset(context, 0, sizeof(*context));
  context->host_allocator = host_allocator;
  context->loads = loads;
  context->emits_issue_window_diagnostic = emits_issue_window_diagnostic;
  context->window_count = load_count;

  iree_status_t status = iree_ok_status();
  iree_host_size_t encoded_load_step_count = 0;
  if (context->window_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, context->window_count,
                                         sizeof(context->windows[0]),
                                         (void**)&context->windows);
    if (iree_status_is_ok(status)) {
      memset(context->windows, 0,
             context->window_count * sizeof(context->windows[0]));
    }
  }
  for (iree_host_size_t step_index = 0;
       step_index < load_step_count && iree_status_is_ok(status);) {
    const id4_pipeline_parameter_load_group_t group =
        id4_pipeline_parameter_load_group_from_steps(
            step_index, load_step_count, load_steps);
    if (id4_pipeline_parameter_load_group_is_encode(group)) {
      if (group.target_slab_index >= context->window_count) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "parameter load group target slab %" PRIhsz
                             " outside encode window count %" PRIhsz,
                             group.target_slab_index, context->window_count);
        break;
      }
      const id4_pipeline_parameter_slab_load_t* load =
          &loads[group.target_slab_index];
      status = id4_pipeline_parameter_encode_window_record_statistics(
          load, &load_steps[group.step_offset], group.step_count,
          encoder_staging_chunk_byte_capacity,
          &context->windows[group.target_slab_index]);
      if (iree_status_is_ok(status)) {
        status = id4_pipeline_parameter_add_host_size(
            group.step_count, &encoded_load_step_count,
            "parameter encoder load step");
      }
    }
    step_index += group.step_count;
  }
  for (iree_host_size_t i = 0;
       i < context->window_count && iree_status_is_ok(status); ++i) {
    status = id4_pipeline_parameter_encode_window_finalize_plan(
        &context->windows[i]);
  }
  if (iree_status_is_ok(status) && encoded_load_step_count != 0) {
    context->prepared_encoder_capacity = encoded_load_step_count;
    status = iree_allocator_malloc_array(host_allocator,
                                         context->prepared_encoder_capacity,
                                         sizeof(context->prepared_encoders[0]),
                                         (void**)&context->prepared_encoders);
    if (iree_status_is_ok(status)) {
      memset(context->prepared_encoders, 0,
             context->prepared_encoder_capacity *
                 sizeof(context->prepared_encoders[0]));
    }
  }
  if (iree_status_is_ok(status) && context->window_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, context->window_count,
                                         sizeof(context->cleanup_semaphores[0]),
                                         (void**)&context->cleanup_semaphores);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(
          host_allocator, context->window_count,
          sizeof(context->cleanup_payload_values[0]),
          (void**)&context->cleanup_payload_values);
    }
  }
  if (!iree_status_is_ok(status)) {
    id4_pipeline_parameter_encode_window_context_deinitialize(context);
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_encode_window_context_finish(
    id4_pipeline_parameter_encode_window_context_t* context,
    iree_hal_semaphore_list_t* out_cleanup_wait_list) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_cleanup_wait_list);
  *out_cleanup_wait_list = iree_hal_semaphore_list_empty();
  if (context->finished) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter encode window context is already finished");
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->window_count && iree_status_is_ok(status); ++i) {
    id4_pipeline_parameter_encode_window_t* window = &context->windows[i];
    if (!window->alloca_submitted) continue;

    const id4_pipeline_parameter_slab_load_t* load = &context->loads[i];
    status = id4_pipeline_parameter_create_semaphore(
        load->device, load->queue_affinity, &window->cleanup_semaphore);
    if (!iree_status_is_ok(status)) break;

    window->cleanup_payload_value = 1;
    iree_hal_semaphore_t* cleanup_semaphore = window->cleanup_semaphore;
    uint64_t cleanup_payload_value = window->cleanup_payload_value;
    iree_hal_semaphore_list_t cleanup_signal_list =
        id4_pipeline_parameter_one_semaphore_list(&cleanup_semaphore,
                                                  &cleanup_payload_value);
    status = iree_hal_device_queue_dealloca(
        load->device, load->queue_affinity,
        id4_pipeline_parameter_encode_window_cleanup_wait_list(window),
        cleanup_signal_list, window->staging_buffer,
        IREE_HAL_DEALLOCA_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      context->cleanup_semaphores[context->cleanup_count] =
          window->cleanup_semaphore;
      context->cleanup_payload_values[context->cleanup_count] =
          window->cleanup_payload_value;
      ++context->cleanup_count;
    }
  }
  if (iree_status_is_ok(status)) {
    context->finished = true;
    *out_cleanup_wait_list = (iree_hal_semaphore_list_t){
        // Number of reusable staging cleanup edges.
        .count = context->cleanup_count,
        // Cleanup semaphores borrowed from the window context.
        .semaphores =
            context->cleanup_count == 0 ? NULL : context->cleanup_semaphores,
        // Cleanup payload values paired with cleanup semaphores.
        .payload_values = context->cleanup_count == 0
                              ? NULL
                              : context->cleanup_payload_values,
    };
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_slab_submit_load_group(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    id4_pipeline_parameter_encode_window_context_t* encode_window_context,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    id4_pipeline_parameter_load_group_context_t group_context,
    id4_pipeline_parameter_load_group_t group, iree_string_view_t stage_name,
    iree_host_size_t target_buffer_count,
    iree_hal_buffer_t* const* target_buffers,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_allocator_t host_allocator) {
  const id4_pipeline_parameter_load_step_t* step =
      &load_steps[group.step_offset];
  if (step->target_slab_index >= load_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter load group %" PRIhsz " target slab %" PRIhsz
        " is outside load count %" PRIhsz,
        group_context.group_index, step->target_slab_index, load_count);
  }
  if (step->target_slab_index >= target_buffer_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter load group %" PRIhsz
                            " target slab %" PRIhsz
                            " is outside target buffer count %" PRIhsz,
                            group_context.group_index, step->target_slab_index,
                            target_buffer_count);
  }
  if (!target_buffers) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter load group target buffer table is required");
  }
  iree_hal_buffer_t* target_buffer = target_buffers[step->target_slab_index];
  if (!target_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter load group %" PRIhsz
                            " target slab %" PRIhsz " has no buffer",
                            group_context.group_index, step->target_slab_index);
  }
  const id4_pipeline_parameter_slab_load_t* load =
      &loads[step->target_slab_index];

  iree_status_t status = iree_ok_status();
  const iree_time_t start_time_ns = iree_time_now();
  if (id4_pipeline_parameter_load_group_is_encode(group)) {
    status = id4_pipeline_parameter_slab_submit_encode_run(
        options, encode_window_context, load, group.step_count, step,
        group_context, group.step_offset, stage_name, target_buffer,
        wait_semaphore_list, signal_semaphore_list, host_allocator);
  } else {
    status = id4_pipeline_parameter_slab_submit_gather(
        options, load, step, stage_name, group_context, group, target_buffer,
        wait_semaphore_list, signal_semaphore_list);
  }
  const iree_time_t end_time_ns = iree_time_now();
  return iree_status_join(
      status, id4_pipeline_parameter_slab_emit_load_group_submit_timing(
                  load, stage_name, group_context, group, start_time_ns,
                  end_time_ns, step->name, options->diagnostics_sink));
}

static iree_status_t id4_pipeline_parameter_slab_create_group_semaphores(
    iree_host_size_t group_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_allocator_t host_allocator, iree_hal_semaphore_t*** out_semaphores,
    iree_host_size_t* out_semaphore_count) {
  IREE_ASSERT_ARGUMENT(out_semaphores);
  IREE_ASSERT_ARGUMENT(out_semaphore_count);
  *out_semaphores = NULL;
  *out_semaphore_count = group_count;
  if (*out_semaphore_count == 0) return iree_ok_status();

  iree_hal_semaphore_t** semaphores = NULL;
  iree_status_t status =
      iree_allocator_malloc_array(host_allocator, *out_semaphore_count,
                                  sizeof(semaphores[0]), (void**)&semaphores);
  if (iree_status_is_ok(status)) {
    memset(semaphores, 0, *out_semaphore_count * sizeof(semaphores[0]));
  }
  for (iree_host_size_t step_index = 0, group_index = 0;
       step_index < load_step_count && iree_status_is_ok(status);
       ++group_index) {
    const id4_pipeline_parameter_load_group_t group =
        id4_pipeline_parameter_load_group_from_steps(
            step_index, load_step_count, load_steps);
    const id4_pipeline_parameter_slab_load_t* load =
        &loads[group.target_slab_index];
    status = iree_hal_semaphore_create(load->device, load->queue_affinity,
                                       /*initial_value=*/0,
                                       IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &semaphores[group_index]);
    step_index += group.step_count;
  }
  if (iree_status_is_ok(status)) {
    *out_semaphores = semaphores;
  } else {
    if (semaphores) {
      for (iree_host_size_t i = 0; i < *out_semaphore_count; ++i) {
        iree_hal_semaphore_release(semaphores[i]);
      }
    }
    iree_allocator_free(host_allocator, semaphores);
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_slab_signal_load_complete(
    const id4_pipeline_parameter_slab_load_t* loads,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  const id4_pipeline_parameter_slab_load_t* load =
      &loads[load_steps[0].target_slab_index];
  return iree_hal_device_queue_barrier(
      load->device, load->queue_affinity, wait_semaphore_list,
      signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t id4_pipeline_parameter_slab_set_allocate_buffers(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_count, iree_string_view_t stage_name,
    id4_pipeline_parameter_slab_set_t* slab_set) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < load_count && iree_status_is_ok(status);
       ++i) {
    status = id4_pipeline_parameter_slab_allocate_buffer(&loads[i],
                                                         &slab_set->buffers[i]);
    if (!iree_status_is_ok(status)) {
      status = iree_status_join(
          status,
          id4_pipeline_parameter_slab_emit_diagnostic(
              &loads[i], stage_name, IREE_SV("parameter_slab.load.error"),
              IREE_SV("parameter slab allocation failed"),
              iree_string_view_empty(), IREE_HOST_SIZE_MAX,
              options->diagnostics_sink));
    }
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_slab_set_create_load_groups(
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t* slab_set) {
  iree_host_size_t load_group_count = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_group_count(
      load_step_count, load_steps, &load_group_count));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_create_group_semaphores(
      load_group_count, loads, load_step_count, load_steps, host_allocator,
      &slab_set->load_group_semaphores, &slab_set->load_group_count));
  if (slab_set->load_group_count == 0) return iree_ok_status();
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, slab_set->load_group_count,
      sizeof(slab_set->load_groups[0]), (void**)&slab_set->load_groups);
  for (iree_host_size_t i = 0;
       i < slab_set->load_group_count && iree_status_is_ok(status); ++i) {
    status = id4_pipeline_parameter_load_group_at(load_step_count, load_steps,
                                                  i, &slab_set->load_groups[i]);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(host_allocator, slab_set->load_group_count,
                                    sizeof(slab_set->load_group_submitted[0]),
                                    (void**)&slab_set->load_group_submitted);
  }
  if (iree_status_is_ok(status)) {
    memset(
        slab_set->load_group_submitted, 0,
        slab_set->load_group_count * sizeof(slab_set->load_group_submitted[0]));
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_slab_set_submit_eager_load_groups(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_string_view_t stage_name,
    const id4_pipeline_parameter_slab_set_t* slab_set,
    iree_hal_semaphore_t** group_semaphores,
    iree_host_size_t group_semaphore_count, iree_allocator_t host_allocator) {
  if (load_step_count == 0) return iree_ok_status();
  if (group_semaphore_count == 0) {
    const id4_pipeline_parameter_load_group_t group =
        id4_pipeline_parameter_load_group_from_steps(
            /*step_index=*/0, load_step_count, load_steps);
    id4_pipeline_parameter_load_group_context_t group_context = {
        // This eager path does not retain plan-local group ordering.
        .group_index = IREE_HOST_SIZE_MAX,
        // This eager path does not track first consumer regions.
        .first_consumer_region_id = IREE_HOST_SIZE_MAX,
        // Eager prepare-time loading happens outside region issue.
        .submit_region_id = IREE_HOST_SIZE_MAX,
    };
    return id4_pipeline_parameter_slab_submit_load_group(
        options, /*encode_window_context=*/NULL, loads, slab_set->load_count,
        load_steps, group_context, group, stage_name, slab_set->count,
        slab_set->buffers, options->wait_semaphore_list,
        options->signal_semaphore_list, host_allocator);
  }

  iree_hal_semaphore_t** completion_semaphores = NULL;
  uint64_t* completion_payload_values = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, group_semaphore_count, sizeof(completion_semaphores[0]),
      (void**)&completion_semaphores);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, group_semaphore_count,
                                         sizeof(completion_payload_values[0]),
                                         (void**)&completion_payload_values);
  }

  iree_host_size_t completion_count = 0;
  iree_hal_semaphore_t* encode_wait_semaphore = NULL;
  uint64_t encode_wait_payload_value = 0;
  id4_pipeline_parameter_encode_window_context_t encode_window_context;
  memset(&encode_window_context, 0, sizeof(encode_window_context));
  bool encode_window_context_initialized = false;
  bool encode_window_context_finish_attempted = false;
  if (iree_status_is_ok(status) &&
      id4_pipeline_parameter_load_steps_require_encoder(load_step_count,
                                                        load_steps)) {
    status = id4_pipeline_parameter_encode_window_context_initialize(
        loads, slab_set->count, load_step_count, load_steps,
        options->encoder_staging_chunk_byte_capacity,
        /*emits_issue_window_diagnostic=*/false, host_allocator,
        &encode_window_context);
    encode_window_context_initialized = iree_status_is_ok(status);
  }
  for (iree_host_size_t step_index = 0, group_index = 0;
       step_index < load_step_count && iree_status_is_ok(status);
       ++group_index) {
    const id4_pipeline_parameter_load_group_t group =
        id4_pipeline_parameter_load_group_from_steps(
            step_index, load_step_count, load_steps);
    iree_hal_semaphore_list_t group_wait_list = options->wait_semaphore_list;
    if (id4_pipeline_parameter_load_group_is_encode(group) &&
        encode_wait_semaphore && !encode_window_context_initialized) {
      group_wait_list = id4_pipeline_parameter_one_semaphore_list(
          &encode_wait_semaphore, &encode_wait_payload_value);
    }

    uint64_t group_signal_payload_value = 1;
    iree_hal_semaphore_t* group_signal_semaphore =
        group_semaphores[group_index];
    iree_hal_semaphore_list_t group_signal_list =
        id4_pipeline_parameter_one_semaphore_list(&group_signal_semaphore,
                                                  &group_signal_payload_value);
    id4_pipeline_parameter_load_group_context_t group_context = {
        // Plan-local load group ordinal.
        .group_index = group_index,
        // Eager prepare-time loading does not track first consumer regions.
        .first_consumer_region_id = IREE_HOST_SIZE_MAX,
        // Eager prepare-time loading happens outside region issue.
        .submit_region_id = IREE_HOST_SIZE_MAX,
    };
    status = id4_pipeline_parameter_slab_submit_load_group(
        options,
        encode_window_context_initialized ? &encode_window_context : NULL,
        loads, slab_set->load_count, load_steps, group_context, group,
        stage_name, slab_set->count, slab_set->buffers, group_wait_list,
        group_signal_list, host_allocator);
    if (iree_status_is_ok(status)) {
      if (id4_pipeline_parameter_load_group_is_encode(group)) {
        encode_wait_semaphore = group_signal_semaphore;
        encode_wait_payload_value = group_signal_payload_value;
      } else {
        completion_semaphores[completion_count] = group_signal_semaphore;
        completion_payload_values[completion_count] =
            group_signal_payload_value;
        ++completion_count;
      }
      step_index += group.step_count;
    }
  }

  iree_hal_semaphore_list_t encode_cleanup_wait_list =
      iree_hal_semaphore_list_empty();
  if (iree_status_is_ok(status) && encode_window_context_initialized) {
    encode_window_context_finish_attempted = true;
    status = id4_pipeline_parameter_encode_window_context_finish(
        &encode_window_context, &encode_cleanup_wait_list);
  }
  if (iree_status_is_ok(status) && encode_wait_semaphore) {
    if (encode_window_context_initialized) {
      for (iree_host_size_t i = 0; i < encode_cleanup_wait_list.count; ++i) {
        completion_semaphores[completion_count] =
            encode_cleanup_wait_list.semaphores[i];
        completion_payload_values[completion_count] =
            encode_cleanup_wait_list.payload_values[i];
        ++completion_count;
      }
    } else {
      completion_semaphores[completion_count] = encode_wait_semaphore;
      completion_payload_values[completion_count] = encode_wait_payload_value;
      ++completion_count;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t wait_list =
        id4_pipeline_parameter_many_semaphore_list(
            completion_count, completion_semaphores, completion_payload_values);
    status = id4_pipeline_parameter_slab_signal_load_complete(
        loads, load_steps, wait_list, options->signal_semaphore_list);
  } else if (encode_window_context_initialized &&
             !encode_window_context_finish_attempted) {
    iree_hal_semaphore_list_t cleanup_wait_list =
        iree_hal_semaphore_list_empty();
    iree_status_t cleanup_status =
        id4_pipeline_parameter_encode_window_context_finish(
            &encode_window_context, &cleanup_wait_list);
    status = iree_status_join(status, cleanup_status);
  }
  id4_pipeline_parameter_encode_window_context_deinitialize(
      &encode_window_context);
  iree_allocator_free(host_allocator, completion_payload_values);
  iree_allocator_free(host_allocator, completion_semaphores);
  return status;
}

static iree_status_t id4_pipeline_parameter_slab_set_validate_load_inputs(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps) {
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_set_load_validate_options(
      options, load_step_count, load_steps));
  if (load_count != 0 && !loads) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab load array is required");
  }
  if (load_count == 0 && load_step_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter load steps require at least one parameter slab load");
  }
  if (load_count != 0 && load_step_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slabs require an explicit parameter load step schedule");
  }
  if (load_step_count != 0 && !load_steps) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter load step array is required");
  }
  for (iree_host_size_t i = 0; i < load_step_count; ++i) {
    IREE_RETURN_IF_ERROR(
        id4_pipeline_parameter_load_step_validate_against_loads(
            &load_steps[i], load_count, loads));
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_slab_set_prepare_internal(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    id4_pipeline_parameter_slab_prepare_flags_t flags,
    iree_string_view_t stage_name, iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set) {
  IREE_ASSERT_ARGUMENT(out_slab_set);
  *out_slab_set = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_set_validate_load_inputs(
      options, load_count, loads, load_step_count, load_steps));
  if (load_count != 0 && options->signal_semaphore_list.count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "deferred parameter slab loading does not signal prepare readiness");
  }

  id4_pipeline_parameter_slab_set_t* slab_set = NULL;
  iree_status_t status = id4_pipeline_parameter_slab_set_create_empty(
      load_count, host_allocator, &slab_set);
  if (iree_status_is_ok(status) &&
      iree_all_bits_set(
          flags,
          ID4_PIPELINE_PARAMETER_SLAB_PREPARE_FLAG_ALLOCATE_RESIDENT_BUFFERS)) {
    status = id4_pipeline_parameter_slab_set_allocate_buffers(
        options, loads, load_count, stage_name, slab_set);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_set_create_load_groups(
        loads, load_step_count, load_steps, host_allocator, slab_set);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_parameter_slab_set_capture_encoder_diagnostics(
        slab_set, options, load_step_count, load_steps);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_pipeline_parameter_slab_set_copy_loads(slab_set, load_count, loads);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_set_copy_load_steps(
        slab_set, load_step_count, load_steps);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_set_retain_load_context(
        slab_set, options, load_step_count, load_steps);
  }
  if (iree_status_is_ok(status)) {
    *out_slab_set = slab_set;
  } else {
    id4_pipeline_parameter_slab_set_release(slab_set);
  }
  return status;
}

iree_status_t id4_pipeline_parameter_slab_set_prepare(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_string_view_t stage_name, iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set) {
  return id4_pipeline_parameter_slab_set_prepare_internal(
      options, load_count, loads, load_step_count, load_steps,
      ID4_PIPELINE_PARAMETER_SLAB_PREPARE_FLAG_ALLOCATE_RESIDENT_BUFFERS,
      stage_name, host_allocator, out_slab_set);
}

iree_status_t id4_pipeline_parameter_slab_set_prepare_load_context(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_string_view_t stage_name, iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set) {
  return id4_pipeline_parameter_slab_set_prepare_internal(
      options, load_count, loads, load_step_count, load_steps,
      /*flags=*/0, stage_name, host_allocator, out_slab_set);
}

iree_status_t id4_pipeline_parameter_slab_set_create_uninitialized(
    iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_string_view_t stage_name,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set) {
  IREE_ASSERT_ARGUMENT(out_slab_set);
  *out_slab_set = NULL;
  if (load_count != 0 && !loads) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab load array is required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      diagnostics_sink, IREE_SV("parameter slab allocation")));
  for (iree_host_size_t i = 0; i < load_count; ++i) {
    if (!loads[i].slab) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter slab load %" PRIhsz " has no allocation plan", i);
    }
    if (!loads[i].device) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter slab load %" PRIhsz " has no target device", i);
    }
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_request_table_validate(
        loads[i].slab, loads[i].request_table));
  }

  id4_pipeline_parameter_slab_set_load_options_t allocation_options;
  memset(&allocation_options, 0, sizeof(allocation_options));
  allocation_options.structure_size = sizeof(allocation_options);
  allocation_options.diagnostics_sink = diagnostics_sink;

  id4_pipeline_parameter_slab_set_t* slab_set = NULL;
  iree_status_t status = id4_pipeline_parameter_slab_set_create_empty(
      load_count, host_allocator, &slab_set);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_set_allocate_buffers(
        &allocation_options, loads, load_count, stage_name, slab_set);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_pipeline_parameter_slab_set_copy_loads(slab_set, load_count, loads);
  }
  if (iree_status_is_ok(status)) {
    *out_slab_set = slab_set;
  } else {
    id4_pipeline_parameter_slab_set_release(slab_set);
  }
  return status;
}

iree_status_t id4_pipeline_parameter_slab_set_load(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_string_view_t stage_name, iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set) {
  IREE_ASSERT_ARGUMENT(out_slab_set);
  *out_slab_set = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_set_validate_load_inputs(
      options, load_count, loads, load_step_count, load_steps));
  if (load_count != 0 && options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slab loading requires a signal semaphore list");
  }

  id4_pipeline_parameter_slab_set_t* slab_set = NULL;
  iree_status_t status = id4_pipeline_parameter_slab_set_create_empty(
      load_count, host_allocator, &slab_set);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_set_allocate_buffers(
        options, loads, load_count, stage_name, slab_set);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_set_create_load_groups(
        loads, load_step_count, load_steps, host_allocator, slab_set);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_parameter_slab_set_capture_encoder_diagnostics(
        slab_set, options, load_step_count, load_steps);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_pipeline_parameter_slab_set_copy_loads(slab_set, load_count, loads);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_set_copy_load_steps(
        slab_set, load_step_count, load_steps);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_set_submit_eager_load_groups(
        options, loads, load_step_count, load_steps, stage_name, slab_set,
        slab_set->load_group_semaphores, slab_set->load_group_count,
        host_allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_slab_set = slab_set;
  } else {
    if (options->signal_semaphore_list.count != 0) {
      iree_hal_semaphore_list_fail(options->signal_semaphore_list,
                                   iree_status_clone(status));
    }
    id4_pipeline_parameter_slab_set_release(slab_set);
  }
  return status;
}

static iree_hal_semaphore_list_t
id4_pipeline_parameter_slab_set_prepare_wait_list(
    id4_pipeline_parameter_slab_set_t* slab_set) {
  return (iree_hal_semaphore_list_t){
      // Number of retained prepare wait edges.
      .count = slab_set->wait_count,
      // Retained prepare wait semaphores.
      .semaphores =
          slab_set->wait_count == 0 ? NULL : slab_set->wait_semaphores,
      // Payload values paired with retained prepare wait semaphores.
      .payload_values =
          slab_set->wait_count == 0 ? NULL : slab_set->wait_payload_values,
  };
}

static id4_pipeline_parameter_slab_set_load_options_t
id4_pipeline_parameter_slab_set_submit_options(
    id4_pipeline_parameter_slab_set_t* slab_set,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_pipeline_parameter_slab_set_load_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.provider = slab_set->provider;
  options.kernel_library = slab_set->kernel_library;
  options.kernel_cache = slab_set->kernel_cache;
  options.executable_cache = slab_set->executable_cache;
  options.command_buffer_mode = slab_set->command_buffer_mode;
  options.encoder_staging_memory_type = slab_set->encoder_staging_memory_type;
  options.encoder_staging_chunk_byte_capacity =
      slab_set->encoder_staging_chunk_byte_capacity;
  options.diagnostic_artifact_flags = slab_set->diagnostic_artifact_flags;
  options.diagnostics_sink = diagnostics_sink;
  return options;
}

iree_status_t id4_pipeline_parameter_slab_issue_context_create(
    id4_pipeline_parameter_slab_set_t* slab_set,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_issue_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;
  if (!slab_set) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab set is required");
  }
  if (!slab_set->provider) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "parameter slab set has no retained load context");
  }
  if (load_step_count != 0 && !load_steps) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter load step array is required");
  }

  id4_pipeline_parameter_slab_issue_context_t* context = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*context), (void**)&context);
  if (iree_status_is_ok(status)) {
    memset(context, 0, sizeof(*context));
    context->host_allocator = host_allocator;
    context->slab_set = slab_set;
    id4_pipeline_parameter_slab_set_retain(context->slab_set);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_encode_window_context_initialize(
        slab_set->loads, slab_set->load_count, load_step_count, load_steps,
        slab_set->encoder_staging_chunk_byte_capacity,
        /*emits_issue_window_diagnostic=*/true, host_allocator,
        &context->encode_windows);
  }
  if (iree_status_is_ok(status)) {
    *out_context = context;
  } else if (context) {
    id4_pipeline_parameter_slab_issue_context_destroy(context);
  }
  return status;
}

iree_status_t id4_pipeline_parameter_slab_issue_context_finish(
    id4_pipeline_parameter_slab_issue_context_t* context,
    iree_hal_semaphore_list_t* out_cleanup_wait_list) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_cleanup_wait_list);
  return id4_pipeline_parameter_encode_window_context_finish(
      &context->encode_windows, out_cleanup_wait_list);
}

void id4_pipeline_parameter_slab_issue_context_release(
    id4_pipeline_parameter_slab_issue_context_t* context) {
  if (!context) return;
  id4_pipeline_parameter_slab_issue_context_destroy(context);
}

iree_status_t id4_pipeline_parameter_slab_issue_context_submit_load_group(
    id4_pipeline_parameter_slab_issue_context_t* context,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    id4_pipeline_parameter_load_group_context_t group_context,
    iree_string_view_t stage_name,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  if (!context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab issue context is required");
  }
  id4_pipeline_parameter_slab_set_t* slab_set = context->slab_set;
  if (!slab_set) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter slab issue context has no retained slab set");
  }
  if (group_context.group_index >= slab_set->load_group_count ||
      !slab_set->load_group_semaphores ||
      !slab_set->load_group_semaphores[group_context.group_index]) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter load group %" PRIhsz " is outside load group count %" PRIhsz,
        group_context.group_index, slab_set->load_group_count);
  }
  uint64_t target_signal_payload_value = 1;
  iree_hal_semaphore_t* target_signal_semaphore =
      slab_set->load_group_semaphores[group_context.group_index];
  iree_hal_semaphore_list_t target_signal_list =
      id4_pipeline_parameter_one_semaphore_list(&target_signal_semaphore,
                                                &target_signal_payload_value);
  return id4_pipeline_parameter_slab_issue_context_submit_load_group_to_buffers(
      context, slab_set->load_count, slab_set->loads, load_step_count,
      load_steps, group_context.group_index, slab_set->count, slab_set->buffers,
      iree_hal_semaphore_list_empty(), target_signal_list, group_context,
      stage_name, diagnostics_sink);
}

iree_status_t
id4_pipeline_parameter_slab_issue_context_submit_load_group_to_buffers(
    id4_pipeline_parameter_slab_issue_context_t* context,
    iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_host_size_t load_group_index, iree_host_size_t target_buffer_count,
    iree_hal_buffer_t* const* buffers,
    iree_hal_semaphore_list_t target_wait_semaphore_list,
    iree_hal_semaphore_list_t target_signal_semaphore_list,
    id4_pipeline_parameter_load_group_context_t group_context,
    iree_string_view_t stage_name,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  if (!context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab issue context is required");
  }
  if (context->encode_windows.finished) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "parameter slab issue context is already finished");
  }
  id4_pipeline_parameter_slab_set_t* slab_set = context->slab_set;
  if (!slab_set) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter slab issue context has no retained slab set");
  }
  if (load_count != 0 && !loads) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab load array is required");
  }
  if (load_count == 0 && load_step_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter load steps require at least one parameter slab load");
  }
  if (load_step_count != 0 && !load_steps) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter load step array is required");
  }
  if (target_buffer_count != load_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter target buffer count %" PRIhsz
                            " must match load count %" PRIhsz,
                            target_buffer_count, load_count);
  }
  if (target_buffer_count != 0 && !buffers) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter target buffer table is required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_validate_semaphore_list(
      target_wait_semaphore_list, IREE_SV("parameter target wait")));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_validate_semaphore_list(
      target_signal_semaphore_list, IREE_SV("parameter target signal")));
  if (target_signal_semaphore_list.count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter load group requires exactly one target signal semaphore");
  }
  const bool uses_retained_targets =
      loads == slab_set->loads && load_count == slab_set->load_count &&
      buffers == slab_set->buffers && target_buffer_count == slab_set->count;
  if (uses_retained_targets &&
      group_context.group_index >= slab_set->load_group_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter load group %" PRIhsz " is outside load group count %" PRIhsz,
        group_context.group_index, slab_set->load_group_count);
  }
  id4_pipeline_parameter_load_group_t group;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_group_at(
      load_step_count, load_steps, load_group_index, &group));
  for (iree_host_size_t i = 0; i < group.step_count; ++i) {
    IREE_RETURN_IF_ERROR(
        id4_pipeline_parameter_load_step_validate_against_loads(
            &load_steps[group.step_offset + i], load_count, loads));
  }
  if (group.target_slab_index >= target_buffer_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter load-step group %" PRIhsz " target slab %" PRIhsz
        " is outside target buffer count %" PRIhsz,
        load_group_index, group.target_slab_index, target_buffer_count);
  }
  if (!buffers[group.target_slab_index]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter load-step group %" PRIhsz
                            " target slab %" PRIhsz " has no buffer",
                            load_group_index, group.target_slab_index);
  }
  if (uses_retained_targets && !slab_set->load_group_submitted) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter slab set has no load group submission state");
  }
  if (uses_retained_targets &&
      slab_set->load_group_submitted[group_context.group_index]) {
    return iree_ok_status();
  }

  id4_pipeline_parameter_slab_set_load_options_t submit_options =
      id4_pipeline_parameter_slab_set_submit_options(slab_set,
                                                     diagnostics_sink);
  const iree_hal_semaphore_list_t prepare_wait_list =
      id4_pipeline_parameter_slab_set_prepare_wait_list(slab_set);
  iree_hal_semaphore_list_t wait_list = prepare_wait_list;
  iree_hal_semaphore_t** combined_wait_semaphores = NULL;
  uint64_t* combined_wait_payload_values = NULL;
  iree_host_size_t combined_wait_count = 0;
  if (!iree_host_size_checked_add(prepare_wait_list.count,
                                  target_wait_semaphore_list.count,
                                  &combined_wait_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter load group wait count overflows");
  }
  if (combined_wait_count != 0) {
    combined_wait_semaphores = (iree_hal_semaphore_t**)iree_alloca(
        combined_wait_count * sizeof(combined_wait_semaphores[0]));
    combined_wait_payload_values = (uint64_t*)iree_alloca(
        combined_wait_count * sizeof(combined_wait_payload_values[0]));
    for (iree_host_size_t i = 0; i < prepare_wait_list.count; ++i) {
      combined_wait_semaphores[i] = prepare_wait_list.semaphores[i];
      combined_wait_payload_values[i] = prepare_wait_list.payload_values[i];
    }
    for (iree_host_size_t i = 0; i < target_wait_semaphore_list.count; ++i) {
      const iree_host_size_t target_index = prepare_wait_list.count + i;
      combined_wait_semaphores[target_index] =
          target_wait_semaphore_list.semaphores[i];
      combined_wait_payload_values[target_index] =
          target_wait_semaphore_list.payload_values[i];
    }
    wait_list = id4_pipeline_parameter_many_semaphore_list(
        combined_wait_count, combined_wait_semaphores,
        combined_wait_payload_values);
  }
  if (uses_retained_targets &&
      id4_pipeline_parameter_load_group_is_encode(group) &&
      slab_set->encode_tail_semaphore) {
    iree_host_size_t encode_wait_count = 0;
    if (!iree_host_size_checked_add(wait_list.count, 1, &encode_wait_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter encoded load group wait count "
                              "overflows");
    }
    iree_hal_semaphore_t** encode_wait_semaphores =
        (iree_hal_semaphore_t**)iree_alloca(encode_wait_count *
                                            sizeof(encode_wait_semaphores[0]));
    uint64_t* encode_wait_payload_values = (uint64_t*)iree_alloca(
        encode_wait_count * sizeof(encode_wait_payload_values[0]));
    for (iree_host_size_t i = 0; i < wait_list.count; ++i) {
      encode_wait_semaphores[i] = wait_list.semaphores[i];
      encode_wait_payload_values[i] = wait_list.payload_values[i];
    }
    encode_wait_semaphores[wait_list.count] = slab_set->encode_tail_semaphore;
    encode_wait_payload_values[wait_list.count] =
        slab_set->encode_tail_payload_value;
    wait_list = id4_pipeline_parameter_many_semaphore_list(
        encode_wait_count, encode_wait_semaphores, encode_wait_payload_values);
  }

  const bool matches_retained_loads = uses_retained_targets;
  id4_pipeline_parameter_encode_window_context_t* encode_window_context =
      matches_retained_loads ? &context->encode_windows : NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_submit_load_group(
      &submit_options, encode_window_context, loads, load_count, load_steps,
      group_context, group, stage_name, target_buffer_count, buffers, wait_list,
      target_signal_semaphore_list, slab_set->host_allocator));
  if (uses_retained_targets) {
    slab_set->load_group_submitted[group_context.group_index] = true;
    if (id4_pipeline_parameter_load_group_is_encode(group)) {
      slab_set->encode_tail_semaphore =
          target_signal_semaphore_list.semaphores[0];
      slab_set->encode_tail_payload_value =
          target_signal_semaphore_list.payload_values[0];
    }
  }
  return iree_ok_status();
}

void id4_pipeline_parameter_slab_set_retain(
    id4_pipeline_parameter_slab_set_t* slab_set) {
  if (!slab_set) return;
  iree_atomic_ref_count_inc(&slab_set->ref_count);
}

void id4_pipeline_parameter_slab_set_release(
    id4_pipeline_parameter_slab_set_t* slab_set) {
  if (slab_set && iree_atomic_ref_count_dec(&slab_set->ref_count) == 1) {
    id4_pipeline_parameter_slab_set_destroy(slab_set);
  }
}

iree_host_size_t id4_pipeline_parameter_slab_set_count(
    const id4_pipeline_parameter_slab_set_t* slab_set) {
  return slab_set ? slab_set->count : 0;
}

iree_hal_buffer_t* id4_pipeline_parameter_slab_set_buffer_at(
    const id4_pipeline_parameter_slab_set_t* slab_set, iree_host_size_t index) {
  if (!slab_set || index >= slab_set->count || !slab_set->buffers) return NULL;
  return slab_set->buffers[index];
}

const id4_pipeline_parameter_slab_plan_t*
id4_pipeline_parameter_slab_set_plan_at(
    const id4_pipeline_parameter_slab_set_t* slab_set, iree_host_size_t index) {
  if (!slab_set || index >= slab_set->load_count || !slab_set->slab_plans) {
    return NULL;
  }
  return &slab_set->slab_plans[index];
}

const id4_pipeline_parameter_request_table_t*
id4_pipeline_parameter_slab_set_request_table_at(
    const id4_pipeline_parameter_slab_set_t* slab_set, iree_host_size_t index) {
  if (!slab_set || index >= slab_set->load_count || !slab_set->request_tables) {
    return NULL;
  }
  return &slab_set->request_tables[index];
}

iree_host_size_t id4_pipeline_parameter_slab_set_load_group_count(
    const id4_pipeline_parameter_slab_set_t* slab_set) {
  return slab_set ? slab_set->load_group_count : 0;
}

iree_status_t id4_pipeline_parameter_slab_set_load_group_at(
    const id4_pipeline_parameter_slab_set_t* slab_set, iree_host_size_t index,
    id4_pipeline_parameter_load_group_t* out_group) {
  IREE_ASSERT_ARGUMENT(out_group);
  memset(out_group, 0, sizeof(*out_group));
  if (!slab_set) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab set is required");
  }
  if (index >= slab_set->load_group_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter load group %" PRIhsz
                            " is outside load group count %" PRIhsz,
                            index, slab_set->load_group_count);
  }
  if (!slab_set->load_groups) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter slab set has no retained load group descriptors");
  }
  *out_group = slab_set->load_groups[index];
  return iree_ok_status();
}

bool id4_pipeline_parameter_slab_set_has_deferred_load_context(
    const id4_pipeline_parameter_slab_set_t* slab_set) {
  return slab_set && slab_set->provider;
}

bool id4_pipeline_parameter_slab_set_has_resident_buffers(
    const id4_pipeline_parameter_slab_set_t* slab_set) {
  if (!slab_set || slab_set->count == 0 || !slab_set->buffers) return false;
  for (iree_host_size_t i = 0; i < slab_set->count; ++i) {
    if (!slab_set->buffers[i]) return false;
  }
  return true;
}

iree_status_t id4_pipeline_parameter_slab_set_load_group_ready_at(
    const id4_pipeline_parameter_slab_set_t* slab_set, iree_host_size_t index,
    iree_hal_semaphore_t** out_semaphore, uint64_t* out_payload_value) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_ASSERT_ARGUMENT(out_payload_value);
  *out_semaphore = NULL;
  *out_payload_value = 0;
  if (!slab_set) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab set is required");
  }
  if (index >= slab_set->load_group_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter load group %" PRIhsz
                            " is outside load group count %" PRIhsz,
                            index, slab_set->load_group_count);
  }
  if (!slab_set->load_group_semaphores ||
      !slab_set->load_group_semaphores[index]) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter load group %" PRIhsz " has no readiness semaphore", index);
  }
  *out_semaphore = slab_set->load_group_semaphores[index];
  *out_payload_value = 1;
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_slab_set_check_load_group_failures(
    const id4_pipeline_parameter_slab_set_t* slab_set,
    iree_string_view_t stage_name,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  if (!slab_set) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab set is required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      diagnostics_sink, IREE_SV("parameter load failure check")));
  if (slab_set->load_group_count == 0) return iree_ok_status();
  if (!slab_set->load_group_semaphores) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter slab set has no retained load group semaphores");
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < slab_set->load_group_count; ++i) {
    iree_hal_semaphore_t* semaphore = slab_set->load_group_semaphores[i];
    if (!semaphore) {
      status = iree_status_join(
          status, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                   "parameter load group %" PRIhsz
                                   " has no retained readiness semaphore",
                                   i));
      continue;
    }
    uint64_t current_payload_value = 0;
    iree_status_t query_status =
        iree_hal_semaphore_query(semaphore, &current_payload_value);
    if (iree_status_is_ok(query_status)) continue;
    status = iree_status_join(status, query_status);
    status = iree_status_join(
        status,
        id4_pipeline_parameter_slab_describe_failed_load_group(
            slab_set, i, stage_name, current_payload_value, diagnostics_sink));
  }
  return status;
}
