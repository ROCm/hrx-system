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
  ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_STEP_CAPACITY = 32,
  ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT = 2,
  ID4_PIPELINE_PARAMETER_ENCODER_MAX_SOURCE_COUNT = 2,
  ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY =
      ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_STEP_CAPACITY *
      ID4_PIPELINE_PARAMETER_ENCODER_MAX_SOURCE_COUNT,
  ID4_PIPELINE_PARAMETER_ENCODER_CLEANUP_WAIT_CAPACITY =
      ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT *
      ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY,
  ID4_PIPELINE_PARAMETER_ENCODER_LINEAR_CONFIG_COUNT = 2,
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

struct id4_pipeline_parameter_slab_set_t {
  // Reference count for shared slab set ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for slab set storage.
  iree_allocator_t host_allocator;
  // Number of loaded slab buffers.
  iree_host_size_t count;
  // Loaded slab buffers retained by this set.
  iree_hal_buffer_t** buffers;
};

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
  if (slab->request_count != 0 && !slab->requests) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab request array is required");
  }
  for (iree_host_size_t i = 0; i < slab->request_count; ++i) {
    const id4_pipeline_parameter_request_t* request = &slab->requests[i];
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
    const id4_pipeline_parameter_slab_plan_t* slab) {
  if (!id4_pipeline_parameter_load_step_is_encode(step))
    return iree_ok_status();
  const id4_pipeline_parameter_request_t* request =
      &slab->requests[step->request_offset];
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
        "parameter encode load step '%.*s' target request must cover dense "
        "BF16 storage",
        (int)step->name.size, step->name.data);
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_load_step_validate(
    const id4_pipeline_parameter_load_step_t* step, iree_host_size_t slab_count,
    const id4_pipeline_parameter_slab_plan_t* slabs) {
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
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_step_validate_sources(step));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_step_validate_request_range(
      step, slab->request_count));
  return id4_pipeline_parameter_load_step_validate_target_request(step, slab);
}

iree_status_t id4_pipeline_parameter_slab_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(user_data);
  IREE_ASSERT_ARGUMENT(out_key);
  IREE_ASSERT_ARGUMENT(out_span);
  id4_pipeline_parameter_slab_enumerator_state_t* state =
      (id4_pipeline_parameter_slab_enumerator_state_t*)user_data;
  if (!state->slab || i >= state->request_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter request index %" PRIhsz
                            " is outside enumerated request count %" PRIhsz,
                            i, state->request_count);
  }
  const iree_host_size_t request_index = state->request_indices
                                             ? state->request_indices[i]
                                             : state->request_offset + i;
  if (request_index >= state->slab->request_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter request index %" PRIhsz
                            " is outside slab request count %" PRIhsz,
                            request_index, state->slab->request_count);
  }
  const id4_pipeline_parameter_request_t* request =
      &state->slab->requests[request_index];
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
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_step_validate_sources(step));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_load_step_validate_request_range(
      step, load->slab->request_count));
  return id4_pipeline_parameter_load_step_validate_target_request(step,
                                                                  load->slab);
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
      // Total slab byte length.
      .slab_byte_length = slab->byte_length,
      // Required slab base alignment.
      .slab_alignment = slab->alignment,
      // Number of requests in the slab.
      .request_count = slab->request_count,
  };
  if (request_index < slab->request_count) {
    const id4_pipeline_parameter_request_t* request =
        &slab->requests[request_index];
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

static iree_status_t id4_pipeline_parameter_submit_source_gather_batch(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* load,
    iree_string_view_t source_scope, iree_host_size_t source_count,
    const id4_pipeline_parameter_encode_chunk_source_t* sources,
    iree_hal_buffer_t* staging_buffer,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  if (!iree_io_parameter_provider_query_support(options->provider,
                                                source_scope)) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "parameter provider does not support source scope '%.*s'",
        (int)source_scope.size, source_scope.data);
  }
  id4_pipeline_parameter_load_source_enumerator_state_t enumerator_state = {
      // Number of source descriptors gathered by this request.
      .source_count = source_count,
      // Source descriptors gathered by this request.
      .sources = sources,
  };
  return iree_io_parameter_provider_gather(
      options->provider, load->device, load->queue_affinity,
      wait_semaphore_list, signal_semaphore_list, source_scope, staging_buffer,
      source_count,
      id4_pipeline_parameter_load_source_enumerator(&enumerator_state));
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

typedef struct id4_pipeline_parameter_encode_run_statistics_t {
  // Byte length required by each bounded staging slot.
  iree_device_size_t staging_slot_byte_length;
  // Number of staging chunks planned for the encode run.
  iree_host_size_t staging_chunk_count;
  // Number of logical provider source tensors gathered into staging.
  iree_host_size_t logical_source_count;
  // Number of provider gather batches submitted by the encode run.
  iree_host_size_t source_gather_batch_count;
  // Total provider source bytes gathered by the encode run.
  iree_device_size_t source_byte_length;
  // Total final target slab bytes populated by the encode run.
  iree_device_size_t target_byte_length;
  // Number of encoder dispatches recorded by the encode run.
  iree_host_size_t encoder_dispatch_count;
} id4_pipeline_parameter_encode_run_statistics_t;

static iree_status_t id4_pipeline_parameter_encode_run_collect_statistics(
    const id4_pipeline_parameter_slab_load_t* load,
    const id4_pipeline_parameter_load_step_t* steps, iree_host_size_t count,
    iree_device_size_t chunk_byte_capacity,
    id4_pipeline_parameter_encode_run_statistics_t* out_statistics) {
  memset(out_statistics, 0, sizeof(*out_statistics));
  out_statistics->encoder_dispatch_count = count;
  for (iree_host_size_t i = 0; i < count;) {
    id4_pipeline_parameter_encode_step_staging_layout_t
        layouts[ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_STEP_CAPACITY];
    iree_host_size_t chunk_step_count = 0;
    iree_device_size_t chunk_byte_length = 0;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_encode_plan_chunk(
        count - i, &steps[i], chunk_byte_capacity, layouts, &chunk_step_count,
        &chunk_byte_length));
    if (chunk_step_count == 0) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "parameter encoder could not plan a staging chunk");
    }
    if (chunk_byte_length > out_statistics->staging_slot_byte_length) {
      out_statistics->staging_slot_byte_length = chunk_byte_length;
    }
    id4_pipeline_parameter_encode_chunk_source_t
        grouped_sources[ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY] =
            {0};
    id4_pipeline_parameter_encode_source_batch_t
        source_batches[ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY] = {
            0};
    iree_host_size_t source_batch_count = 0;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_encode_group_chunk_sources(
        chunk_step_count, &steps[i], layouts, /*slot_base_offset=*/0,
        grouped_sources, source_batches, &source_batch_count));
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_host_size(
        1, &out_statistics->staging_chunk_count, "staging chunk"));
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_host_size(
        source_batch_count, &out_statistics->source_gather_batch_count,
        "source gather batch"));
    for (iree_host_size_t j = 0; j < chunk_step_count; ++j) {
      const id4_pipeline_parameter_load_step_t* step = &steps[i + j];
      IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_host_size(
          step->source_count, &out_statistics->logical_source_count,
          "logical source"));
      for (iree_host_size_t k = 0; k < step->source_count; ++k) {
        IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_device_size(
            step->sources[k].byte_length, &out_statistics->source_byte_length,
            "source"));
      }
      const id4_pipeline_parameter_request_t* target_request =
          &load->slab->requests[step->request_offset];
      IREE_RETURN_IF_ERROR(id4_pipeline_parameter_add_device_size(
          target_request->span.length, &out_statistics->target_byte_length,
          "target"));
    }
    i += chunk_step_count;
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_slab_emit_encode_window_diagnostic(
    const id4_pipeline_parameter_slab_load_t* load,
    iree_string_view_t stage_name, iree_host_size_t load_step_offset,
    iree_host_size_t load_step_count, iree_device_size_t staging_byte_length,
    const id4_pipeline_parameter_encode_run_statistics_t* statistics,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_pipeline_parameter_slab_diagnostic_t parameter_slab =
      id4_pipeline_parameter_slab_make_diagnostic(load, load->slab->scope,
                                                  IREE_HOST_SIZE_MAX);
  id4_pipeline_parameter_load_diagnostic_t parameter_load = {
      // Plan-local slab index populated by the loading window.
      .slab_index = load->slab_index,
      // First load-step ordinal represented by the window.
      .load_step_offset = load_step_offset,
      // Number of load steps represented by the window.
      .load_step_count = load_step_count,
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

static iree_status_t id4_pipeline_parameter_slab_submit_encode_run(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* load, iree_host_size_t step_count,
    const id4_pipeline_parameter_load_step_t* steps,
    iree_host_size_t load_step_offset, iree_string_view_t stage_name,
    iree_hal_buffer_t* target_buffer,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  id4_pipeline_parameter_encode_run_statistics_t statistics;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_encode_run_collect_statistics(
      load, steps, step_count, options->encoder_staging_chunk_byte_capacity,
      &statistics));
  const iree_device_size_t staging_slot_byte_length =
      statistics.staging_slot_byte_length;
  iree_device_size_t staging_byte_length = 0;
  if (!iree_device_size_checked_mul(
          staging_slot_byte_length,
          ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT,
          &staging_byte_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter encoder staging allocation byte length overflows");
  }
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_slab_emit_encode_window_diagnostic(
          load, stage_name, load_step_offset, step_count, staging_byte_length,
          &statistics, options->diagnostics_sink));

  iree_hal_semaphore_t* run_semaphore = NULL;
  id4_pipeline_parameter_encode_staging_slot_t
      slots[ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT] = {0};
  iree_hal_semaphore_t* cleanup_semaphore = NULL;
  iree_hal_buffer_t* staging_buffer = NULL;
  iree_status_t status = id4_pipeline_parameter_create_semaphore(
      load->device, load->queue_affinity, &run_semaphore);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_create_staging_slot_semaphores(
        load->device, load->queue_affinity, slots);
  }
  if (iree_status_is_ok(status)) {
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
  staging_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  staging_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  staging_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  staging_params.queue_affinity = load->queue_affinity;
  staging_params.min_alignment = 16;

  bool staging_alloca_submitted = false;
  iree_hal_semaphore_t* cleanup_wait_semaphores
      [ID4_PIPELINE_PARAMETER_ENCODER_CLEANUP_WAIT_CAPACITY] = {0};
  uint64_t cleanup_wait_payload_values
      [ID4_PIPELINE_PARAMETER_ENCODER_CLEANUP_WAIT_CAPACITY] = {0};
  iree_host_size_t cleanup_wait_count = 0;
  if (iree_status_is_ok(status)) {
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
  iree_host_size_t chunk_ordinal = 0;
  while (step_offset < step_count && iree_status_is_ok(status)) {
    id4_pipeline_parameter_encode_step_staging_layout_t
        layouts[ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_STEP_CAPACITY];
    iree_host_size_t chunk_step_count = 0;
    iree_device_size_t chunk_byte_length = 0;
    status = id4_pipeline_parameter_encode_plan_chunk(
        step_count - step_offset, &steps[step_offset],
        options->encoder_staging_chunk_byte_capacity, layouts,
        &chunk_step_count, &chunk_byte_length);
    if (!iree_status_is_ok(status)) break;
    if (chunk_step_count == 0) {
      status =
          iree_make_status(IREE_STATUS_INTERNAL,
                           "parameter encoder could not plan a staging chunk");
      break;
    }
    if (chunk_byte_length > staging_slot_byte_length) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "parameter encoder chunk byte length exceeds staging slot");
      break;
    }

    const iree_host_size_t slot =
        chunk_ordinal % ID4_PIPELINE_PARAMETER_ENCODER_STAGING_SLOT_COUNT;
    id4_pipeline_parameter_encode_staging_slot_t* slot_state = &slots[slot];
    iree_device_size_t slot_base_offset = 0;
    if (!iree_device_size_checked_mul(staging_slot_byte_length, slot,
                                      &slot_base_offset)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "parameter encoder staging slot byte offset overflows");
      break;
    }
    iree_hal_semaphore_t* source_wait_semaphore = NULL;
    uint64_t source_wait_payload_value = 0;
    if (slot_state->encode_payload_value == 0) {
      source_wait_semaphore = run_semaphore;
      source_wait_payload_value = staging_ready_payload_value;
    } else {
      source_wait_semaphore = slot_state->encode_semaphore;
      source_wait_payload_value = slot_state->encode_payload_value;
    }
    iree_hal_semaphore_list_t source_wait_list =
        id4_pipeline_parameter_one_semaphore_list(&source_wait_semaphore,
                                                  &source_wait_payload_value);
    id4_pipeline_parameter_encode_chunk_source_t
        grouped_sources[ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY] =
            {0};
    id4_pipeline_parameter_encode_source_batch_t
        source_batches[ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY] = {
            0};
    iree_host_size_t source_batch_count = 0;
    status = id4_pipeline_parameter_encode_group_chunk_sources(
        chunk_step_count, &steps[step_offset], layouts, slot_base_offset,
        grouped_sources, source_batches, &source_batch_count);
    if (!iree_status_is_ok(status)) break;
    uint64_t source_signal_payload_values
        [ID4_PIPELINE_PARAMETER_ENCODER_CHUNK_SOURCE_CAPACITY] = {0};
    const uint64_t source_payload_value = slot_state->source_payload_value + 1;
    for (iree_host_size_t i = 0; i < source_batch_count; ++i) {
      source_signal_payload_values[i] = source_payload_value;
    }

    for (iree_host_size_t batch_index = 0;
         batch_index < source_batch_count && iree_status_is_ok(status);
         ++batch_index) {
      const id4_pipeline_parameter_encode_source_batch_t* batch =
          &source_batches[batch_index];
      iree_hal_semaphore_list_t source_signal_list =
          id4_pipeline_parameter_one_semaphore_list(
              &slot_state->source_semaphores[batch_index],
              &source_signal_payload_values[batch_index]);
      status = id4_pipeline_parameter_submit_source_gather_batch(
          options, load, batch->source_scope, batch->source_count,
          &grouped_sources[batch->source_offset], staging_buffer,
          source_wait_list, source_signal_list);
      if (iree_status_is_ok(status)) {
        if (batch_index == 0) slot_state->cleanup_wait_count = 0;
        slot_state->cleanup_wait_semaphores[batch_index] =
            slot_state->source_semaphores[batch_index];
        slot_state->cleanup_wait_payload_values[batch_index] =
            source_signal_payload_values[batch_index];
        slot_state->cleanup_wait_count = batch_index + 1;
        cleanup_wait_count =
            id4_pipeline_parameter_encode_cleanup_waits_flatten(
                slots, cleanup_wait_semaphores, cleanup_wait_payload_values);
      }
    }
    if (iree_status_is_ok(status)) {
      slot_state->source_payload_value = source_payload_value;
    }

    iree_hal_command_buffer_t* command_buffer = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_create(
          load->device,
          options->command_buffer_mode | IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
          IREE_HAL_COMMAND_CATEGORY_DISPATCH, load->queue_affinity,
          /*binding_capacity=*/0, &command_buffer);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_begin(command_buffer);
    }
    for (iree_host_size_t i = 0;
         i < chunk_step_count && iree_status_is_ok(status); ++i) {
      const id4_pipeline_parameter_load_step_t* step = &steps[step_offset + i];
      const id4_pipeline_parameter_request_t* target_request =
          &load->slab->requests[step->request_offset];
      const id4_pipeline_parameter_load_source_t* weight_source =
          &step->sources[0];

      id4_pipeline_kernel_executable_t* executable = NULL;
      iree_hal_executable_function_t function =
          iree_hal_executable_function_invalid();
      status = id4_pipeline_parameter_prepare_encoder(options, load, step,
                                                      &executable, &function);
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
              staging_buffer, slot_base_offset + layouts[i].source_offsets[j],
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
            id4_pipeline_kernel_executable_hal_executable(executable), function,
            id4_pipeline_kernel_executable_dispatch_config(executable),
            iree_const_byte_span_empty(), binding_list,
            IREE_HAL_DISPATCH_FLAG_NONE);
      }
      id4_pipeline_kernel_executable_release(executable);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_end(command_buffer);
    }
    uint64_t encode_payload_value = slot_state->encode_payload_value + 1;
    iree_hal_semaphore_list_t encode_signal_list =
        id4_pipeline_parameter_one_semaphore_list(&slot_state->encode_semaphore,
                                                  &encode_payload_value);
    if (iree_status_is_ok(status)) {
      iree_hal_semaphore_list_t encode_wait_list =
          id4_pipeline_parameter_many_semaphore_list(
              source_batch_count, slot_state->source_semaphores,
              source_signal_payload_values);
      status = iree_hal_device_queue_execute(
          load->device, load->queue_affinity, encode_wait_list,
          encode_signal_list, command_buffer,
          iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE);
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
        step_offset += chunk_step_count;
        ++chunk_ordinal;
      }
    }
    iree_hal_command_buffer_release(command_buffer);
  }

  if (iree_status_is_ok(status)) {
    cleanup_wait_count = id4_pipeline_parameter_encode_cleanup_waits_flatten(
        slots, cleanup_wait_semaphores, cleanup_wait_payload_values);
    if (cleanup_wait_count == 0) {
      cleanup_wait_semaphores[0] = run_semaphore;
      cleanup_wait_payload_values[0] = staging_ready_payload_value;
      cleanup_wait_count = 1;
    }
    iree_hal_semaphore_list_t run_complete_wait_list =
        id4_pipeline_parameter_encode_cleanup_wait_list(
            cleanup_wait_count, cleanup_wait_semaphores,
            cleanup_wait_payload_values);
    status = iree_hal_device_queue_dealloca(
        load->device, load->queue_affinity, run_complete_wait_list,
        signal_semaphore_list, staging_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
  } else if (staging_alloca_submitted && staging_buffer) {
    iree_hal_semaphore_list_t cleanup_wait_list =
        id4_pipeline_parameter_encode_cleanup_wait_list(
            cleanup_wait_count, cleanup_wait_semaphores,
            cleanup_wait_payload_values);
    iree_status_t cleanup_status = iree_hal_device_queue_dealloca(
        load->device, load->queue_affinity, cleanup_wait_list,
        cleanup_signal_list, staging_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
    if (iree_status_is_ok(cleanup_status)) {
      cleanup_status = iree_hal_semaphore_wait(
          cleanup_semaphore, cleanup_payload_value, iree_infinite_timeout(),
          IREE_ASYNC_WAIT_FLAG_NONE);
    }
    status = iree_status_join(status, cleanup_status);
  }

  iree_hal_buffer_release(staging_buffer);
  iree_hal_semaphore_release(cleanup_semaphore);
  id4_pipeline_parameter_release_staging_slot_semaphores(slots);
  iree_hal_semaphore_release(run_semaphore);
  return status;
}

static iree_host_size_t id4_pipeline_parameter_slab_encode_run_count(
    iree_host_size_t start_index, iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps) {
  const iree_host_size_t target_slab_index =
      load_steps[start_index].target_slab_index;
  iree_host_size_t end_index = start_index;
  while (end_index < load_step_count &&
         id4_pipeline_parameter_load_step_is_encode(&load_steps[end_index]) &&
         load_steps[end_index].target_slab_index == target_slab_index) {
    ++end_index;
  }
  return end_index - start_index;
}

static iree_status_t id4_pipeline_parameter_slab_submit_gather(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* load,
    const id4_pipeline_parameter_load_step_t* step,
    iree_string_view_t stage_name, iree_hal_buffer_t* target_buffer,
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
      // Slab plan supplying request keys and spans.
      .slab = load->slab,
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

typedef struct id4_pipeline_parameter_load_group_t {
  // First load-step ordinal represented by this submitted group.
  iree_host_size_t step_offset;
  // Number of load steps represented by this submitted group.
  iree_host_size_t step_count;
  // True when the group uses bounded encoder staging.
  bool is_encode;
} id4_pipeline_parameter_load_group_t;

static id4_pipeline_parameter_load_group_t
id4_pipeline_parameter_load_group_from_steps(
    iree_host_size_t step_index, iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps) {
  id4_pipeline_parameter_load_group_t group = {
      // First load-step ordinal represented by this group.
      .step_offset = step_index,
      // One direct gather by default.
      .step_count = 1,
      // Updated below for encoded groups.
      .is_encode = false,
  };
  if (id4_pipeline_parameter_load_step_is_encode(&load_steps[step_index])) {
    group.step_count = id4_pipeline_parameter_slab_encode_run_count(
        step_index, load_step_count, load_steps);
    group.is_encode = true;
  }
  return group;
}

static iree_host_size_t id4_pipeline_parameter_load_group_count(
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps) {
  iree_host_size_t group_count = 0;
  for (iree_host_size_t i = 0; i < load_step_count;) {
    const id4_pipeline_parameter_load_group_t group =
        id4_pipeline_parameter_load_group_from_steps(i, load_step_count,
                                                     load_steps);
    i += group.step_count;
    ++group_count;
  }
  return group_count;
}

static iree_status_t id4_pipeline_parameter_slab_submit_load_group(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* loads,
    const id4_pipeline_parameter_load_step_t* load_steps,
    id4_pipeline_parameter_load_group_t group, iree_string_view_t stage_name,
    const id4_pipeline_parameter_slab_set_t* slab_set,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  const id4_pipeline_parameter_load_step_t* step =
      &load_steps[group.step_offset];
  const id4_pipeline_parameter_slab_load_t* load =
      &loads[step->target_slab_index];

  iree_status_t status = iree_ok_status();
  if (group.is_encode) {
    status = id4_pipeline_parameter_slab_submit_encode_run(
        options, load, group.step_count, step, group.step_offset, stage_name,
        slab_set->buffers[step->target_slab_index], wait_semaphore_list,
        signal_semaphore_list);
  } else {
    status = id4_pipeline_parameter_slab_submit_gather(
        options, load, step, stage_name,
        slab_set->buffers[step->target_slab_index], wait_semaphore_list,
        signal_semaphore_list);
  }
  return status;
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
        &loads[load_steps[step_index].target_slab_index];
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

static void id4_pipeline_parameter_slab_release_group_semaphores(
    iree_hal_semaphore_t** semaphores, iree_host_size_t semaphore_count,
    iree_allocator_t host_allocator) {
  for (iree_host_size_t i = 0; i < semaphore_count; ++i) {
    iree_hal_semaphore_release(semaphores[i]);
  }
  iree_allocator_free(host_allocator, semaphores);
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

static iree_status_t id4_pipeline_parameter_slab_load_groups_submit(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_string_view_t stage_name,
    const id4_pipeline_parameter_slab_set_t* slab_set,
    iree_hal_semaphore_t** group_semaphores,
    iree_host_size_t group_semaphore_count) {
  if (load_step_count == 0) return iree_ok_status();
  if (group_semaphore_count == 0) {
    const id4_pipeline_parameter_load_group_t group =
        id4_pipeline_parameter_load_group_from_steps(
            /*step_index=*/0, load_step_count, load_steps);
    return id4_pipeline_parameter_slab_submit_load_group(
        options, loads, load_steps, group, stage_name, slab_set,
        options->wait_semaphore_list, options->signal_semaphore_list);
  }

  iree_hal_semaphore_t** completion_semaphores = NULL;
  uint64_t* completion_payload_values = NULL;
  if (group_semaphore_count != 0) {
    completion_semaphores = (iree_hal_semaphore_t**)iree_alloca(
        group_semaphore_count * sizeof(completion_semaphores[0]));
    completion_payload_values = (uint64_t*)iree_alloca(
        group_semaphore_count * sizeof(completion_payload_values[0]));
  }

  iree_status_t status = iree_ok_status();
  iree_host_size_t completion_count = 0;
  iree_hal_semaphore_t* encode_wait_semaphore = NULL;
  uint64_t encode_wait_payload_value = 0;
  for (iree_host_size_t step_index = 0, group_index = 0;
       step_index < load_step_count && iree_status_is_ok(status);
       ++group_index) {
    const id4_pipeline_parameter_load_group_t group =
        id4_pipeline_parameter_load_group_from_steps(
            step_index, load_step_count, load_steps);
    iree_hal_semaphore_list_t group_wait_list = options->wait_semaphore_list;
    if (group.is_encode && encode_wait_semaphore) {
      group_wait_list = id4_pipeline_parameter_one_semaphore_list(
          &encode_wait_semaphore, &encode_wait_payload_value);
    }

    uint64_t group_signal_payload_value = 1;
    iree_hal_semaphore_t* group_signal_semaphore =
        group_semaphores[group_index];
    iree_hal_semaphore_list_t group_signal_list =
        id4_pipeline_parameter_one_semaphore_list(&group_signal_semaphore,
                                                  &group_signal_payload_value);
    status = id4_pipeline_parameter_slab_submit_load_group(
        options, loads, load_steps, group, stage_name, slab_set,
        group_wait_list, group_signal_list);
    if (iree_status_is_ok(status)) {
      if (group.is_encode) {
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

  if (iree_status_is_ok(status) && encode_wait_semaphore) {
    completion_semaphores[completion_count] = encode_wait_semaphore;
    completion_payload_values[completion_count] = encode_wait_payload_value;
    ++completion_count;
  }
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t wait_list =
        id4_pipeline_parameter_many_semaphore_list(
            completion_count, completion_semaphores, completion_payload_values);
    status = id4_pipeline_parameter_slab_signal_load_complete(
        loads, load_steps, wait_list, options->signal_semaphore_list);
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
  if (load_count != 0 && options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter slab loading requires a signal semaphore list");
  }

  id4_pipeline_parameter_slab_set_t* slab_set = NULL;
  iree_status_t status = id4_pipeline_parameter_slab_set_create_empty(
      load_count, host_allocator, &slab_set);
  for (iree_host_size_t i = 0; i < load_count && iree_status_is_ok(status);
       ++i) {
    status = id4_pipeline_parameter_slab_allocate_buffer(&loads[i],
                                                         &slab_set->buffers[i]);
    if (!iree_status_is_ok(status)) {
      status = iree_status_join(
          status,
          id4_pipeline_parameter_slab_emit_diagnostic(
              &loads[i], stage_name, IREE_SV("parameter_slab.load.error"),
              IREE_SV("parameter slab allocation failed"), loads[i].slab->scope,
              IREE_HOST_SIZE_MAX, options->diagnostics_sink));
    }
  }

  const iree_host_size_t load_group_count =
      id4_pipeline_parameter_load_group_count(load_step_count, load_steps);
  const iree_host_size_t scheduled_group_semaphore_count =
      load_group_count > 1 ? load_group_count : 0;
  iree_host_size_t group_semaphore_count = 0;
  iree_hal_semaphore_t** group_semaphores = NULL;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_create_group_semaphores(
        scheduled_group_semaphore_count, loads, load_step_count, load_steps,
        host_allocator, &group_semaphores, &group_semaphore_count);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_load_groups_submit(
        options, loads, load_step_count, load_steps, stage_name, slab_set,
        group_semaphores, group_semaphore_count);
  }
  id4_pipeline_parameter_slab_release_group_semaphores(
      group_semaphores, group_semaphore_count, host_allocator);
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
  if (!slab_set || index >= slab_set->count) return NULL;
  return slab_set->buffers[index];
}
