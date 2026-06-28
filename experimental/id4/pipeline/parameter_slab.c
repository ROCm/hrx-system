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
  ID4_PIPELINE_PARAMETER_ENCODER_FP8_TO_BF16_CONFIG_COUNT = 2,
};

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
  if (step->kind !=
      ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16) {
    return iree_ok_status();
  }
  const id4_pipeline_parameter_request_t* request =
      &slab->requests[step->request_offset];
  const id4_pipeline_parameter_load_source_t* weight_source = &step->sources[0];
  if (weight_source->shape.rank != 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter encode load step '%.*s' weight source must be rank 2",
        (int)step->name.size, step->name.data);
  }
  if (weight_source->byte_length > IREE_DEVICE_SIZE_MAX / sizeof(uint16_t)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter encode load step '%.*s' target byte length overflows",
        (int)step->name.size, step->name.data);
  }
  const iree_device_size_t expected_target_byte_length =
      weight_source->byte_length * sizeof(uint16_t);
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
  const iree_host_size_t request_index = state->request_offset + i;
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
    if (load_steps[i].kind ==
        ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16) {
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

typedef struct id4_pipeline_parameter_load_source_enumerator_state_t {
  // Source tensor gathered into staging storage.
  const id4_pipeline_parameter_load_source_t* source;
  // Staging buffer offset receiving the source tensor.
  iree_device_size_t buffer_offset;
} id4_pipeline_parameter_load_source_enumerator_state_t;

static iree_status_t id4_pipeline_parameter_load_source_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  id4_pipeline_parameter_load_source_enumerator_state_t* state =
      (id4_pipeline_parameter_load_source_enumerator_state_t*)user_data;
  if (i != 0) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter load source index %" PRIhsz " out of range", i);
  }
  *out_key = state->source->key;
  *out_span = id4_pipeline_parameter_span(
      /*parameter_offset=*/0, state->buffer_offset, state->source->byte_length);
  return iree_ok_status();
}

static iree_io_parameter_enumerator_t
id4_pipeline_parameter_load_source_enumerator(
    id4_pipeline_parameter_load_source_enumerator_state_t* state) {
  iree_io_parameter_enumerator_t enumerator = {
      // Callback used by IREE parameter provider gather APIs.
      .fn = id4_pipeline_parameter_load_source_enumerate,
      // Enumerator state holding one source descriptor.
      .user_data = state,
  };
  return enumerator;
}

typedef struct id4_pipeline_parameter_encoder_config_t {
  // Number of config bindings used by this encoder specialization.
  iree_host_size_t count;
  // Fixed-capacity config binding storage.
  id4_pipeline_kernel_config_binding_t
      bindings[ID4_PIPELINE_PARAMETER_ENCODER_FP8_TO_BF16_CONFIG_COUNT];
  // Fixed-capacity string storage backing binding values.
  char value_storage[ID4_PIPELINE_PARAMETER_ENCODER_FP8_TO_BF16_CONFIG_COUNT]
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

static iree_status_t id4_pipeline_parameter_encoder_make_config(
    const id4_pipeline_parameter_load_step_t* step,
    id4_pipeline_parameter_encoder_config_t* out_config) {
  memset(out_config, 0, sizeof(*out_config));
  const id4_pipeline_parameter_load_source_t* weight_source = &step->sources[0];
  iree_string_view_t output_size = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_format_u64(
      weight_source->shape.dims[0], out_config->value_storage[0],
      IREE_ARRAYSIZE(out_config->value_storage[0]), &output_size));
  out_config->bindings[0] = id4_pipeline_make_kernel_config_binding(
      IREE_SV("id4.parameter.fp8_e4m3_scaled_to_bf16.output_size"),
      output_size);
  iree_string_view_t input_size = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_format_u64(
      weight_source->shape.dims[1], out_config->value_storage[1],
      IREE_ARRAYSIZE(out_config->value_storage[1]), &input_size));
  out_config->bindings[1] = id4_pipeline_make_kernel_config_binding(
      IREE_SV("id4.parameter.fp8_e4m3_scaled_to_bf16.input_size"), input_size);
  out_config->count = ID4_PIPELINE_PARAMETER_ENCODER_FP8_TO_BF16_CONFIG_COUNT;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_prepare_fp8_e4m3_to_bf16(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* load,
    const id4_pipeline_parameter_load_step_t* step,
    id4_pipeline_kernel_executable_t** out_executable,
    iree_hal_executable_function_t* out_function) {
  *out_executable = NULL;
  *out_function = iree_hal_executable_function_invalid();

  const id4_pipeline_kernel_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_library_lookup(
      options->kernel_library, IREE_SV("parameter/fp8_e4m3_scaled_to_bf16"),
      &module));
  id4_pipeline_parameter_encoder_config_t config;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_encoder_make_config(step, &config));

  id4_pipeline_kernel_cache_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.executable_cache = options->executable_cache;
  prepare_options.queue_affinity = load->queue_affinity;
  prepare_options.caching_mode = IREE_HAL_EXECUTABLE_CACHING_MODE_NONE;
  prepare_options.source_identifier = module->source_identifier;
  prepare_options.source_contents = module->source_contents;
  prepare_options.module_path = module->module_path;
  prepare_options.function_name =
      IREE_SV("id4_parameter_fp8_e4m3_scaled_to_bf16");
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

static iree_status_t id4_pipeline_parameter_submit_source_gather(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* load,
    const id4_pipeline_parameter_load_source_t* source,
    iree_device_size_t staging_offset, iree_hal_buffer_t* staging_buffer,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  if (!iree_io_parameter_provider_query_support(options->provider,
                                                source->source_scope)) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "parameter provider does not support source scope '%.*s'",
        (int)source->source_scope.size, source->source_scope.data);
  }
  id4_pipeline_parameter_load_source_enumerator_state_t enumerator_state = {
      // Source descriptor gathered by this request.
      .source = source,
      // Staging buffer byte offset receiving the source.
      .buffer_offset = staging_offset,
  };
  return iree_io_parameter_provider_gather(
      options->provider, load->device, load->queue_affinity,
      wait_semaphore_list, signal_semaphore_list, source->source_scope,
      staging_buffer, /*count=*/1,
      id4_pipeline_parameter_load_source_enumerator(&enumerator_state));
}

typedef struct id4_pipeline_parameter_encode_staging_layout_t {
  // Staging offset where the F32 row-scale tensor starts.
  iree_device_size_t scale_offset;
  // Total staging byte length required for the FP8 source and F32 row scales.
  iree_device_size_t byte_length;
} id4_pipeline_parameter_encode_staging_layout_t;

static iree_status_t id4_pipeline_parameter_encode_staging_layout(
    const id4_pipeline_parameter_load_step_t* step,
    id4_pipeline_parameter_encode_staging_layout_t* out_layout) {
  const id4_pipeline_parameter_load_source_t* weight_source = &step->sources[0];
  const id4_pipeline_parameter_load_source_t* scale_source = &step->sources[1];
  iree_device_size_t scale_offset = 0;
  if (!iree_device_size_checked_align(weight_source->byte_length, 16,
                                      &scale_offset)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter load step '%.*s' staging scale offset overflows",
        (int)step->name.size, step->name.data);
  }
  iree_device_size_t byte_length = 0;
  if (!iree_device_size_checked_add(scale_offset, scale_source->byte_length,
                                    &byte_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter load step '%.*s' staging byte length overflows",
        (int)step->name.size, step->name.data);
  }
  out_layout->scale_offset = scale_offset;
  out_layout->byte_length = byte_length;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_encode_run_staging_size(
    const id4_pipeline_parameter_load_step_t* steps, iree_host_size_t count,
    iree_device_size_t* out_byte_length) {
  *out_byte_length = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    id4_pipeline_parameter_encode_staging_layout_t layout;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_parameter_encode_staging_layout(&steps[i], &layout));
    if (layout.byte_length > *out_byte_length) {
      *out_byte_length = layout.byte_length;
    }
  }
  return iree_ok_status();
}

static iree_hal_semaphore_list_t id4_pipeline_parameter_two_semaphore_list(
    iree_hal_semaphore_t** semaphores, uint64_t* payload_values) {
  return (iree_hal_semaphore_list_t){
      // Two semaphores in the list.
      .count = 2,
      // Semaphore pointer list.
      .semaphores = semaphores,
      // Required or published payload values.
      .payload_values = payload_values,
  };
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

static iree_status_t id4_pipeline_parameter_slab_submit_encode_fp8_to_bf16_run(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* load, iree_host_size_t step_count,
    const id4_pipeline_parameter_load_step_t* steps,
    iree_string_view_t stage_name, iree_hal_buffer_t* target_buffer,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_device_size_t staging_byte_length = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_encode_run_staging_size(
      steps, step_count, &staging_byte_length));

  iree_hal_semaphore_t* run_semaphore = NULL;
  iree_hal_semaphore_t* weight_gather_semaphore = NULL;
  iree_hal_semaphore_t* scale_gather_semaphore = NULL;
  iree_hal_semaphore_t* cleanup_semaphore = NULL;
  iree_hal_buffer_t* staging_buffer = NULL;
  iree_status_t status = id4_pipeline_parameter_create_semaphore(
      load->device, load->queue_affinity, &run_semaphore);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_create_semaphore(
        load->device, load->queue_affinity, &weight_gather_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_create_semaphore(
        load->device, load->queue_affinity, &scale_gather_semaphore);
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
  iree_hal_semaphore_t* cleanup_wait_semaphores[2] = {NULL, NULL};
  uint64_t cleanup_wait_payload_values[2] = {0, 0};
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

  for (iree_host_size_t i = 0; i < step_count && iree_status_is_ok(status);
       ++i) {
    const id4_pipeline_parameter_load_step_t* step = &steps[i];
    const id4_pipeline_parameter_request_t* target_request =
        &load->slab->requests[step->request_offset];
    const id4_pipeline_parameter_load_source_t* weight_source =
        &step->sources[0];
    const id4_pipeline_parameter_load_source_t* scale_source =
        &step->sources[1];

    id4_pipeline_parameter_encode_staging_layout_t layout;
    status = id4_pipeline_parameter_encode_staging_layout(step, &layout);
    if (!iree_status_is_ok(status)) break;

    id4_pipeline_kernel_executable_t* executable = NULL;
    iree_hal_executable_function_t function =
        iree_hal_executable_function_invalid();
    status = id4_pipeline_parameter_prepare_fp8_e4m3_to_bf16(
        options, load, step, &executable, &function);
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_parameter_slab_emit_diagnostic(
          load, stage_name, IREE_SV("parameter_slab.encode"),
          IREE_SV("encoding FP8 e4m3 parameter to BF16"),
          weight_source->source_scope, step->request_offset,
          options->diagnostics_sink);
    }

    uint64_t source_wait_payload_value = i + 1;
    iree_hal_semaphore_list_t source_wait_list =
        id4_pipeline_parameter_one_semaphore_list(&run_semaphore,
                                                  &source_wait_payload_value);
    uint64_t gather_payload_value = i + 1;
    iree_hal_semaphore_list_t weight_signal_list =
        id4_pipeline_parameter_one_semaphore_list(&weight_gather_semaphore,
                                                  &gather_payload_value);
    iree_hal_semaphore_list_t scale_signal_list =
        id4_pipeline_parameter_one_semaphore_list(&scale_gather_semaphore,
                                                  &gather_payload_value);
    iree_hal_semaphore_t* gather_signal_semaphores[2] = {
        weight_gather_semaphore,
        scale_gather_semaphore,
    };
    uint64_t gather_signal_payload_values[2] = {
        gather_payload_value,
        gather_payload_value,
    };
    iree_hal_semaphore_list_t gather_signal_list =
        id4_pipeline_parameter_two_semaphore_list(gather_signal_semaphores,
                                                  gather_signal_payload_values);
    uint64_t encode_payload_value = i + 2;
    iree_hal_semaphore_list_t encode_signal_list =
        id4_pipeline_parameter_one_semaphore_list(&run_semaphore,
                                                  &encode_payload_value);

    bool weight_gather_submitted = false;
    bool scale_gather_submitted = false;
    bool encode_submitted = false;
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_parameter_submit_source_gather(
          options, load, weight_source, /*staging_offset=*/0, staging_buffer,
          source_wait_list, weight_signal_list);
      weight_gather_submitted = iree_status_is_ok(status);
      if (weight_gather_submitted) {
        cleanup_wait_semaphores[0] = weight_gather_semaphore;
        cleanup_wait_payload_values[0] = gather_payload_value;
        cleanup_wait_count = 1;
      }
    }
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_parameter_submit_source_gather(
          options, load, scale_source, layout.scale_offset, staging_buffer,
          source_wait_list, scale_signal_list);
      scale_gather_submitted = iree_status_is_ok(status);
      if (scale_gather_submitted) {
        cleanup_wait_semaphores[0] = weight_gather_semaphore;
        cleanup_wait_payload_values[0] = gather_payload_value;
        cleanup_wait_semaphores[1] = scale_gather_semaphore;
        cleanup_wait_payload_values[1] = gather_payload_value;
        cleanup_wait_count = weight_gather_submitted ? 2 : 1;
      }
    }
    if (iree_status_is_ok(status)) {
      iree_hal_buffer_ref_t bindings[3] = {
          iree_hal_make_buffer_ref(staging_buffer, 0,
                                   weight_source->byte_length),
          iree_hal_make_buffer_ref(staging_buffer, layout.scale_offset,
                                   scale_source->byte_length),
          iree_hal_make_buffer_ref(target_buffer,
                                   target_request->span.buffer_offset,
                                   target_request->span.length),
      };
      iree_hal_buffer_ref_list_t binding_list = {
          // Weight source, row-scale source, and BF16 target.
          .count = IREE_ARRAYSIZE(bindings),
          // Direct source and target buffer refs.
          .values = bindings,
      };
      status = iree_hal_device_queue_dispatch(
          load->device, load->queue_affinity, gather_signal_list,
          encode_signal_list,
          id4_pipeline_kernel_executable_hal_executable(executable), function,
          id4_pipeline_kernel_executable_dispatch_config(executable),
          iree_const_byte_span_empty(), binding_list,
          IREE_HAL_DISPATCH_FLAG_NONE);
      encode_submitted = iree_status_is_ok(status);
      if (encode_submitted) {
        cleanup_wait_semaphores[0] = run_semaphore;
        cleanup_wait_payload_values[0] = encode_payload_value;
        cleanup_wait_count = 1;
      }
    }
    id4_pipeline_kernel_executable_release(executable);
  }

  if (iree_status_is_ok(status)) {
    uint64_t run_complete_payload_value = step_count + 1;
    iree_hal_semaphore_list_t run_complete_wait_list =
        id4_pipeline_parameter_one_semaphore_list(&run_semaphore,
                                                  &run_complete_payload_value);
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
  iree_hal_semaphore_release(scale_gather_semaphore);
  iree_hal_semaphore_release(weight_gather_semaphore);
  iree_hal_semaphore_release(run_semaphore);
  return status;
}

static iree_host_size_t id4_pipeline_parameter_slab_encode_run_count(
    iree_host_size_t start_index, iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps) {
  const iree_host_size_t target_slab_index =
      load_steps[start_index].target_slab_index;
  iree_host_size_t end_index = start_index;
  while (
      end_index < load_step_count &&
      load_steps[end_index].kind ==
          ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16 &&
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
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_emit_diagnostic(
        load, stage_name, IREE_SV("parameter_slab.gather"),
        IREE_SV("gathering parameter request"), step->source_scope,
        step->request_offset + j, options->diagnostics_sink));
  }
  id4_pipeline_parameter_slab_enumerator_state_t enumerator_state = {
      // Slab plan supplying request keys and spans.
      .slab = load->slab,
      // First request ordinal loaded by this step.
      .request_offset = step->request_offset,
      // Number of requests loaded by this step.
      .request_count = step->request_count,
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

static iree_hal_semaphore_list_t id4_pipeline_parameter_load_step_wait_list(
    iree_host_size_t step_index, iree_hal_semaphore_list_t initial_wait_list,
    iree_hal_semaphore_t** chain_semaphores, uint64_t* chain_wait_value,
    iree_hal_semaphore_t** out_chain_wait_semaphore) {
  if (step_index == 0) return initial_wait_list;
  *out_chain_wait_semaphore = chain_semaphores[step_index - 1];
  *chain_wait_value = 1;
  return id4_pipeline_parameter_one_semaphore_list(out_chain_wait_semaphore,
                                                   chain_wait_value);
}

static iree_hal_semaphore_list_t id4_pipeline_parameter_load_step_signal_list(
    iree_host_size_t next_step_index, iree_host_size_t load_step_count,
    iree_hal_semaphore_list_t final_signal_list,
    iree_hal_semaphore_t** chain_semaphores, uint64_t* chain_signal_value,
    iree_hal_semaphore_t** out_chain_signal_semaphore) {
  if (next_step_index == load_step_count) return final_signal_list;
  *out_chain_signal_semaphore = chain_semaphores[next_step_index - 1];
  *chain_signal_value = 1;
  return id4_pipeline_parameter_one_semaphore_list(out_chain_signal_semaphore,
                                                   chain_signal_value);
}

static iree_status_t id4_pipeline_parameter_slab_submit_load_step(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* loads,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_host_size_t load_step_count, iree_host_size_t step_index,
    iree_host_size_t* inout_next_step_index, iree_string_view_t stage_name,
    const id4_pipeline_parameter_slab_set_t* slab_set,
    iree_hal_semaphore_t** chain_semaphores,
    iree_hal_semaphore_list_t initial_wait_list,
    iree_hal_semaphore_list_t final_signal_list) {
  const id4_pipeline_parameter_load_step_t* step = &load_steps[step_index];
  const id4_pipeline_parameter_slab_load_t* load =
      &loads[step->target_slab_index];
  iree_host_size_t next_step_index = step_index + 1;
  if (step->kind ==
      ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16) {
    next_step_index = step_index + id4_pipeline_parameter_slab_encode_run_count(
                                       step_index, load_step_count, load_steps);
  }

  iree_hal_semaphore_t* chain_wait_semaphore = NULL;
  uint64_t chain_wait_value = 1;
  iree_hal_semaphore_list_t step_wait_semaphore_list =
      id4_pipeline_parameter_load_step_wait_list(
          step_index, initial_wait_list, chain_semaphores, &chain_wait_value,
          &chain_wait_semaphore);
  iree_hal_semaphore_t* chain_signal_semaphore = NULL;
  uint64_t chain_signal_value = 1;
  iree_hal_semaphore_list_t step_signal_semaphore_list =
      id4_pipeline_parameter_load_step_signal_list(
          next_step_index, load_step_count, final_signal_list, chain_semaphores,
          &chain_signal_value, &chain_signal_semaphore);

  iree_status_t status = iree_ok_status();
  if (step->kind ==
      ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16) {
    status = id4_pipeline_parameter_slab_submit_encode_fp8_to_bf16_run(
        options, load, next_step_index - step_index, step, stage_name,
        slab_set->buffers[step->target_slab_index], step_wait_semaphore_list,
        step_signal_semaphore_list);
  } else {
    status = id4_pipeline_parameter_slab_submit_gather(
        options, load, step, stage_name,
        slab_set->buffers[step->target_slab_index], step_wait_semaphore_list,
        step_signal_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    *inout_next_step_index = next_step_index;
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_slab_load_steps_submit(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_string_view_t stage_name,
    const id4_pipeline_parameter_slab_set_t* slab_set,
    iree_hal_semaphore_t** chain_semaphores) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < load_step_count && iree_status_is_ok(status);) {
    iree_host_size_t next_step_index = i;
    status = id4_pipeline_parameter_slab_submit_load_step(
        options, loads, load_steps, load_step_count, i, &next_step_index,
        stage_name, slab_set, chain_semaphores, options->wait_semaphore_list,
        options->signal_semaphore_list);
    i = next_step_index;
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_slab_create_chain_semaphores(
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_allocator_t host_allocator, iree_hal_semaphore_t*** out_semaphores,
    iree_host_size_t* out_semaphore_count) {
  IREE_ASSERT_ARGUMENT(out_semaphores);
  IREE_ASSERT_ARGUMENT(out_semaphore_count);
  *out_semaphores = NULL;
  *out_semaphore_count = load_step_count > 1 ? load_step_count - 1 : 0;
  if (*out_semaphore_count == 0) return iree_ok_status();

  iree_hal_semaphore_t** semaphores = NULL;
  iree_status_t status =
      iree_allocator_malloc_array(host_allocator, *out_semaphore_count,
                                  sizeof(semaphores[0]), (void**)&semaphores);
  for (iree_host_size_t i = 0;
       i < *out_semaphore_count && iree_status_is_ok(status); ++i) {
    const id4_pipeline_parameter_slab_load_t* load =
        &loads[load_steps[i].target_slab_index];
    status = iree_hal_semaphore_create(
        load->device, load->queue_affinity,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphores[i]);
  }
  if (iree_status_is_ok(status)) {
    *out_semaphores = semaphores;
  } else {
    for (iree_host_size_t i = 0; i < *out_semaphore_count; ++i) {
      iree_hal_semaphore_release(semaphores[i]);
    }
    iree_allocator_free(host_allocator, semaphores);
  }
  return status;
}

static void id4_pipeline_parameter_slab_release_chain_semaphores(
    iree_hal_semaphore_t** semaphores, iree_host_size_t semaphore_count,
    iree_allocator_t host_allocator) {
  for (iree_host_size_t i = 0; i < semaphore_count; ++i) {
    iree_hal_semaphore_release(semaphores[i]);
  }
  iree_allocator_free(host_allocator, semaphores);
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

  iree_host_size_t chain_semaphore_count = 0;
  iree_hal_semaphore_t** chain_semaphores = NULL;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_create_chain_semaphores(
        load_step_count, loads, load_steps, host_allocator, &chain_semaphores,
        &chain_semaphore_count);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_load_steps_submit(
        options, loads, load_step_count, load_steps, stage_name, slab_set,
        chain_semaphores);
  }
  id4_pipeline_parameter_slab_release_chain_semaphores(
      chain_semaphores, chain_semaphore_count, host_allocator);
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
