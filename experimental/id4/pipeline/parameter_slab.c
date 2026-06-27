// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_slab.h"

#include <stddef.h>
#include <string.h>

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
  return id4_pipeline_parameter_load_step_validate_request_range(
      step, slab->request_count);
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
  return id4_pipeline_parameter_load_step_validate_request_range(
      step, load->slab->request_count);
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
    iree_io_parameter_provider_t* provider,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_string_view_t stage_name,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(out_slab_set);
  *out_slab_set = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      diagnostics_sink, IREE_SV("parameter slab load")));
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
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_validate_semaphore_list(
      wait_semaphore_list, IREE_SV("parameter slab wait")));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_slab_validate_semaphore_list(
      signal_semaphore_list, IREE_SV("parameter slab signal")));
  if (load_count != 0 && signal_semaphore_list.count == 0) {
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
              IREE_HOST_SIZE_MAX, diagnostics_sink));
    }
  }

  iree_host_size_t chain_semaphore_count = 0;
  iree_hal_semaphore_t** chain_semaphores = NULL;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_create_chain_semaphores(
        load_step_count, loads, load_steps, host_allocator, &chain_semaphores,
        &chain_semaphore_count);
  }
  for (iree_host_size_t i = 0; i < load_step_count && iree_status_is_ok(status);
       ++i) {
    const id4_pipeline_parameter_load_step_t* step = &load_steps[i];
    const id4_pipeline_parameter_slab_load_t* load =
        &loads[step->target_slab_index];
    if (step->kind ==
        ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16) {
      const iree_string_view_t scope =
          id4_pipeline_parameter_load_step_primary_scope(step);
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "parameter load step '%.*s' FP8 e4m3 to BF16 encoder is not "
          "implemented",
          (int)step->name.size, step->name.data);
      status = iree_status_join(
          status, id4_pipeline_parameter_slab_emit_diagnostic(
                      load, stage_name, IREE_SV("parameter_slab.load.error"),
                      IREE_SV("parameter load step encoder is not implemented"),
                      scope, step->request_offset, diagnostics_sink));
      break;
    }
    if (!iree_io_parameter_provider_query_support(provider,
                                                  step->source_scope)) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "parameter provider does not support source scope '%.*s'",
          (int)step->source_scope.size, step->source_scope.data);
      status = iree_status_join(
          status,
          id4_pipeline_parameter_slab_emit_diagnostic(
              load, stage_name, IREE_SV("parameter_slab.load.error"),
              IREE_SV("parameter provider does not support source "
                      "scope"),
              step->source_scope, IREE_HOST_SIZE_MAX, diagnostics_sink));
      break;
    }
    status = id4_pipeline_parameter_slab_emit_diagnostic(
        load, stage_name, IREE_SV("parameter_slab.load"),
        IREE_SV("loading parameter slab"), step->source_scope,
        IREE_HOST_SIZE_MAX, diagnostics_sink);
    for (iree_host_size_t j = 0;
         j < step->request_count && iree_status_is_ok(status); ++j) {
      status = id4_pipeline_parameter_slab_emit_diagnostic(
          load, stage_name, IREE_SV("parameter_slab.gather"),
          IREE_SV("gathering parameter request"), step->source_scope,
          step->request_offset + j, diagnostics_sink);
    }
    if (!iree_status_is_ok(status)) break;
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
    iree_hal_semaphore_t* chain_wait_semaphore =
        i == 0 ? NULL : chain_semaphores[i - 1];
    uint64_t chain_wait_value = 1;
    iree_hal_semaphore_list_t gather_wait_semaphore_list =
        i == 0 ? wait_semaphore_list
               : (iree_hal_semaphore_list_t){
                     // One internal chain semaphore.
                     .count = 1,
                     // Semaphore waited on before this slab gather.
                     .semaphores = &chain_wait_semaphore,
                     // Payload value required before this slab gather.
                     .payload_values = &chain_wait_value,
                 };
    iree_hal_semaphore_t* chain_signal_semaphore =
        i + 1 == load_step_count ? NULL : chain_semaphores[i];
    uint64_t chain_signal_value = 1;
    iree_hal_semaphore_list_t gather_signal_semaphore_list =
        i + 1 == load_step_count
            ? signal_semaphore_list
            : (iree_hal_semaphore_list_t){
                  // One internal chain semaphore.
                  .count = 1,
                  // Semaphore signaled after this slab gather.
                  .semaphores = &chain_signal_semaphore,
                  // Payload value published after this slab gather.
                  .payload_values = &chain_signal_value,
              };
    status = iree_io_parameter_provider_gather(
        provider, load->device, load->queue_affinity,
        gather_wait_semaphore_list, gather_signal_semaphore_list,
        step->source_scope, slab_set->buffers[step->target_slab_index],
        step->request_count, enumerator);
    if (!iree_status_is_ok(status)) {
      status = iree_status_join(
          status,
          id4_pipeline_parameter_slab_emit_diagnostic(
              load, stage_name, IREE_SV("parameter_slab.load.error"),
              IREE_SV("parameter gather submission failed"), step->source_scope,
              IREE_HOST_SIZE_MAX, diagnostics_sink));
    }
  }
  id4_pipeline_parameter_slab_release_chain_semaphores(
      chain_semaphores, chain_semaphore_count, host_allocator);
  if (iree_status_is_ok(status)) {
    *out_slab_set = slab_set;
  } else {
    if (signal_semaphore_list.count != 0) {
      iree_hal_semaphore_list_fail(signal_semaphore_list,
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
