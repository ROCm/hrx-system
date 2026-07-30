// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "experimental/qwen/runtime/model_shape.h"
#include "experimental/qwen/runtime/request_state.h"
#include "iree/base/internal/atomics.h"

typedef struct qwen_request_semaphore_list_storage_t {
  // Materialized semaphore pointer array.
  iree_hal_semaphore_t** semaphores;
  // Materialized semaphore payload array.
  uint64_t* payload_values;
  // List view over the materialized arrays.
  iree_hal_semaphore_list_t list;
} qwen_request_semaphore_list_storage_t;

struct qwen_request_t {
  // Reference count for shared request ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for request-owned host allocations.
  iree_allocator_t host_allocator;
  // Model retained for the complete request lifetime.
  qwen_model_t* model;
  // Device-local request storage.
  iree_hal_buffer_t* storage_buffer;
  // Host-visible layer-output staging buffer.
  iree_hal_buffer_t* output_staging_buffer;
  // Persistent mapping of |output_staging_buffer|.
  iree_hal_buffer_mapping_t output_staging_mapping;
  // True when |output_staging_mapping| must be unmapped during destruction.
  bool output_staging_is_mapped;
  // Internal readiness and operation timeline.
  iree_hal_semaphore_t* timeline_semaphore;
  // Latest reserved value on |timeline_semaphore|.
  uint64_t timeline_value;
  // Timeline value that last published a complete layer output.
  uint64_t output_ready_value;
  // Exact physical token count.
  iree_host_size_t token_count;
  // Exact K/V row capacity.
  iree_host_size_t context_capacity;
  // Deterministic persistent storage layout.
  qwen_request_storage_layout_t storage_layout;
};

static void qwen_request_semaphore_list_storage_deinitialize(
    qwen_request_semaphore_list_storage_t* storage,
    iree_allocator_t host_allocator) {
  if (!storage) return;
  iree_allocator_free(host_allocator, storage->payload_values);
  iree_allocator_free(host_allocator, storage->semaphores);
  memset(storage, 0, sizeof(*storage));
}

static iree_status_t qwen_request_semaphore_list_prepend(
    iree_hal_semaphore_t* semaphore, uint64_t payload_value,
    iree_hal_semaphore_list_t suffix, iree_allocator_t host_allocator,
    qwen_request_semaphore_list_storage_t* out_storage) {
  memset(out_storage, 0, sizeof(*out_storage));
  if (suffix.count == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen semaphore list count overflows");
  }
  const iree_host_size_t count = suffix.count + 1;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, count, sizeof(*out_storage->semaphores),
      (void**)&out_storage->semaphores);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, count,
                                         sizeof(*out_storage->payload_values),
                                         (void**)&out_storage->payload_values);
  }
  if (iree_status_is_ok(status)) {
    out_storage->semaphores[0] = semaphore;
    out_storage->payload_values[0] = payload_value;
    if (suffix.count != 0) {
      memcpy(&out_storage->semaphores[1], suffix.semaphores,
             suffix.count * sizeof(*suffix.semaphores));
      memcpy(&out_storage->payload_values[1], suffix.payload_values,
             suffix.count * sizeof(*suffix.payload_values));
    }
    out_storage->list = (iree_hal_semaphore_list_t){
        .count = count,
        .semaphores = out_storage->semaphores,
        .payload_values = out_storage->payload_values,
    };
  } else {
    qwen_request_semaphore_list_storage_deinitialize(out_storage,
                                                     host_allocator);
  }
  return status;
}

static void qwen_request_destroy(qwen_request_t* request) {
  if (request->output_staging_is_mapped) {
    iree_hal_buffer_unmap_range(&request->output_staging_mapping);
  }
  iree_hal_semaphore_release(request->timeline_semaphore);
  iree_hal_buffer_release(request->output_staging_buffer);
  iree_hal_buffer_release(request->storage_buffer);
  qwen_model_release(request->model);
  iree_allocator_t host_allocator = request->host_allocator;
  iree_allocator_free(host_allocator, request);
}

void qwen_request_options_initialize(qwen_request_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (qwen_request_options_t){
      .structure_size = sizeof(*out_options),
      .next = NULL,
      .token_count = 1,
      .context_capacity = 1,
  };
}

iree_status_t qwen_request_create(
    qwen_model_t* model, const qwen_request_options_t* options,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_allocator_t host_allocator, qwen_request_t** out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  *out_request = NULL;
  if (!model || !options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen model and request options are required");
  }
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request options structure is too small");
  }
  if (options->next) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request option extensions are unsupported");
  }

  qwen_request_storage_layout_t storage_layout;
  IREE_RETURN_IF_ERROR(qwen_request_storage_layout_calculate(
      options->token_count, options->context_capacity, &storage_layout));

  qwen_request_t* request = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*request),
                                             (void**)&request));
  memset(request, 0, sizeof(*request));
  iree_atomic_ref_count_init(&request->ref_count);
  request->host_allocator = host_allocator;
  request->model = model;
  qwen_model_retain(model);
  request->token_count = options->token_count;
  request->context_capacity = options->context_capacity;
  request->storage_layout = storage_layout;

  iree_hal_device_t* device = qwen_model_device(model);
  const iree_hal_queue_affinity_t queue_affinity =
      qwen_model_queue_affinity(model);
  iree_status_t status = iree_hal_semaphore_create(
      device, queue_affinity, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &request->timeline_semaphore);

  iree_hal_buffer_params_t staging_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
               IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
               IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM,
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = queue_affinity,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), staging_params,
        storage_layout.hidden_state.length, &request->output_staging_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_range(
        request->output_staging_buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
        IREE_HAL_MEMORY_ACCESS_READ, /*byte_offset=*/0,
        storage_layout.hidden_state.length, &request->output_staging_mapping);
  }
  if (iree_status_is_ok(status)) {
    request->output_staging_is_mapped = true;
  }

  qwen_request_semaphore_list_storage_t signal_storage;
  memset(&signal_storage, 0, sizeof(signal_storage));
  if (iree_status_is_ok(status)) {
    status = qwen_request_semaphore_list_prepend(
        request->timeline_semaphore, /*payload_value=*/1, signal_semaphore_list,
        host_allocator, &signal_storage);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_params_t storage_params = {
        .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                 IREE_HAL_BUFFER_USAGE_TRANSFER,
        .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        .queue_affinity = queue_affinity,
        .min_alignment = 64,
    };
    status = iree_hal_device_queue_alloca(
        device, queue_affinity, wait_semaphore_list, signal_storage.list,
        /*pool=*/NULL, storage_params, storage_layout.persistent_byte_length,
        IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME, &request->storage_buffer);
  }
  qwen_request_semaphore_list_storage_deinitialize(&signal_storage,
                                                   host_allocator);

  if (iree_status_is_ok(status)) {
    request->timeline_value = 1;
    *out_request = request;
  } else {
    qwen_request_destroy(request);
  }
  return status;
}

void qwen_request_retain(qwen_request_t* request) {
  if (request) {
    iree_atomic_ref_count_inc(&request->ref_count);
  }
}

void qwen_request_release(qwen_request_t* request) {
  if (request && iree_atomic_ref_count_dec(&request->ref_count) == 1) {
    qwen_request_destroy(request);
  }
}

iree_status_t qwen_request_reset_hidden_state(
    qwen_request_t* request, iree_const_byte_span_t hidden_state_data,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  if (!request) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request is required");
  }
  if (hidden_state_data.data_length !=
      request->storage_layout.hidden_state.length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen hidden-state input has %" PRIhsz " bytes; expected %" PRIu64,
        hidden_state_data.data_length,
        (uint64_t)request->storage_layout.hidden_state.length);
  }
  if (request->timeline_value == IREE_HAL_SEMAPHORE_MAX_VALUE) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Qwen request timeline is exhausted");
  }

  uint8_t* upload_data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      request->host_allocator,
      (iree_host_size_t)request->storage_layout.reset_upload_byte_length,
      (void**)&upload_data));
  memset(upload_data, 0,
         (iree_host_size_t)request->storage_layout.reset_upload_byte_length);
  memcpy(upload_data + request->storage_layout.hidden_state.offset,
         hidden_state_data.data, hidden_state_data.data_length);
  qwen_request_control_t* control =
      (qwen_request_control_t*)(upload_data +
                                request->storage_layout.control.offset);
  *control = (qwen_request_control_t){
      .context_base = 0,
  };

  qwen_request_semaphore_list_storage_t wait_storage;
  qwen_request_semaphore_list_storage_t signal_storage;
  memset(&wait_storage, 0, sizeof(wait_storage));
  memset(&signal_storage, 0, sizeof(signal_storage));
  const uint64_t next_timeline_value = request->timeline_value + 1;
  iree_status_t status = qwen_request_semaphore_list_prepend(
      request->timeline_semaphore, request->timeline_value, wait_semaphore_list,
      request->host_allocator, &wait_storage);
  if (iree_status_is_ok(status)) {
    status = qwen_request_semaphore_list_prepend(
        request->timeline_semaphore, next_timeline_value, signal_semaphore_list,
        request->host_allocator, &signal_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_update(
        qwen_model_device(request->model),
        qwen_model_queue_affinity(request->model), wait_storage.list,
        signal_storage.list, upload_data, /*source_offset=*/0,
        request->storage_buffer, /*target_offset=*/0,
        request->storage_layout.reset_upload_byte_length,
        IREE_HAL_UPDATE_FLAG_NONE);
  }
  qwen_request_semaphore_list_storage_deinitialize(&signal_storage,
                                                   request->host_allocator);
  qwen_request_semaphore_list_storage_deinitialize(&wait_storage,
                                                   request->host_allocator);
  iree_allocator_free(request->host_allocator, upload_data);

  if (iree_status_is_ok(status)) {
    request->timeline_value = next_timeline_value;
    request->output_ready_value = 0;
  }
  return status;
}

iree_status_t qwen_request_read_hidden_state(qwen_request_t* request,
                                             iree_byte_span_t target_data) {
  if (!request) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request is required");
  }
  if (target_data.data_length != request->storage_layout.hidden_state.length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen hidden-state output has %" PRIhsz " bytes; expected %" PRIu64,
        target_data.data_length,
        (uint64_t)request->storage_layout.hidden_state.length);
  }
  if (request->output_ready_value == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen request has no completed layer output");
  }
  uint64_t completed_value = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_query(request->timeline_semaphore, &completed_value));
  if (completed_value < request->output_ready_value) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen layer output timeline value %" PRIu64
                            " has not completed; current value is %" PRIu64,
                            request->output_ready_value, completed_value);
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_mapping_invalidate_range(
      &request->output_staging_mapping, /*byte_offset=*/0,
      request->storage_layout.hidden_state.length));
  memcpy(target_data.data, request->output_staging_mapping.contents.data,
         target_data.data_length);
  return iree_ok_status();
}

iree_host_size_t qwen_request_token_count(const qwen_request_t* request) {
  return request ? request->token_count : 0;
}

iree_host_size_t qwen_request_context_capacity(const qwen_request_t* request) {
  return request ? request->context_capacity : 0;
}

iree_device_size_t qwen_request_persistent_byte_length(
    const qwen_request_t* request) {
  return request ? request->storage_layout.persistent_byte_length : 0;
}

qwen_model_t* qwen_request_model(const qwen_request_t* request) {
  return request ? request->model : NULL;
}

iree_hal_buffer_t* qwen_request_storage_buffer(const qwen_request_t* request) {
  return request ? request->storage_buffer : NULL;
}

iree_hal_buffer_t* qwen_request_output_staging_buffer(
    const qwen_request_t* request) {
  return request ? request->output_staging_buffer : NULL;
}

const qwen_request_storage_layout_t* qwen_request_storage_layout(
    const qwen_request_t* request) {
  return request ? &request->storage_layout : NULL;
}

iree_hal_semaphore_t* qwen_request_timeline_semaphore(
    const qwen_request_t* request) {
  return request ? request->timeline_semaphore : NULL;
}

uint64_t qwen_request_timeline_value(const qwen_request_t* request) {
  return request ? request->timeline_value : 0;
}

void qwen_request_commit_program_signal(qwen_request_t* request,
                                        uint64_t signal_value) {
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT(signal_value == request->timeline_value + 1);
  request->timeline_value = signal_value;
  request->output_ready_value = signal_value;
}

void qwen_request_fail(qwen_request_t* request, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(request);
  iree_hal_semaphore_fail(request->timeline_semaphore, status);
}
