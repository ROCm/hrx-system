// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/model.h"

#include <string.h>

#include "experimental/qwen/runtime/loom_jit.h"
#include "experimental/qwen/runtime/parameters.h"
#include "iree/base/internal/atomics.h"
#include "iree/io/parameter_provider.h"

struct qwen_model_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  // Single-device group retained for the complete model lifetime.
  iree_hal_device_group_t* device_group;
  // Sole device borrowed from |device_group|.
  iree_hal_device_t* device;
  // Queue affinity used by every model operation.
  iree_hal_queue_affinity_t queue_affinity;
  // Shared compiler/executable cache used by every prepared program.
  qwen_loom_jit_t* loom_jit;

  // Provider retained until the model is destroyed.
  iree_io_parameter_provider_t* parameter_provider;
  // Indeterminate-lifetime resident parameter and auxiliary allocation.
  iree_hal_buffer_t* parameter_buffer;
  // Typed fixed-schema layout of |parameter_buffer|.
  qwen_parameter_layout_t parameter_layout;

  // Timeline publishing allocation, gather, and auxiliary readiness.
  iree_hal_semaphore_t* readiness_semaphore;
  // Final payload published after the immutable auxiliary upload.
  uint64_t readiness_value;
};

typedef struct qwen_parameter_gather_enumerator_t {
  // Fixed layout supplying target spans.
  const qwen_parameter_layout_t* layout;
  // Reused storage for one formatted layer key.
  char key_storage[QWEN_PARAMETER_KEY_CAPACITY];
} qwen_parameter_gather_enumerator_t;

static iree_status_t qwen_parameter_gather_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  qwen_parameter_gather_enumerator_t* enumerator =
      (qwen_parameter_gather_enumerator_t*)user_data;
  return qwen_parameter_layout_enumerate(
      enumerator->layout, i, enumerator->key_storage, out_key, out_span);
}

static iree_hal_semaphore_list_t qwen_model_one_semaphore_list(
    iree_hal_semaphore_t** semaphore, uint64_t* value) {
  iree_hal_semaphore_list_t list = {
      .count = 1,
      .semaphores = semaphore,
      .payload_values = value,
  };
  return list;
}

static iree_status_t qwen_model_append_signal_semaphore(
    iree_hal_semaphore_list_t source_list,
    iree_hal_semaphore_t* additional_semaphore, uint64_t additional_value,
    iree_allocator_t host_allocator,
    iree_hal_semaphore_list_t* out_combined_list) {
  *out_combined_list = iree_hal_semaphore_list_empty();
  iree_host_size_t combined_count = 0;
  if (!iree_host_size_checked_add(source_list.count, 1, &combined_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen signal semaphore count overflow");
  }

  iree_hal_semaphore_t** semaphores = NULL;
  uint64_t* payload_values = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, combined_count, sizeof(*semaphores), (void**)&semaphores);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, combined_count,
                                         sizeof(*payload_values),
                                         (void**)&payload_values);
  }
  if (iree_status_is_ok(status)) {
    semaphores[0] = additional_semaphore;
    payload_values[0] = additional_value;
    if (source_list.count != 0) {
      memcpy(semaphores + 1, source_list.semaphores,
             source_list.count * sizeof(*semaphores));
      memcpy(payload_values + 1, source_list.payload_values,
             source_list.count * sizeof(*payload_values));
    }
    *out_combined_list = (iree_hal_semaphore_list_t){
        .count = combined_count,
        .semaphores = semaphores,
        .payload_values = payload_values,
    };
  } else {
    iree_allocator_free(host_allocator, payload_values);
    iree_allocator_free(host_allocator, semaphores);
  }
  return status;
}

static void qwen_model_free_signal_semaphore_list(
    iree_allocator_t host_allocator,
    iree_hal_semaphore_list_t* semaphore_list) {
  iree_allocator_free(host_allocator, semaphore_list->payload_values);
  iree_allocator_free(host_allocator, semaphore_list->semaphores);
  *semaphore_list = iree_hal_semaphore_list_empty();
}

static void qwen_model_destroy(qwen_model_t* model) {
  iree_allocator_t host_allocator = model->host_allocator;
  qwen_loom_jit_release(model->loom_jit);
  iree_hal_semaphore_release(model->readiness_semaphore);
  iree_hal_buffer_release(model->parameter_buffer);
  iree_io_parameter_provider_release(model->parameter_provider);
  iree_hal_device_group_release(model->device_group);
  iree_allocator_free(host_allocator, model);
}

void qwen_model_options_initialize(qwen_model_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
  out_options->structure_size = sizeof(*out_options);
  out_options->queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
}

static iree_status_t qwen_model_validate_load_options(
    const qwen_model_options_t* options,
    const qwen_parameter_source_t* source) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen model options are required");
  }
  if (options->structure_size != sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen model options have size %" PRIhsz
                            "; expected %" PRIhsz,
                            options->structure_size, sizeof(*options));
  }
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Qwen model option extensions are not implemented");
  }
  if (!options->device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen model requires a HAL device group");
  }
  const iree_host_size_t device_count =
      iree_hal_device_group_device_count(options->device_group);
  if (device_count != 1 || options->device_index != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen model requires device 0 in a one-device group; received "
        "%" PRIhsz " devices and index %" PRIhsz,
        device_count, options->device_index);
  }
  if (options->queue_affinity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen model queue affinity must not be zero");
  }
  if (!source || !source->index || !source->provider) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen model requires a parameter index and provider");
  }
  if (!iree_io_parameter_provider_query_support(source->provider,
                                                source->scope)) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "Qwen parameter provider does not support scope '%.*s'",
        (int)source->scope.size, source->scope.data);
  }
  return iree_ok_status();
}

iree_status_t qwen_model_load(const qwen_model_options_t* options,
                              const qwen_parameter_source_t* source,
                              iree_hal_semaphore_list_t wait_semaphore_list,
                              iree_hal_semaphore_list_t signal_semaphore_list,
                              iree_allocator_t host_allocator,
                              qwen_model_t** out_model) {
  IREE_ASSERT_ARGUMENT(out_model);
  *out_model = NULL;
  IREE_RETURN_IF_ERROR(qwen_model_validate_load_options(options, source));

  qwen_parameter_layout_t parameter_layout;
  IREE_RETURN_IF_ERROR(
      qwen_parameter_layout_build(source->index, &parameter_layout));

  qwen_model_t* model = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*model), (void**)&model));
  memset(model, 0, sizeof(*model));
  iree_atomic_ref_count_init(&model->ref_count);
  model->host_allocator = host_allocator;
  model->device_group = options->device_group;
  iree_hal_device_group_retain(model->device_group);
  model->device = iree_hal_device_group_device_at(model->device_group,
                                                  options->device_index);
  model->queue_affinity = options->queue_affinity;
  model->parameter_provider = source->provider;
  iree_io_parameter_provider_retain(model->parameter_provider);
  model->parameter_layout = parameter_layout;
  model->readiness_value = 3;

  const qwen_loom_jit_options_t jit_options = {
      .structure_size = sizeof(jit_options),
      .device = model->device,
      .queue_affinity = model->queue_affinity,
      .entry_limit = QWEN_LOOM_JIT_DEFAULT_ENTRY_LIMIT,
      .sanitizer_checks = options->sanitizer_checks,
  };
  iree_status_t status =
      qwen_loom_jit_create(&jit_options, host_allocator, &model->loom_jit);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        model->device, model->queue_affinity, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &model->readiness_semaphore);
  }

  uint64_t allocation_ready_value = 1;
  uint64_t parameters_ready_value = 2;
  iree_hal_semaphore_list_t allocation_ready_list =
      qwen_model_one_semaphore_list(&model->readiness_semaphore,
                                    &allocation_ready_value);
  iree_hal_semaphore_list_t parameters_ready_list =
      qwen_model_one_semaphore_list(&model->readiness_semaphore,
                                    &parameters_ready_value);

  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_params_t buffer_params = {
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
        .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        .queue_affinity = model->queue_affinity,
        .min_alignment = 64,
    };
    status = iree_hal_device_queue_alloca(
        model->device, model->queue_affinity, wait_semaphore_list,
        allocation_ready_list, /*pool=*/NULL, buffer_params,
        model->parameter_layout.statistics.allocation_bytes,
        IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME, &model->parameter_buffer);
  }

  qwen_parameter_gather_enumerator_t gather_enumerator = {
      .layout = &model->parameter_layout,
  };
  if (iree_status_is_ok(status)) {
    const iree_io_parameter_gather_t gather = {
        .source_scope = source->scope,
        .target_buffer = model->parameter_buffer,
        .count = QWEN_PARAMETER_COUNT,
        .enumerator =
            {
                .fn = qwen_parameter_gather_enumerate,
                .user_data = &gather_enumerator,
            },
        .wait_semaphore_list = allocation_ready_list,
        .signal_semaphore_list = parameters_ready_list,
    };
    status = iree_io_parameter_provider_gather_batch(
        model->parameter_provider, model->device, model->queue_affinity,
        /*gather_count=*/1, &gather);
  }

  float rope_inverse_frequencies[QWEN_MODEL_ROPE_FREQUENCY_COUNT];
  qwen_parameter_calculate_rope_inverse_frequencies(rope_inverse_frequencies);
  iree_hal_semaphore_list_t ready_signal_list = iree_hal_semaphore_list_empty();
  if (iree_status_is_ok(status)) {
    status = qwen_model_append_signal_semaphore(
        signal_semaphore_list, model->readiness_semaphore,
        model->readiness_value, host_allocator, &ready_signal_list);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_update(
        model->device, model->queue_affinity, parameters_ready_list,
        ready_signal_list, rope_inverse_frequencies, /*source_offset=*/0,
        model->parameter_buffer,
        model->parameter_layout.rope_inverse_frequencies.offset,
        model->parameter_layout.rope_inverse_frequencies.length,
        IREE_HAL_UPDATE_FLAG_NONE);
  }
  qwen_model_free_signal_semaphore_list(host_allocator, &ready_signal_list);

  if (iree_status_is_ok(status)) {
    *out_model = model;
  } else {
    qwen_model_release(model);
  }
  return status;
}

void qwen_model_retain(qwen_model_t* model) {
  if (IREE_LIKELY(model)) {
    iree_atomic_ref_count_inc(&model->ref_count);
  }
}

void qwen_model_release(qwen_model_t* model) {
  if (IREE_LIKELY(model) && iree_atomic_ref_count_dec(&model->ref_count) == 1) {
    qwen_model_destroy(model);
  }
}

iree_hal_device_group_t* qwen_model_device_group(const qwen_model_t* model) {
  IREE_ASSERT_ARGUMENT(model);
  return model->device_group;
}

iree_hal_device_t* qwen_model_device(const qwen_model_t* model) {
  IREE_ASSERT_ARGUMENT(model);
  return model->device;
}

iree_hal_queue_affinity_t qwen_model_queue_affinity(const qwen_model_t* model) {
  IREE_ASSERT_ARGUMENT(model);
  return model->queue_affinity;
}

iree_hal_semaphore_list_t qwen_model_ready_semaphore_list(
    const qwen_model_t* model) {
  IREE_ASSERT_ARGUMENT(model);
  return qwen_model_one_semaphore_list(
      (iree_hal_semaphore_t**)&model->readiness_semaphore,
      (uint64_t*)&model->readiness_value);
}

qwen_loom_jit_t* qwen_model_loom_jit(const qwen_model_t* model) {
  IREE_ASSERT_ARGUMENT(model);
  return model->loom_jit;
}

qwen_model_statistics_t qwen_model_statistics(const qwen_model_t* model) {
  IREE_ASSERT_ARGUMENT(model);
  return (qwen_model_statistics_t){
      .encoded_parameter_bytes =
          model->parameter_layout.statistics.encoded_parameter_bytes,
      .parameter_padding_bytes =
          model->parameter_layout.statistics.parameter_padding_bytes,
      .immutable_auxiliary_bytes =
          model->parameter_layout.statistics.immutable_auxiliary_bytes,
      .allocation_bytes = model->parameter_layout.statistics.allocation_bytes,
  };
}

iree_hal_buffer_t* qwen_model_parameter_buffer(const qwen_model_t* model) {
  IREE_ASSERT_ARGUMENT(model);
  return model->parameter_buffer;
}

const qwen_parameter_layout_t* qwen_model_parameter_layout(
    const qwen_model_t* model) {
  IREE_ASSERT_ARGUMENT(model);
  return &model->parameter_layout;
}
