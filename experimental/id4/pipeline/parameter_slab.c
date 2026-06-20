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

iree_status_t id4_pipeline_parameter_slab_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(user_data);
  IREE_ASSERT_ARGUMENT(out_key);
  IREE_ASSERT_ARGUMENT(out_span);
  id4_pipeline_parameter_slab_enumerator_state_t* state =
      (id4_pipeline_parameter_slab_enumerator_state_t*)user_data;
  if (!state->slab || i >= state->slab->request_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter request index %" PRIhsz " is outside slab request count", i);
  }
  const id4_pipeline_parameter_request_t* request = &state->slab->requests[i];
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
    status = iree_allocator_malloc(host_allocator,
                                   count * sizeof(slab_set->buffers[0]),
                                   (void**)&slab_set->buffers);
  }
  if (iree_status_is_ok(status) && count != 0) {
    memset(slab_set->buffers, 0, count * sizeof(slab_set->buffers[0]));
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

static iree_status_t id4_pipeline_parameter_slab_create_chain_semaphores(
    iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_allocator_t host_allocator, iree_hal_semaphore_t*** out_semaphores,
    iree_host_size_t* out_semaphore_count) {
  IREE_ASSERT_ARGUMENT(out_semaphores);
  IREE_ASSERT_ARGUMENT(out_semaphore_count);
  *out_semaphores = NULL;
  *out_semaphore_count = load_count > 1 ? load_count - 1 : 0;
  if (*out_semaphore_count == 0) return iree_ok_status();

  iree_hal_semaphore_t** semaphores = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, *out_semaphore_count * sizeof(semaphores[0]),
      (void**)&semaphores);
  if (iree_status_is_ok(status)) {
    memset(semaphores, 0, *out_semaphore_count * sizeof(semaphores[0]));
  }
  for (iree_host_size_t i = 0;
       i < *out_semaphore_count && iree_status_is_ok(status); ++i) {
    const id4_pipeline_parameter_slab_load_t* load = &loads[i];
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
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(out_slab_set);
  *out_slab_set = NULL;
  if (load_count != 0 && !loads) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter slab load array is required");
  }

  id4_pipeline_parameter_slab_set_t* slab_set = NULL;
  iree_status_t status = id4_pipeline_parameter_slab_set_create_empty(
      load_count, host_allocator, &slab_set);
  for (iree_host_size_t i = 0; i < load_count && iree_status_is_ok(status);
       ++i) {
    status = id4_pipeline_parameter_slab_allocate_buffer(&loads[i],
                                                         &slab_set->buffers[i]);
  }

  iree_host_size_t chain_semaphore_count = 0;
  iree_hal_semaphore_t** chain_semaphores = NULL;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_create_chain_semaphores(
        load_count, loads, host_allocator, &chain_semaphores,
        &chain_semaphore_count);
  }
  for (iree_host_size_t i = 0; i < load_count && iree_status_is_ok(status);
       ++i) {
    const id4_pipeline_parameter_slab_load_t* load = &loads[i];
    if (!iree_io_parameter_provider_query_support(provider,
                                                  load->slab->scope)) {
      status =
          iree_make_status(IREE_STATUS_NOT_FOUND,
                           "parameter provider does not support scope '%.*s'",
                           (int)load->slab->scope.size, load->slab->scope.data);
      break;
    }
    id4_pipeline_parameter_slab_enumerator_state_t enumerator_state = {
        // Slab plan supplying request keys and spans.
        .slab = load->slab,
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
        i + 1 == load_count ? NULL : chain_semaphores[i];
    uint64_t chain_signal_value = 1;
    iree_hal_semaphore_list_t gather_signal_semaphore_list =
        i + 1 == load_count
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
        load->slab->scope, slab_set->buffers[i], load->slab->request_count,
        enumerator);
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
