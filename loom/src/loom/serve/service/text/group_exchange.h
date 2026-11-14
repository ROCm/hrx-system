// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_GROUP_EXCHANGE_H_
#define LOOM_SERVE_SERVICE_TEXT_GROUP_EXCHANGE_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_text_group_exchange_t
    loom_serve_text_group_exchange_t;

// One-generation resident ABI: request {generation@0, ACK@64,
// admission@128}; response contains sixteen generation words at stride 64.
// The caller supplies HAL-qualified uncached shared storage with persistent
// host access. A group owns its exchange until all accepted lanes retire.
iree_status_t loom_serve_text_group_exchange_create(
    iree_host_size_t capacity, iree_allocator_t allocator,
    loom_serve_text_group_exchange_t** out_exchange);
iree_status_t loom_serve_text_group_exchange_add(
    loom_serve_text_group_exchange_t* exchange, iree_hal_device_t* device,
    iree_hal_buffer_t* buffer, iree_device_size_t request_offset,
    iree_device_size_t response_offset, iree_hal_semaphore_t** out_ready);
iree_status_t loom_serve_text_group_exchange_add_resident(
    loom_serve_text_group_exchange_t* exchange, iree_hal_device_t* device,
    iree_hal_buffer_t* buffer, iree_device_size_t request_offset,
    iree_device_size_t response_offset, uint32_t generations);

// Publishes all admissions before numerical enqueue, then bridges DMA payload
// generations to independent HAL readiness. timeout bounds a whole group.
iree_status_t loom_serve_text_group_exchange_start(
    loom_serve_text_group_exchange_t* exchange, iree_duration_t timeout);
// Stops the bridge and publishes ABORT + final ACK on cancellation, including
// after payload readiness. Never authorizes storage reuse or terminal success.
void loom_serve_text_group_exchange_abort(
    loom_serve_text_group_exchange_t* exchange);
iree_status_t loom_serve_text_group_exchange_check(
    loom_serve_text_group_exchange_t* exchange);
// Only after accepted execution retirement (or before any numerical enqueue).
void loom_serve_text_group_exchange_free(
    loom_serve_text_group_exchange_t* exchange);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif
