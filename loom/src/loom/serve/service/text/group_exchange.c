// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/group_exchange.h"

#include "iree/base/threading/thread.h"

typedef struct loom_serve_text_exchange_slot_t {
  iree_hal_buffer_t* buffer;
  iree_device_size_t request_offset;
  iree_device_size_t response_offset;
  iree_hal_buffer_mapping_t request;
  iree_hal_buffer_mapping_t response;
  iree_hal_semaphore_t* ready;
  uint32_t generations;
  bool continuous;
} loom_serve_text_exchange_slot_t;

struct loom_serve_text_group_exchange_t {
  iree_allocator_t allocator;
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_thread_t* thread;
  iree_atomic_int32_t cancel;
  iree_atomic_int32_t stop;
  iree_time_t deadline;
  bool started;
  loom_serve_text_exchange_slot_t slots[];
};

iree_status_t loom_serve_text_group_exchange_create(
    iree_host_size_t capacity, iree_allocator_t allocator,
    loom_serve_text_group_exchange_t** out_exchange) {
  *out_exchange = NULL;
  if (!capacity || capacity > 65536) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid resident exchange capacity");
  }
  loom_serve_text_group_exchange_t* exchange = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      allocator, sizeof(*exchange), capacity, sizeof(*exchange->slots),
      (void**)&exchange));
  exchange->allocator = allocator;
  exchange->capacity = capacity;
  *out_exchange = exchange;
  return iree_ok_status();
}

static bool loom_serve_text_exchange_overlaps(iree_device_size_t a,
                                              iree_device_size_t al,
                                              iree_device_size_t b,
                                              iree_device_size_t bl) {
  return a < b + bl && b < a + al;
}

iree_status_t loom_serve_text_group_exchange_add(
    loom_serve_text_group_exchange_t* exchange, iree_hal_device_t* device,
    iree_hal_buffer_t* buffer, iree_device_size_t request_offset,
    iree_device_size_t response_offset, iree_hal_semaphore_t** out_ready) {
  *out_ready = NULL;
  const iree_device_size_t length = iree_hal_buffer_byte_length(buffer);
  if (exchange->started || exchange->count == exchange->capacity ||
      request_offset % 64 || response_offset % 64 || request_offset > length ||
      length - request_offset < 192 || response_offset > length ||
      length - response_offset < 1024 ||
      loom_serve_text_exchange_overlaps(request_offset, 192, response_offset,
                                        1024)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid resident exchange ranges");
  }
  for (iree_host_size_t i = 0; i < exchange->count; ++i) {
    const loom_serve_text_exchange_slot_t* slot = &exchange->slots[i];
    if (iree_hal_buffer_test_overlap(buffer, request_offset, 192, slot->buffer,
                                     slot->request_offset,
                                     192) != IREE_HAL_BUFFER_OVERLAP_DISJOINT ||
        iree_hal_buffer_test_overlap(buffer, request_offset, 192, slot->buffer,
                                     slot->response_offset, 1024) !=
            IREE_HAL_BUFFER_OVERLAP_DISJOINT ||
        iree_hal_buffer_test_overlap(buffer, response_offset, 1024,
                                     slot->buffer, slot->request_offset,
                                     192) != IREE_HAL_BUFFER_OVERLAP_DISJOINT ||
        iree_hal_buffer_test_overlap(
            buffer, response_offset, 1024, slot->buffer, slot->response_offset,
            1024) != IREE_HAL_BUFFER_OVERLAP_DISJOINT) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "resident exchange slots alias");
    }
  }
  loom_serve_text_exchange_slot_t* slot = &exchange->slots[exchange->count++];
  slot->buffer = buffer;
  slot->generations = 1;
  iree_hal_buffer_retain(buffer);
  slot->request_offset = request_offset;
  slot->response_offset = response_offset;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, 0,
      request_offset, 192, &slot->request));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, 0,
      response_offset, 1024, &slot->response));
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &slot->ready));
  memset(slot->request.contents.data, 0, 192);
  memset(slot->response.contents.data, 0, 1024);
  iree_unaligned_store_le_u32(slot->request.contents.data, 1);
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_mapping_flush_range(&slot->request, 0, 192));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_mapping_flush_range(&slot->response, 0, 1024));
  *out_ready = slot->ready;
  return iree_ok_status();
}

iree_status_t loom_serve_text_group_exchange_add_resident(
    loom_serve_text_group_exchange_t* exchange, iree_hal_device_t* device,
    iree_hal_buffer_t* buffer, iree_device_size_t request_offset,
    iree_device_size_t response_offset, uint32_t generations) {
  if (!generations || generations >= INT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid resident generation count");
  }
  iree_hal_semaphore_t* ready = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_text_group_exchange_add(
      exchange, device, buffer, request_offset, response_offset, &ready));
  loom_serve_text_exchange_slot_t* slot = &exchange->slots[exchange->count - 1];
  slot->generations = generations;
  slot->continuous = true;
  iree_unaligned_store_le_u32(slot->request.contents.data, 0);
  return iree_hal_buffer_mapping_flush_range(&slot->request, 0, 4);
}

static void loom_serve_text_exchange_abort_slots(
    loom_serve_text_group_exchange_t* exchange) {
  iree_atomic_store(&exchange->cancel, 1, iree_memory_order_release);
  for (iree_host_size_t i = 0; i < exchange->count; ++i) {
    loom_serve_text_exchange_slot_t* slot = &exchange->slots[i];
    if (slot->request.contents.data) {
      volatile uint32_t* words =
          (volatile uint32_t*)slot->request.contents.data;
      words[32] = UINT32_MAX;
      words[0] = slot->continuous ? INT32_MAX : 1;
      words[16] = slot->generations;
      iree_status_ignore(
          iree_hal_buffer_mapping_flush_range(&slot->request, 0, 192));
    }
    if (slot->ready) {
      iree_hal_semaphore_fail(
          slot->ready,
          iree_make_status(IREE_STATUS_ABORTED, "resident exchange aborted"));
    }
  }
}

static int loom_serve_text_exchange_poll(void* self) {
  loom_serve_text_group_exchange_t* exchange = self;
  for (iree_host_size_t i = 0; i < exchange->count; ++i) {
    loom_serve_text_exchange_slot_t* slot = &exchange->slots[i];
    for (;;) {
      if (iree_atomic_load(&exchange->stop, iree_memory_order_acquire)) {
        return 0;
      }
      if (iree_atomic_load(&exchange->cancel, iree_memory_order_acquire) ||
          iree_time_now() >= exchange->deadline) {
        loom_serve_text_exchange_abort_slots(exchange);
        return 0;
      }
      if (slot->continuous) {
        iree_status_t status =
            iree_hal_buffer_mapping_invalidate_range(&slot->request, 64, 4);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          loom_serve_text_exchange_abort_slots(exchange);
          return 0;
        }
        const iree_atomic_int32_t* words =
            (const iree_atomic_int32_t*)slot->request.contents.data;
        if ((uint32_t)iree_atomic_load(&words[16], iree_memory_order_acquire) ==
            slot->generations) {
          break;
        }
        iree_wait_until(iree_time_now() + 100 * 1000);
        continue;
      }
      iree_status_t status =
          iree_hal_buffer_mapping_invalidate_range(&slot->response, 0, 1024);
      if (!iree_status_is_ok(status)) {
        iree_hal_semaphore_fail(slot->ready, status);
        loom_serve_text_exchange_abort_slots(exchange);
        return 0;
      }
      const iree_atomic_int32_t* words =
          (const iree_atomic_int32_t*)slot->response.contents.data;
      bool ready = true;
      for (iree_host_size_t owner = 0; owner < 16; ++owner) {
        ready &= iree_atomic_load(&words[owner * 16],
                                  iree_memory_order_acquire) == 1;
      }
      if (ready) {
        status = iree_hal_semaphore_signal(slot->ready, 1, NULL);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          loom_serve_text_exchange_abort_slots(exchange);
          return 0;
        }
        break;
      }
      iree_wait_until(iree_time_now() + 100 * 1000);
    }
  }
  return 0;
}

iree_status_t loom_serve_text_group_exchange_start(
    loom_serve_text_group_exchange_t* exchange, iree_duration_t timeout) {
  if (exchange->started || !exchange->count || timeout <= 0 ||
      timeout > IREE_TIME_INFINITE_FUTURE - iree_time_now()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid resident exchange start");
  }
  exchange->started = true;
  exchange->deadline = iree_time_now() + timeout;
  for (iree_host_size_t i = 0; i < exchange->count; ++i) {
    loom_serve_text_exchange_slot_t* slot = &exchange->slots[i];
    iree_unaligned_store_le_u32(slot->request.contents.data + 128, 1);
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_mapping_flush_range(&slot->request, 128, 4));
  }
  return iree_thread_create(
      loom_serve_text_exchange_poll, exchange,
      (iree_thread_create_params_t){.name = IREE_SV("loom-resident")},
      exchange->allocator, &exchange->thread);
}

void loom_serve_text_group_exchange_abort(
    loom_serve_text_group_exchange_t* exchange) {
  if (!exchange) {
    return;
  }
  iree_atomic_store(&exchange->cancel, 1, iree_memory_order_release);
  iree_thread_release(exchange->thread);
  exchange->thread = NULL;
  loom_serve_text_exchange_abort_slots(exchange);
}

void loom_serve_text_group_exchange_free(
    loom_serve_text_group_exchange_t* exchange) {
  if (!exchange) {
    return;
  }
  iree_atomic_store(&exchange->stop, 1, iree_memory_order_release);
  iree_thread_release(exchange->thread);
  for (iree_host_size_t i = 0; i < exchange->count; ++i) {
    loom_serve_text_exchange_slot_t* slot = &exchange->slots[i];
    if (slot->request.contents.data) {
      iree_status_ignore(iree_hal_buffer_unmap_range(&slot->request));
    }
    if (slot->response.contents.data) {
      iree_status_ignore(iree_hal_buffer_unmap_range(&slot->response));
    }
    iree_hal_semaphore_release(slot->ready);
    iree_hal_buffer_release(slot->buffer);
  }
  iree_allocator_free(exchange->allocator, exchange);
}

iree_status_t loom_serve_text_group_exchange_check(
    loom_serve_text_group_exchange_t* exchange) {
  if (exchange &&
      iree_atomic_load(&exchange->cancel, iree_memory_order_acquire)) {
    return iree_make_status(IREE_STATUS_ABORTED, "resident exchange aborted");
  }
  return iree_ok_status();
}
