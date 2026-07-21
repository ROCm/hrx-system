// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "common/internal.h"
#include "common/printf_format.h"
#include "common/rocm_hostcall_packet.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/thread.h"

#define IREE_HAL_STREAMING_HOSTCALL_SERVICE_FUNCTION_CALL 1u
#define IREE_HAL_STREAMING_HOSTCALL_SERVICE_PRINTF 2u
#define IREE_HAL_STREAMING_HOSTCALL_SERVICE_DEVMEM 3u
#define IREE_HAL_STREAMING_HOSTCALL_SERVICE_SANITIZER 4u

typedef struct iree_hal_streaming_hostcall_message_t {
  // Accumulated qword payload for one in-flight device-library message.
  uint64_t* data;
  // Number of qwords currently accumulated in |data|.
  iree_host_size_t count;
  // Capacity of |data| in qwords.
  iree_host_size_t capacity;
  // True while this message ID is active.
  bool live;
} iree_hal_streaming_hostcall_message_t;

typedef struct iree_hal_streaming_hostcall_message_table_t {
  // Indexed message table. Device descriptors carry the table index.
  iree_hal_streaming_hostcall_message_t* messages;
  // Number of entries in |messages|.
  iree_host_size_t count;
  // Capacity of |messages|.
  iree_host_size_t capacity;
  // Allocator used for table and message payload storage.
  iree_allocator_t host_allocator;
} iree_hal_streaming_hostcall_message_table_t;

typedef struct iree_hal_streaming_rocm_hostcall_service_t {
  // Driver-owned notification used to wake the hostcall listener.
  iree_hal_host_notification_t* notification;
  // Queue-visible hostcall buffer.
  iree_hal_streaming_buffer_t* buffer;
  // Host view of |buffer|'s ABI header.
  iree_hal_streaming_hostcall_buffer_header_t* hostcall_buffer;
  // Host view of the packet header array.
  iree_hal_streaming_hostcall_packet_header_t* packet_headers;
  // Host view of the packet payload array.
  iree_hal_streaming_hostcall_payload_t* packet_payloads;
  // Immutable number of entries in the host packet arrays.
  uint32_t packet_count;
  // Listener thread that processes packets while kernels are running.
  iree_thread_t* listener_thread;
  // Non-zero after shutdown requests the listener to exit.
  iree_atomic_int32_t stop_requested;
  // Accumulated device-library printf messages.
  iree_hal_streaming_hostcall_message_table_t messages;
  // Host allocator used for the service object and dynamic message storage.
  iree_allocator_t host_allocator;
} iree_hal_streaming_rocm_hostcall_service_t;

static_assert(sizeof(iree_atomic_uint32_t) == sizeof(uint32_t),
              "hostcall control field must remain a 32-bit ABI field");
static_assert(sizeof(iree_atomic_uint64_t) == sizeof(uint64_t),
              "hostcall ready stack field must remain a 64-bit ABI field");

static iree_host_size_t iree_hal_streaming_hostcall_align(
    iree_host_size_t value, iree_host_size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

iree_status_t iree_hal_streaming_rocm_hostcall_calculate_packet_count(
    const iree_hal_streaming_device_t* device, uint32_t* out_packet_count) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_packet_count);
  *out_packet_count = 0;

  if (IREE_UNLIKELY(device->raw_compute_unit_count == 0 ||
                    device->maximum_resident_subgroup_count == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "hostcall sizing requires physical compute-unit and resident-wave "
        "limits");
  }

  // One packet is required for every wave that can be resident on the queue.
  // The raw execution properties deliberately remain separate from HIP's
  // compatibility-adjusted public device attributes.
  const uint64_t minimum_packet_count =
      (uint64_t)device->raw_compute_unit_count *
      device->maximum_resident_subgroup_count;
  if (IREE_UNLIKELY(minimum_packet_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "hostcall packet count exceeds ABI range");
  }

  // Tagged stack pointers encode the packet index with a bitmask. Round up so
  // the allocated table covers every representable index selected by that mask.
  uint32_t packet_count = 2;
  while (packet_count < minimum_packet_count) {
    if (IREE_UNLIKELY(packet_count > UINT32_MAX / 2)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "hostcall packet count exceeds ABI range");
    }
    packet_count *= 2;
  }

  *out_packet_count = packet_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_hostcall_buffer_layout(
    uint32_t packet_count, iree_host_size_t* out_headers_offset,
    iree_host_size_t* out_payloads_offset, iree_host_size_t* out_total_size) {
  if (IREE_UNLIKELY(packet_count == 0 ||
                    (packet_count & (packet_count - 1)) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hostcall packet count must be a non-zero power of two");
  }

  iree_host_size_t headers_offset = iree_hal_streaming_hostcall_align(
      sizeof(iree_hal_streaming_hostcall_buffer_header_t),
      iree_alignof(iree_hal_streaming_hostcall_packet_header_t));
  iree_host_size_t headers_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          packet_count, sizeof(iree_hal_streaming_hostcall_packet_header_t),
          &headers_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "hostcall packet header table size overflow");
  }
  iree_host_size_t headers_end = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(headers_offset, headers_size,
                                                &headers_end))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "hostcall packet header table range overflow");
  }

  iree_host_size_t payloads_offset = iree_hal_streaming_hostcall_align(
      headers_end, iree_alignof(iree_hal_streaming_hostcall_payload_t));
  iree_host_size_t payloads_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          packet_count, sizeof(iree_hal_streaming_hostcall_payload_t),
          &payloads_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "hostcall packet payload table size overflow");
  }
  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(payloads_offset, payloads_size,
                                                &total_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "hostcall buffer size overflow");
  }

  *out_headers_offset = headers_offset;
  *out_payloads_offset = payloads_offset;
  *out_total_size = total_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_hostcall_message_table_initialize(
    iree_allocator_t host_allocator,
    iree_hal_streaming_hostcall_message_table_t* out_table) {
  memset(out_table, 0, sizeof(*out_table));
  out_table->host_allocator = host_allocator;
  return iree_ok_status();
}

static void iree_hal_streaming_hostcall_message_table_deinitialize(
    iree_hal_streaming_hostcall_message_table_t* table) {
  for (iree_host_size_t i = 0; i < table->count; ++i) {
    iree_allocator_free(table->host_allocator, table->messages[i].data);
  }
  iree_allocator_free(table->host_allocator, table->messages);
  memset(table, 0, sizeof(*table));
}

static iree_status_t iree_hal_streaming_hostcall_message_table_grow(
    iree_hal_streaming_hostcall_message_table_t* table,
    iree_host_size_t minimum_capacity) {
  if (table->capacity >= minimum_capacity) return iree_ok_status();
  iree_host_size_t new_capacity = table->capacity ? table->capacity * 2 : 16;
  while (new_capacity < minimum_capacity) new_capacity *= 2;
  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          new_capacity, sizeof(table->messages[0]), &allocation_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "hostcall message table size overflow");
  }
  const iree_host_size_t old_capacity = table->capacity;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(
      table->host_allocator, allocation_size, (void**)&table->messages));
  memset(table->messages + old_capacity, 0,
         (new_capacity - old_capacity) * sizeof(table->messages[0]));
  table->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_hostcall_message_allocate(
    iree_hal_streaming_hostcall_message_table_t* table,
    uint64_t* out_message_id,
    iree_hal_streaming_hostcall_message_t** out_message) {
  for (iree_host_size_t i = 0; i < table->count; ++i) {
    iree_hal_streaming_hostcall_message_t* message = &table->messages[i];
    if (message->live) continue;
    message->count = 0;
    message->live = true;
    *out_message_id = i;
    *out_message = message;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_hostcall_message_table_grow(table, table->count + 1));
  iree_hal_streaming_hostcall_message_t* message =
      &table->messages[table->count];
  message->count = 0;
  message->live = true;
  *out_message_id = table->count++;
  *out_message = message;
  return iree_ok_status();
}

static iree_hal_streaming_hostcall_message_t*
iree_hal_streaming_hostcall_message_lookup(
    iree_hal_streaming_hostcall_message_table_t* table, uint64_t message_id) {
  if (message_id >= table->count) return NULL;
  iree_hal_streaming_hostcall_message_t* message = &table->messages[message_id];
  return message->live ? message : NULL;
}

static iree_status_t iree_hal_streaming_hostcall_message_append(
    iree_hal_streaming_hostcall_message_table_t* table,
    iree_hal_streaming_hostcall_message_t* message, const uint64_t* data,
    iree_host_size_t count) {
  if (count == 0) return iree_ok_status();
  iree_host_size_t required_count = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(message->count, count,
                                                &required_count))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "hostcall message length overflow");
  }
  if (message->capacity < required_count) {
    iree_host_size_t new_capacity =
        message->capacity ? message->capacity * 2 : 16;
    while (new_capacity < required_count) new_capacity *= 2;
    iree_host_size_t allocation_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            new_capacity, sizeof(message->data[0]), &allocation_size))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "hostcall message storage size overflow");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        table->host_allocator, allocation_size, (void**)&message->data));
    message->capacity = new_capacity;
  }
  memcpy(message->data + message->count, data, count * sizeof(data[0]));
  message->count = required_count;
  return iree_ok_status();
}

static void iree_hal_streaming_hostcall_message_discard(
    iree_hal_streaming_hostcall_message_t* message) {
  message->count = 0;
  message->live = false;
}

static uint64_t iree_hal_streaming_hostcall_descriptor_field(uint64_t value,
                                                             uint32_t offset,
                                                             uint32_t width) {
  return (value >> offset) & ((1ull << width) - 1);
}

static uint64_t iree_hal_streaming_hostcall_descriptor_set_field(
    uint64_t descriptor, uint64_t value, uint32_t offset, uint32_t width) {
  const uint64_t reset_mask = ~(((1ull << width) - 1) << offset);
  return (descriptor & reset_mask) | (value << offset);
}

static bool iree_hal_streaming_hostcall_handle_printf(
    iree_hal_streaming_hostcall_message_table_t* table, uint64_t* payload) {
  enum {
    kDescriptorOffsetBegin = 0,
    kDescriptorOffsetEnd = 1,
    kDescriptorOffsetReserved = 2,
    kDescriptorOffsetLength = 5,
    kDescriptorOffsetId = 8,
  };

  uint64_t descriptor = payload[0];
  const bool begin = iree_hal_streaming_hostcall_descriptor_field(
                         descriptor, kDescriptorOffsetBegin, 1) != 0;
  const bool end = iree_hal_streaming_hostcall_descriptor_field(
                       descriptor, kDescriptorOffsetEnd, 1) != 0;
  const uint64_t reserved = iree_hal_streaming_hostcall_descriptor_field(
      descriptor, kDescriptorOffsetReserved, 3);
  const uint64_t length = iree_hal_streaming_hostcall_descriptor_field(
      descriptor, kDescriptorOffsetLength, 3);
  if (IREE_UNLIKELY(reserved != 0)) {
    payload[0] = (uint64_t)-1;
    return false;
  }

  iree_hal_streaming_hostcall_message_t* message = NULL;
  if (begin) {
    uint64_t message_id = 0;
    if (!iree_status_is_ok(iree_hal_streaming_hostcall_message_allocate(
            table, &message_id, &message))) {
      payload[0] = (uint64_t)-1;
      return false;
    }
    descriptor &= ~(1ull << kDescriptorOffsetBegin);
    descriptor = iree_hal_streaming_hostcall_descriptor_set_field(
        descriptor, message_id, kDescriptorOffsetId, 56);
    payload[0] = descriptor;
  } else {
    const uint64_t message_id = iree_hal_streaming_hostcall_descriptor_field(
        descriptor, kDescriptorOffsetId, 56);
    message = iree_hal_streaming_hostcall_message_lookup(table, message_id);
    if (!message) {
      payload[0] = (uint64_t)-1;
      return false;
    }
  }

  if (!iree_status_is_ok(iree_hal_streaming_hostcall_message_append(
          table, message, payload + 1, (iree_host_size_t)length))) {
    payload[0] = (uint64_t)-1;
    iree_hal_streaming_hostcall_message_discard(message);
    return false;
  }

  if (!end) return true;

  if (message->count == 0) {
    payload[0] = (uint64_t)-1;
    iree_hal_streaming_hostcall_message_discard(message);
    return false;
  }

  const uint64_t control = message->data[0];
  FILE* stream = (control & 1u) ? stderr : stdout;
  if (control & ~1ull) {
    payload[0] = (uint64_t)-1;
    iree_hal_streaming_hostcall_message_discard(message);
    return false;
  }
  const uint8_t* message_bytes = (const uint8_t*)(message->data + 1);
  iree_host_size_t message_length = 0;
  int result = -1;
  if (iree_host_size_checked_mul(message->count - 1, sizeof(uint64_t),
                                 &message_length)) {
    const char* format_end = memchr(message_bytes, '\0', message_length);
    if (format_end) {
      const iree_host_size_t format_length =
          (iree_host_size_t)(format_end - (const char*)message_bytes);
      const iree_host_size_t argument_offset = (format_length + 8u) & ~7u;
      if (argument_offset <= message_length) {
        iree_status_t status = iree_hal_streaming_printf_format(
            stream,
            iree_make_string_view((const char*)message_bytes, format_length),
            message_bytes + argument_offset, message_length - argument_offset,
            &result);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          result = -1;
        }
      }
    }
  }
  fflush(stream);
  payload[0] = (uint64_t)(int64_t)result;
  payload[1] = 0;
  iree_hal_streaming_hostcall_message_discard(message);
  return result >= 0;
}

static void iree_hal_streaming_hostcall_complete_unsupported(
    uint32_t service, uint64_t* payload) {
  switch (service) {
    case IREE_HAL_STREAMING_HOSTCALL_SERVICE_FUNCTION_CALL:
    case IREE_HAL_STREAMING_HOSTCALL_SERVICE_DEVMEM:
    case IREE_HAL_STREAMING_HOSTCALL_SERVICE_SANITIZER:
      payload[0] = 0;
      payload[1] = 0;
      return;
    default:
      payload[0] = (uint64_t)-1;
      payload[1] = 0;
      return;
  }
}

static void iree_hal_streaming_hostcall_process_payload(
    iree_hal_streaming_rocm_hostcall_service_t* service, uint32_t service_id,
    uint64_t* payload) {
  if (service_id == IREE_HAL_STREAMING_HOSTCALL_SERVICE_PRINTF) {
    (void)iree_hal_streaming_hostcall_handle_printf(&service->messages,
                                                    payload);
    return;
  }
  iree_hal_streaming_hostcall_complete_unsupported(service_id, payload);
}

static void iree_hal_streaming_hostcall_process_packets(
    iree_hal_streaming_rocm_hostcall_service_t* service) {
  uint64_t ready_stack = iree_atomic_exchange(
      &service->hostcall_buffer->ready_stack, 0, iree_memory_order_acquire);
  iree_hal_streaming_hostcall_packet_iterator_t iterator;
  iree_hal_streaming_hostcall_packet_iterator_initialize(
      service->packet_headers, service->packet_count, ready_stack, &iterator);
  uint32_t packet_index = 0;
  while (
      !iree_atomic_load(&service->stop_requested, iree_memory_order_acquire) &&
      iree_hal_streaming_hostcall_packet_iterator_advance(&iterator,
                                                          &packet_index)) {
    iree_hal_streaming_hostcall_packet_header_t* header =
        &service->packet_headers[packet_index];

    uint64_t activemask = header->activemask;
    while (activemask != 0) {
      const int lane = __builtin_ctzll(activemask);
      activemask &= ~(1ull << lane);
      iree_hal_streaming_hostcall_process_payload(
          service, header->service,
          service->packet_payloads[packet_index].slots[lane]);
    }

    const uint32_t control =
        iree_atomic_load(&header->control, iree_memory_order_relaxed);
    iree_atomic_store(&header->control, control & ~1u,
                      iree_memory_order_release);
  }
}

static int iree_hal_streaming_hostcall_listener_main(void* user_data) {
  iree_hal_streaming_rocm_hostcall_service_t* service =
      (iree_hal_streaming_rocm_hostcall_service_t*)user_data;
  uint64_t signal_value = IREE_HAL_HOST_NOTIFICATION_INITIAL_VALUE;
  while (
      !iree_atomic_load(&service->stop_requested, iree_memory_order_acquire)) {
    const uint64_t new_value =
        iree_hal_host_notification_wait(service->notification, signal_value);
    if (iree_atomic_load(&service->stop_requested, iree_memory_order_acquire)) {
      break;
    }
    if (new_value == signal_value) continue;
    signal_value = new_value;
    iree_hal_streaming_hostcall_process_packets(service);
  }
  return 0;
}

static void iree_hal_streaming_hostcall_buffer_initialize(
    iree_hal_streaming_buffer_t* buffer, uint64_t doorbell_token,
    uint32_t packet_count, iree_host_size_t headers_offset,
    iree_host_size_t payloads_offset,
    iree_hal_streaming_rocm_hostcall_service_t* service) {
  service->hostcall_buffer =
      (iree_hal_streaming_hostcall_buffer_header_t*)buffer->host_ptr;
  service->packet_headers =
      (iree_hal_streaming_hostcall_packet_header_t*)((uint8_t*)
                                                         buffer->host_ptr +
                                                     headers_offset);
  service->packet_payloads =
      (iree_hal_streaming_hostcall_payload_t*)((uint8_t*)buffer->host_ptr +
                                               payloads_offset);
  service->packet_count = packet_count;
  const uint64_t packet_index_mask = packet_count - 1;

  service->hostcall_buffer->headers = buffer->device_ptr + headers_offset;
  service->hostcall_buffer->payloads = buffer->device_ptr + payloads_offset;
  service->hostcall_buffer->doorbell = doorbell_token;
  service->hostcall_buffer->index_mask = packet_index_mask;
  iree_atomic_store(&service->hostcall_buffer->ready_stack, 0,
                    iree_memory_order_relaxed);

  uint64_t next = packet_index_mask + 1;
  service->packet_headers[0].next = 0;
  for (uint32_t i = 1; i < packet_count; ++i) {
    service->packet_headers[i].next = next;
    next = i;
  }
  service->hostcall_buffer->free_stack = next;
}

static void iree_hal_streaming_rocm_hostcall_service_destroy(
    iree_hal_streaming_rocm_hostcall_service_t* service) {
  if (!service) return;
  if (service->listener_thread) {
    iree_atomic_store(&service->stop_requested, 1, iree_memory_order_release);
    iree_hal_host_notification_wake(service->notification);
    iree_thread_join(service->listener_thread);
    iree_thread_release(service->listener_thread);
  }
  iree_hal_host_notification_release(service->notification);
  iree_hal_streaming_memory_release_transient_buffer(service->buffer);
  iree_hal_streaming_hostcall_message_table_deinitialize(&service->messages);
  iree_allocator_free(service->host_allocator, service);
}

static iree_status_t iree_hal_streaming_rocm_hostcall_service_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_rocm_hostcall_service_t** out_service) {
  *out_service = NULL;

  iree_hal_streaming_rocm_hostcall_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      context->host_allocator, sizeof(*service), (void**)&service));
  memset(service, 0, sizeof(*service));
  service->host_allocator = context->host_allocator;
  iree_atomic_store(&service->stop_requested, 0, iree_memory_order_relaxed);
  iree_status_t status = iree_hal_streaming_hostcall_message_table_initialize(
      context->host_allocator, &service->messages);

  if (iree_status_is_ok(status)) {
    status = iree_hal_host_notification_create(context->device,
                                               &service->notification);
  }

  iree_host_size_t headers_offset = 0;
  iree_host_size_t payloads_offset = 0;
  iree_host_size_t buffer_size = 0;
  uint32_t packet_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_rocm_hostcall_calculate_packet_count(
        context->device_entry, &packet_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_hostcall_buffer_layout(
        packet_count, &headers_offset, &payloads_offset, &buffer_size);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_memory_allocate_context_runtime_host(
        context, buffer_size, &service->buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_hostcall_buffer_initialize(
        service->buffer,
        iree_hal_host_notification_device_token(service->notification),
        packet_count, headers_offset, payloads_offset, service);
  }

  if (iree_status_is_ok(status)) {
    // Hostcall is a live rendezvous: device-library code can wait inside the
    // dispatch until the host clears packet ready bits. A stream callback
    // queued after the dispatch would run too late, so the listener must exist
    // before kernels receive the hidden hostcall buffer pointer.
    const iree_thread_create_params_t thread_params = {
        .name = IREE_SV("hrx-hostcall"),
        .priority_class = IREE_THREAD_PRIORITY_CLASS_HIGH,
    };
    status = iree_thread_create(iree_hal_streaming_hostcall_listener_main,
                                service, thread_params, context->host_allocator,
                                &service->listener_thread);
  }

  if (iree_status_is_ok(status)) {
    *out_service = service;
  } else {
    iree_hal_streaming_rocm_hostcall_service_destroy(service);
  }
  return status;
}

iree_status_t iree_hal_streaming_context_rocm_hostcall_buffer(
    iree_hal_streaming_context_t* context, uint64_t* out_buffer_device_ptr) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_buffer_device_ptr);
  *out_buffer_device_ptr = 0;

  if (IREE_UNLIKELY(!context->device_entry->host_native_atomic_supported)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "ROCm hostcall requires native host/device atomic support");
  }

  const uint64_t cached_buffer_device_ptr =
      iree_hal_streaming_context_cached_rocm_hostcall_buffer(context);
  if (cached_buffer_device_ptr != 0) {
    *out_buffer_device_ptr = cached_buffer_device_ptr;
    return iree_ok_status();
  }

  iree_slim_mutex_lock(&context->mutex);
  iree_status_t status = iree_ok_status();
  if (!context->rocm_device_runtime.hostcall_service) {
    status = iree_hal_streaming_rocm_hostcall_service_create(
        context, &context->rocm_device_runtime.hostcall_service);
  }
  if (iree_status_is_ok(status)) {
    *out_buffer_device_ptr =
        context->rocm_device_runtime.hostcall_service->buffer->device_ptr;
    iree_atomic_store(&context->rocm_device_runtime.hostcall_buffer_device_ptr,
                      *out_buffer_device_ptr, iree_memory_order_release);
  }
  iree_slim_mutex_unlock(&context->mutex);
  return status;
}

void iree_hal_streaming_context_deinitialize_rocm_hostcall_service(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  iree_slim_mutex_lock(&context->mutex);
  iree_hal_streaming_rocm_hostcall_service_t* service =
      context->rocm_device_runtime.hostcall_service;
  context->rocm_device_runtime.hostcall_service = NULL;
  iree_atomic_store(&context->rocm_device_runtime.hostcall_buffer_device_ptr, 0,
                    iree_memory_order_release);
  iree_slim_mutex_unlock(&context->mutex);
  iree_hal_streaming_rocm_hostcall_service_destroy(service);
}
