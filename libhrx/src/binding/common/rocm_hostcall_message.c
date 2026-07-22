// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/rocm_hostcall_message.h"

#include <stdio.h>
#include <string.h>

#include "common/printf_format.h"
#include "iree/base/internal/math.h"

void iree_hal_streaming_hostcall_message_table_initialize(
    iree_allocator_t host_allocator,
    iree_hal_streaming_hostcall_message_table_t* out_table) {
  memset(out_table, 0, sizeof(*out_table));
  out_table->host_allocator = host_allocator;
  out_table->free_head = IREE_HOST_SIZE_MAX;
}

void iree_hal_streaming_hostcall_message_table_deinitialize(
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

  iree_host_size_t new_capacity = table->capacity ? table->capacity : 16;
  while (new_capacity < minimum_capacity) {
    if (IREE_UNLIKELY(new_capacity > IREE_HOST_SIZE_MAX / 2)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "hostcall message table capacity overflow");
    }
    new_capacity *= 2;
  }
  iree_host_size_t allocation_size = 0;
  iree_host_size_t total_allocation_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(new_capacity,
                                                sizeof(table->messages[0]),
                                                &allocation_size) ||
                    !iree_host_size_checked_add(allocation_size,
                                                table->allocated_payload_bytes,
                                                &total_allocation_size) ||
                    total_allocation_size >
                        IREE_HAL_STREAMING_HOSTCALL_MAX_MESSAGE_TABLE_BYTES)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "hostcall message table allocation limit exceeded");
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
  if (table->free_head != IREE_HOST_SIZE_MAX) {
    const iree_host_size_t message_id = table->free_head;
    iree_hal_streaming_hostcall_message_t* message =
        &table->messages[message_id];
    table->free_head = message->next_free;
    message->count = 0;
    message->live = true;
    *out_message_id = message_id;
    *out_message = message;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_hostcall_message_table_grow(table, table->count + 1));
  iree_hal_streaming_hostcall_message_t* message =
      &table->messages[table->count];
  message->count = 0;
  message->next_free = IREE_HOST_SIZE_MAX;
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
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(message->count, count, &required_count) ||
          required_count > IREE_HAL_STREAMING_HOSTCALL_MAX_MESSAGE_BYTES /
                               sizeof(message->data[0]))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "hostcall message allocation limit exceeded");
  }
  if (message->capacity < required_count) {
    iree_host_size_t new_capacity = message->capacity ? message->capacity : 16;
    while (new_capacity < required_count) new_capacity *= 2;

    iree_host_size_t allocation_size = 0;
    iree_host_size_t old_allocation_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(new_capacity,
                                                  sizeof(message->data[0]),
                                                  &allocation_size) ||
                      !iree_host_size_checked_mul(message->capacity,
                                                  sizeof(message->data[0]),
                                                  &old_allocation_size))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "hostcall message storage size overflow");
    }
    const iree_host_size_t allocation_growth =
        allocation_size - old_allocation_size;
    const iree_host_size_t message_table_size =
        table->capacity * sizeof(table->messages[0]);
    iree_host_size_t total_allocation_size = 0;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_add(message_table_size,
                                        table->allocated_payload_bytes,
                                        &total_allocation_size) ||
            !iree_host_size_checked_add(total_allocation_size,
                                        allocation_growth,
                                        &total_allocation_size) ||
            total_allocation_size >
                IREE_HAL_STREAMING_HOSTCALL_MAX_MESSAGE_TABLE_BYTES)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "hostcall message table allocation limit exceeded");
    }

    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        table->host_allocator, allocation_size, (void**)&message->data));
    message->capacity = new_capacity;
    table->allocated_payload_bytes += allocation_growth;
  }
  memcpy(message->data + message->count, data, count * sizeof(data[0]));
  message->count = required_count;
  return iree_ok_status();
}

static void iree_hal_streaming_hostcall_message_discard(
    iree_hal_streaming_hostcall_message_table_t* table,
    iree_hal_streaming_hostcall_message_t* message) {
  const iree_host_size_t message_id =
      (iree_host_size_t)(message - table->messages);
  message->count = 0;
  message->live = false;
  message->next_free = table->free_head;
  table->free_head = message_id;
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

bool iree_hal_streaming_hostcall_message_handle_printf(
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
    iree_hal_streaming_hostcall_message_discard(table, message);
    return false;
  }

  if (!end) return true;
  if (message->count == 0) {
    payload[0] = (uint64_t)-1;
    iree_hal_streaming_hostcall_message_discard(table, message);
    return false;
  }

  const uint64_t control = message->data[0];
  FILE* stream = (control & 1u) ? stderr : stdout;
  if (control & ~1ull) {
    payload[0] = (uint64_t)-1;
    iree_hal_streaming_hostcall_message_discard(table, message);
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
  iree_hal_streaming_hostcall_message_discard(table, message);
  return result >= 0;
}
