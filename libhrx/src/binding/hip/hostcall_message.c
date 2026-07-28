// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/hostcall_message.h"

#include <inttypes.h>
#include <stdbool.h>
#include <string.h>

enum {
  IREE_HIP_HOSTCALL_DESCRIPTOR_BEGIN_OFFSET = 0,
  IREE_HIP_HOSTCALL_DESCRIPTOR_END_OFFSET = 1,
  IREE_HIP_HOSTCALL_DESCRIPTOR_RESERVED_OFFSET = 2,
  IREE_HIP_HOSTCALL_DESCRIPTOR_LENGTH_OFFSET = 5,
  IREE_HIP_HOSTCALL_DESCRIPTOR_ID_OFFSET = 8,
  IREE_HIP_HOSTCALL_DESCRIPTOR_ID_WIDTH = 56,
  IREE_HIP_HOSTCALL_MESSAGE_BLOCK_SIZE = 128,
  IREE_HIP_HOSTCALL_MESSAGE_BLOCK_QWORD_COUNT =
      (IREE_HIP_HOSTCALL_MESSAGE_BLOCK_SIZE - sizeof(iree_arena_block_t)) /
      sizeof(uint64_t),
};

static_assert((IREE_HIP_HOSTCALL_MESSAGE_BLOCK_SIZE -
               sizeof(iree_arena_block_t)) %
                      sizeof(uint64_t) ==
                  0,
              "message block payload must contain a whole number of qwords");

static uint64_t iree_hip_hostcall_descriptor_field(uint64_t value,
                                                   uint32_t offset,
                                                   uint32_t width) {
  return (value >> offset) & ((UINT64_C(1) << width) - 1);
}

static uint64_t iree_hip_hostcall_descriptor_set_field(uint64_t descriptor,
                                                       uint64_t value,
                                                       uint32_t offset,
                                                       uint32_t width) {
  const uint64_t field_mask = ((UINT64_C(1) << width) - 1) << offset;
  return (descriptor & ~field_mask) | ((value << offset) & field_mask);
}

static iree_hip_hostcall_message_t* iree_hip_hostcall_message_lookup_active(
    iree_hip_hostcall_message_table_t* table, uint64_t message_id) {
  if (message_id >= table->count) return NULL;
  iree_hip_hostcall_message_t* message = &table->messages[message_id];
  return message->state == IREE_HIP_HOSTCALL_MESSAGE_STATE_ACTIVE ? message
                                                                  : NULL;
}

static void iree_hip_hostcall_message_discard(
    iree_hip_hostcall_message_table_t* table,
    iree_hip_hostcall_message_t* message) {
  const iree_host_size_t message_id =
      (iree_host_size_t)(message - table->messages);
  if (message->block_head) {
    iree_arena_block_pool_release(&table->block_pool, message->block_head,
                                  message->block_tail);
  }
  message->block_head = NULL;
  message->block_tail = NULL;
  message->count = 0;
  message->state = IREE_HIP_HOSTCALL_MESSAGE_STATE_FREE;
  message->next_free = table->free_head;
  table->free_head = message_id;
}

static iree_status_t iree_hip_hostcall_message_table_grow(
    iree_hip_hostcall_message_table_t* table,
    iree_host_size_t minimum_capacity) {
  if (table->capacity >= minimum_capacity) return iree_ok_status();

  iree_host_size_t new_capacity = table->capacity ? table->capacity : 16;
  while (new_capacity < minimum_capacity) {
    if (IREE_UNLIKELY(new_capacity > IREE_HOST_SIZE_MAX / 2)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "HIP hostcall message table capacity overflow");
    }
    new_capacity *= 2;
  }

  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          new_capacity, sizeof(table->messages[0]), &allocation_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "HIP hostcall message table size overflow");
  }

  const iree_host_size_t old_capacity = table->capacity;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(
      table->host_allocator, allocation_size, (void**)&table->messages));
  memset(table->messages + old_capacity, 0,
         (new_capacity - old_capacity) * sizeof(table->messages[0]));
  table->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_hip_hostcall_message_allocate(
    iree_hip_hostcall_message_table_t* table, uint64_t* out_message_id,
    iree_hip_hostcall_message_t** out_message) {
  if (table->free_head != IREE_HOST_SIZE_MAX) {
    const iree_host_size_t message_id = table->free_head;
    iree_hip_hostcall_message_t* message = &table->messages[message_id];
    table->free_head = message->next_free;
    message->count = 0;
    message->state = IREE_HIP_HOSTCALL_MESSAGE_STATE_ACTIVE;
    *out_message_id = message_id;
    *out_message = message;
    return iree_ok_status();
  }

  if (IREE_UNLIKELY(table->count >=
                    (UINT64_C(1) << IREE_HIP_HOSTCALL_DESCRIPTOR_ID_WIDTH))) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "HIP hostcall message count exceeds the device descriptor ID range");
  }
  iree_host_size_t new_count = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(table->count, 1, &new_count))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "HIP hostcall message count overflow");
  }
  IREE_RETURN_IF_ERROR(iree_hip_hostcall_message_table_grow(table, new_count));
  iree_hip_hostcall_message_t* message = &table->messages[table->count];
  message->count = 0;
  message->next_free = IREE_HOST_SIZE_MAX;
  message->state = IREE_HIP_HOSTCALL_MESSAGE_STATE_ACTIVE;
  *out_message_id = table->count;
  *out_message = message;
  table->count = new_count;
  return iree_ok_status();
}

static iree_status_t iree_hip_hostcall_message_append(
    iree_hip_hostcall_message_table_t* table,
    iree_hip_hostcall_message_t* message, const uint64_t* data,
    iree_host_size_t count) {
  if (count == 0) return iree_ok_status();

  iree_host_size_t required_count = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(message->count, count, &required_count) ||
          required_count > IREE_HOST_SIZE_MAX / sizeof(uint64_t))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "HIP hostcall message size overflow");
  }

  iree_host_size_t source_offset = 0;
  while (source_offset < count) {
    const iree_host_size_t tail_offset =
        message->count % IREE_HIP_HOSTCALL_MESSAGE_BLOCK_QWORD_COUNT;
    if (!message->block_tail || tail_offset == 0) {
      iree_arena_block_t* block = NULL;
      void* block_ptr = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_block_pool_acquire(&table->block_pool,
                                                         &block, &block_ptr));
      if (message->block_tail) {
        message->block_tail->next = block;
      } else {
        message->block_head = block;
      }
      message->block_tail = block;
    }

    const iree_host_size_t block_count =
        iree_min(count - source_offset,
                 IREE_HIP_HOSTCALL_MESSAGE_BLOCK_QWORD_COUNT - tail_offset);
    uint64_t* block_data = (uint64_t*)iree_arena_block_ptr(&table->block_pool,
                                                           message->block_tail);
    memcpy(block_data + tail_offset, data + source_offset,
           block_count * sizeof(data[0]));
    message->count += block_count;
    source_offset += block_count;
  }

  IREE_ASSERT(message->count == required_count);
  return iree_ok_status();
}

void iree_hip_hostcall_message_table_initialize(
    iree_allocator_t host_allocator,
    iree_hip_hostcall_message_table_t* out_table) {
  IREE_ASSERT_ARGUMENT(out_table);
  memset(out_table, 0, sizeof(*out_table));
  out_table->free_head = IREE_HOST_SIZE_MAX;
  out_table->host_allocator = host_allocator;
  iree_arena_block_pool_initialize(IREE_HIP_HOSTCALL_MESSAGE_BLOCK_SIZE,
                                   host_allocator, &out_table->block_pool);
}

void iree_hip_hostcall_message_table_deinitialize(
    iree_hip_hostcall_message_table_t* table) {
  IREE_ASSERT_ARGUMENT(table);
  for (iree_host_size_t i = 0; i < table->count; ++i) {
    iree_hip_hostcall_message_t* message = &table->messages[i];
    if (message->block_head) {
      iree_arena_block_pool_release(&table->block_pool, message->block_head,
                                    message->block_tail);
    }
  }
  iree_allocator_free(table->host_allocator, table->messages);
  iree_arena_block_pool_deinitialize(&table->block_pool);
  memset(table, 0, sizeof(*table));
}

iree_status_t iree_hip_hostcall_message_consume_fragment(
    iree_hip_hostcall_message_table_t* table,
    const uint64_t fragment[IREE_HIP_HOSTCALL_PACKET_SLOT_QWORD_COUNT],
    iree_hip_hostcall_message_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(fragment);
  IREE_ASSERT_ARGUMENT(out_result);
  memset(out_result, 0, sizeof(*out_result));

  uint64_t descriptor = fragment[0];
  const bool begin =
      iree_hip_hostcall_descriptor_field(
          descriptor, IREE_HIP_HOSTCALL_DESCRIPTOR_BEGIN_OFFSET, 1) != 0;
  const bool end =
      iree_hip_hostcall_descriptor_field(
          descriptor, IREE_HIP_HOSTCALL_DESCRIPTOR_END_OFFSET, 1) != 0;
  const uint64_t message_id = iree_hip_hostcall_descriptor_field(
      descriptor, IREE_HIP_HOSTCALL_DESCRIPTOR_ID_OFFSET,
      IREE_HIP_HOSTCALL_DESCRIPTOR_ID_WIDTH);
  const uint64_t reserved = iree_hip_hostcall_descriptor_field(
      descriptor, IREE_HIP_HOSTCALL_DESCRIPTOR_RESERVED_OFFSET, 3);
  if (IREE_UNLIKELY(reserved != 0)) {
    iree_hip_hostcall_message_t* message =
        begin ? NULL
              : iree_hip_hostcall_message_lookup_active(table, message_id);
    if (message) iree_hip_hostcall_message_discard(table, message);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HIP hostcall descriptor reserved bits are nonzero");
  }

  iree_hip_hostcall_message_t* message = NULL;
  if (begin) {
    uint64_t allocated_message_id = 0;
    IREE_RETURN_IF_ERROR(iree_hip_hostcall_message_allocate(
        table, &allocated_message_id, &message));
    descriptor &= ~(UINT64_C(1) << IREE_HIP_HOSTCALL_DESCRIPTOR_BEGIN_OFFSET);
    descriptor = iree_hip_hostcall_descriptor_set_field(
        descriptor, allocated_message_id,
        IREE_HIP_HOSTCALL_DESCRIPTOR_ID_OFFSET,
        IREE_HIP_HOSTCALL_DESCRIPTOR_ID_WIDTH);
  } else {
    message = iree_hip_hostcall_message_lookup_active(table, message_id);
    if (IREE_UNLIKELY(!message)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HIP hostcall descriptor references unknown message ID %" PRIu64,
          message_id);
    }
  }

  const iree_host_size_t fragment_qword_count =
      (iree_host_size_t)iree_hip_hostcall_descriptor_field(
          descriptor, IREE_HIP_HOSTCALL_DESCRIPTOR_LENGTH_OFFSET, 3);
  iree_status_t status = iree_hip_hostcall_message_append(
      table, message, fragment + 1, fragment_qword_count);
  if (!iree_status_is_ok(status)) {
    iree_hip_hostcall_message_discard(table, message);
    return status;
  }

  if (!end) {
    out_result->type = IREE_HIP_HOSTCALL_MESSAGE_RESULT_CONTINUE;
    out_result->continuation_descriptor = descriptor;
    return iree_ok_status();
  }

  message->state = IREE_HIP_HOSTCALL_MESSAGE_STATE_COMPLETE;
  out_result->type = IREE_HIP_HOSTCALL_MESSAGE_RESULT_COMPLETE;
  out_result->message_id = (iree_host_size_t)(message - table->messages);
  out_result->count = message->count;
  return iree_ok_status();
}

void iree_hip_hostcall_message_copy(
    const iree_hip_hostcall_message_table_t* table, iree_host_size_t message_id,
    iree_byte_span_t target) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT(message_id < table->count);
  const iree_hip_hostcall_message_t* message = &table->messages[message_id];
  IREE_ASSERT(message->state == IREE_HIP_HOSTCALL_MESSAGE_STATE_COMPLETE);
  IREE_ASSERT(target.data_length == message->count * sizeof(uint64_t));

  iree_host_size_t remaining_count = message->count;
  uint8_t* target_data = target.data;
  const iree_arena_block_t* block = message->block_head;
  while (remaining_count != 0) {
    IREE_ASSERT(block);
    const iree_host_size_t block_count =
        iree_min(remaining_count,
                 (iree_host_size_t)IREE_HIP_HOSTCALL_MESSAGE_BLOCK_QWORD_COUNT);
    const uint64_t* block_data =
        (const uint64_t*)iree_arena_block_ptr(&table->block_pool, block);
    memcpy(target_data, block_data, block_count * sizeof(block_data[0]));
    target_data += block_count * sizeof(block_data[0]);
    remaining_count -= block_count;
    block = block->next;
  }
}

void iree_hip_hostcall_message_release(iree_hip_hostcall_message_table_t* table,
                                       iree_host_size_t message_id) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT(message_id < table->count);
  iree_hip_hostcall_message_t* message = &table->messages[message_id];
  IREE_ASSERT(message->state == IREE_HIP_HOSTCALL_MESSAGE_STATE_COMPLETE);
  iree_hip_hostcall_message_discard(table, message);
}
