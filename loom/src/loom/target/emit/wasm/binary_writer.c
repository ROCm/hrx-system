// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/binary_writer.h"

#include <string.h>

void loom_wasm_binary_writer_initialize(iree_allocator_t allocator,
                                        loom_wasm_binary_writer_t* out_writer) {
  *out_writer = (loom_wasm_binary_writer_t){
      .allocator = allocator,
  };
}

void loom_wasm_binary_writer_deinitialize(loom_wasm_binary_writer_t* writer) {
  if (!writer) {
    return;
  }
  iree_allocator_free(writer->allocator, writer->data);
  *writer = (loom_wasm_binary_writer_t){0};
}

static iree_status_t loom_wasm_binary_writer_reserve(
    loom_wasm_binary_writer_t* writer, iree_host_size_t additional_length) {
  iree_host_size_t minimum_capacity = 0;
  if (!iree_host_size_checked_add(writer->length, additional_length,
                                  &minimum_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm binary length overflow");
  }
  if (minimum_capacity <= writer->capacity) {
    return iree_ok_status();
  }

  iree_host_size_t doubled_capacity =
      writer->capacity ? writer->capacity * 2 : 128;
  if (doubled_capacity < writer->capacity ||
      doubled_capacity < minimum_capacity) {
    doubled_capacity = minimum_capacity;
  }
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(
      writer->allocator, doubled_capacity, (void**)&writer->data));
  writer->capacity = doubled_capacity;
  return iree_ok_status();
}

iree_status_t loom_wasm_binary_write_u8(loom_wasm_binary_writer_t* writer,
                                        uint8_t value) {
  IREE_RETURN_IF_ERROR(loom_wasm_binary_writer_reserve(writer, 1));
  writer->data[writer->length++] = value;
  return iree_ok_status();
}

iree_status_t loom_wasm_binary_write_bytes(loom_wasm_binary_writer_t* writer,
                                           const uint8_t* data,
                                           iree_host_size_t data_length) {
  if (data_length == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_wasm_binary_writer_reserve(writer, data_length));
  memcpy(writer->data + writer->length, data, data_length);
  writer->length += data_length;
  return iree_ok_status();
}

iree_status_t loom_wasm_binary_write_u32_leb(loom_wasm_binary_writer_t* writer,
                                             uint32_t value) {
  do {
    uint8_t byte = (uint8_t)(value & 0x7Fu);
    value >>= 7;
    if (value != 0) {
      byte |= 0x80u;
    }
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(writer, byte));
  } while (value != 0);
  return iree_ok_status();
}

iree_status_t loom_wasm_binary_write_i32_leb(loom_wasm_binary_writer_t* writer,
                                             int32_t value) {
  bool more = true;
  while (more) {
    uint8_t byte = (uint8_t)(value & 0x7F);
    value >>= 7;
    const bool sign_bit_set = (byte & 0x40u) != 0;
    more = !((value == 0 && !sign_bit_set) || (value == -1 && sign_bit_set));
    if (more) {
      byte |= 0x80u;
    }
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(writer, byte));
  }
  return iree_ok_status();
}

iree_status_t loom_wasm_binary_write_u64_le(loom_wasm_binary_writer_t* writer,
                                            uint64_t value) {
  uint8_t data[8] = {
      (uint8_t)(value & 0xFFu),         (uint8_t)((value >> 8) & 0xFFu),
      (uint8_t)((value >> 16) & 0xFFu), (uint8_t)((value >> 24) & 0xFFu),
      (uint8_t)((value >> 32) & 0xFFu), (uint8_t)((value >> 40) & 0xFFu),
      (uint8_t)((value >> 48) & 0xFFu), (uint8_t)(value >> 56),
  };
  return loom_wasm_binary_write_bytes(writer, data, sizeof(data));
}
