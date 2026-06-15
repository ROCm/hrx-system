// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer.h"

#include <string.h>

#include "loom/format/bytecode/varint.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

#define LOOM_BYTECODE_DEFAULT_PRODUCER "loom-c"

// Maximum nesting depth for regions. Prevents stack exhaustion from
// pathologically deep IR (e.g., 1000 nested scf.execute_region ops).
// Each level of nesting uses ~256 bytes of stack (local variables in
// the write_region/write_block/write_operation chain).
#define LOOM_BYTECODE_MAX_REGION_DEPTH 256

//===----------------------------------------------------------------------===//
// Page-buffered stream writer
//===----------------------------------------------------------------------===//
//
// Amortizes iree_io_stream_write vtable dispatch cost by accumulating
// small writes in a local 4KB page buffer. Flushes to the stream when
// full. All section data flows through this writer.

#define LOOM_BYTECODE_PAGE_SIZE 4096

typedef struct loom_bytecode_page_writer_t {
  iree_io_stream_t* stream;
  uint8_t page[LOOM_BYTECODE_PAGE_SIZE];
  iree_host_size_t position;  // Bytes currently in the page.
  uint64_t total_written;     // Logical total (flushed + position).
                              // uint64_t to match the format's 64-bit
                              // offsets and avoid overflow on 32-bit.
} loom_bytecode_page_writer_t;

static void loom_bytecode_page_writer_initialize(
    loom_bytecode_page_writer_t* writer, iree_io_stream_t* stream) {
  writer->stream = stream;
  writer->position = 0;
  writer->total_written = 0;
}

static iree_status_t loom_bytecode_page_writer_flush(
    loom_bytecode_page_writer_t* writer) {
  if (writer->position == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      iree_io_stream_write(writer->stream, writer->position, writer->page));
  writer->position = 0;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_page_writer_write(
    loom_bytecode_page_writer_t* writer, const void* data,
    iree_host_size_t length) {
  if (length == 0) return iree_ok_status();
  writer->total_written += length;
  const uint8_t* source = (const uint8_t*)data;

  // Fast path: fits in current page.
  if (IREE_LIKELY(writer->position + length <= LOOM_BYTECODE_PAGE_SIZE)) {
    memcpy(writer->page + writer->position, source, length);
    writer->position += length;
    return iree_ok_status();
  }

  // Fill the remainder of the current page.
  iree_host_size_t remaining = LOOM_BYTECODE_PAGE_SIZE - writer->position;
  if (remaining > 0) {
    memcpy(writer->page + writer->position, source, remaining);
    writer->position = LOOM_BYTECODE_PAGE_SIZE;
    source += remaining;
    length -= remaining;
  }

  // Flush the full page.
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_flush(writer));

  // Write full pages directly to the stream (skip the page buffer).
  while (length >= LOOM_BYTECODE_PAGE_SIZE) {
    IREE_RETURN_IF_ERROR(
        iree_io_stream_write(writer->stream, LOOM_BYTECODE_PAGE_SIZE, source));
    source += LOOM_BYTECODE_PAGE_SIZE;
    length -= LOOM_BYTECODE_PAGE_SIZE;
  }

  // Copy remainder into the empty page.
  if (length > 0) {
    memcpy(writer->page, source, length);
    writer->position = length;
  }
  return iree_ok_status();
}

// Typed write helpers. Each encodes into a small stack buffer, then
// writes through the page writer.

static iree_status_t loom_bytecode_page_writer_write_u8(
    loom_bytecode_page_writer_t* writer, uint8_t value) {
  return loom_bytecode_page_writer_write(writer, &value, 1);
}

static iree_status_t loom_bytecode_page_writer_write_u16_le(
    loom_bytecode_page_writer_t* writer, uint16_t value) {
  uint8_t bytes[2] = {(uint8_t)value, (uint8_t)(value >> 8)};
  return loom_bytecode_page_writer_write(writer, bytes, 2);
}

static iree_status_t loom_bytecode_page_writer_write_u32_le(
    loom_bytecode_page_writer_t* writer, uint32_t value) {
  uint8_t bytes[4] = {(uint8_t)value, (uint8_t)(value >> 8),
                      (uint8_t)(value >> 16), (uint8_t)(value >> 24)};
  return loom_bytecode_page_writer_write(writer, bytes, 4);
}

static iree_status_t loom_bytecode_page_writer_write_u64_le(
    loom_bytecode_page_writer_t* writer, uint64_t value) {
  uint8_t bytes[8] = {
      (uint8_t)value,         (uint8_t)(value >> 8),  (uint8_t)(value >> 16),
      (uint8_t)(value >> 24), (uint8_t)(value >> 32), (uint8_t)(value >> 40),
      (uint8_t)(value >> 48), (uint8_t)(value >> 56),
  };
  return loom_bytecode_page_writer_write(writer, bytes, 8);
}

static iree_status_t loom_bytecode_page_writer_write_uvarint(
    loom_bytecode_page_writer_t* writer, uint64_t value) {
  uint8_t buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t span = iree_make_byte_span(buffer, sizeof(buffer));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_uvarint_encode(value, span, &length));
  return loom_bytecode_page_writer_write(writer, buffer, length);
}

static iree_status_t loom_bytecode_page_writer_write_svarint(
    loom_bytecode_page_writer_t* writer, int64_t value) {
  uint8_t buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t span = iree_make_byte_span(buffer, sizeof(buffer));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_svarint_encode(value, span, &length));
  return loom_bytecode_page_writer_write(writer, buffer, length);
}

static iree_status_t loom_bytecode_page_writer_write_string(
    loom_bytecode_page_writer_t* writer, iree_string_view_t text) {
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, text.size));
  return loom_bytecode_page_writer_write(writer, text.data, text.size);
}

static iree_status_t loom_bytecode_page_writer_write_zeros(
    loom_bytecode_page_writer_t* writer, iree_host_size_t count) {
  static const uint8_t zeros[64] = {0};
  while (count > 0) {
    iree_host_size_t chunk = count < sizeof(zeros) ? count : sizeof(zeros);
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(writer, zeros, chunk));
    count -= chunk;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_page_writer_pad_to_alignment(
    loom_bytecode_page_writer_t* writer, iree_host_size_t alignment) {
  iree_host_size_t remainder = writer->total_written % alignment;
  if (remainder == 0) return iree_ok_status();
  return loom_bytecode_page_writer_write_zeros(writer, alignment - remainder);
}

// Writes a null-terminated string followed by padding to 8-byte alignment.
static iree_status_t loom_bytecode_page_writer_write_null_terminated_string(
    loom_bytecode_page_writer_t* writer, iree_string_view_t text) {
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write(writer, text.data, text.size));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 0));
  return iree_ok_status();
}

static iree_status_t loom_bytecode_page_writer_write_comment_list(
    loom_bytecode_page_writer_t* writer, const iree_string_view_t* comments,
    iree_host_size_t comment_count) {
  if (comment_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "comment count %" PRIhsz " exceeds maximum %u",
                            comment_count, (unsigned)UINT16_MAX);
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, comment_count));
  for (iree_host_size_t i = 0; i < comment_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, comments[i].size));
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(
        writer, comments[i].data, comments[i].size));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// String builder emit helpers (for SYMBOLS section buffering)
//===----------------------------------------------------------------------===//

static iree_status_t loom_bytecode_emit_u8(iree_string_builder_t* builder,
                                           uint8_t value) {
  char* head = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(builder, 1, &head));
  if (head) head[0] = (char)value;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_emit_u16_le(iree_string_builder_t* builder,
                                               uint16_t value) {
  char* head = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(builder, 2, &head));
  if (head) {
    head[0] = (char)(value & 0xFF);
    head[1] = (char)((value >> 8) & 0xFF);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_emit_u32_le(iree_string_builder_t* builder,
                                               uint32_t value) {
  char* head = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(builder, 4, &head));
  if (head) {
    head[0] = (char)(value & 0xFF);
    head[1] = (char)((value >> 8) & 0xFF);
    head[2] = (char)((value >> 16) & 0xFF);
    head[3] = (char)((value >> 24) & 0xFF);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_emit_u64_le(iree_string_builder_t* builder,
                                               uint64_t value) {
  char* head = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(builder, 8, &head));
  if (head) {
    for (int i = 0; i < 8; ++i) {
      head[i] = (char)((value >> (i * 8)) & 0xFF);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_emit_uvarint(iree_string_builder_t* builder,
                                                uint64_t value) {
  uint8_t buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t span = iree_make_byte_span(buffer, sizeof(buffer));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_uvarint_encode(value, span, &length));
  return iree_string_builder_append_string(
      builder, iree_make_string_view((const char*)buffer, length));
}

static iree_status_t loom_bytecode_emit_svarint(iree_string_builder_t* builder,
                                                int64_t value) {
  uint8_t buffer[LOOM_VARINT_MAX_LENGTH];
  iree_byte_span_t span = iree_make_byte_span(buffer, sizeof(buffer));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_svarint_encode(value, span, &length));
  return iree_string_builder_append_string(
      builder, iree_make_string_view((const char*)buffer, length));
}

static void loom_bytecode_patch_u64_le(iree_string_builder_t* builder,
                                       iree_host_size_t offset,
                                       uint64_t value) {
  char* buffer = builder->buffer;
  for (int i = 0; i < 8; ++i) {
    buffer[offset + i] = (char)((value >> (i * 8)) & 0xFF);
  }
}

static iree_status_t loom_bytecode_emit_comment_list(
    iree_string_builder_t* builder, const iree_string_view_t* comments,
    iree_host_size_t comment_count) {
  if (comment_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "comment count %" PRIhsz " exceeds maximum %u",
                            comment_count, (unsigned)UINT16_MAX);
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, comment_count));
  for (iree_host_size_t i = 0; i < comment_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, comments[i].size));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, comments[i]));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Numbering context
//===----------------------------------------------------------------------===//

// Sentinel for "not yet assigned" in mapping arrays.
#define LOOM_WRITER_ID_NONE UINT32_MAX

// External string entry: a string from a vtable B-string, not in
// the module's string table.
typedef struct loom_bytecode_external_string_t {
  // External string contents.
  iree_string_view_t view;
  // Writer string ID assigned to view.
  uint32_t writer_id;
} loom_bytecode_external_string_t;

// Op name entry in the numbering context.
typedef struct loom_bytecode_op_entry_t {
  // Internal op kind represented by this bytecode op-table entry.
  loom_op_kind_t kind;
  // Dense writer op-table index.
  uint32_t writer_op_id;
  // Writer string ID naming this op kind.
  uint32_t string_writer_id;
} loom_bytecode_op_entry_t;

// Hash table entry mapping a structural loom_type_t to its module type index.
typedef struct loom_bytecode_type_index_entry_t {
  // Structural hash of module->types.entries[module_index].
  uint32_t hash;
  // Module type table index, or LOOM_WRITER_ID_NONE for an empty slot.
  uint32_t module_index;
} loom_bytecode_type_index_entry_t;

typedef struct loom_bytecode_numbering_t {
  // Module being serialized.
  const loom_module_t* module;
  // Temporary arena that owns numbering tables.
  iree_arena_allocator_t* arena;
  // File-level location mode controlling operation location references.
  loom_bytecode_location_mode_t location_mode;

  // Writer string ID to string view table for the STRINGS section.
  iree_string_view_t* string_entries;
  // Number of writer strings assigned.
  iree_host_size_t string_count;
  // Allocated capacity of string_entries.
  iree_host_size_t string_capacity;

  // Module string ID to writer string ID map.
  uint32_t* module_string_map;

  // External strings not in the module table.
  loom_bytecode_external_string_t* external_strings;
  // Number of external strings assigned.
  iree_host_size_t external_string_count;
  // Allocated capacity of external_strings.
  iree_host_size_t external_string_capacity;

  // Type mapping: module type index → writer type_id.
  uint32_t* type_map;
  // Reverse lookup from wire type value to a representative module type index.
  loom_bytecode_type_index_entry_t* type_index_entries;
  // Capacity of type_index_entries. Always a power of two when non-zero.
  iree_host_size_t type_index_capacity;
  // Writer type_id → module type index (for section writing).
  iree_host_size_t* type_order;
  // Number of writer types assigned.
  iree_host_size_t type_count;
  // Allocated capacity of type_order.
  iree_host_size_t type_order_capacity;

  // Op name registry.
  loom_bytecode_op_entry_t* op_entries;
  // Number of op names assigned.
  iree_host_size_t op_count;
  // Allocated capacity of op_entries.
  iree_host_size_t op_capacity;
} loom_bytecode_numbering_t;

static uint32_t loom_bytecode_type_hash_mix_bytes(uint32_t hash,
                                                  const void* data,
                                                  iree_host_size_t length) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

static uint32_t loom_bytecode_type_hash_mix_u8(uint32_t hash, uint8_t value) {
  return loom_bytecode_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_bytecode_type_hash_mix_u16(uint32_t hash, uint16_t value) {
  return loom_bytecode_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_bytecode_type_hash_mix_u32(uint32_t hash, uint32_t value) {
  return loom_bytecode_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_bytecode_type_hash_mix_u64(uint32_t hash, uint64_t value) {
  return loom_bytecode_type_hash_mix_bytes(hash, &value, sizeof(value));
}

static uint32_t loom_bytecode_type_wire_hash(loom_type_t type) {
  uint32_t hash = 2166136261u;
  loom_type_kind_t kind = loom_type_kind(type);
  hash = loom_bytecode_type_hash_mix_u8(hash, (uint8_t)kind);
  hash = loom_bytecode_type_hash_mix_u8(hash,
                                        (uint8_t)loom_type_element_type(type));
  hash = loom_bytecode_type_hash_mix_u8(hash, loom_type_rank(type));

  switch (kind) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW: {
      if (loom_type_has_ssa_encoding(type)) {
        hash = loom_bytecode_type_hash_mix_u8(
            hash, LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA);
      } else if (loom_type_has_static_encoding(type)) {
        hash = loom_bytecode_type_hash_mix_u8(
            hash, LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC);
        hash = loom_bytecode_type_hash_mix_u16(hash, type.encoding_id);
      } else {
        hash = loom_bytecode_type_hash_mix_u8(
            hash, LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE);
      }
      for (uint8_t i = 0; i < loom_type_rank(type); ++i) {
        uint64_t dim = loom_type_dim(type, i);
        if (loom_dim_is_dynamic(dim)) {
          hash = loom_bytecode_type_hash_mix_u8(hash, 1);
        } else {
          hash = loom_bytecode_type_hash_mix_u8(hash, 0);
          hash = loom_bytecode_type_hash_mix_u64(hash, dim);
        }
      }
      return hash;
    }
    case LOOM_TYPE_POOL: {
      uint64_t dim = loom_type_dim(type, 0);
      if (loom_dim_is_dynamic(dim)) {
        hash = loom_bytecode_type_hash_mix_u8(hash, 1);
      } else {
        hash = loom_bytecode_type_hash_mix_u8(hash, 0);
        hash = loom_bytecode_type_hash_mix_u64(hash, dim);
      }
      return hash;
    }
    case LOOM_TYPE_GROUP:
      return loom_bytecode_type_hash_mix_u8(
          hash, (uint8_t)loom_type_group_scope(type));
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) return hash;
      hash = loom_bytecode_type_hash_mix_u16(hash, data->arg_count);
      hash = loom_bytecode_type_hash_mix_u16(hash, data->result_count);
      uint16_t type_count = (uint16_t)(data->arg_count + data->result_count);
      for (uint16_t i = 0; i < type_count; ++i) {
        hash = loom_bytecode_type_hash_mix_u32(
            hash, loom_bytecode_type_wire_hash(data->types[i]));
      }
      return hash;
    }
    case LOOM_TYPE_DIALECT: {
      hash = loom_bytecode_type_hash_mix_u32(hash,
                                             loom_type_dialect_name_id(type));
      uint16_t param_count = loom_type_dialect_param_count(type);
      hash = loom_bytecode_type_hash_mix_u16(hash, param_count);
      const loom_type_t* params = loom_type_dialect_params(type);
      for (uint16_t i = 0; params && i < param_count; ++i) {
        hash = loom_bytecode_type_hash_mix_u32(
            hash, loom_bytecode_type_wire_hash(params[i]));
      }
      return hash;
    }
    case LOOM_TYPE_REGISTER:
      hash = loom_bytecode_type_hash_mix_u64(hash,
                                             loom_type_register_payload0(type));
      return loom_bytecode_type_hash_mix_u64(hash,
                                             loom_type_register_payload1(type));
    case LOOM_TYPE_STORAGE:
      return loom_bytecode_type_hash_mix_u8(
          hash, (uint8_t)loom_type_storage_space(type));
    case LOOM_TYPE_ENCODING:
      return loom_bytecode_type_hash_mix_u8(
          hash, (uint8_t)loom_type_encoding_role(type));
    default:
      return hash;
  }
}

static bool loom_bytecode_type_dim_wire_equal(uint64_t a, uint64_t b) {
  bool a_dynamic = loom_dim_is_dynamic(a);
  bool b_dynamic = loom_dim_is_dynamic(b);
  if (a_dynamic || b_dynamic) return a_dynamic == b_dynamic;
  return a == b;
}

static bool loom_bytecode_type_encoding_wire_equal(loom_type_t a,
                                                   loom_type_t b) {
  if (loom_type_has_ssa_encoding(a) || loom_type_has_ssa_encoding(b)) {
    return loom_type_has_ssa_encoding(a) == loom_type_has_ssa_encoding(b);
  }
  if (loom_type_has_static_encoding(a) || loom_type_has_static_encoding(b)) {
    return loom_type_has_static_encoding(a) ==
               loom_type_has_static_encoding(b) &&
           a.encoding_id == b.encoding_id;
  }
  return !loom_type_has_encoding(a) && !loom_type_has_encoding(b);
}

static bool loom_bytecode_type_wire_equal(loom_type_t a, loom_type_t b) {
  loom_type_kind_t kind = loom_type_kind(a);
  if (kind != loom_type_kind(b)) return false;
  if (loom_type_element_type(a) != loom_type_element_type(b)) return false;
  if (loom_type_rank(a) != loom_type_rank(b)) return false;

  switch (kind) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW:
      if (!loom_bytecode_type_encoding_wire_equal(a, b)) return false;
      for (uint8_t i = 0; i < loom_type_rank(a); ++i) {
        if (!loom_bytecode_type_dim_wire_equal(loom_type_dim(a, i),
                                               loom_type_dim(b, i))) {
          return false;
        }
      }
      return true;
    case LOOM_TYPE_POOL:
      return loom_bytecode_type_dim_wire_equal(loom_type_dim(a, 0),
                                               loom_type_dim(b, 0));
    case LOOM_TYPE_GROUP:
      return loom_type_group_scope(a) == loom_type_group_scope(b);
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* a_data = loom_type_func_data(a);
      const loom_func_type_data_t* b_data = loom_type_func_data(b);
      if (!a_data || !b_data) return a_data == b_data;
      if (a_data->arg_count != b_data->arg_count ||
          a_data->result_count != b_data->result_count) {
        return false;
      }
      uint16_t type_count =
          (uint16_t)(a_data->arg_count + a_data->result_count);
      for (uint16_t i = 0; i < type_count; ++i) {
        if (!loom_bytecode_type_wire_equal(a_data->types[i],
                                           b_data->types[i])) {
          return false;
        }
      }
      return true;
    }
    case LOOM_TYPE_DIALECT: {
      if (loom_type_dialect_name_id(a) != loom_type_dialect_name_id(b)) {
        return false;
      }
      uint16_t param_count = loom_type_dialect_param_count(a);
      if (param_count != loom_type_dialect_param_count(b)) return false;
      const loom_type_t* a_params = loom_type_dialect_params(a);
      const loom_type_t* b_params = loom_type_dialect_params(b);
      if (!a_params || !b_params) return a_params == b_params;
      for (uint16_t i = 0; i < param_count; ++i) {
        if (!loom_bytecode_type_wire_equal(a_params[i], b_params[i])) {
          return false;
        }
      }
      return true;
    }
    case LOOM_TYPE_REGISTER:
      return loom_type_register_payload0(a) == loom_type_register_payload0(b) &&
             loom_type_register_payload1(a) == loom_type_register_payload1(b);
    case LOOM_TYPE_STORAGE:
      return loom_type_storage_space(a) == loom_type_storage_space(b);
    case LOOM_TYPE_ENCODING:
      return loom_type_encoding_role(a) == loom_type_encoding_role(b);
    default:
      return true;
  }
}

// Appends a string_view to the ordered string list, growing if needed.
static iree_status_t loom_bytecode_numbering_append_string(
    loom_bytecode_numbering_t* numbering, iree_string_view_t view,
    uint32_t* out_writer_id) {
  if (numbering->string_count >= numbering->string_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->string_count, /*minimum_capacity=*/16,
        sizeof(iree_string_view_t), &numbering->string_capacity,
        (void**)&numbering->string_entries));
  }
  if (numbering->string_count >= (1u << 24)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode string count exceeds format maximum "
                            "(16M)");
  }
  uint32_t id = (uint32_t)numbering->string_count;
  numbering->string_entries[numbering->string_count++] = view;
  *out_writer_id = id;
  return iree_ok_status();
}

// Initializes the numbering context. All allocations come from |arena|,
// which the caller owns. No individual frees needed — the arena handles
// bulk deallocation.
static iree_status_t loom_bytecode_numbering_initialize(
    loom_bytecode_numbering_t* numbering, const loom_module_t* module,
    iree_arena_allocator_t* arena) {
  memset(numbering, 0, sizeof(*numbering));
  numbering->module = module;
  numbering->arena = arena;

  // Module string map: parallel array for O(1) module_string_id → writer_id.
  if (module->strings.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, module->strings.count, sizeof(uint32_t),
        (void**)&numbering->module_string_map));
    memset(numbering->module_string_map, 0xFF,
           module->strings.count * sizeof(uint32_t));
  }

  // Writer string id 0 is reserved as "no SSA name" in value definitions.
  // Keep the empty string in slot 0 so named values never alias the sentinel.
  uint32_t empty_string_writer_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_append_string(
      numbering, iree_string_view_empty(), &empty_string_writer_id));
  if (numbering->module_string_map != NULL) {
    for (iree_host_size_t i = 0; i < module->strings.count; ++i) {
      if (iree_string_view_is_empty(module->strings.entries[i])) {
        numbering->module_string_map[i] = empty_string_writer_id;
        break;
      }
    }
  }

  // Type map: parallel array for O(1) module_type_index → writer_type_id.
  if (module->types.count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, module->types.count, sizeof(uint32_t),
                                  (void**)&numbering->type_map));
    memset(numbering->type_map, 0xFF, module->types.count * sizeof(uint32_t));

    iree_host_size_t type_index_capacity =
        iree_host_size_next_power_of_two((module->types.count * 4 + 2) / 3);
    if (type_index_capacity < 16) type_index_capacity = 16;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, type_index_capacity, sizeof(loom_bytecode_type_index_entry_t),
        (void**)&numbering->type_index_entries));
    for (iree_host_size_t i = 0; i < type_index_capacity; ++i) {
      numbering->type_index_entries[i].hash = 0;
      numbering->type_index_entries[i].module_index = LOOM_WRITER_ID_NONE;
    }
    numbering->type_index_capacity = type_index_capacity;
    iree_host_size_t mask = type_index_capacity - 1;
    for (iree_host_size_t i = 0; i < module->types.count; ++i) {
      uint32_t hash = loom_bytecode_type_wire_hash(module->types.entries[i]);
      iree_host_size_t slot = hash & mask;
      while (numbering->type_index_entries[slot].module_index !=
             LOOM_WRITER_ID_NONE) {
        slot = (slot + 1) & mask;
      }
      numbering->type_index_entries[slot].hash = hash;
      numbering->type_index_entries[slot].module_index = (uint32_t)i;
    }
  }

  return iree_ok_status();
}

// Interns a module string by its module string_id. Returns writer string ID.
static iree_status_t loom_bytecode_numbering_intern_module_string(
    loom_bytecode_numbering_t* numbering, loom_string_id_t string_id,
    uint32_t* out_writer_id) {
  if (string_id == LOOM_STRING_ID_INVALID) {
    *out_writer_id = 0;
    return iree_ok_status();
  }
  if (string_id >= numbering->module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "string_id %u out of range (module has %" PRIhsz
                            " strings)",
                            string_id, numbering->module->strings.count);
  }
  if (numbering->module_string_map[string_id] != LOOM_WRITER_ID_NONE) {
    *out_writer_id = numbering->module_string_map[string_id];
    return iree_ok_status();
  }
  iree_string_view_t view = numbering->module->strings.entries[string_id];
  uint32_t writer_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_append_string(numbering, view, &writer_id));
  numbering->module_string_map[string_id] = writer_id;
  *out_writer_id = writer_id;
  return iree_ok_status();
}

// Interns an arbitrary string_view (for vtable names not in the module).
static iree_status_t loom_bytecode_numbering_intern_string_view(
    loom_bytecode_numbering_t* numbering, iree_string_view_t view,
    uint32_t* out_writer_id) {
  if (iree_string_view_is_empty(view)) {
    *out_writer_id = 0;
    return iree_ok_status();
  }
  // Check external strings first (linear scan, small list).
  for (iree_host_size_t i = 0; i < numbering->external_string_count; ++i) {
    if (iree_string_view_equal(numbering->external_strings[i].view, view)) {
      *out_writer_id = numbering->external_strings[i].writer_id;
      return iree_ok_status();
    }
  }
  // Also check if it happens to match a module string. Module strings are
  // assigned in first-use order, not source intern-table order, so the bytecode
  // stays canonical across text forms that intern strings differently.
  for (iree_host_size_t i = 0; i < numbering->module->strings.count; ++i) {
    if (iree_string_view_equal(numbering->module->strings.entries[i], view)) {
      return loom_bytecode_numbering_intern_module_string(
          numbering, (loom_string_id_t)i, out_writer_id);
    }
  }
  // New external string.
  uint32_t writer_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_append_string(numbering, view, &writer_id));
  if (numbering->external_string_count >= numbering->external_string_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->external_string_count,
        /*minimum_capacity=*/16, sizeof(loom_bytecode_external_string_t),
        &numbering->external_string_capacity,
        (void**)&numbering->external_strings));
  }
  numbering->external_strings[numbering->external_string_count++] =
      (loom_bytecode_external_string_t){.view = view, .writer_id = writer_id};
  *out_writer_id = writer_id;
  return iree_ok_status();
}

// Finds the representative module type table index for a given wire type.
static uint32_t loom_bytecode_find_type_index(
    const loom_bytecode_numbering_t* numbering, loom_type_t type) {
  if (numbering->type_index_capacity == 0) return LOOM_WRITER_ID_NONE;
  uint32_t hash = loom_bytecode_type_wire_hash(type);
  iree_host_size_t mask = numbering->type_index_capacity - 1;
  iree_host_size_t slot = hash & mask;
  while (numbering->type_index_entries[slot].module_index !=
         LOOM_WRITER_ID_NONE) {
    const loom_bytecode_type_index_entry_t* entry =
        &numbering->type_index_entries[slot];
    if (entry->hash == hash &&
        loom_bytecode_type_wire_equal(
            numbering->module->types.entries[entry->module_index], type)) {
      return entry->module_index;
    }
    slot = (slot + 1) & mask;
  }
  return LOOM_WRITER_ID_NONE;
}

// Interns a type, recursing into sub-types first (topological order).
static iree_status_t loom_bytecode_numbering_intern_type(
    loom_bytecode_numbering_t* numbering, loom_type_t type,
    uint32_t* out_writer_id) {
  uint32_t module_index = loom_bytecode_find_type_index(numbering, type);
  if (module_index == LOOM_WRITER_ID_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type not found in module type table (kind=%u, "
                            "rank=%u, module types=%" PRIhsz ")",
                            (unsigned)loom_type_kind(type),
                            (unsigned)loom_type_rank(type),
                            numbering->module->types.count);
  }
  if (numbering->type_map[module_index] != LOOM_WRITER_ID_NONE) {
    *out_writer_id = numbering->type_map[module_index];
    return iree_ok_status();
  }

  // Recurse into sub-types first (topological ordering).
  uint32_t unused_id = 0;
  loom_type_kind_t kind = loom_type_kind(type);
  switch (kind) {
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW: {
      // Element type is a scalar — intern it.
      loom_type_t element_type = loom_type_scalar(loom_type_element_type(type));
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
          numbering, element_type, &unused_id));
      break;
    }
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* func_data = loom_type_func_data(type);
      for (uint16_t i = 0; i < func_data->arg_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
            numbering, func_data->types[i], &unused_id));
      }
      for (uint16_t i = 0; i < func_data->result_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
            numbering, func_data->types[func_data->arg_count + i], &unused_id));
      }
      break;
    }
    case LOOM_TYPE_GROUP:
      // Group scope is serialized as a byte, not a string. No interning.
      break;
    case LOOM_TYPE_DIALECT: {
      // Intern the dialect type name string.
      loom_string_id_t name_id = loom_type_dialect_name_id(type);
      if (name_id < numbering->module->strings.count) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
            numbering, numbering->module->strings.entries[name_id],
            &unused_id));
      }
      // Recurse into type parameters.
      uint16_t param_count = loom_type_dialect_param_count(type);
      const loom_type_t* params = loom_type_dialect_params(type);
      for (uint16_t i = 0; i < param_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
            numbering, params[i], &unused_id));
      }
      break;
    }
    case LOOM_TYPE_REGISTER:
      break;
    default:
      break;
  }

  // Intern the parent type.
  if (numbering->type_count >= (1u << 16)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode type count exceeds format maximum (64K)");
  }
  uint32_t writer_id = numbering->type_count;
  numbering->type_map[module_index] = writer_id;
  if (numbering->type_count >= numbering->type_order_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->type_count, /*minimum_capacity=*/16,
        sizeof(iree_host_size_t), &numbering->type_order_capacity,
        (void**)&numbering->type_order));
  }
  numbering->type_order[numbering->type_count] = module_index;
  numbering->type_count++;
  *out_writer_id = writer_id;
  return iree_ok_status();
}

// Interns an op kind. Returns writer op name ID.
static iree_status_t loom_bytecode_numbering_intern_op(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op,
    uint32_t* out_writer_op_id) {
  // Check existing entries.
  for (uint32_t i = 0; i < numbering->op_count; ++i) {
    if (numbering->op_entries[i].kind == op->kind) {
      *out_writer_op_id = numbering->op_entries[i].writer_op_id;
      return iree_ok_status();
    }
  }
  // New op kind.
  iree_string_view_t name = loom_op_name(numbering->module, op);
  uint32_t string_writer_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
      numbering, name, &string_writer_id));

  if (numbering->op_count >= (1u << 24)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode op count exceeds format maximum (16M)");
  }
  uint32_t writer_op_id = (uint32_t)numbering->op_count;
  if (numbering->op_count >= numbering->op_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        numbering->arena, numbering->op_count, /*minimum_capacity=*/16,
        sizeof(loom_bytecode_op_entry_t), &numbering->op_capacity,
        (void**)&numbering->op_entries));
  }
  numbering->op_entries[numbering->op_count++] = (loom_bytecode_op_entry_t){
      .kind = op->kind,
      .writer_op_id = writer_op_id,
      .string_writer_id = string_writer_id,
  };
  *out_writer_op_id = writer_op_id;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Value numbering (per-function)
//===----------------------------------------------------------------------===//

typedef struct loom_bytecode_value_numbering_entry_t {
  // Module value ID being mapped.
  loom_value_id_t value_id;
  // Body-local value number assigned to |value_id|.
  uint32_t number;
} loom_bytecode_value_numbering_entry_t;

typedef struct loom_bytecode_value_numbering_t {
  // Module containing numbered values.
  const loom_module_t* module;
  // Arena that owns |entries|.
  iree_arena_allocator_t* arena;
  // Sorted map from module value ID to body-local value number.
  loom_bytecode_value_numbering_entry_t* entries;
  // Number of initialized entries.
  iree_host_size_t count;
  // Allocated entry capacity.
  iree_host_size_t capacity;
  // Next value number to assign.
  uint32_t next_number;
} loom_bytecode_value_numbering_t;

static void loom_bytecode_value_numbering_initialize(
    loom_bytecode_value_numbering_t* value_numbering,
    const loom_module_t* module, iree_arena_allocator_t* arena) {
  *value_numbering = (loom_bytecode_value_numbering_t){
      .module = module,
      .arena = arena,
  };
}

static iree_host_size_t loom_bytecode_value_numbering_lower_bound(
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_value_id_t value_id, bool* out_found) {
  iree_host_size_t lo = 0;
  iree_host_size_t hi = value_numbering->count;
  while (lo < hi) {
    iree_host_size_t mid = lo + (hi - lo) / 2;
    if (value_numbering->entries[mid].value_id < value_id) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  *out_found = lo < value_numbering->count &&
               value_numbering->entries[lo].value_id == value_id;
  return lo;
}

static iree_status_t loom_bytecode_value_numbering_ensure_capacity(
    loom_bytecode_value_numbering_t* value_numbering,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= value_numbering->capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(
      value_numbering->arena, value_numbering->count, minimum_capacity,
      sizeof(*value_numbering->entries), &value_numbering->capacity,
      (void**)&value_numbering->entries);
}

static iree_status_t loom_bytecode_value_numbering_assign_value(
    loom_bytecode_value_numbering_t* value_numbering,
    loom_value_id_t value_id) {
  if (value_id >= value_numbering->module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value_id %u exceeds module value count %" PRIhsz,
                            value_id, value_numbering->module->values.count);
  }
  bool found = false;
  iree_host_size_t entry_index = loom_bytecode_value_numbering_lower_bound(
      value_numbering, value_id, &found);
  if (found) {
    return iree_ok_status();
  }
  if (value_numbering->next_number == LOOM_WRITER_ID_NONE) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bytecode local value number exceeds uint32");
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_ensure_capacity(
      value_numbering, value_numbering->count + 1));
  memmove(&value_numbering->entries[entry_index + 1],
          &value_numbering->entries[entry_index],
          (value_numbering->count - entry_index) *
              sizeof(*value_numbering->entries));
  value_numbering->entries[entry_index] =
      (loom_bytecode_value_numbering_entry_t){
          .value_id = value_id,
          .number = value_numbering->next_number++,
      };
  ++value_numbering->count;
  return iree_ok_status();
}

// Resolves a module value_id to its function-local value number.
// Returns INVALID_ARGUMENT if the value_id is out of bounds or was
// never assigned a number (indicates a malformed IR graph).
static iree_status_t loom_bytecode_resolve_value_number(
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_value_id_t value_id, uint32_t* out_number) {
  if (value_id >= value_numbering->module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value_id %u exceeds module value count %" PRIhsz,
                            value_id, value_numbering->module->values.count);
  }
  bool found = false;
  iree_host_size_t entry_index = loom_bytecode_value_numbering_lower_bound(
      value_numbering, value_id, &found);
  if (!found) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value_id %u has no assigned value number "
                            "(undefined or not in scope)",
                            value_id);
  }
  *out_number = value_numbering->entries[entry_index].number;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_value_numbering_assign_region(
    loom_bytecode_value_numbering_t* value_numbering,
    const loom_region_t* region) {
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    // Block arguments define values.
    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
          value_numbering, loom_block_arg_id(block, arg_index)));
    }
    // Op results define values.
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t result_index = 0; result_index < op->result_count;
           ++result_index) {
        IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
            value_numbering, results[result_index]));
      }
      // Recurse into nested regions.
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t region_index = 0; region_index < op->region_count;
           ++region_index) {
        if (regions[region_index]) {
          IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_region(
              value_numbering, regions[region_index]));
        }
      }
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Numbering walk
//===----------------------------------------------------------------------===//
//
// Walks the module in the same order as the Python writer's
// _number_module to produce identical string/type/op ordering.

// Forward declarations for recursive numbering.
static iree_status_t loom_bytecode_number_region(
    loom_bytecode_numbering_t* numbering, const loom_region_t* region,
    uint32_t depth);
static iree_status_t loom_bytecode_number_operation(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op, uint32_t depth);
static iree_status_t loom_bytecode_number_encoding(
    loom_bytecode_numbering_t* numbering, uint16_t encoding_id);

static iree_status_t loom_bytecode_get_enum_ordinal(
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    uint8_t* out_ordinal) {
  if (attr.raw > UINT8_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "enum attribute value exceeds uint8_t range");
  }
  *out_ordinal = (uint8_t)attr.raw;
  if (!descriptor ||
      iree_all_bits_set(descriptor->flags, LOOM_ATTR_OPEN_ENUM) ||
      descriptor->enum_case_count == 0) {
    return iree_ok_status();
  }
  if (*out_ordinal >= descriptor->enum_case_count ||
      (descriptor->enum_case_names &&
       !descriptor->enum_case_names[*out_ordinal])) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "enum attribute value has no declared case");
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_attr_value(
    loom_bytecode_numbering_t* numbering, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor) {
  (void)descriptor;
  uint32_t unused_id = 0;
  switch (attr.kind) {
    case LOOM_ATTR_STRING: {
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, attr.string_id, &unused_id));
      break;
    }
    case LOOM_ATTR_ENUM: {
      break;
    }
    case LOOM_ATTR_SYMBOL: {
      loom_symbol_ref_t ref = attr.symbol;
      if (loom_symbol_ref_is_valid(ref) &&
          ref.symbol_id < numbering->module->symbols.count) {
        const loom_symbol_t* target_symbol =
            &numbering->module->symbols.entries[ref.symbol_id];
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, target_symbol->name_id, &unused_id));
      }
      break;
    }
    case LOOM_ATTR_TYPE: {
      if (attr.type_id >= numbering->module->types.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "type attribute id %u out of range (module has %" PRIhsz " types)",
            (unsigned)attr.type_id, numbering->module->types.count);
      }
      loom_type_t type = numbering->module->types.entries[attr.type_id];
      IREE_RETURN_IF_ERROR(
          loom_bytecode_numbering_intern_type(numbering, type, &unused_id));
      break;
    }
    case LOOM_ATTR_ENCODING: {
      IREE_RETURN_IF_ERROR(loom_bytecode_number_encoding(
          numbering, loom_attr_as_encoding_id(attr)));
      break;
    }
    case LOOM_ATTR_DICT: {
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, attr.dict_entries[i].name_id, &unused_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_number_attr_value(
            numbering, attr.dict_entries[i].value, NULL));
      }
      break;
    }
    default:
      break;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_encoding(
    loom_bytecode_numbering_t* numbering, uint16_t encoding_id) {
  if (encoding_id == 0 || encoding_id > numbering->module->encodings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "encoding_id %u out of range (module has %" PRIhsz " encodings)",
        (unsigned)encoding_id, numbering->module->encodings.count);
  }
  uint32_t unused_id = 0;
  const loom_encoding_t* encoding =
      &numbering->module->encodings.entries[encoding_id - 1];
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
      numbering, encoding->name_id, &unused_id));
  if (encoding->alias_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, encoding->alias_id, &unused_id));
  }
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    const loom_named_attr_t* attr = &encoding->attributes[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, attr->name_id, &unused_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_number_attr_value(numbering, attr->value, NULL));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_op_attr_is_present(
    const loom_op_t* op, const loom_attr_descriptor_t* descriptor,
    loom_attribute_t attr, bool* out_present) {
  if (attr.kind != LOOM_ATTR_ABSENT) {
    *out_present = true;
    return iree_ok_status();
  }
  if (descriptor && iree_all_bits_set(descriptor->flags, LOOM_ATTR_OPTIONAL)) {
    *out_present = false;
    return iree_ok_status();
  }
  iree_string_view_t attr_name =
      descriptor ? loom_attr_descriptor_name(descriptor) : IREE_SV("<unknown>");
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "op kind 0x%04x has absent required attribute '%.*s'",
                          (unsigned)op->kind, (int)attr_name.size,
                          attr_name.data);
}

static bool loom_bytecode_attr_descriptor_is_symbol(
    const loom_attr_descriptor_t* descriptor) {
  return descriptor && descriptor->attr_kind == LOOM_ATTR_SYMBOL;
}

static bool loom_bytecode_attr_is_symbol_identity(
    const loom_op_vtable_t* vtable, uint8_t attr_index) {
  return vtable && vtable->symbol_def &&
         attr_index == vtable->symbol_def->name_attr_index;
}

static uint8_t loom_bytecode_find_symbol_attr_index(
    const loom_op_vtable_t* vtable) {
  if (!vtable || !vtable->attr_descriptors) return LOOM_ATTR_INDEX_NONE;
  if (vtable->symbol_def) {
    uint8_t attr_index = vtable->symbol_def->name_attr_index;
    if (attr_index < vtable->attribute_count &&
        loom_bytecode_attr_descriptor_is_symbol(
            &vtable->attr_descriptors[attr_index])) {
      return attr_index;
    }
    return LOOM_ATTR_INDEX_NONE;
  }
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    if (loom_bytecode_attr_descriptor_is_symbol(&vtable->attr_descriptors[i])) {
      return i;
    }
  }
  return LOOM_ATTR_INDEX_NONE;
}

typedef struct loom_bytecode_global_value_list_t {
  // Temporary arena used for growing the value ID list.
  iree_arena_allocator_t* arena;
  // Module whose value table owns all IDs in the list.
  const loom_module_t* module;
  // Ordered declaration-local values used by one global symbol payload.
  loom_value_id_t* values;
  // Number of populated value IDs.
  iree_host_size_t count;
  // Allocated capacity of values.
  iree_host_size_t capacity;
} loom_bytecode_global_value_list_t;

static iree_status_t loom_bytecode_global_value_list_reserve(
    loom_bytecode_global_value_list_t* list,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= list->capacity) return iree_ok_status();
  iree_host_size_t new_capacity = list->capacity ? list->capacity : 4;
  while (new_capacity < minimum_capacity) {
    if (new_capacity > IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "global declaration-local value list overflow");
    }
    new_capacity *= 2;
  }
  loom_value_id_t* new_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      list->arena, new_capacity, sizeof(loom_value_id_t), (void**)&new_values));
  if (list->count > 0) {
    memcpy(new_values, list->values, list->count * sizeof(loom_value_id_t));
  }
  list->values = new_values;
  list->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_global_value_list_push_unique(
    loom_bytecode_global_value_list_t* list, loom_value_id_t value_id) {
  if (value_id >= list->module->values.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "global declaration-local value id %u out of range (module has %" PRIhsz
        " values)",
        value_id, list->module->values.count);
  }
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    if (list->values[i] == value_id) return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_global_value_list_reserve(list, list->count + 1));
  list->values[list->count++] = value_id;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_collect_global_type_value_ref(
    loom_value_id_t value_id, void* user_data) {
  return loom_bytecode_global_value_list_push_unique(
      (loom_bytecode_global_value_list_t*)user_data, value_id);
}

static iree_status_t loom_bytecode_collect_global_value_type_refs(
    loom_bytecode_global_value_list_t* list, iree_host_size_t* scan_index) {
  while (*scan_index < list->count) {
    loom_value_id_t value_id = list->values[(*scan_index)++];
    loom_type_t type = list->module->values.entries[value_id].type;
    IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
        type, loom_bytecode_collect_global_type_value_ref, list));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_collect_global_attr_value_refs(
    loom_bytecode_global_value_list_t* list, loom_attribute_t attr) {
  switch (attr.kind) {
    case LOOM_ATTR_PREDICATE_LIST:
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_predicate_t* predicate = &attr.predicate_list[i];
        if (predicate->arg_count > IREE_ARRAYSIZE(predicate->arg_tags)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "predicate arg count %u exceeds capacity",
                                  (unsigned)predicate->arg_count);
        }
        for (uint8_t arg_index = 0; arg_index < predicate->arg_count;
             ++arg_index) {
          if (predicate->arg_tags[arg_index] != LOOM_PRED_ARG_VALUE) {
            continue;
          }
          IREE_RETURN_IF_ERROR(loom_bytecode_global_value_list_push_unique(
              list, (loom_value_id_t)predicate->args[arg_index]));
        }
      }
      break;
    case LOOM_ATTR_DICT:
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_collect_global_attr_value_refs(
            list, attr.dict_entries[i].value));
      }
      break;
    default:
      break;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_collect_global_values(
    iree_arena_allocator_t* arena, const loom_module_t* module,
    const loom_op_t* op, loom_bytecode_global_value_list_t* out_values) {
  *out_values = (loom_bytecode_global_value_list_t){
      .arena = arena,
      .module = module,
  };

  const loom_value_id_t* result_ids = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_global_value_list_push_unique(out_values, result_ids[i]));
  }

  iree_host_size_t scan_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_collect_global_value_type_refs(out_values, &scan_index));

  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || (op->attribute_count > 0 && !vtable->attr_descriptors)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "global symbol op kind 0x%04x has missing attr descriptors",
        (unsigned)op->kind);
  }
  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_bytecode_collect_global_attr_value_refs(out_values, attrs[i]));
  }

  return loom_bytecode_collect_global_value_type_refs(out_values, &scan_index);
}

static iree_status_t loom_bytecode_number_global(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op,
    const loom_bytecode_global_value_list_t* local_values) {
  uint32_t unused_id = 0;

  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &unused_id));

  const loom_module_t* module = numbering->module;
  for (iree_host_size_t i = 0; i < local_values->count; ++i) {
    const loom_value_t* value =
        &module->values.entries[local_values->values[i]];
    if (value->name_id != LOOM_STRING_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, value->name_id, &unused_id));
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
        numbering, value->type, &unused_id));
  }

  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(module->context, op->kind);
  if (!vtable || (op->attribute_count > 0 && !vtable->attr_descriptors)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "global symbol op kind 0x%04x has missing attr descriptors",
        (unsigned)op->kind);
  }

  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      continue;
    }

    iree_string_view_t key_name = loom_attr_descriptor_name(descriptor);
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, key_name, &unused_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_number_attr_value(numbering, attrs[i], descriptor));
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_validate_record_symbol_op(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !iree_all_bits_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE) ||
      !vtable->symbol_def ||
      !loom_symbol_definition_implements(vtable->symbol_def,
                                         LOOM_SYMBOL_INTERFACE_RECORD)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "record symbol op kind 0x%04x must define a RECORD symbol",
        (unsigned)op->kind);
  }
  if (op->operand_count != 0 || op->result_count != 0 ||
      op->tied_result_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "record symbol op kind 0x%04x must not have operands, results, or "
        "tied results",
        (unsigned)op->kind);
  }
  if (vtable->region_count > 1 ||
      iree_any_bit_set(vtable->vtable_flags, LOOM_OP_VTABLE_VARIADIC_REGIONS)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "record symbol op kind 0x%04x must declare at most one fixed region",
        (unsigned)op->kind);
  }
  if (op->region_count != vtable->region_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "record symbol op kind 0x%04x region count does not match its vtable",
        (unsigned)op->kind);
  }
  if (op->region_count == 1 && !loom_op_regions(op)[0]) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "record symbol op kind 0x%04x must have a materialized body region",
        (unsigned)op->kind);
  }
  if (loom_bytecode_find_symbol_attr_index(vtable) == LOOM_ATTR_INDEX_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "record symbol op kind 0x%04x must declare a symbol attribute",
        (unsigned)op->kind);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_record(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op) {
  uint32_t unused_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_validate_record_symbol_op(numbering->module, op));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &unused_id));

  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(numbering->module->context, op->kind);
  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      continue;
    }

    iree_string_view_t key_name = loom_attr_descriptor_name(descriptor);
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, key_name, &unused_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_number_attr_value(numbering, attrs[i], descriptor));
  }

  if (op->region_count == 1) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_number_region(numbering, loom_op_regions(op)[0], 0));
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_function(
    loom_bytecode_numbering_t* numbering, loom_func_like_t func_like) {
  uint32_t unused_id = 0;

  // Defining func-like op name.
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, func_like.op, &unused_id));

  // Arg names and types.
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(func_like, &arg_count);
  for (uint16_t i = 0; i < arg_count; ++i) {
    loom_value_id_t value_id = arg_ids[i];
    if (value_id < numbering->module->values.count) {
      loom_string_id_t name_id =
          numbering->module->values.entries[value_id].name_id;
      if (name_id != LOOM_STRING_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, name_id, &unused_id));
      }
    }
  }
  for (uint16_t i = 0; i < arg_count; ++i) {
    loom_type_t arg_type = numbering->module->values.entries[arg_ids[i]].type;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_numbering_intern_type(numbering, arg_type, &unused_id));
  }

  // Result types (recursive).
  uint16_t result_count = func_like.op->result_count;
  const loom_value_id_t* result_ids = loom_op_const_results(func_like.op);
  for (uint16_t i = 0; i < result_count; ++i) {
    loom_type_t result_type =
        numbering->module->values.entries[result_ids[i]].type;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
        numbering, result_type, &unused_id));
  }

  // Function metadata predicates reference signature SSA values. Their value
  // numbers are resolved while the metadata section is emitted.

  // Root regions. Write the FuncLike body first so its signature arguments are
  // the first function-local value numbers, but preserve each declared region
  // index in the IR payload.
  loom_region_t** regions = loom_op_regions(func_like.op);
  uint8_t body_region_index = func_like.vtable->body_region_index;
  if (body_region_index != LOOM_REGION_INDEX_NONE &&
      body_region_index < func_like.op->region_count &&
      regions[body_region_index]) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_number_region(numbering, regions[body_region_index], 0));
  }
  for (uint8_t i = 0; i < func_like.op->region_count; ++i) {
    if (i == body_region_index || !regions[i]) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_number_region(numbering, regions[i], 0));
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_region(
    loom_bytecode_numbering_t* numbering, const loom_region_t* region,
    uint32_t depth) {
  if (depth >= LOOM_BYTECODE_MAX_REGION_DEPTH) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "region nesting exceeds maximum depth %d",
                            LOOM_BYTECODE_MAX_REGION_DEPTH);
  }
  uint32_t unused_id = 0;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);

    // Block label.
    if (block->label_id != LOOM_STRING_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, block->label_id, &unused_id));
    }

    // Block arg names and types.
    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      loom_value_id_t value_id = loom_block_arg_id(block, arg_index);
      const loom_value_t* value = &numbering->module->values.entries[value_id];
      if (value->name_id != LOOM_STRING_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, value->name_id, &unused_id));
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
          numbering, value->type, &unused_id));
    }

    // Operations.
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_number_operation(numbering, op, depth));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_number_operation(
    loom_bytecode_numbering_t* numbering, const loom_op_t* op, uint32_t depth) {
  uint32_t unused_id = 0;

  // Op name (into both op table and string table).
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &unused_id));

  // Result names and types.
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_t* value = &numbering->module->values.entries[results[i]];
    if (value->name_id != LOOM_STRING_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, value->name_id, &unused_id));
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
        numbering, value->type, &unused_id));
  }

  // Attribute keys and values.
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(numbering->module->context, op->kind);
  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor =
        (vtable && vtable->attr_descriptors) ? &vtable->attr_descriptors[i]
                                             : NULL;
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present) continue;

    // Attribute key name (from vtable descriptor).
    if (descriptor) {
      iree_string_view_t key_name = loom_attr_descriptor_name(descriptor);
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
          numbering, key_name, &unused_id));
    }
    // Attribute value strings.
    IREE_RETURN_IF_ERROR(
        loom_bytecode_number_attr_value(numbering, attrs[i], descriptor));
  }

  // Nested regions.
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_number_region(numbering, regions[i], depth + 1));
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Type kind mapping (C enum → bytecode kind byte)
//===----------------------------------------------------------------------===//

static iree_status_t loom_bytecode_type_kind_byte(loom_type_kind_t kind,
                                                  uint8_t* out_byte) {
  switch (kind) {
    case LOOM_TYPE_NONE:
      *out_byte = LOOM_BYTECODE_TYPE_NONE;
      return iree_ok_status();
    case LOOM_TYPE_SCALAR:
      *out_byte = LOOM_BYTECODE_TYPE_SCALAR;
      return iree_ok_status();
    case LOOM_TYPE_TILE:
      *out_byte = LOOM_BYTECODE_TYPE_TILE;
      return iree_ok_status();
    case LOOM_TYPE_TENSOR:
      *out_byte = LOOM_BYTECODE_TYPE_TENSOR;
      return iree_ok_status();
    case LOOM_TYPE_VECTOR:
      *out_byte = LOOM_BYTECODE_TYPE_VECTOR;
      return iree_ok_status();
    case LOOM_TYPE_VIEW:
      *out_byte = LOOM_BYTECODE_TYPE_VIEW;
      return iree_ok_status();
    case LOOM_TYPE_BUFFER:
      *out_byte = LOOM_BYTECODE_TYPE_BUFFER;
      return iree_ok_status();
    case LOOM_TYPE_GROUP:
      *out_byte = LOOM_BYTECODE_TYPE_GROUP;
      return iree_ok_status();
    case LOOM_TYPE_FUNCTION:
      *out_byte = LOOM_BYTECODE_TYPE_FUNCTION;
      return iree_ok_status();
    case LOOM_TYPE_DIALECT:
      *out_byte = LOOM_BYTECODE_TYPE_DIALECT;
      return iree_ok_status();
    case LOOM_TYPE_REGISTER:
      *out_byte = LOOM_BYTECODE_TYPE_REGISTER;
      return iree_ok_status();
    case LOOM_TYPE_STORAGE:
      *out_byte = LOOM_BYTECODE_TYPE_STORAGE;
      return iree_ok_status();
    case LOOM_TYPE_ENCODING:
      *out_byte = LOOM_BYTECODE_TYPE_ENCODING;
      return iree_ok_status();
    case LOOM_TYPE_POOL:
      *out_byte = LOOM_BYTECODE_TYPE_POOL;
      return iree_ok_status();
    default:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unknown type kind %d",
                          (int)kind);
}

static iree_status_t loom_bytecode_encoding_role_byte(loom_encoding_role_t role,
                                                      uint8_t* out_byte) {
  switch (role) {
    case LOOM_ENCODING_ROLE_UNKNOWN:
      *out_byte = LOOM_BYTECODE_ENCODING_ROLE_UNKNOWN;
      return iree_ok_status();
    case LOOM_ENCODING_ROLE_ADDRESS_LAYOUT:
      *out_byte = LOOM_BYTECODE_ENCODING_ROLE_LAYOUT;
      return iree_ok_status();
    case LOOM_ENCODING_ROLE_STORAGE_SCHEMA:
      *out_byte = LOOM_BYTECODE_ENCODING_ROLE_SCHEMA;
      return iree_ok_status();
    case LOOM_ENCODING_ROLE_PHYSICAL_STORAGE:
      *out_byte = LOOM_BYTECODE_ENCODING_ROLE_STORAGE;
      return iree_ok_status();
    case LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM:
      *out_byte = LOOM_BYTECODE_ENCODING_ROLE_TRANSFORM;
      return iree_ok_status();
    case LOOM_ENCODING_ROLE_COUNT_:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown encoding role %d", (int)role);
}

static iree_status_t loom_bytecode_symbol_kind_byte(loom_symbol_kind_t kind,
                                                    uint8_t* out_byte) {
  switch (kind) {
    case LOOM_SYMBOL_FUNC_DEF:
      *out_byte = LOOM_BYTECODE_SYMBOL_FUNC_DEF;
      return iree_ok_status();
    case LOOM_SYMBOL_FUNC_DECL:
      *out_byte = LOOM_BYTECODE_SYMBOL_FUNC_DECL;
      return iree_ok_status();
    case LOOM_SYMBOL_FUNC_TEMPLATE:
      *out_byte = LOOM_BYTECODE_SYMBOL_FUNC_TEMPLATE;
      return iree_ok_status();
    case LOOM_SYMBOL_FUNC_UKERNEL:
      *out_byte = LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL;
      return iree_ok_status();
    case LOOM_SYMBOL_GLOBAL:
      *out_byte = LOOM_BYTECODE_SYMBOL_GLOBAL;
      return iree_ok_status();
    case LOOM_SYMBOL_EXECUTABLE:
      *out_byte = LOOM_BYTECODE_SYMBOL_EXECUTABLE;
      return iree_ok_status();
    case LOOM_SYMBOL_RECORD:
      *out_byte = LOOM_BYTECODE_SYMBOL_RECORD;
      return iree_ok_status();
    default:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown symbol kind %u", (unsigned)kind);
}

//===----------------------------------------------------------------------===//
// Section writers
//===----------------------------------------------------------------------===//

// Writes the IR section: function bodies streamed through the page writer.
// Returns per-symbol (offset, length) pairs for the SYMBOLS section.
typedef struct loom_bytecode_ir_offset_t {
  // Byte offset of the function body from the IR section start.
  uint64_t offset;
  // Byte length of the function body.
  uint32_t length;
} loom_bytecode_ir_offset_t;

typedef struct loom_bytecode_body_counts_t {
  // Number of SSA values described by this allocation summary.
  uint64_t value_count;
  // Number of serialized regions described by this allocation summary.
  uint64_t region_count;
  // Number of serialized blocks described by this allocation summary.
  uint64_t block_count;
  // Number of serialized live operations described by this allocation summary.
  uint64_t op_count;
} loom_bytecode_body_counts_t;

static void loom_bytecode_count_region_tree(
    const loom_region_t* region, loom_bytecode_body_counts_t* counts) {
  ++counts->region_count;
  counts->block_count += region->block_count;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    counts->value_count += block->arg_count;
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      ++counts->op_count;
      counts->value_count += op->result_count;
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t region_index = 0; region_index < op->region_count;
           ++region_index) {
        if (regions[region_index]) {
          loom_bytecode_count_region_tree(regions[region_index], counts);
        }
      }
    }
  }
}

static void loom_bytecode_count_op_region_forest(
    const loom_op_t* op, uint8_t first_region_index,
    loom_bytecode_body_counts_t* counts) {
  if (op->region_count == 0) {
    return;
  }
  loom_region_t** regions = loom_op_regions(op);
  if (first_region_index != LOOM_REGION_INDEX_NONE &&
      first_region_index < op->region_count && regions[first_region_index]) {
    loom_bytecode_count_region_tree(regions[first_region_index], counts);
  }
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (i == first_region_index || !regions[i]) {
      continue;
    }
    loom_bytecode_count_region_tree(regions[i], counts);
  }
}

static uint8_t loom_bytecode_count_root_regions(const loom_op_t* op) {
  uint8_t count = 0;
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      ++count;
    }
  }
  return count;
}

static iree_status_t loom_bytecode_count_serialized_bodies(
    loom_bytecode_numbering_t* numbering, loom_bytecode_body_counts_t* counts) {
  const loom_module_t* module = numbering->module;
  *counts = (loom_bytecode_body_counts_t){
      .value_count = 0,
      .region_count = 0,
      .block_count = 0,
      .op_count = 0,
  };
  for (iree_host_size_t symbol_index = 0; symbol_index < module->symbols.count;
       ++symbol_index) {
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_index];
    loom_symbol_kind_t bytecode_kind = loom_symbol_bytecode_kind(symbol);
    bool is_function_like =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
        loom_symbol_kind_is_function_like(bytecode_kind);
    bool is_global =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_GLOBAL) ||
        bytecode_kind == LOOM_SYMBOL_GLOBAL;
    bool is_record =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_RECORD) ||
        bytecode_kind == LOOM_SYMBOL_RECORD;
    if (!is_function_like || !symbol->defining_op) {
      if (is_global && symbol->defining_op) {
        loom_bytecode_global_value_list_t local_values = {0};
        IREE_RETURN_IF_ERROR(loom_bytecode_collect_global_values(
            numbering->arena, module, symbol->defining_op, &local_values));
        counts->value_count += local_values.count;
      } else if (is_record && symbol->defining_op &&
                 symbol->defining_op->region_count == 1) {
        loom_region_t* body = loom_op_regions(symbol->defining_op)[0];
        if (body) {
          loom_bytecode_count_region_tree(body, counts);
        }
      }
      continue;
    }
    loom_func_like_t func_like =
        loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_isa(func_like)) {
      continue;
    }
    counts->value_count += func_like.op->result_count;
    if (loom_bytecode_count_root_regions(func_like.op) != 0) {
      loom_bytecode_count_op_region_forest(
          func_like.op, func_like.vtable->body_region_index, counts);
    } else {
      uint16_t arg_count = 0;
      loom_func_like_arg_ids(func_like, &arg_count);
      counts->value_count += arg_count;
    }
  }
  return iree_ok_status();
}

// Forward declarations for recursive IR writing.
static iree_status_t loom_bytecode_write_region(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_region_t* region, uint32_t depth);

static iree_status_t loom_bytecode_write_value_def(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_value_t* value) {
  uint32_t name_writer_id = 0;
  if (value->name_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, value->name_id, &name_writer_id));
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, name_writer_id));

  uint32_t type_writer_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
      numbering, value->type, &type_writer_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, type_writer_id));

  // Dim bindings: count dynamic dims, then emit value refs.
  loom_type_t type = value->type;
  uint8_t rank = loom_type_rank(type);
  uint32_t dynamic_count = 0;
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(type, i)) ++dynamic_count;
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, dynamic_count));
  for (uint8_t i = 0; i < rank; ++i) {
    uint64_t packed = loom_type_dim(type, i);
    if (!loom_dim_is_dynamic(packed)) continue;
    loom_value_id_t dim_value_id = loom_dim_value_id(packed);
    uint32_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
        value_numbering, dim_value_id, &value_number));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_svarint(writer, (int64_t)value_number));
  }

  // Encoding binding.
  if (loom_type_has_ssa_encoding(type)) {
    uint16_t encoding_value_id = loom_type_encoding_value_id(type);
    uint32_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
        value_numbering, encoding_value_id, &value_number));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, 1 + value_number));
  } else {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(writer, 0));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_emit_value_def(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_value_t* value) {
  uint32_t name_writer_id = 0;
  if (value->name_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, value->name_id, &name_writer_id));
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, (uint64_t)name_writer_id));

  uint32_t type_writer_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
      numbering, value->type, &type_writer_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, (uint64_t)type_writer_id));

  loom_type_t type = value->type;
  uint8_t rank = loom_type_rank(type);
  uint32_t dynamic_count = 0;
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(type, i)) ++dynamic_count;
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, (uint64_t)dynamic_count));
  for (uint8_t i = 0; i < rank; ++i) {
    uint64_t packed = loom_type_dim(type, i);
    if (!loom_dim_is_dynamic(packed)) continue;
    loom_value_id_t dim_value_id = loom_dim_value_id(packed);
    uint32_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
        value_numbering, dim_value_id, &value_number));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_emit_svarint(builder, (int64_t)value_number));
  }

  if (loom_type_has_ssa_encoding(type)) {
    uint16_t encoding_value_id = loom_type_encoding_value_id(type);
    uint32_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
        value_numbering, encoding_value_id, &value_number));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_emit_uvarint(builder, 1 + (uint64_t)value_number));
  } else {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, 0));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_attr_value(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor) {
  switch (attr.kind) {
    case LOOM_ATTR_I64: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 0));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_svarint(writer, attr.i64));
      break;
    }
    case LOOM_ATTR_F64: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 1));
      uint8_t bytes[8];
      memcpy(bytes, &attr.f64, 8);
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(writer, bytes, 8));
      break;
    }
    case LOOM_ATTR_STRING: {
      uint32_t string_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, attr.string_id, &string_writer_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 2));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, string_writer_id));
      break;
    }
    case LOOM_ATTR_BOOL: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 3));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_u8(writer, attr.raw ? 1 : 0));
      break;
    }
    case LOOM_ATTR_ENUM: {
      uint8_t ordinal = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_get_enum_ordinal(attr, descriptor, &ordinal));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 4));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, ordinal));
      break;
    }
    case LOOM_ATTR_I64_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 5));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_svarint(writer, attr.i64_array[i]));
      }
      break;
    }
    case LOOM_ATTR_SYMBOL: {
      loom_symbol_ref_t ref = attr.symbol;
      uint32_t string_writer_id = 0;
      if (loom_symbol_ref_is_valid(ref) &&
          ref.symbol_id < numbering->module->symbols.count) {
        const loom_symbol_t* target =
            &numbering->module->symbols.entries[ref.symbol_id];
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, target->name_id, &string_writer_id));
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 6));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, string_writer_id));
      break;
    }
    case LOOM_ATTR_TYPE: {
      if (attr.type_id >= numbering->module->types.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "type attribute id %u out of range (module has %" PRIhsz " types)",
            (unsigned)attr.type_id, numbering->module->types.count);
      }
      loom_type_t type = numbering->module->types.entries[attr.type_id];
      uint32_t type_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
          numbering, type, &type_writer_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 7));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, type_writer_id));
      break;
    }
    case LOOM_ATTR_PREDICATE_LIST: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 8));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_predicate_t* predicate = &attr.predicate_list[i];
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u8(writer, predicate->kind));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u8(writer, predicate->arg_count));
        for (uint8_t arg_index = 0; arg_index < predicate->arg_count;
             ++arg_index) {
          uint8_t tag = predicate->arg_tags[arg_index];
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, tag));
          switch (tag) {
            case LOOM_PRED_ARG_VALUE: {
              if (value_numbering) {
                uint32_t value_number = 0;
                IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
                    value_numbering,
                    (loom_value_id_t)predicate->args[arg_index],
                    &value_number));
                IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
                    writer, value_number));
              } else {
                loom_string_id_t name_id =
                    (loom_string_id_t)predicate->args[arg_index];
                uint32_t string_writer_id = 0;
                IREE_RETURN_IF_ERROR(
                    loom_bytecode_numbering_intern_module_string(
                        numbering, name_id, &string_writer_id));
                IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
                    writer, string_writer_id));
              }
              break;
            }
            case LOOM_PRED_ARG_CONST: {
              IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_svarint(
                  writer, predicate->args[arg_index]));
              break;
            }
            default:
              return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                      "unknown predicate arg tag %d", (int)tag);
          }
        }
      }
      break;
    }
    case LOOM_ATTR_DICT: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 9));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.count));
      // Dict attrs are stored canonically at IR construction time; emit that
      // order directly so logically equivalent dicts produce stable bytes.
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_named_attr_t* entry = &attr.dict_entries[i];
        uint32_t key_writer_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, entry->name_id, &key_writer_id));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_uvarint(writer, key_writer_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value(
            writer, numbering, value_numbering, entry->value, NULL));
      }
      break;
    }
    case LOOM_ATTR_ENCODING: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, 10));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.encoding_id));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported attribute kind %d", (int)attr.kind);
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_emit_attr_value(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor) {
  switch (attr.kind) {
    case LOOM_ATTR_I64: {
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, 0));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_svarint(builder, attr.i64));
      break;
    }
    case LOOM_ATTR_F64: {
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, 1));
      uint64_t bits = 0;
      memcpy(&bits, &attr.f64, sizeof(bits));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u64_le(builder, bits));
      break;
    }
    case LOOM_ATTR_STRING: {
      uint32_t string_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, attr.string_id, &string_writer_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, 2));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, string_writer_id));
      break;
    }
    case LOOM_ATTR_BOOL: {
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, 3));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, attr.raw ? 1 : 0));
      break;
    }
    case LOOM_ATTR_ENUM: {
      uint8_t ordinal = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_get_enum_ordinal(attr, descriptor, &ordinal));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, 4));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, ordinal));
      break;
    }
    case LOOM_ATTR_I64_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, 5));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_emit_svarint(builder, attr.i64_array[i]));
      }
      break;
    }
    case LOOM_ATTR_SYMBOL: {
      loom_symbol_ref_t ref = attr.symbol;
      uint32_t string_writer_id = 0;
      if (loom_symbol_ref_is_valid(ref) &&
          ref.symbol_id < numbering->module->symbols.count) {
        const loom_symbol_t* target =
            &numbering->module->symbols.entries[ref.symbol_id];
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, target->name_id, &string_writer_id));
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, 6));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, string_writer_id));
      break;
    }
    case LOOM_ATTR_TYPE: {
      if (attr.type_id >= numbering->module->types.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "type attribute id %u out of range (module has %" PRIhsz " types)",
            (unsigned)attr.type_id, numbering->module->types.count);
      }
      loom_type_t type = numbering->module->types.entries[attr.type_id];
      uint32_t type_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
          numbering, type, &type_writer_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, 7));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, type_writer_id));
      break;
    }
    case LOOM_ATTR_PREDICATE_LIST: {
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, 8));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_predicate_t* predicate = &attr.predicate_list[i];
        IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, predicate->kind));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_emit_u8(builder, predicate->arg_count));
        for (uint8_t arg_index = 0; arg_index < predicate->arg_count;
             ++arg_index) {
          uint8_t tag = predicate->arg_tags[arg_index];
          IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, tag));
          switch (tag) {
            case LOOM_PRED_ARG_VALUE: {
              if (value_numbering) {
                uint32_t value_number = 0;
                IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
                    value_numbering,
                    (loom_value_id_t)predicate->args[arg_index],
                    &value_number));
                IREE_RETURN_IF_ERROR(
                    loom_bytecode_emit_uvarint(builder, value_number));
              } else {
                loom_string_id_t name_id =
                    (loom_string_id_t)predicate->args[arg_index];
                uint32_t string_writer_id = 0;
                IREE_RETURN_IF_ERROR(
                    loom_bytecode_numbering_intern_module_string(
                        numbering, name_id, &string_writer_id));
                IREE_RETURN_IF_ERROR(
                    loom_bytecode_emit_uvarint(builder, string_writer_id));
              }
              break;
            }
            case LOOM_PRED_ARG_CONST: {
              IREE_RETURN_IF_ERROR(loom_bytecode_emit_svarint(
                  builder, predicate->args[arg_index]));
              break;
            }
            default:
              return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                      "unknown predicate arg tag %d", (int)tag);
          }
        }
      }
      break;
    }
    case LOOM_ATTR_DICT: {
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, 9));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_named_attr_t* entry = &attr.dict_entries[i];
        uint32_t key_writer_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, entry->name_id, &key_writer_id));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_emit_uvarint(builder, key_writer_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_emit_attr_value(
            builder, numbering, value_numbering, entry->value, NULL));
      }
      break;
    }
    case LOOM_ATTR_ENCODING: {
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, 10));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, attr.encoding_id));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported attribute kind %d", (int)attr.kind);
  }
  return iree_ok_status();
}

static uint8_t loom_bytecode_instance_flags_mask(
    const loom_op_vtable_t* vtable) {
  if (!iree_all_bits_set(vtable->vtable_flags,
                         LOOM_OP_VTABLE_HAS_INSTANCE_FLAGS)) {
    return 0;
  }
  if (vtable->instance_flags_case_count >= 8) return UINT8_MAX;
  return (uint8_t)((1u << vtable->instance_flags_case_count) - 1u);
}

static iree_status_t loom_bytecode_find_successor_block_index(
    const loom_op_t* op, const loom_block_t* target,
    uint16_t* out_block_index) {
  if (!target) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "operation successor target is NULL");
  }
  if (!op->parent_block || !op->parent_block->parent_region) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "operation with successors is not attached to a region");
  }
  const loom_region_t* region = op->parent_block->parent_region;
  if (target->parent_region && target->parent_region != region) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "operation successor target belongs to a different region");
  }
  for (uint16_t i = 0; i < region->block_count; ++i) {
    if (loom_region_const_block(region, i) == target) {
      *out_block_index = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "operation successor target is not in its region");
}

static iree_status_t loom_bytecode_write_operation(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering, const loom_op_t* op,
    uint32_t depth) {
  const loom_module_t* module = numbering->module;
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(module->context, op->kind);
  if (!vtable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "op kind 0x%04x has no registered vtable",
                            (unsigned)op->kind);
  }

  // Operation table index, plus one so 0 remains an invalid reference.
  uint32_t writer_op_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &writer_op_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, writer_op_id + 1));

  uint8_t instance_flags_mask = loom_bytecode_instance_flags_mask(vtable);
  if (iree_any_bit_set(op->instance_flags, (uint8_t)~instance_flags_mask)) {
    iree_string_view_t name = loom_op_vtable_name(vtable);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "op %.*s has undeclared instance flag bits 0x%02x", (int)name.size,
        name.data,
        (unsigned)(op->instance_flags & (uint8_t)~instance_flags_mask));
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_u8(writer, op->instance_flags));

  loom_location_id_t location =
      numbering->location_mode == LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS
          ? LOOM_LOCATION_UNKNOWN
          : op->location;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, location));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, op, &comment_count);
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_comment_list(
      writer, comments, comment_count));

  // Operands.
  const loom_value_id_t* operands = loom_op_const_operands(op);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->operand_count));
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    uint32_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
        value_numbering, operands[i], &value_number));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, value_number));
  }
  uint8_t operand_segment_count = loom_op_vtable_operand_segment_count(vtable);
  const uint16_t* operand_segment_counts =
      loom_op_const_operand_segment_counts(op);
  for (uint8_t i = 0; i < operand_segment_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        writer, operand_segment_counts[i]));
  }

  // Successors are encoded as region-local block ordinals. This keeps bytecode
  // independent from optional text labels and makes forward edges cheap.
  loom_block_t* const* successors = loom_op_const_successors(op);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->successor_count));
  for (uint8_t i = 0; i < op->successor_count; ++i) {
    uint16_t block_index = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_find_successor_block_index(
        op, successors[i], &block_index));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, block_index));
  }

  // Results.
  const loom_value_id_t* results = loom_op_const_results(op);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->result_count));
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_t* value = &module->values.entries[results[i]];
    IREE_RETURN_IF_ERROR(loom_bytecode_write_value_def(writer, numbering,
                                                       value_numbering, value));
  }

  // Tied results.
  const loom_tied_result_t* tied = loom_op_tied_results(op);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->tied_result_count));
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, tied[i].result_index));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, tied[i].operand_index));
  }

  // Attributes: each has a key name from the vtable descriptor and a
  // tagged value from the op's trailing data.
  const loom_attribute_t* attrs = loom_op_attrs(op);
  if (op->attribute_count > 0 && !vtable->attr_descriptors) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "op kind 0x%04x has %d attributes but no attr_descriptors in vtable",
        (unsigned)op->kind, (int)op->attribute_count);
  }
  uint8_t present_attr_count = 0;
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_op_attr_is_present(
        op, &vtable->attr_descriptors[i], attrs[i], &present));
    if (present) ++present_attr_count;
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, present_attr_count));
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_op_attr_is_present(
        op, &vtable->attr_descriptors[i], attrs[i], &present));
    if (!present) continue;

    iree_string_view_t key_name =
        loom_attr_descriptor_name(&vtable->attr_descriptors[i]);
    uint32_t key_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, key_name, &key_writer_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, key_writer_id));
    // Value.
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value(
        writer, numbering, value_numbering, attrs[i], descriptor));
  }

  // Regions.
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->region_count));
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      IREE_RETURN_IF_ERROR(loom_bytecode_write_region(
          writer, numbering, value_numbering, regions[i], depth + 1));
    } else {
      // Empty region: 0 blocks.
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(writer, 0));
    }
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_block(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_block_t* block, uint32_t depth) {
  const loom_module_t* module = numbering->module;

  // Label.
  bool has_label = block->label_id != LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_u8(writer, has_label ? 1 : 0));
  if (has_label) {
    uint32_t label_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, block->label_id, &label_writer_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, label_writer_id));
  }

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_block_comments(module, block, &comment_count);
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_comment_list(
      writer, comments, comment_count));

  // Block args.
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, block->arg_count));
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    loom_value_id_t value_id = loom_block_arg_id(block, i);
    const loom_value_t* value = &module->values.entries[value_id];
    IREE_RETURN_IF_ERROR(loom_bytecode_write_value_def(writer, numbering,
                                                       value_numbering, value));
  }

  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, block->op_count));
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    IREE_RETURN_IF_ERROR(loom_bytecode_write_operation(
        writer, numbering, value_numbering, op, depth));
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_region(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_region_t* region, uint32_t depth) {
  if (depth >= LOOM_BYTECODE_MAX_REGION_DEPTH) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "region nesting exceeds maximum depth %d",
                            LOOM_BYTECODE_MAX_REGION_DEPTH);
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, region->block_count));
  for (uint16_t i = 0; i < region->block_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_write_block(writer, numbering, value_numbering,
                                  loom_region_const_block(region, i), depth));
  }
  return iree_ok_status();
}

// Writes the IR section and returns per-symbol offsets.
static iree_status_t loom_bytecode_write_ir_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering,
    loom_bytecode_ir_offset_t* ir_offsets) {
  const loom_module_t* module = numbering->module;
  iree_host_size_t section_start = page_writer->total_written;

  for (iree_host_size_t symbol_index = 0; symbol_index < module->symbols.count;
       ++symbol_index) {
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_index];
    loom_symbol_kind_t bytecode_kind = loom_symbol_bytecode_kind(symbol);
    bool is_function_like =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
        loom_symbol_kind_is_function_like(bytecode_kind);
    bool is_record =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_RECORD) ||
        bytecode_kind == LOOM_SYMBOL_RECORD;
    if (!symbol->defining_op) {
      ir_offsets[symbol_index].offset = 0;
      ir_offsets[symbol_index].length = 0;
      continue;
    }

    uint8_t first_region_index = LOOM_REGION_INDEX_NONE;
    uint8_t root_region_count = 0;
    if (is_function_like) {
      loom_func_like_t func_like =
          loom_func_like_cast(module, symbol->defining_op);
      if (!loom_func_like_isa(func_like)) {
        ir_offsets[symbol_index].offset = 0;
        ir_offsets[symbol_index].length = 0;
        continue;
      }
      root_region_count = loom_bytecode_count_root_regions(func_like.op);
      if (root_region_count == 0) {
        ir_offsets[symbol_index].offset = 0;
        ir_offsets[symbol_index].length = 0;
        continue;
      }
      first_region_index = func_like.vtable->body_region_index;
      if (first_region_index != LOOM_REGION_INDEX_NONE &&
          (first_region_index >= func_like.op->region_count ||
           !loom_op_regions(func_like.op)[first_region_index])) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "func-like symbol op kind 0x%04x has root regions but no body "
            "region",
            func_like.op->kind);
      }

      // Intern function signature and regions (matching Python walk order).
      IREE_RETURN_IF_ERROR(loom_bytecode_number_function(numbering, func_like));
    } else if (is_record && symbol->defining_op->region_count == 1) {
      root_region_count = loom_bytecode_count_root_regions(symbol->defining_op);
      if (root_region_count == 0) {
        ir_offsets[symbol_index].offset = 0;
        ir_offsets[symbol_index].length = 0;
        continue;
      }

      // Intern record metadata and body (matching Python walk order).
      IREE_RETURN_IF_ERROR(
          loom_bytecode_number_record(numbering, symbol->defining_op));
    } else {
      ir_offsets[symbol_index].offset = 0;
      ir_offsets[symbol_index].length = 0;
      continue;
    }

    loom_bytecode_body_counts_t body_counts = {0};
    loom_bytecode_count_op_region_forest(symbol->defining_op,
                                         first_region_index, &body_counts);

    loom_bytecode_value_numbering_t value_numbering;
    loom_bytecode_value_numbering_initialize(&value_numbering, module,
                                             numbering->arena);
    IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_ensure_capacity(
        &value_numbering, body_counts.value_count));

    // Assign value numbers in the same definition order used by the reader.
    loom_region_t** regions = loom_op_regions(symbol->defining_op);
    if (first_region_index != LOOM_REGION_INDEX_NONE &&
        first_region_index < symbol->defining_op->region_count &&
        regions[first_region_index]) {
      IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_region(
          &value_numbering, regions[first_region_index]));
    }
    for (uint8_t i = 0; i < symbol->defining_op->region_count; ++i) {
      if (i == first_region_index || !regions[i]) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_region(
          &value_numbering, regions[i]));
    }

    // Write the region summary and root region list. Each root region carries
    // its declared op region index so source op region order survives even when
    // the body is serialized first for signature value numbering.
    iree_host_size_t body_start = page_writer->total_written;
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, body_counts.value_count));
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, body_counts.region_count));
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, body_counts.block_count));
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, body_counts.op_count));
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, root_region_count));
    if (first_region_index != LOOM_REGION_INDEX_NONE &&
        first_region_index < symbol->defining_op->region_count &&
        regions[first_region_index]) {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, first_region_index));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_write_region(page_writer, numbering, &value_numbering,
                                     regions[first_region_index], 0));
    }
    for (uint8_t i = 0; i < symbol->defining_op->region_count; ++i) {
      if (i == first_region_index || !regions[i]) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(page_writer, i));
      IREE_RETURN_IF_ERROR(loom_bytecode_write_region(
          page_writer, numbering, &value_numbering, regions[i], 0));
    }
    iree_host_size_t body_length = page_writer->total_written - body_start;

    if (body_length > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "function body length %" PRIhsz
                              " exceeds uint32 maximum",
                              body_length);
    }
    ir_offsets[symbol_index].offset = body_start - section_start;
    ir_offsets[symbol_index].length = (uint32_t)body_length;
  }

  return iree_ok_status();
}

// Writes the function-like metadata fields for a single symbol entry.
// Called from loom_bytecode_write_symbols_section for each function-like
// symbol that has a defining op.
static bool loom_bytecode_func_metadata_attr_is_shared(
    const loom_op_vtable_t* vtable, const loom_func_like_vtable_t* func_like,
    uint8_t attr_index) {
  if (loom_bytecode_attr_is_symbol_identity(vtable, attr_index)) return true;
  iree_string_view_t name =
      loom_attr_descriptor_name(&vtable->attr_descriptors[attr_index]);
  if (iree_string_view_equal(name, IREE_SV("import_module")) ||
      iree_string_view_equal(name, IREE_SV("import_symbol")) ||
      attr_index == func_like->visibility_attr_index ||
      attr_index == func_like->cc_attr_index ||
      attr_index == func_like->purity_attr_index ||
      attr_index == func_like->predicates_attr_index) {
    return true;
  }
  if ((vtable->symbol_kind == LOOM_SYMBOL_FUNC_TEMPLATE ||
       vtable->symbol_kind == LOOM_SYMBOL_FUNC_UKERNEL) &&
      (attr_index == func_like->implements_attr_index ||
       attr_index == func_like->priority_attr_index)) {
    return true;
  }
  return false;
}

static iree_status_t loom_bytecode_write_func_payload_attrs(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_module_t* module, loom_func_like_t func_like,
    loom_bytecode_value_numbering_t* signature_numbering) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, func_like.op);
  const loom_attribute_t* attrs = loom_op_attrs(func_like.op);
  uint8_t present_attr_count = 0;
  for (uint8_t i = 0; i < func_like.op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_op_attr_is_present(
        func_like.op, descriptor, attrs[i], &present));
    if (present && !loom_bytecode_func_metadata_attr_is_shared(
                       vtable, func_like.vtable, i)) {
      ++present_attr_count;
    }
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, present_attr_count));
  for (uint8_t i = 0; i < func_like.op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_op_attr_is_present(
        func_like.op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_func_metadata_attr_is_shared(
                        vtable, func_like.vtable, i)) {
      continue;
    }
    uint32_t key_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, loom_attr_descriptor_name(descriptor), &key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_attr_value(
        builder, numbering, signature_numbering, attrs[i], descriptor));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_func_metadata(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_module_t* module, loom_func_like_t func_like,
    loom_bytecode_value_numbering_t* signature_numbering,
    loom_bytecode_ir_offset_t ir_offset) {
  uint32_t writer_op_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_op(
      numbering, func_like.op, &writer_op_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, (uint64_t)writer_op_id + 1));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, func_like.op, &comment_count);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_comment_list(builder, comments, comment_count));

  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_u8(builder, loom_func_like_cc(func_like)));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_u8(builder, loom_func_like_purity(func_like)));

  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(func_like, &arg_count);
  uint16_t result_count = func_like.op->result_count;
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, arg_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, result_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_ensure_capacity(
      signature_numbering, (iree_host_size_t)arg_count + result_count));

  for (uint16_t i = 0; i < arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
        signature_numbering, arg_ids[i]));
  }
  const loom_value_id_t* result_ids = loom_op_const_results(func_like.op);
  for (uint16_t i = 0; i < result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
        signature_numbering, result_ids[i]));
  }

  // Arg value definitions.
  for (uint16_t i = 0; i < arg_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_emit_value_def(builder, numbering, signature_numbering,
                                     &module->values.entries[arg_ids[i]]));
  }

  // Result value definitions with tied info.
  const loom_tied_result_t* tied_results = loom_op_tied_results(func_like.op);
  uint16_t tied_result_count = func_like.op->tied_result_count;
  for (uint16_t i = 0; i < result_count; ++i) {
    bool is_tied = false;
    uint16_t tied_operand_index = 0;
    for (uint16_t t = 0; t < tied_result_count; ++t) {
      if (tied_results[t].result_index == i) {
        is_tied = true;
        tied_operand_index = tied_results[t].operand_index;
        break;
      }
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, is_tied ? 1 : 0));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_emit_value_def(builder, numbering, signature_numbering,
                                     &module->values.entries[result_ids[i]]));
    if (is_tied) {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, tied_operand_index));
    }
  }

  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, tied_result_count));

  // Predicates.
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(func_like, &predicate_count);
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, predicate_count));
  for (uint16_t i = 0; i < predicate_count; ++i) {
    const loom_predicate_t* predicate = &predicates[i];
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, predicate->kind));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, predicate->arg_count));
    for (uint8_t arg_index = 0; arg_index < predicate->arg_count; ++arg_index) {
      uint8_t tag = predicate->arg_tags[arg_index];
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, tag));
      switch (tag) {
        case LOOM_PRED_ARG_VALUE: {
          uint32_t value_number = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
              signature_numbering, (loom_value_id_t)predicate->args[arg_index],
              &value_number));
          IREE_RETURN_IF_ERROR(
              loom_bytecode_emit_uvarint(builder, value_number));
          break;
        }
        case LOOM_PRED_ARG_CONST: {
          IREE_RETURN_IF_ERROR(
              loom_bytecode_emit_svarint(builder, predicate->args[arg_index]));
          break;
        }
        default:
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unknown predicate arg tag %d", (int)tag);
      }
    }
  }

  // Template/ukernel dispatch metadata: the name of the op kind this function
  // provides an implementation for, and its matching priority. Written only
  // for templates and ukernels; absent for def/decl.
  const loom_op_vtable_t* vtable = loom_op_vtable(module, func_like.op);
  loom_string_id_t implements_id = loom_func_like_implements(func_like);
  if (vtable->symbol_kind == LOOM_SYMBOL_FUNC_TEMPLATE ||
      vtable->symbol_kind == LOOM_SYMBOL_FUNC_UKERNEL) {
    if (implements_id == LOOM_STRING_ID_INVALID) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "template/ukernel function symbol must have implements metadata");
    }
    uint32_t implements_string_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, implements_id, &implements_string_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_emit_uvarint(builder, implements_string_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(
        builder, (uint64_t)loom_func_like_priority(func_like)));
  }

  IREE_RETURN_IF_ERROR(loom_bytecode_write_func_payload_attrs(
      builder, numbering, module, func_like, signature_numbering));

  bool has_body = ir_offset.length > 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, has_body ? 1 : 0));
  if (has_body) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u64_le(builder, ir_offset.offset));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u32_le(builder, ir_offset.length));
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_global_metadata(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_module_t* module, const loom_op_t* op,
    loom_bytecode_value_numbering_t* value_numbering) {
  loom_bytecode_global_value_list_t local_values = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_collect_global_values(
      numbering->arena, module, op, &local_values));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_number_global(numbering, op, &local_values));
  IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_ensure_capacity(
      value_numbering, local_values.count));

  uint32_t writer_op_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &writer_op_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, (uint64_t)writer_op_id + 1));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, op, &comment_count);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_comment_list(builder, comments, comment_count));

  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, op->result_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, local_values.count));
  for (iree_host_size_t i = 0; i < local_values.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
        value_numbering, local_values.values[i]));
  }
  for (iree_host_size_t i = 0; i < local_values.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_value_def(
        builder, numbering, value_numbering,
        &module->values.entries[local_values.values[i]]));
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  const loom_attribute_t* attrs = loom_op_attrs(op);
  uint8_t present_attr_count = 0;
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (present && !loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      ++present_attr_count;
    }
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, present_attr_count));
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      continue;
    }

    uint32_t key_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, loom_attr_descriptor_name(descriptor), &key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_attr_value(
        builder, numbering, value_numbering, attrs[i], descriptor));
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_record_metadata(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_module_t* module, const loom_op_t* op,
    loom_bytecode_ir_offset_t ir_offset) {
  IREE_RETURN_IF_ERROR(loom_bytecode_validate_record_symbol_op(module, op));
  IREE_RETURN_IF_ERROR(loom_bytecode_number_record(numbering, op));

  uint32_t writer_op_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &writer_op_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, (uint64_t)writer_op_id + 1));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, op, &comment_count);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_comment_list(builder, comments, comment_count));

  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  const loom_attribute_t* attrs = loom_op_attrs(op);
  uint8_t present_attr_count = 0;
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (present && !loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      ++present_attr_count;
    }
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, present_attr_count));
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      continue;
    }

    uint32_t key_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, loom_attr_descriptor_name(descriptor), &key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_attr_value(builder, numbering, NULL,
                                                       attrs[i], descriptor));
  }

  bool has_body = ir_offset.length > 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, has_body ? 1 : 0));
  if (has_body) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u64_le(builder, ir_offset.offset));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u32_le(builder, ir_offset.length));
  }

  return iree_ok_status();
}

static loom_attribute_t loom_bytecode_find_op_attr_by_name(
    const loom_op_vtable_t* vtable, const loom_op_t* op,
    iree_string_view_t name) {
  if (!vtable || !vtable->attr_descriptors) return loom_attr_absent();
  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (iree_string_view_equal(
            loom_attr_descriptor_name(&vtable->attr_descriptors[i]), name)) {
      return attrs[i];
    }
  }
  return loom_attr_absent();
}

static iree_status_t loom_bytecode_find_string_attr_by_name(
    const loom_op_vtable_t* vtable, const loom_op_t* op,
    iree_string_view_t name, loom_string_id_t* out_string_id) {
  loom_attribute_t attr = loom_bytecode_find_op_attr_by_name(vtable, op, name);
  if (loom_attr_is_absent(attr)) {
    *out_string_id = LOOM_STRING_ID_INVALID;
    return iree_ok_status();
  }
  if (attr.kind != LOOM_ATTR_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function symbol attribute %.*s must be a string",
                            (int)name.size, name.data);
  }
  *out_string_id = loom_attr_as_string_id(attr);
  return iree_ok_status();
}

typedef struct loom_bytecode_symbol_linkage_t {
  bool is_public;                     // Symbol is externally visible.
  bool is_import;                     // Symbol resolves to another module.
  bool has_import_symbol;             // Source symbol was explicitly authored.
  loom_string_id_t import_module_id;  // Source module for imported symbols.
  loom_string_id_t import_symbol_id;  // Source symbol for imported symbols.
} loom_bytecode_symbol_linkage_t;

static bool loom_bytecode_symbol_has_visibility_attr(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (!symbol->defining_op) return false;
  const loom_op_vtable_t* vtable = loom_op_vtable(module, symbol->defining_op);
  if (!vtable || !vtable->attr_descriptors) return false;
  const loom_attribute_t* attrs = loom_op_const_attrs(symbol->defining_op);
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (!iree_string_view_equal(loom_attr_descriptor_name(descriptor),
                                IREE_SV("visibility"))) {
      continue;
    }
    if (descriptor->attr_kind != LOOM_ATTR_ENUM ||
        i >= symbol->defining_op->attribute_count) {
      return false;
    }
    return loom_attr_as_enum(attrs[i]) != 0;
  }
  return false;
}

static iree_status_t loom_bytecode_symbol_linkage(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_bytecode_symbol_linkage_t* out_linkage) {
  *out_linkage = (loom_bytecode_symbol_linkage_t){
      .is_public = iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC) ||
                   loom_bytecode_symbol_has_visibility_attr(module, symbol),
      .is_import = false,
      .has_import_symbol = false,
      .import_module_id = LOOM_STRING_ID_INVALID,
      .import_symbol_id = LOOM_STRING_ID_INVALID,
  };
  if (!symbol->defining_op) return iree_ok_status();

  loom_func_like_t func_like = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(func_like)) return iree_ok_status();

  const loom_op_vtable_t* op_vtable = loom_op_vtable(module, func_like.op);
  IREE_RETURN_IF_ERROR(loom_bytecode_find_string_attr_by_name(
      op_vtable, func_like.op, IREE_SV("import_module"),
      &out_linkage->import_module_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_find_string_attr_by_name(
      op_vtable, func_like.op, IREE_SV("import_symbol"),
      &out_linkage->import_symbol_id));
  out_linkage->has_import_symbol =
      out_linkage->import_symbol_id != LOOM_STRING_ID_INVALID;

  if (out_linkage->import_module_id == LOOM_STRING_ID_INVALID) {
    if (out_linkage->import_symbol_id != LOOM_STRING_ID_INVALID) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "function symbol import_symbol requires import_module");
    }
    return iree_ok_status();
  }

  out_linkage->is_import = true;
  if (out_linkage->import_symbol_id == LOOM_STRING_ID_INVALID) {
    out_linkage->import_symbol_id = symbol->name_id;
  }
  return iree_ok_status();
}

// Writes the SYMBOLS section into a string builder (for offset table patching).
static iree_status_t loom_bytecode_write_symbols_section(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_ir_offset_t* ir_offsets) {
  const loom_module_t* module = numbering->module;

  // Classify symbols.
  uint32_t import_count = 0;
  uint32_t export_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    loom_bytecode_symbol_linkage_t linkage;
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_linkage(
        module, &module->symbols.entries[i], &linkage));
    if (linkage.is_import) {
      ++import_count;
    } else if (linkage.is_public) {
      ++export_count;
    }
  }

  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, module->symbols.count));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, import_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, export_count));

  // Reserve import/export offset tables (patched after writing entries).
  iree_host_size_t import_table_offset = iree_string_builder_size(builder);
  for (uint32_t i = 0; i < import_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u64_le(builder, 0));
  }
  iree_host_size_t export_table_offset = iree_string_builder_size(builder);
  for (uint32_t i = 0; i < export_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u64_le(builder, 0));
  }

  iree_host_size_t entries_start = iree_string_builder_size(builder);
  uint32_t import_index = 0;
  uint32_t export_index = 0;

  for (iree_host_size_t symbol_index = 0; symbol_index < module->symbols.count;
       ++symbol_index) {
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_index];
    loom_bytecode_symbol_linkage_t linkage;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_symbol_linkage(module, symbol, &linkage));
    uint64_t entry_offset = iree_string_builder_size(builder) - entries_start;

    // Track import/export offsets.
    if (linkage.is_import) {
      loom_bytecode_patch_u64_le(
          builder, import_table_offset + (iree_host_size_t)import_index * 8,
          entry_offset);
      ++import_index;
    } else if (linkage.is_public) {
      loom_bytecode_patch_u64_le(
          builder, export_table_offset + (iree_host_size_t)export_index * 8,
          entry_offset);
      ++export_index;
    }

    // Name.
    uint32_t name_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, symbol->name_id, &name_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, name_writer_id));

    // Kind.
    uint8_t kind_byte = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_kind_byte(
        loom_symbol_bytecode_kind(symbol), &kind_byte));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, kind_byte));

    // Visibility.
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(
        builder, linkage.is_public ? LOOM_BYTECODE_SYMBOL_VISIBILITY_PUBLIC
                                   : LOOM_BYTECODE_SYMBOL_VISIBILITY_PRIVATE));

    // Flags.
    uint16_t bytecode_flags =
        linkage.is_public ? LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC : 0;
    if (linkage.is_import) {
      bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_IMPORT;
      if (linkage.has_import_symbol) {
        bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_IMPORT_SYMBOL;
      }
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_u16_le(builder, bytecode_flags));
    if (linkage.is_import) {
      uint32_t import_module_string_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, linkage.import_module_id, &import_module_string_id));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, import_module_string_id));
      uint32_t import_symbol_string_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, linkage.import_symbol_id, &import_symbol_string_id));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, import_symbol_string_id));
    }

    // Function metadata.
    loom_symbol_kind_t metadata_kind = loom_symbol_bytecode_kind(symbol);
    bool has_function_metadata =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
        loom_symbol_kind_is_function_like(metadata_kind);
    bool has_global_metadata =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_GLOBAL) ||
        metadata_kind == LOOM_SYMBOL_GLOBAL;
    bool has_record_metadata =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_RECORD) ||
        metadata_kind == LOOM_SYMBOL_RECORD;
    if (has_function_metadata && symbol->defining_op) {
      loom_func_like_t func_like =
          loom_func_like_cast(module, symbol->defining_op);
      if (loom_func_like_isa(func_like)) {
        loom_bytecode_value_numbering_t signature_numbering;
        loom_bytecode_value_numbering_initialize(&signature_numbering, module,
                                                 numbering->arena);
        IREE_RETURN_IF_ERROR(loom_bytecode_write_func_metadata(
            builder, numbering, module, func_like, &signature_numbering,
            ir_offsets[symbol_index]));
      }
    } else if (has_global_metadata && symbol->defining_op) {
      loom_bytecode_value_numbering_t signature_numbering;
      loom_bytecode_value_numbering_initialize(&signature_numbering, module,
                                               numbering->arena);
      IREE_RETURN_IF_ERROR(loom_bytecode_write_global_metadata(
          builder, numbering, module, symbol->defining_op,
          &signature_numbering));
    } else if (has_record_metadata && symbol->defining_op) {
      IREE_RETURN_IF_ERROR(loom_bytecode_write_record_metadata(
          builder, numbering, module, symbol->defining_op,
          ir_offsets[symbol_index]));
    }
  }

  return iree_ok_status();
}

// Writes the STRINGS section through the page writer.
static iree_status_t loom_bytecode_write_strings_section(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering) {
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, numbering->string_count));
  for (iree_host_size_t i = 0; i < numbering->string_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_string(
        page_writer, numbering->string_entries[i]));
  }
  return iree_ok_status();
}

// Writes the SOURCES section through the page writer.
static iree_status_t loom_bytecode_write_sources_section(
    loom_bytecode_page_writer_t* page_writer, const loom_module_t* module) {
  iree_host_size_t source_count = module->sources.count;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(page_writer, source_count));
  for (iree_host_size_t i = 0; i < source_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_string(
        page_writer, module->sources.entries[i]));
  }
  return iree_ok_status();
}

// Writes the TYPES section through the page writer.
static iree_status_t loom_bytecode_write_types_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering) {
  const loom_module_t* module = numbering->module;
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, numbering->type_count));

  for (uint32_t writer_type_id = 0; writer_type_id < numbering->type_count;
       ++writer_type_id) {
    iree_host_size_t module_index = numbering->type_order[writer_type_id];
    loom_type_t type = module->types.entries[module_index];
    loom_type_kind_t kind = loom_type_kind(type);

    uint8_t kind_byte = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_type_kind_byte(kind, &kind_byte));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u8(page_writer, kind_byte));

    switch (kind) {
      case LOOM_TYPE_NONE:
        break;
      case LOOM_TYPE_SCALAR: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
            page_writer, (uint8_t)loom_type_element_type(type)));
        break;
      }
      case LOOM_TYPE_TILE:
      case LOOM_TYPE_TENSOR:
      case LOOM_TYPE_VECTOR:
      case LOOM_TYPE_VIEW: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
            page_writer, (uint8_t)loom_type_element_type(type)));
        uint8_t rank = loom_type_rank(type);
        if (kind == LOOM_TYPE_VECTOR) {
          if (rank == 0) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "vector types must have rank >= 1");
          }
          if (type.encoding_id != 0 || type.encoding_flags != 0) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "vector types must not carry encoding or layout attachments");
          }
        }
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u8(page_writer, rank));
        // Encoding.
        if (loom_type_has_ssa_encoding(type)) {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
              page_writer, LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA));
          IREE_RETURN_IF_ERROR(
              loom_bytecode_page_writer_write_uvarint(page_writer, 0));
        } else if (loom_type_has_static_encoding(type)) {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
              page_writer, LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, type.encoding_id));
        } else {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
              page_writer, LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE));
          IREE_RETURN_IF_ERROR(
              loom_bytecode_page_writer_write_uvarint(page_writer, 0));
        }
        // Dims.
        for (uint8_t i = 0; i < rank; ++i) {
          if (loom_type_dim_is_dynamic_at(type, i)) {
            IREE_RETURN_IF_ERROR(
                loom_bytecode_page_writer_write_u8(page_writer, 1));
          } else {
            IREE_RETURN_IF_ERROR(
                loom_bytecode_page_writer_write_u8(page_writer, 0));
            IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
                page_writer, (uint64_t)loom_type_dim_static_size_at(type, i)));
          }
        }
        break;
      }
      case LOOM_TYPE_GROUP: {
        loom_group_scope_t scope = loom_type_group_scope(type);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u8(page_writer, (uint8_t)scope));
        break;
      }
      case LOOM_TYPE_FUNCTION: {
        const loom_func_type_data_t* func_data = loom_type_func_data(type);
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, func_data->arg_count));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, func_data->result_count));
        iree_host_size_t type_count =
            (iree_host_size_t)func_data->arg_count + func_data->result_count;
        for (iree_host_size_t i = 0; i < type_count; ++i) {
          uint32_t sub_type_id = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
              numbering, func_data->types[i], &sub_type_id));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, sub_type_id));
        }
        break;
      }
      case LOOM_TYPE_DIALECT: {
        loom_string_id_t name_id = loom_type_dialect_name_id(type);
        uint32_t name_writer_id = 0;
        if (name_id < numbering->module->strings.count) {
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
              numbering, numbering->module->strings.entries[name_id],
              &name_writer_id));
        }
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, name_writer_id));
        uint16_t param_count = loom_type_dialect_param_count(type);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_uvarint(page_writer, param_count));
        const loom_type_t* params = loom_type_dialect_params(type);
        for (uint16_t i = 0; i < param_count; ++i) {
          uint32_t param_type_id = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
              numbering, params[i], &param_type_id));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, param_type_id));
        }
        break;
      }
      case LOOM_TYPE_REGISTER: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, loom_type_register_payload0(type)));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, loom_type_register_payload1(type)));
        break;
      }
      case LOOM_TYPE_STORAGE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
            page_writer, (uint8_t)loom_type_storage_space(type)));
        break;
      }
      case LOOM_TYPE_ENCODING: {
        uint8_t role_byte = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_encoding_role_byte(
            loom_type_encoding_role(type), &role_byte));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u8(page_writer, role_byte));
        break;
      }
      case LOOM_TYPE_BUFFER:
        // No additional data.
        break;
      case LOOM_TYPE_POOL: {
        uint64_t dim = loom_type_dim(type, 0);
        if (loom_dim_is_dynamic(dim)) {
          IREE_RETURN_IF_ERROR(
              loom_bytecode_page_writer_write_u8(page_writer, 1));
        } else {
          IREE_RETURN_IF_ERROR(
              loom_bytecode_page_writer_write_u8(page_writer, 0));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, (uint64_t)loom_dim_static_size(dim)));
        }
        break;
      }
      default:
        // Unreachable: loom_bytecode_type_kind_byte above rejects
        // unknown kinds before we get here.
        break;
    }
  }

  return iree_ok_status();
}

// Writes the ENCODINGS section: encoding family registry + instances.
static iree_status_t loom_bytecode_write_encodings_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering) {
  const loom_module_t* module = numbering->module;

  // Build the encoding family registry from unique encoding names.
  // Small (typically <10 families), so linear dedup is fine.
  typedef struct {
    loom_string_id_t name_id;
    uint32_t writer_string_id;
  } encoding_family_entry_t;
  encoding_family_entry_t family_entries[256];
  iree_host_size_t family_count = 0;

  for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
    loom_string_id_t name_id = module->encodings.entries[i].name_id;
    bool found = false;
    for (iree_host_size_t k = 0; k < family_count; ++k) {
      if (family_entries[k].name_id == name_id) {
        found = true;
        break;
      }
    }
    if (!found) {
      if (family_count >= 256) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "more than 256 unique encoding families");
      }
      uint32_t writer_string_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, name_id, &writer_string_id));
      family_entries[family_count++] = (encoding_family_entry_t){
          .name_id = name_id, .writer_string_id = writer_string_id};
    }
  }

  // Encoding family count and family name string IDs.
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(page_writer, family_count));
  for (iree_host_size_t k = 0; k < family_count; ++k) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, family_entries[k].writer_string_id));
  }

  // Encoding instances.
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, module->encodings.count));
  for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
    const loom_encoding_t* encoding = &module->encodings.entries[i];

    // Find the family index for this encoding's name.
    uint32_t family_index = 0;
    for (iree_host_size_t k = 0; k < family_count; ++k) {
      if (family_entries[k].name_id == encoding->name_id) {
        family_index = (uint32_t)k;
        break;
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(page_writer, family_index));

    // Alias string ID plus one (0 = no alias).
    uint32_t alias_string_id_plus1 = 0;
    if (encoding->alias_id != LOOM_STRING_ID_INVALID) {
      uint32_t alias_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, encoding->alias_id, &alias_writer_id));
      alias_string_id_plus1 = alias_writer_id + 1;
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, alias_string_id_plus1));

    // Parameters as structured named attributes, using the same
    // attribute serialization as the IR section.
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, encoding->attribute_count));
    for (uint8_t p = 0; p < encoding->attribute_count; ++p) {
      const loom_named_attr_t* attr = &encoding->attributes[p];
      uint32_t param_name_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, attr->name_id, &param_name_writer_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, param_name_writer_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value(
          page_writer, numbering, NULL, attr->value, NULL));
    }
  }

  return iree_ok_status();
}

// Writes the OPS section through the page writer.
static iree_status_t loom_bytecode_write_ops_section(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering) {
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, numbering->op_count));
  for (uint32_t i = 0; i < numbering->op_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, numbering->op_entries[i].string_writer_id));
  }
  return iree_ok_status();
}

// Writes the LOCATIONS section: location_count + per-entry serialization.
static iree_status_t loom_bytecode_write_locations_section(
    loom_bytecode_page_writer_t* page_writer, const loom_module_t* module) {
  iree_host_size_t location_count = module->locations.count;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(page_writer, location_count));
  for (iree_host_size_t i = 0; i < location_count; ++i) {
    const loom_location_entry_t* entry = &module->locations.entries[i];
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u8(page_writer, (uint8_t)entry->kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u8(page_writer, entry->flags));
    switch (entry->kind) {
      case LOOM_LOCATION_NONE:
        break;
      case LOOM_LOCATION_FILE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.source_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.start_line));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.start_col));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.end_line));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.end_col));
        break;
      }
      case LOOM_LOCATION_FUSED: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->fused.count));
        for (uint32_t c = 0; c < entry->fused.count; ++c) {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, entry->fused.children[c]));
        }
        break;
      }
      case LOOM_LOCATION_OPAQUE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->opaque.source_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->opaque.data_length));
        if (entry->opaque.data_length > 0) {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(
              page_writer, entry->opaque.data, entry->opaque.data_length));
        }
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown location kind %d", (int)entry->kind);
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Top-level writer
//===----------------------------------------------------------------------===//

static iree_status_t loom_bytecode_validate_module(
    const loom_module_t* module) {
  if (!module->context) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module has no context (needed for op vtables)");
  }
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    loom_symbol_kind_t bytecode_kind = loom_symbol_bytecode_kind(symbol);
    if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_GLOBAL) ||
        bytecode_kind == LOOM_SYMBOL_GLOBAL) {
      if (!symbol->defining_op) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "GLOBAL symbol %" PRIhsz " has no defining op",
                                i);
      }
      const loom_op_t* op = symbol->defining_op;
      const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
      if (!vtable ||
          !iree_all_bits_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE) ||
          !vtable->symbol_def ||
          vtable->symbol_def->bytecode_kind != LOOM_SYMBOL_GLOBAL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "GLOBAL symbol %" PRIhsz
                                " defining op does not use the GLOBAL "
                                "bytecode payload",
                                i);
      }
      if (op->operand_count != 0 || op->region_count != 0 ||
          op->tied_result_count != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "GLOBAL symbol %" PRIhsz
            " defining op must not have operands, regions, or tied results",
            i);
      }
      if (op->result_count == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "GLOBAL symbol %" PRIhsz " defining op must have results", i);
      }
      if (op->attribute_count > 0 && !vtable->attr_descriptors) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "GLOBAL symbol %" PRIhsz
            " defining op has attributes but no descriptors",
            i);
      }
    }
    if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_EXECUTABLE) ||
        bytecode_kind == LOOM_SYMBOL_EXECUTABLE) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "EXECUTABLE symbols not yet supported");
    }
    if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_RECORD) ||
        bytecode_kind == LOOM_SYMBOL_RECORD) {
      if (!symbol->defining_op) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "RECORD symbol %" PRIhsz " has no defining op",
                                i);
      }
      IREE_RETURN_IF_ERROR(
          loom_bytecode_validate_record_symbol_op(module, symbol->defining_op));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_write_module(
    const loom_module_t* module, iree_io_stream_t* stream,
    const loom_bytecode_write_options_t* options,
    iree_arena_block_pool_t* block_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, loom_bytecode_validate_module(module));

  // Check stream capabilities.
  iree_io_stream_mode_t mode = iree_io_stream_mode(stream);
  if (!(mode & IREE_IO_STREAM_MODE_WRITABLE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "stream is not writable");
  }
  if (!(mode & IREE_IO_STREAM_MODE_SEEKABLE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "stream is not seekable (needed for directory "
                            "patching)");
  }

  iree_string_view_t producer = (options && options->producer.size > 0)
                                    ? options->producer
                                    : IREE_SV(LOOM_BYTECODE_DEFAULT_PRODUCER);
  loom_bytecode_location_mode_t location_mode =
      options ? options->location_mode
              : LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS;
  if (location_mode > LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported bytecode location mode %u",
                            (unsigned)location_mode);
  }
  if (location_mode == LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "FULL_LOCATIONS bytecode mode requires field span emission");
  }

  // Temporary arena for all working memory. All numbering tables,
  // value maps, and scratch allocations come from here. Deinitialized
  // at the end, returning all blocks to the shared pool in O(1).
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);

  // Initialize the page writer.
  loom_bytecode_page_writer_t page_writer;
  loom_bytecode_page_writer_initialize(&page_writer, stream);

  // Initialize numbering context.
  loom_bytecode_numbering_t numbering;
  iree_status_t status =
      loom_bytecode_numbering_initialize(&numbering, module, &arena);
  numbering.location_mode = location_mode;

  // Pass 1: Number module metadata. Function signatures and bodies are numbered
  // during IR section writing.
  if (iree_status_is_ok(status)) {
    uint32_t unused_id = 0;
    status = loom_bytecode_numbering_intern_module_string(
        &numbering, module->name_id, &unused_id);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0;
         i < module->symbols.count && iree_status_is_ok(status); ++i) {
      uint32_t unused_id = 0;
      status = loom_bytecode_numbering_intern_module_string(
          &numbering, module->symbols.entries[i].name_id, &unused_id);
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0;
         i < module->encodings.count && iree_status_is_ok(status); ++i) {
      status = loom_bytecode_number_encoding(&numbering, (uint16_t)(i + 1));
    }
  }

  // File header: magic, version, location mode, module count, producer string.
  iree_string_view_t module_name = module->strings.entries[module->name_id];

  if (iree_status_is_ok(status)) {
    // Magic.
    status = loom_bytecode_page_writer_write(&page_writer, LOOM_BYTECODE_MAGIC,
                                             LOOM_BYTECODE_MAGIC_LENGTH);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u8(&page_writer,
                                                LOOM_BYTECODE_FORMAT_VERSION);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u8(&page_writer, location_mode);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u16_le(&page_writer,
                                                    1);  // module_count
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u32_le(
        &page_writer, (uint32_t)module_name.size);  // string_pool_length
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_page_writer_write_u32_le(&page_writer, 0);  // reserved
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_null_terminated_string(
        &page_writer, producer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_pad_to_alignment(&page_writer, 8);
  }

  // Module directory entry. module_offset and module_length are
  // written as placeholders and patched after all sections are written.
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_page_writer_write_u32_le(&page_writer, 0);  // name_offset
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u16_le(&page_writer,
                                                    (uint16_t)module_name.size);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u16_le(&page_writer,
                                                    0);  // module_flags
  }
  // module_offset placeholder: we'll patch this as part of the directory entry.
  iree_host_size_t module_dir_offset_position = 0;
  if (iree_status_is_ok(status)) {
    module_dir_offset_position = page_writer.total_written;
    status = loom_bytecode_page_writer_write_u64_le(&page_writer,
                                                    0);  // module_offset
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u64_le(&page_writer,
                                                    0);  // module_length
  }

  // File string pool: module name(s).
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write(&page_writer, module_name.data,
                                             module_name.size);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_pad_to_alignment(&page_writer, 8);
  }

  // Module data starts at this offset.
  iree_host_size_t module_start = page_writer.total_written;

  static const loom_bytecode_section_kind_t section_write_order[] = {
      LOOM_BYTECODE_SECTION_IR,      LOOM_BYTECODE_SECTION_SYMBOLS,
      LOOM_BYTECODE_SECTION_STRINGS, LOOM_BYTECODE_SECTION_SOURCES,
      LOOM_BYTECODE_SECTION_TYPES,   LOOM_BYTECODE_SECTION_ENCODINGS,
      LOOM_BYTECODE_SECTION_OPS,     LOOM_BYTECODE_SECTION_LOCATIONS,
  };
  uint32_t section_count = IREE_ARRAYSIZE(section_write_order);
  if (location_mode == LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS) {
    --section_count;
  }
  loom_bytecode_body_counts_t module_counts = {0};
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_count_serialized_bodies(&numbering, &module_counts);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_page_writer_write_uvarint(&page_writer, section_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_uvarint(&page_writer,
                                                     module_counts.value_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_uvarint(
        &page_writer, module_counts.region_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_uvarint(&page_writer,
                                                     module_counts.block_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_uvarint(&page_writer,
                                                     module_counts.op_count);
  }

  // Section directory placeholder — patched after all sections are written.
  iree_host_size_t section_dir_patch_position = page_writer.total_written;
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_zeros(
        &page_writer,
        section_count * sizeof(loom_bytecode_section_dir_entry_t));
  }

  // Track section offsets and lengths.
  uint64_t section_offsets[LOOM_BYTECODE_SECTION_COUNT] = {0};
  uint64_t section_lengths[LOOM_BYTECODE_SECTION_COUNT] = {0};

  // Allocate IR offset tracking from the arena.
  loom_bytecode_ir_offset_t* ir_offsets = NULL;
  if (iree_status_is_ok(status) && module->symbols.count > 0) {
    status = iree_arena_allocate_array(&arena, module->symbols.count,
                                       sizeof(loom_bytecode_ir_offset_t),
                                       (void**)&ir_offsets);
    if (iree_status_is_ok(status)) {
      memset(ir_offsets, 0,
             module->symbols.count * sizeof(loom_bytecode_ir_offset_t));
    }
  }

  // IR section: function bodies streamed through the page writer.
  // Written first so the numbering tables grow as entities are encountered.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_IR] =
        page_writer.total_written - module_start;
    status =
        loom_bytecode_write_ir_section(&page_writer, &numbering, ir_offsets);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_IR] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_IR];
    }
  }

  // Symbols section: buffered in a string builder because the import/export
  // offset tables at the start reference entry positions that come later.
  // The SYMBOLS section uses a string_builder (which needs realloc, so it
  // can't use the arena). Use the module's context allocator.
  iree_string_builder_t symbols_builder;
  iree_string_builder_initialize(module->context->allocator, &symbols_builder);
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_write_symbols_section(&symbols_builder, &numbering,
                                                 ir_offsets);
  }
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_SYMBOLS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_page_writer_write(
        &page_writer, iree_string_builder_buffer(&symbols_builder),
        iree_string_builder_size(&symbols_builder));
    section_lengths[LOOM_BYTECODE_SECTION_SYMBOLS] =
        iree_string_builder_size(&symbols_builder);
  }
  iree_string_builder_deinitialize(&symbols_builder);

  // Strings section: all interned strings from the numbering context.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_STRINGS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_strings_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_STRINGS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_STRINGS];
    }
  }

  // Sources section: module-local source identifiers (filenames, tags).
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_SOURCES] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_sources_section(&page_writer, module);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_SOURCES] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_SOURCES];
    }
  }

  // Types section: interned type table in topological order.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_TYPES] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_types_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_TYPES] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_TYPES];
    }
  }

  // Encodings section: kind registry + parameterized instances.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_ENCODINGS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_encodings_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_ENCODINGS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_ENCODINGS];
    }
  }

  // Ops section: op kind name registry.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_OPS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_ops_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_OPS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_OPS];
    }
  }

  // Locations section.
  if (iree_status_is_ok(status) &&
      location_mode != LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS) {
    section_offsets[LOOM_BYTECODE_SECTION_LOCATIONS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_locations_section(&page_writer, module);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_LOCATIONS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_LOCATIONS];
    }
  }

  // Flush remaining page buffer, then seek back to patch the section
  // directory and module directory with correct offsets and lengths.
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_flush(&page_writer);
  }

  // Patch section directory.
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET,
                            (iree_io_stream_pos_t)section_dir_patch_position);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(section_write_order) && iree_status_is_ok(status);
         ++i) {
      loom_bytecode_section_kind_t kind = section_write_order[i];
      if (kind == LOOM_BYTECODE_SECTION_LOCATIONS &&
          location_mode == LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS) {
        continue;
      }
      uint8_t entry[sizeof(loom_bytecode_section_dir_entry_t)] = {0};
      entry[0] = (uint8_t)kind;
      entry[1] = (uint8_t)((uint16_t)kind >> 8);
      uint64_t section_offset = section_offsets[kind];
      uint64_t section_length = section_lengths[kind];
      for (int byte_index = 0; byte_index < 8; ++byte_index) {
        entry[8 + byte_index] = (uint8_t)(section_offset >> (byte_index * 8));
        entry[16 + byte_index] = (uint8_t)(section_length >> (byte_index * 8));
      }
      status = iree_io_stream_write(stream, sizeof(entry), entry);
    }
  }

  // Patch module directory: module_offset and module_length.
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET,
                            (iree_io_stream_pos_t)module_dir_offset_position);
  }
  if (iree_status_is_ok(status)) {
    uint64_t module_offset = module_start;
    status = iree_io_stream_write(stream, 8, &module_offset);
  }
  if (iree_status_is_ok(status)) {
    uint64_t module_length = page_writer.total_written - module_start;
    status = iree_io_stream_write(stream, 8, &module_length);
  }

  // All numbering tables, value maps, and ir_offsets were arena-allocated.
  // One call returns all blocks to the shared pool.
  iree_arena_deinitialize(&arena);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
