// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/tooling/route_trace_file.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/alignment.h"

#define QWEN_ROUTE_TRACE_FILE_HEADER_LENGTH 96u

static const uint8_t qwen_route_trace_file_magic[8] = {
    'Q', 'W', 'E', 'N', 'R', 'T', '0', '1',
};

typedef struct qwen_route_trace_file_layout_t {
  uint64_t element_count;
  uint64_t plane_byte_length;
  uint64_t ids_offset;
  uint64_t weights_offset;
  uint64_t file_byte_length;
} qwen_route_trace_file_layout_t;

static iree_status_t qwen_route_trace_file_calculate_layout(
    const qwen_route_trace_file_metadata_t* metadata,
    qwen_route_trace_file_layout_t* out_layout) {
  if (metadata->model != QWEN_ROUTE_TRACE_MODEL_QWEN3_30B_A3B) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported Qwen route-trace model %u",
                            (uint32_t)metadata->model);
  }
  if (metadata->context_capacity == 0 ||
      metadata->context_capacity > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "route-trace context capacity must be in [1, %u]",
                            UINT32_MAX);
  }
  if (metadata->prompt_token_count == 0 ||
      metadata->generated_token_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "route-trace prompt and generated token counts must be nonzero");
  }
  if (metadata->prompt_token_count > UINT32_MAX ||
      metadata->generated_token_count > UINT32_MAX ||
      metadata->parameter_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "route-trace metadata exceeds 32-bit file fields");
  }
  if (metadata->generated_token_count - 1 >
      SIZE_MAX - metadata->prompt_token_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "route-trace captured token count overflows");
  }
  const iree_host_size_t expected_captured_token_count =
      metadata->prompt_token_count + metadata->generated_token_count - 1;
  if (metadata->captured_token_count != expected_captured_token_count ||
      metadata->captured_token_count > metadata->context_capacity) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "route trace captures %" PRIhsz " rows; expected %" PRIhsz
        " within context capacity %" PRIhsz,
        metadata->captured_token_count, expected_captured_token_count,
        metadata->context_capacity);
  }

  uint64_t element_count = (uint64_t)metadata->context_capacity;
  if (element_count > UINT64_MAX / QWEN_MODEL_LAYER_COUNT ||
      element_count * QWEN_MODEL_LAYER_COUNT >
          UINT64_MAX / QWEN_MODEL_ROUTE_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "route-trace element count overflows");
  }
  element_count *= QWEN_MODEL_LAYER_COUNT * QWEN_MODEL_ROUTE_COUNT;
  if (element_count > UINT64_MAX / sizeof(uint32_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "route-trace plane byte length overflows");
  }
  const uint64_t plane_byte_length = element_count * sizeof(uint32_t);
  const uint64_t ids_offset = QWEN_ROUTE_TRACE_FILE_HEADER_LENGTH;
  const uint64_t weights_offset = ids_offset + plane_byte_length;
  if (weights_offset < ids_offset ||
      plane_byte_length > UINT64_MAX - weights_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "route-trace file byte length overflows");
  }
  const uint64_t file_byte_length = weights_offset + plane_byte_length;
  if (file_byte_length > SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "route-trace file does not fit host address space");
  }
  *out_layout = (qwen_route_trace_file_layout_t){
      .element_count = element_count,
      .plane_byte_length = plane_byte_length,
      .ids_offset = ids_offset,
      .weights_offset = weights_offset,
      .file_byte_length = file_byte_length,
  };
  return iree_ok_status();
}

static uint64_t qwen_route_trace_hash_bytes(uint64_t hash, const void* data,
                                            iree_host_size_t data_length) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < data_length; ++i) {
    hash ^= bytes[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static uint64_t qwen_route_trace_hash_u64(uint64_t hash, uint64_t value) {
  uint8_t bytes[sizeof(value)];
  iree_unaligned_store_le_u64(bytes, value);
  return qwen_route_trace_hash_bytes(hash, bytes, sizeof(bytes));
}

iree_status_t qwen_route_trace_parameter_layout_fingerprint(
    iree_io_parameter_index_t* parameter_index, uint64_t* out_fingerprint) {
  IREE_ASSERT_ARGUMENT(parameter_index);
  IREE_ASSERT_ARGUMENT(out_fingerprint);
  *out_fingerprint = 0;

  const iree_host_size_t parameter_count =
      iree_io_parameter_index_count(parameter_index);
  uint64_t hash = UINT64_C(14695981039346656037);
  hash = qwen_route_trace_hash_u64(hash, parameter_count);
  for (iree_host_size_t i = 0; i < parameter_count; ++i) {
    const iree_io_parameter_index_entry_t* entry = NULL;
    IREE_RETURN_IF_ERROR(
        iree_io_parameter_index_get(parameter_index, i, &entry));
    hash = qwen_route_trace_hash_u64(hash, entry->key.size);
    hash = qwen_route_trace_hash_bytes(hash, entry->key.data, entry->key.size);
    hash = qwen_route_trace_hash_u64(hash, entry->metadata.data_length);
    hash = qwen_route_trace_hash_bytes(hash, entry->metadata.data,
                                       entry->metadata.data_length);
    hash = qwen_route_trace_hash_u64(hash, entry->length);
    hash = qwen_route_trace_hash_u64(hash, entry->type);
    if (entry->type == IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE) {
      hash = qwen_route_trace_hash_u64(hash, entry->storage.file.offset);
    } else if (entry->type ==
               IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT) {
      hash =
          qwen_route_trace_hash_u64(hash, entry->storage.splat.pattern_length);
      hash = qwen_route_trace_hash_bytes(hash, entry->storage.splat.pattern,
                                         entry->storage.splat.pattern_length);
    }
  }
  *out_fingerprint = hash;
  return iree_ok_status();
}

iree_status_t qwen_route_trace_file_build(
    const qwen_route_trace_file_metadata_t* metadata,
    iree_const_byte_span_t route_payload, iree_allocator_t host_allocator,
    iree_byte_span_t* out_file_data) {
  IREE_ASSERT_ARGUMENT(metadata);
  IREE_ASSERT_ARGUMENT(out_file_data);
  *out_file_data = iree_byte_span_empty();

  qwen_route_trace_file_layout_t layout;
  IREE_RETURN_IF_ERROR(
      qwen_route_trace_file_calculate_layout(metadata, &layout));
  const uint64_t expected_payload_length =
      layout.plane_byte_length + layout.plane_byte_length;
  if (route_payload.data_length != expected_payload_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "route-trace payload has %" PRIhsz
                            "; expected %" PRIu64 " bytes",
                            route_payload.data_length, expected_payload_length);
  }

  uint8_t* file_data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, (iree_host_size_t)layout.file_byte_length,
      (void**)&file_data));
  memcpy(file_data, qwen_route_trace_file_magic,
         sizeof(qwen_route_trace_file_magic));
  iree_unaligned_store_le_u32(file_data + 8,
                              QWEN_ROUTE_TRACE_FILE_HEADER_LENGTH);
  iree_unaligned_store_le_u32(file_data + 12, QWEN_ROUTE_TRACE_FILE_VERSION);
  iree_unaligned_store_le_u32(file_data + 16, metadata->model);
  iree_unaligned_store_le_u32(file_data + 20, QWEN_MODEL_LAYER_COUNT);
  iree_unaligned_store_le_u32(file_data + 24, QWEN_MODEL_EXPERT_COUNT);
  iree_unaligned_store_le_u32(file_data + 28, QWEN_MODEL_ROUTE_COUNT);
  iree_unaligned_store_le_u32(file_data + 32,
                              (uint32_t)metadata->context_capacity);
  iree_unaligned_store_le_u32(file_data + 36,
                              (uint32_t)metadata->captured_token_count);
  iree_unaligned_store_le_u32(file_data + 40,
                              (uint32_t)metadata->prompt_token_count);
  iree_unaligned_store_le_u32(file_data + 44,
                              (uint32_t)metadata->generated_token_count);
  iree_unaligned_store_le_u32(file_data + 48,
                              (uint32_t)metadata->parameter_count);
  iree_unaligned_store_le_u32(file_data + 52, 0);
  iree_unaligned_store_le_u64(file_data + 56,
                              metadata->parameter_layout_fingerprint);
  iree_unaligned_store_le_u64(file_data + 64,
                              metadata->encoded_parameter_bytes);
  iree_unaligned_store_le_u64(file_data + 72, layout.element_count);
  iree_unaligned_store_le_u64(file_data + 80, layout.ids_offset);
  iree_unaligned_store_le_u64(file_data + 88, layout.weights_offset);
  memcpy(file_data + layout.ids_offset, route_payload.data,
         route_payload.data_length);

  *out_file_data =
      iree_make_byte_span(file_data, (iree_host_size_t)layout.file_byte_length);
  return iree_ok_status();
}

iree_status_t qwen_route_trace_file_parse(
    iree_const_byte_span_t file_data, qwen_route_trace_file_view_t* out_view) {
  IREE_ASSERT_ARGUMENT(out_view);
  memset(out_view, 0, sizeof(*out_view));
  if (file_data.data_length < QWEN_ROUTE_TRACE_FILE_HEADER_LENGTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen route-trace file has %" PRIhsz "; expected at least %u bytes",
        file_data.data_length, QWEN_ROUTE_TRACE_FILE_HEADER_LENGTH);
  }
  if (memcmp(file_data.data, qwen_route_trace_file_magic,
             sizeof(qwen_route_trace_file_magic)) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen route-trace file magic is invalid");
  }
  const uint32_t header_length = iree_unaligned_load_le_u32(file_data.data + 8);
  const uint32_t version = iree_unaligned_load_le_u32(file_data.data + 12);
  if (header_length != QWEN_ROUTE_TRACE_FILE_HEADER_LENGTH ||
      version != QWEN_ROUTE_TRACE_FILE_VERSION) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Qwen route-trace header version %u length %u is unsupported", version,
        header_length);
  }
  if (iree_unaligned_load_le_u32(file_data.data + 20) !=
          QWEN_MODEL_LAYER_COUNT ||
      iree_unaligned_load_le_u32(file_data.data + 24) !=
          QWEN_MODEL_EXPERT_COUNT ||
      iree_unaligned_load_le_u32(file_data.data + 28) !=
          QWEN_MODEL_ROUTE_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen route-trace model geometry is invalid");
  }
  if (iree_unaligned_load_le_u32(file_data.data + 52) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen route-trace reserved header bits are set");
  }

  qwen_route_trace_file_metadata_t metadata = {
      .model = (qwen_route_trace_model_t)iree_unaligned_load_le_u32(
          file_data.data + 16),
      .context_capacity = iree_unaligned_load_le_u32(file_data.data + 32),
      .captured_token_count = iree_unaligned_load_le_u32(file_data.data + 36),
      .prompt_token_count = iree_unaligned_load_le_u32(file_data.data + 40),
      .generated_token_count = iree_unaligned_load_le_u32(file_data.data + 44),
      .parameter_count = iree_unaligned_load_le_u32(file_data.data + 48),
      .parameter_layout_fingerprint =
          iree_unaligned_load_le_u64(file_data.data + 56),
      .encoded_parameter_bytes =
          iree_unaligned_load_le_u64(file_data.data + 64),
  };
  qwen_route_trace_file_layout_t layout;
  IREE_RETURN_IF_ERROR(
      qwen_route_trace_file_calculate_layout(&metadata, &layout));
  if (iree_unaligned_load_le_u64(file_data.data + 72) != layout.element_count ||
      iree_unaligned_load_le_u64(file_data.data + 80) != layout.ids_offset ||
      iree_unaligned_load_le_u64(file_data.data + 88) !=
          layout.weights_offset ||
      file_data.data_length != layout.file_byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen route-trace payload layout is invalid");
  }

  *out_view = (qwen_route_trace_file_view_t){
      .metadata = metadata,
      .route_ids =
          iree_make_const_byte_span(file_data.data + layout.ids_offset,
                                    (iree_host_size_t)layout.plane_byte_length),
      .route_weights =
          iree_make_const_byte_span(file_data.data + layout.weights_offset,
                                    (iree_host_size_t)layout.plane_byte_length),
  };
  return iree_ok_status();
}
