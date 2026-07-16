// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_layout.h"

#include <string.h>

#include "experimental/id4/pipeline/program_region.h"
#include "iree/schemas/parameter_archive.h"

enum {
  ID4_PIPELINE_PARAMETER_LAYOUT_METADATA_VERSION = 0,
  ID4_PIPELINE_PARAMETER_LAYOUT_METADATA_LENGTH =
      6 * sizeof(uint32_t) +
      (ID4_PIPELINE_TENSOR_MAX_RANK + 3) * sizeof(uint64_t),
};

static uint64_t id4_pipeline_parameter_layout_hash_bytes(
    uint64_t hash, const void* data, iree_host_size_t data_length) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < data_length; ++i) {
    hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
  }
  return hash;
}

static uint64_t id4_pipeline_parameter_layout_hash_u64(uint64_t hash,
                                                       uint64_t value) {
  uint8_t bytes[sizeof(value)];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(bytes); ++i) {
    bytes[i] = (uint8_t)(value >> (i * 8));
  }
  return id4_pipeline_parameter_layout_hash_bytes(hash, bytes, sizeof(bytes));
}

static uint64_t id4_pipeline_parameter_layout_hash_string(
    uint64_t hash, iree_string_view_t value) {
  hash = id4_pipeline_parameter_layout_hash_u64(hash, value.size);
  return id4_pipeline_parameter_layout_hash_bytes(hash, value.data, value.size);
}

static uint64_t id4_pipeline_parameter_layout_source_schema_fingerprint(
    const id4_pipeline_program_parameter_op_t* parameter_op) {
  uint64_t hash = UINT64_C(14695981039346656037);
  hash =
      id4_pipeline_parameter_layout_hash_u64(hash, parameter_op->source_count);
  for (iree_host_size_t i = 0; i < parameter_op->source_count; ++i) {
    const id4_pipeline_program_parameter_source_t* source =
        &parameter_op->sources[i];
    hash =
        id4_pipeline_parameter_layout_hash_string(hash, source->source_scope);
    hash = id4_pipeline_parameter_layout_hash_string(hash, source->key);
    hash = id4_pipeline_parameter_layout_hash_u64(hash, source->dtype);
    hash = id4_pipeline_parameter_layout_hash_u64(hash, source->shape.rank);
    for (uint32_t j = 0; j < source->shape.rank; ++j) {
      hash =
          id4_pipeline_parameter_layout_hash_u64(hash, source->shape.dims[j]);
    }
  }
  return hash;
}

static iree_status_t id4_pipeline_parameter_layout_validate_plan(
    const id4_pipeline_plan_t* plan) {
  if (!plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter layout plan is required");
  }
  if (!id4_pipeline_plan_source_program(plan)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "baked parameter layouts require a program-backed plan");
  }
  if (id4_pipeline_plan_parameter_slab_count(plan) == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "baked parameter layouts require parameter slabs");
  }
  if (id4_pipeline_plan_parameter_tensor_count(plan) == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "baked parameter layouts require planned parameter tensors");
  }
  return iree_ok_status();
}

static id4_pipeline_tensor_shape_t id4_pipeline_parameter_layout_shape(
    id4_pipeline_program_shape_t source) {
  id4_pipeline_tensor_shape_t target;
  memset(&target, 0, sizeof(target));
  target.rank = source.rank;
  memcpy(target.dims, source.dims, sizeof(target.dims));
  return target;
}

static iree_status_t id4_pipeline_parameter_layout_parameter_op(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_tensor_plan_t* parameter_tensor,
    const id4_pipeline_program_parameter_op_t** out_parameter_op) {
  *out_parameter_op = NULL;
  const id4_pipeline_program_t* program =
      id4_pipeline_plan_source_program(plan);
  const id4_pipeline_program_tensor_record_t* tensor =
      id4_pipeline_program_tensor_at(program,
                                     parameter_tensor->program_tensor_ordinal);
  if (!tensor) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter tensor %.*s program tensor %u is missing",
        (int)parameter_tensor->layout.name.size,
        parameter_tensor->layout.name.data,
        parameter_tensor->program_tensor_ordinal);
  }
  const id4_pipeline_program_op_t* producer = id4_pipeline_program_operation_at(
      program, tensor->producer_operation_ordinal);
  if (!producer || producer->kind != ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER ||
      producer->payload.parameter.tensor.ordinal !=
          parameter_tensor->program_tensor_ordinal) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter tensor %.*s is not produced by a parameter operation",
        (int)parameter_tensor->layout.name.size,
        parameter_tensor->layout.name.data);
  }
  *out_parameter_op = &producer->payload.parameter;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_layout_candidate(
    const id4_pipeline_plan_t* plan, iree_host_size_t parameter_tensor_index,
    id4_pipeline_parameter_layout_entry_t* out_entry) {
  memset(out_entry, 0, sizeof(*out_entry));
  out_entry->parameter_slab_index = IREE_HOST_SIZE_MAX;
  const id4_pipeline_parameter_tensor_plan_t* parameter_tensor =
      id4_pipeline_plan_parameter_tensor_at(plan, parameter_tensor_index);
  if (!parameter_tensor) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter tensor %" PRIhsz " is missing",
                            parameter_tensor_index);
  }
  const id4_pipeline_program_parameter_op_t* parameter_op = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_parameter_op(
      plan, parameter_tensor, &parameter_op));

  const bool retains_source_layout =
      parameter_op->encoding ==
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT &&
      parameter_op->source_span_count != 0;
  out_entry->source_schema_fingerprint =
      id4_pipeline_parameter_layout_source_schema_fingerprint(parameter_op);
  if (retains_source_layout) {
    if (parameter_op->source_count != 1) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "sliced direct parameter %.*s must have exactly one source",
          (int)parameter_tensor->layout.name.size,
          parameter_tensor->layout.name.data);
    }
    const id4_pipeline_program_parameter_source_t* source =
        &parameter_op->sources[0];
    iree_device_size_t source_byte_length = 0;
    IREE_RETURN_IF_ERROR(id4_pipeline_program_tensor_byte_length(
        source->dtype, source->shape, &source_byte_length));
    out_entry->kind = ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_SOURCE;
    out_entry->key = source->key;
    out_entry->encoding = ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT;
    out_entry->dtype = id4_pipeline_program_region_convert_dtype(source->dtype);
    out_entry->shape = id4_pipeline_parameter_layout_shape(source->shape);
    out_entry->byte_length = source_byte_length;
    out_entry->alignment = 0;
    out_entry->source_scope = source->source_scope;
    return iree_ok_status();
  }

  out_entry->kind = ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_EXECUTION;
  out_entry->key = parameter_tensor->layout.name;
  out_entry->encoding = parameter_op->encoding;
  if (parameter_op->encoding ==
      ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT) {
    const id4_pipeline_program_parameter_source_t* source =
        &parameter_op->sources[0];
    out_entry->dtype = id4_pipeline_program_region_convert_dtype(source->dtype);
    out_entry->shape = id4_pipeline_parameter_layout_shape(source->shape);
    IREE_RETURN_IF_ERROR(id4_pipeline_program_tensor_byte_length(
        source->dtype, source->shape, &out_entry->byte_length));
  } else {
    out_entry->dtype = parameter_tensor->layout.dtype;
    out_entry->shape = parameter_tensor->layout.shape;
    out_entry->byte_length = parameter_tensor->layout.byte_length;
  }
  out_entry->alignment = parameter_tensor->layout.alignment;
  out_entry->source_scope = iree_string_view_empty();
  out_entry->parameter_slab_index = parameter_tensor->parameter_slab_index;
  out_entry->parameter_slab_offset = parameter_tensor->offset;
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_layout_make_archive_request(
    const id4_pipeline_plan_t* plan, iree_host_size_t parameter_tensor_index,
    const id4_pipeline_parameter_request_t* planned_request,
    iree_io_parameter_span_t target_span,
    id4_pipeline_parameter_request_t* out_request) {
  IREE_ASSERT_ARGUMENT(planned_request);
  IREE_ASSERT_ARGUMENT(out_request);
  memset(out_request, 0, sizeof(*out_request));
  const id4_pipeline_parameter_tensor_plan_t* parameter_tensor =
      id4_pipeline_plan_parameter_tensor_at(plan, parameter_tensor_index);
  if (!parameter_tensor) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter layout tensor %" PRIhsz " is missing",
                            parameter_tensor_index);
  }
  if (target_span.length != planned_request->span.length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter layout archive target length %" PRIu64
                            " does not match planned request length %" PRIu64,
                            target_span.length, planned_request->span.length);
  }

  id4_pipeline_parameter_layout_entry_t entry;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_candidate(
      plan, parameter_tensor_index, &entry));
  uint64_t archive_parameter_offset = 0;
  if (entry.kind == ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_SOURCE) {
    archive_parameter_offset = planned_request->span.parameter_offset;
  } else if (entry.kind == ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_EXECUTION) {
    if (planned_request->span.buffer_offset < parameter_tensor->offset) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "parameter layout request for %.*s starts before tensor storage",
          (int)parameter_tensor->layout.name.size,
          parameter_tensor->layout.name.data);
    }
    archive_parameter_offset =
        planned_request->span.buffer_offset - parameter_tensor->offset;
  } else {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "parameter layout entry kind %u is invalid",
                            entry.kind);
  }
  if (archive_parameter_offset > entry.byte_length ||
      target_span.length > entry.byte_length - archive_parameter_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter layout archive request for %.*s exceeds entry storage",
        (int)entry.key.size, entry.key.data);
  }
  target_span.parameter_offset = archive_parameter_offset;
  *out_request = id4_pipeline_parameter_request(entry.key, target_span);
  return iree_ok_status();
}

static bool id4_pipeline_parameter_layout_shapes_equal(
    id4_pipeline_tensor_shape_t lhs, id4_pipeline_tensor_shape_t rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint32_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dims[i] != rhs.dims[i]) return false;
  }
  return true;
}

static bool id4_pipeline_parameter_layout_entries_compatible(
    const id4_pipeline_parameter_layout_entry_t* lhs,
    const id4_pipeline_parameter_layout_entry_t* rhs) {
  return lhs->kind == rhs->kind && lhs->encoding == rhs->encoding &&
         lhs->dtype == rhs->dtype &&
         id4_pipeline_parameter_layout_shapes_equal(lhs->shape, rhs->shape) &&
         lhs->byte_length == rhs->byte_length &&
         lhs->alignment == rhs->alignment &&
         lhs->source_schema_fingerprint == rhs->source_schema_fingerprint &&
         iree_string_view_equal(lhs->source_scope, rhs->source_scope);
}

static iree_status_t id4_pipeline_parameter_layout_is_first_candidate(
    const id4_pipeline_plan_t* plan, iree_host_size_t parameter_tensor_index,
    const id4_pipeline_parameter_layout_entry_t* candidate,
    bool* out_is_first) {
  *out_is_first = true;
  for (iree_host_size_t i = 0; i < parameter_tensor_index; ++i) {
    id4_pipeline_parameter_layout_entry_t prior;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_parameter_layout_candidate(plan, i, &prior));
    if (!iree_string_view_equal(candidate->key, prior.key)) continue;
    if (!id4_pipeline_parameter_layout_entries_compatible(candidate, &prior)) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "baked parameter key %.*s has incompatible physical layouts",
          (int)candidate->key.size, candidate->key.data);
    }
    *out_is_first = false;
    return iree_ok_status();
  }
  return iree_ok_status();
}

// Visits one unique durable parameter-layout entry.
typedef iree_status_t(IREE_API_PTR* id4_pipeline_parameter_layout_visit_fn_t)(
    void* user_data, iree_host_size_t index,
    const id4_pipeline_parameter_layout_entry_t* entry);

static iree_status_t id4_pipeline_parameter_layout_enumerate(
    const id4_pipeline_plan_t* plan,
    id4_pipeline_parameter_layout_visit_fn_t visit, void* user_data,
    iree_host_size_t* out_count) {
  *out_count = 0;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_validate_plan(plan));
  const iree_host_size_t parameter_tensor_count =
      id4_pipeline_plan_parameter_tensor_count(plan);
  for (iree_host_size_t i = 0; i < parameter_tensor_count; ++i) {
    id4_pipeline_parameter_layout_entry_t candidate;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_parameter_layout_candidate(plan, i, &candidate));
    bool is_first = false;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_is_first_candidate(
        plan, i, &candidate, &is_first));
    if (!is_first) continue;
    if (visit) {
      IREE_RETURN_IF_ERROR(visit(user_data, *out_count, &candidate));
    }
    ++*out_count;
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_layout_entry_count(
    const id4_pipeline_plan_t* plan, iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(out_count);
  return id4_pipeline_parameter_layout_enumerate(plan, /*visit=*/NULL,
                                                 /*user_data=*/NULL, out_count);
}

iree_status_t id4_pipeline_parameter_layout_entry_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index,
    id4_pipeline_parameter_layout_entry_t* out_entry) {
  IREE_ASSERT_ARGUMENT(out_entry);
  memset(out_entry, 0, sizeof(*out_entry));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_validate_plan(plan));
  iree_host_size_t entry_index = 0;
  const iree_host_size_t parameter_tensor_count =
      id4_pipeline_plan_parameter_tensor_count(plan);
  for (iree_host_size_t i = 0; i < parameter_tensor_count; ++i) {
    id4_pipeline_parameter_layout_entry_t candidate;
    IREE_RETURN_IF_ERROR(
        id4_pipeline_parameter_layout_candidate(plan, i, &candidate));
    bool is_first = false;
    IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_is_first_candidate(
        plan, i, &candidate, &is_first));
    if (!is_first) continue;
    if (entry_index++ == index) {
      *out_entry = candidate;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "parameter layout entry %" PRIhsz
                          " is outside entry count %" PRIhsz,
                          index, entry_index);
}

static iree_status_t id4_pipeline_parameter_layout_accumulate_statistics(
    void* user_data, iree_host_size_t index,
    const id4_pipeline_parameter_layout_entry_t* entry) {
  (void)index;
  id4_pipeline_parameter_layout_statistics_t* statistics =
      (id4_pipeline_parameter_layout_statistics_t*)user_data;
  switch (entry->kind) {
    case ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_SOURCE:
      if (!iree_device_size_checked_add(statistics->source_byte_length,
                                        entry->byte_length,
                                        &statistics->source_byte_length)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "parameter layout source byte length overflows");
      }
      ++statistics->source_entry_count;
      return iree_ok_status();
    case ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_EXECUTION:
      if (!iree_device_size_checked_add(statistics->execution_byte_length,
                                        entry->byte_length,
                                        &statistics->execution_byte_length)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "parameter layout execution byte length overflows");
      }
      ++statistics->execution_entry_count;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "parameter layout entry kind %u is invalid",
                              entry->kind);
  }
}

iree_status_t id4_pipeline_parameter_layout_query_statistics(
    const id4_pipeline_plan_t* plan,
    id4_pipeline_parameter_layout_statistics_t* out_statistics) {
  IREE_ASSERT_ARGUMENT(out_statistics);
  memset(out_statistics, 0, sizeof(*out_statistics));
  iree_host_size_t entry_count = 0;
  return id4_pipeline_parameter_layout_enumerate(
      plan, id4_pipeline_parameter_layout_accumulate_statistics, out_statistics,
      &entry_count);
}

static void id4_pipeline_parameter_layout_store_u32(uint32_t value,
                                                    uint8_t** inout_data) {
  uint8_t* data = *inout_data;
  data[0] = (uint8_t)(value >> 0);
  data[1] = (uint8_t)(value >> 8);
  data[2] = (uint8_t)(value >> 16);
  data[3] = (uint8_t)(value >> 24);
  *inout_data += sizeof(value);
}

static void id4_pipeline_parameter_layout_store_u64(uint64_t value,
                                                    uint8_t** inout_data) {
  uint8_t* data = *inout_data;
  for (iree_host_size_t i = 0; i < sizeof(value); ++i) {
    data[i] = (uint8_t)(value >> (i * 8));
  }
  *inout_data += sizeof(value);
}

static iree_const_byte_span_t id4_pipeline_parameter_layout_metadata(
    const id4_pipeline_parameter_layout_entry_t* entry,
    uint8_t storage[ID4_PIPELINE_PARAMETER_LAYOUT_METADATA_LENGTH]) {
  uint8_t* data = storage;
  id4_pipeline_parameter_layout_store_u32(0x4C503449u, &data);  // "I4PL"
  id4_pipeline_parameter_layout_store_u32(
      ID4_PIPELINE_PARAMETER_LAYOUT_METADATA_VERSION, &data);
  id4_pipeline_parameter_layout_store_u32(entry->kind, &data);
  id4_pipeline_parameter_layout_store_u32(entry->encoding, &data);
  id4_pipeline_parameter_layout_store_u32(entry->dtype, &data);
  id4_pipeline_parameter_layout_store_u32(entry->shape.rank, &data);
  for (uint32_t i = 0; i < ID4_PIPELINE_TENSOR_MAX_RANK; ++i) {
    id4_pipeline_parameter_layout_store_u64(entry->shape.dims[i], &data);
  }
  id4_pipeline_parameter_layout_store_u64(entry->byte_length, &data);
  id4_pipeline_parameter_layout_store_u64(entry->alignment, &data);
  id4_pipeline_parameter_layout_store_u64(entry->source_schema_fingerprint,
                                          &data);
  IREE_ASSERT((iree_host_size_t)(data - storage) ==
              ID4_PIPELINE_PARAMETER_LAYOUT_METADATA_LENGTH);
  return iree_make_const_byte_span(storage, data - storage);
}

static iree_status_t id4_pipeline_parameter_layout_add_archive_entry(
    void* user_data, iree_host_size_t index,
    const id4_pipeline_parameter_layout_entry_t* entry) {
  (void)index;
  iree_io_parameter_archive_builder_t* archive_builder =
      (iree_io_parameter_archive_builder_t*)user_data;
  uint8_t metadata_storage[ID4_PIPELINE_PARAMETER_LAYOUT_METADATA_LENGTH];
  const iree_const_byte_span_t metadata =
      id4_pipeline_parameter_layout_metadata(entry, metadata_storage);
  const iree_io_physical_size_t minimum_alignment = iree_max(
      (iree_io_physical_size_t)IREE_IO_PARAMETER_ARCHIVE_DEFAULT_DATA_ALIGNMENT,
      (iree_io_physical_size_t)entry->alignment);
  return iree_io_parameter_archive_builder_add_data_entry(
      archive_builder, entry->key, metadata, minimum_alignment,
      entry->byte_length);
}

iree_status_t id4_pipeline_parameter_layout_add_archive_entries(
    const id4_pipeline_plan_t* plan,
    iree_io_parameter_archive_builder_t* archive_builder) {
  IREE_ASSERT_ARGUMENT(archive_builder);
  iree_host_size_t entry_count = 0;
  return id4_pipeline_parameter_layout_enumerate(
      plan, id4_pipeline_parameter_layout_add_archive_entry, archive_builder,
      &entry_count);
}

static iree_status_t id4_pipeline_parameter_layout_validate_index_entry(
    void* user_data, iree_host_size_t index,
    const id4_pipeline_parameter_layout_entry_t* expected) {
  (void)index;
  iree_io_parameter_index_t* parameter_index =
      (iree_io_parameter_index_t*)user_data;
  const iree_io_parameter_index_entry_t* actual = NULL;
  IREE_RETURN_IF_ERROR(
      iree_io_parameter_index_lookup(parameter_index, expected->key, &actual));
  if (actual->length != expected->byte_length) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "baked parameter %.*s length %" PRIu64
                            " does not match required length %" PRIu64,
                            (int)expected->key.size, expected->key.data,
                            actual->length, (uint64_t)expected->byte_length);
  }
  uint8_t metadata_storage[ID4_PIPELINE_PARAMETER_LAYOUT_METADATA_LENGTH];
  const iree_const_byte_span_t expected_metadata =
      id4_pipeline_parameter_layout_metadata(expected, metadata_storage);
  if (actual->metadata.data_length != expected_metadata.data_length ||
      memcmp(actual->metadata.data, expected_metadata.data,
             expected_metadata.data_length) != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "baked parameter %.*s physical layout metadata does not match",
        (int)expected->key.size, expected->key.data);
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_layout_validate_index(
    const id4_pipeline_plan_t* plan, iree_io_parameter_index_t* index) {
  if (!index) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "baked parameter index is required");
  }
  iree_host_size_t entry_count = 0;
  return id4_pipeline_parameter_layout_enumerate(
      plan, id4_pipeline_parameter_layout_validate_index_entry, index,
      &entry_count);
}

typedef struct id4_pipeline_parameter_layout_entry_table_t {
  // Maximum number of entries available in |values|.
  iree_host_size_t capacity;
  // Unique entries written in program order.
  id4_pipeline_parameter_layout_entry_t* values;
} id4_pipeline_parameter_layout_entry_table_t;

static iree_status_t id4_pipeline_parameter_layout_store_entry(
    void* user_data, iree_host_size_t index,
    const id4_pipeline_parameter_layout_entry_t* entry) {
  id4_pipeline_parameter_layout_entry_table_t* table =
      (id4_pipeline_parameter_layout_entry_table_t*)user_data;
  if (index >= table->capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter layout entry table capacity exceeded");
  }
  table->values[index] = *entry;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_layout_build_entry_table(
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    iree_host_size_t* out_count,
    id4_pipeline_parameter_layout_entry_t** out_entries) {
  *out_count = 0;
  *out_entries = NULL;
  const iree_host_size_t capacity =
      id4_pipeline_plan_parameter_tensor_count(plan);
  id4_pipeline_parameter_layout_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, capacity, sizeof(entries[0]), (void**)&entries));
  id4_pipeline_parameter_layout_entry_table_t table = {
      // One candidate per planned parameter tensor.
      .capacity = capacity,
      // Caller-owned entry storage.
      .values = entries,
  };
  iree_status_t status = id4_pipeline_parameter_layout_enumerate(
      plan, id4_pipeline_parameter_layout_store_entry, &table, out_count);
  if (iree_status_is_ok(status)) {
    *out_entries = entries;
  } else {
    iree_allocator_free(host_allocator, entries);
  }
  return status;
}

static iree_hal_semaphore_list_t id4_pipeline_parameter_layout_one_semaphore(
    iree_hal_semaphore_t** semaphore, uint64_t* payload_value) {
  return (iree_hal_semaphore_list_t){
      // One timeline semaphore edge.
      .count = 1,
      // Timeline semaphore storage.
      .semaphores = semaphore,
      // Timeline payload storage.
      .payload_values = payload_value,
  };
}

static iree_status_t id4_pipeline_parameter_layout_validate_populate_options(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_layout_populate_options_t* options,
    iree_host_size_t entry_count,
    const id4_pipeline_parameter_layout_entry_t* entries,
    iree_hal_device_t** out_device,
    iree_hal_queue_affinity_t* out_queue_affinity) {
  *out_device = NULL;
  *out_queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter layout populate options are required");
  }
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter layout populate options are too small");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "parameter layout populate extension structures are not supported");
  }
  if (!options->target_index || !options->target_provider ||
      !options->parameter_slabs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter layout target and resident slabs are required");
  }
  if (iree_string_view_is_empty(options->target_scope) ||
      !iree_io_parameter_provider_query_support(options->target_provider,
                                                options->target_scope)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter layout target provider scope is unsupported");
  }
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter layout population requires a signal semaphore list");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_parameter_slabs(
      plan, options->parameter_slabs));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_validate_index(
      plan, options->target_index));
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("parameter layout populate")));

  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    const id4_pipeline_parameter_layout_entry_t* entry = &entries[i];
    if (entry->kind != ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_SOURCE) {
      continue;
    }
    if (!options->source_provider) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter layout source provider is required by %.*s",
          (int)entry->key.size, entry->key.data);
    }
    if (!iree_io_parameter_provider_query_support(options->source_provider,
                                                  entry->source_scope)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter layout source provider does not support scope %.*s",
          (int)entry->source_scope.size, entry->source_scope.data);
    }
    if (options->staging_chunk_byte_capacity == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter layout source entries require a nonzero staging chunk "
          "byte capacity");
    }
  }

  iree_hal_device_group_t* device_group = id4_pipeline_plan_device_group(plan);
  const iree_host_size_t slab_count =
      id4_pipeline_plan_parameter_slab_count(plan);
  for (iree_host_size_t i = 0; i < slab_count; ++i) {
    const id4_pipeline_parameter_slab_plan_t* slab =
        id4_pipeline_plan_parameter_slab_at(plan, i);
    const id4_pipeline_device_placement_t* placement =
        slab ? id4_pipeline_plan_placement_at(plan, slab->placement_id) : NULL;
    iree_hal_device_t* device = placement
                                    ? iree_hal_device_group_device_at(
                                          device_group, placement->device_index)
                                    : NULL;
    if (!slab || !placement || !device) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "parameter layout slab %" PRIhsz " placement is incomplete", i);
    }
    if (!*out_device) {
      *out_device = device;
      *out_queue_affinity = placement->queue_affinity;
    } else if (*out_device != device ||
               *out_queue_affinity != placement->queue_affinity) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "one baked parameter archive population cannot span device queues");
    }
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_parameter_layout_build_execution_scatter_tables(
    const id4_pipeline_plan_t* plan, iree_host_size_t entry_count,
    const id4_pipeline_parameter_layout_entry_t* entries,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_request_table_t** out_tables,
    id4_pipeline_parameter_request_t** out_requests) {
  *out_tables = NULL;
  *out_requests = NULL;
  const iree_host_size_t slab_count =
      id4_pipeline_plan_parameter_slab_count(plan);
  iree_host_size_t execution_entry_count = 0;
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    execution_entry_count +=
        entries[i].kind == ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_EXECUTION
            ? 1
            : 0;
  }

  id4_pipeline_parameter_request_table_t* tables = NULL;
  id4_pipeline_parameter_request_t* requests = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, slab_count, sizeof(tables[0]), (void**)&tables));
  iree_status_t status = iree_ok_status();
  if (execution_entry_count != 0) {
    status =
        iree_allocator_malloc_array(host_allocator, execution_entry_count,
                                    sizeof(requests[0]), (void**)&requests);
  }
  if (iree_status_is_ok(status)) {
    memset(tables, 0, slab_count * sizeof(tables[0]));
    if (execution_entry_count != 0) {
      memset(requests, 0, execution_entry_count * sizeof(requests[0]));
    }
  }

  iree_host_size_t request_offset = 0;
  for (iree_host_size_t slab_index = 0;
       slab_index < slab_count && iree_status_is_ok(status); ++slab_index) {
    const iree_host_size_t table_offset = request_offset;
    for (iree_host_size_t i = 0; i < entry_count; ++i) {
      const id4_pipeline_parameter_layout_entry_t* entry = &entries[i];
      if (entry->kind != ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_EXECUTION ||
          entry->parameter_slab_index != slab_index) {
        continue;
      }
      requests[request_offset++] = id4_pipeline_parameter_request(
          entry->key, id4_pipeline_parameter_span(
                          /*parameter_offset=*/0, entry->parameter_slab_offset,
                          entry->byte_length));
    }
    tables[slab_index] = id4_pipeline_make_parameter_request_table(
        request_offset - table_offset,
        request_offset == table_offset ? NULL : &requests[table_offset]);
  }
  if (iree_status_is_ok(status) && request_offset != execution_entry_count) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "parameter layout execution scatter entry count mismatch");
  }
  if (iree_status_is_ok(status)) {
    *out_tables = tables;
    *out_requests = requests;
  } else {
    iree_allocator_free(host_allocator, requests);
    iree_allocator_free(host_allocator, tables);
  }
  return status;
}

iree_status_t id4_pipeline_parameter_layout_populate(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_layout_populate_options_t* options,
    iree_allocator_t host_allocator) {
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_validate_plan(plan));
  iree_host_size_t entry_count = 0;
  id4_pipeline_parameter_layout_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_build_entry_table(
      plan, host_allocator, &entry_count, &entries));
  iree_hal_device_t* device = NULL;
  iree_hal_queue_affinity_t queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  id4_pipeline_parameter_request_table_t* execution_tables = NULL;
  id4_pipeline_parameter_request_t* execution_requests = NULL;
  iree_status_t status =
      id4_pipeline_parameter_layout_validate_populate_options(
          plan, options, entry_count, entries, &device, &queue_affinity);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_layout_build_execution_scatter_tables(
        plan, entry_count, entries, host_allocator, &execution_tables,
        &execution_requests);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, entries);
    return status;
  }

  iree_hal_semaphore_t* timeline_semaphore = NULL;
  iree_hal_buffer_t* staging_buffer = NULL;
  bool staging_alloca_submitted = false;
  uint64_t timeline_payload_value = 0;
  bool chain_started = false;
  status = iree_hal_semaphore_create(
      device, queue_affinity, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &timeline_semaphore);

  iree_device_size_t staging_byte_length = 0;
  for (iree_host_size_t i = 0; i < entry_count && iree_status_is_ok(status);
       ++i) {
    const id4_pipeline_parameter_layout_entry_t* entry = &entries[i];
    if (entry->kind == ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_SOURCE) {
      staging_byte_length = iree_max(
          staging_byte_length,
          iree_min(entry->byte_length, options->staging_chunk_byte_capacity));
    }
  }
  if (iree_status_is_ok(status) && staging_byte_length != 0) {
    iree_hal_buffer_params_t staging_params = {0};
    staging_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    staging_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    staging_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    staging_params.queue_affinity = queue_affinity;
    staging_params.min_alignment = 16;
    uint64_t signal_payload_value = timeline_payload_value + 1;
    iree_hal_semaphore_list_t signal_list =
        id4_pipeline_parameter_layout_one_semaphore(&timeline_semaphore,
                                                    &signal_payload_value);
    status = iree_hal_device_queue_alloca(
        device, queue_affinity, options->wait_semaphore_list, signal_list,
        /*pool=*/NULL, staging_params, staging_byte_length,
        IREE_HAL_ALLOCA_FLAG_NONE, &staging_buffer);
    staging_alloca_submitted = iree_status_is_ok(status);
    chain_started = staging_alloca_submitted;
    if (staging_alloca_submitted) {
      timeline_payload_value = signal_payload_value;
    }
  }

  for (iree_host_size_t i = 0; i < entry_count && iree_status_is_ok(status);
       ++i) {
    const id4_pipeline_parameter_layout_entry_t* entry = &entries[i];
    if (entry->kind != ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_SOURCE) {
      continue;
    }
    uint64_t entry_offset = 0;
    while (entry_offset < entry->byte_length && iree_status_is_ok(status)) {
      const iree_device_size_t chunk_byte_length =
          iree_min((iree_device_size_t)(entry->byte_length - entry_offset),
                   staging_byte_length);
      iree_hal_semaphore_list_t wait_list =
          id4_pipeline_parameter_layout_one_semaphore(&timeline_semaphore,
                                                      &timeline_payload_value);
      uint64_t read_signal_payload_value = timeline_payload_value + 1;
      iree_hal_semaphore_list_t signal_list =
          id4_pipeline_parameter_layout_one_semaphore(
              &timeline_semaphore, &read_signal_payload_value);
      status = iree_io_parameter_provider_read(
          options->source_provider, device, queue_affinity, wait_list,
          signal_list, entry->source_scope, entry->key, entry_offset,
          staging_buffer, /*target_offset=*/0, chunk_byte_length);
      if (!iree_status_is_ok(status)) break;
      timeline_payload_value = read_signal_payload_value;
      wait_list = id4_pipeline_parameter_layout_one_semaphore(
          &timeline_semaphore, &timeline_payload_value);
      uint64_t write_signal_payload_value = timeline_payload_value + 1;
      signal_list = id4_pipeline_parameter_layout_one_semaphore(
          &timeline_semaphore, &write_signal_payload_value);
      status = iree_io_parameter_provider_write(
          options->target_provider, device, queue_affinity, wait_list,
          signal_list, staging_buffer, /*source_offset=*/0,
          options->target_scope, entry->key, entry_offset, chunk_byte_length);
      if (iree_status_is_ok(status)) {
        timeline_payload_value = write_signal_payload_value;
        entry_offset += chunk_byte_length;
      }
    }
  }

  const iree_host_size_t slab_count =
      id4_pipeline_plan_parameter_slab_count(plan);
  for (iree_host_size_t i = 0; i < slab_count && iree_status_is_ok(status);
       ++i) {
    if (execution_tables[i].count == 0) continue;
    iree_hal_semaphore_list_t wait_list =
        chain_started ? id4_pipeline_parameter_layout_one_semaphore(
                            &timeline_semaphore, &timeline_payload_value)
                      : options->wait_semaphore_list;
    uint64_t signal_payload_value = timeline_payload_value + 1;
    iree_hal_semaphore_list_t signal_list =
        id4_pipeline_parameter_layout_one_semaphore(&timeline_semaphore,
                                                    &signal_payload_value);
    id4_pipeline_parameter_slab_enumerator_state_t enumerator_state = {
        // Execution entries stored in this resident slab.
        .request_table = &execution_tables[i],
        // First execution entry for this slab.
        .request_offset = 0,
        // All execution entries for this slab.
        .request_count = execution_tables[i].count,
        // Execution entries are contiguous in table order.
        .request_indices = NULL,
    };
    status = iree_io_parameter_provider_scatter(
        options->target_provider, device, queue_affinity, wait_list,
        signal_list,
        id4_pipeline_parameter_slab_set_buffer_at(options->parameter_slabs, i),
        options->target_scope, execution_tables[i].count,
        id4_pipeline_parameter_slab_enumerator(&enumerator_state));
    chain_started = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      timeline_payload_value = signal_payload_value;
    }
  }

  if (staging_alloca_submitted) {
    iree_hal_semaphore_list_t wait_list =
        id4_pipeline_parameter_layout_one_semaphore(&timeline_semaphore,
                                                    &timeline_payload_value);
    uint64_t signal_payload_value = timeline_payload_value + 1;
    iree_hal_semaphore_list_t signal_list =
        id4_pipeline_parameter_layout_one_semaphore(&timeline_semaphore,
                                                    &signal_payload_value);
    iree_status_t dealloca_status = iree_hal_device_queue_dealloca(
        device, queue_affinity, wait_list, signal_list, staging_buffer,
        IREE_HAL_DEALLOCA_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      status = dealloca_status;
      chain_started = iree_status_is_ok(status);
      if (iree_status_is_ok(status)) {
        timeline_payload_value = signal_payload_value;
      }
    } else {
      status = iree_status_join(status, dealloca_status);
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_semaphore_list_t wait_list =
        chain_started ? id4_pipeline_parameter_layout_one_semaphore(
                            &timeline_semaphore, &timeline_payload_value)
                      : options->wait_semaphore_list;
    status = iree_hal_device_queue_barrier(device, queue_affinity, wait_list,
                                           options->signal_semaphore_list,
                                           IREE_HAL_EXECUTE_FLAG_NONE);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(options->signal_semaphore_list,
                                 iree_status_clone(status));
  }
  iree_hal_buffer_release(staging_buffer);
  iree_hal_semaphore_release(timeline_semaphore);
  iree_allocator_free(host_allocator, execution_requests);
  iree_allocator_free(host_allocator, execution_tables);
  iree_allocator_free(host_allocator, entries);
  return status;
}

static iree_status_t id4_pipeline_parameter_layout_validate_load_options(
    const id4_pipeline_parameter_layout_load_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter layout load options are required");
  }
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter layout load options are too small");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "parameter layout load extension structures are not supported");
  }
  if (!options->index || !options->provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter layout index and provider are required");
  }
  if (iree_string_view_is_empty(options->scope) ||
      !iree_io_parameter_provider_query_support(options->provider,
                                                options->scope)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter layout provider scope is unsupported");
  }
  const iree_hal_semaphore_list_t semaphore_lists[] = {
      options->wait_semaphore_list,
      options->signal_semaphore_list,
  };
  const iree_string_view_t semaphore_list_names[] = {
      IREE_SVL("parameter layout wait"),
      IREE_SVL("parameter layout signal"),
  };
  for (iree_host_size_t list_index = 0;
       list_index < IREE_ARRAYSIZE(semaphore_lists); ++list_index) {
    const iree_hal_semaphore_list_t list = semaphore_lists[list_index];
    if (list.count != 0 && (!list.semaphores || !list.payload_values)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s semaphore list is malformed",
                              (int)semaphore_list_names[list_index].size,
                              semaphore_list_names[list_index].data);
    }
    for (iree_host_size_t i = 0; i < list.count; ++i) {
      if (!list.semaphores[i]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "%.*s semaphore %" PRIhsz " is NULL",
                                (int)semaphore_list_names[list_index].size,
                                semaphore_list_names[list_index].data, i);
      }
    }
  }
  if (semaphore_lists[1].count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter layout loading requires a signal semaphore list");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("parameter layout load"));
}

static iree_status_t id4_pipeline_parameter_layout_make_load(
    const id4_pipeline_plan_t* plan, iree_host_size_t slab_index,
    id4_pipeline_parameter_slab_load_t* out_load) {
  IREE_RETURN_IF_ERROR(
      id4_pipeline_plan_parameter_slab_load_at(plan, slab_index, out_load));
  if (out_load->request_table->count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "parameter layout slab %" PRIhsz
                            " has no parameter requests",
                            slab_index);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_layout_make_loads(
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_load_t** out_loads) {
  *out_loads = NULL;
  const iree_host_size_t load_count =
      id4_pipeline_plan_parameter_slab_count(plan);
  id4_pipeline_parameter_slab_load_t* loads = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, load_count, sizeof(loads[0]), (void**)&loads));
  iree_hal_device_t* first_device = NULL;
  iree_hal_queue_affinity_t first_queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < load_count && iree_status_is_ok(status);
       ++i) {
    status = id4_pipeline_parameter_layout_make_load(plan, i, &loads[i]);
    if (!iree_status_is_ok(status)) break;
    if (!first_device) {
      first_device = loads[i].device;
      first_queue_affinity = loads[i].queue_affinity;
    } else if (first_device != loads[i].device ||
               first_queue_affinity != loads[i].queue_affinity) {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "one baked parameter archive load cannot span device queues");
      break;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_loads = loads;
  } else {
    iree_allocator_free(host_allocator, loads);
  }
  return status;
}

static iree_status_t
id4_pipeline_parameter_layout_populate_archive_tensor_requests(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_tensor_plan_t* parameter_tensor,
    const id4_pipeline_parameter_slab_load_t* target_load,
    id4_pipeline_parameter_request_table_t* archive_table) {
  if (parameter_tensor->request_offset > archive_table->count ||
      parameter_tensor->request_count >
          archive_table->count - parameter_tensor->request_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter tensor %.*s request range exceeds slab %" PRIhsz,
        (int)parameter_tensor->layout.name.size,
        parameter_tensor->layout.name.data, target_load->slab_index);
  }
  const id4_pipeline_program_parameter_op_t* parameter_op = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_parameter_op(
      plan, parameter_tensor, &parameter_op));
  const id4_pipeline_parameter_request_table_t* original_table =
      target_load->request_table;
  id4_pipeline_parameter_request_t* archive_values =
      (id4_pipeline_parameter_request_t*)archive_table->values;
  if (parameter_op->encoding ==
          ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT &&
      parameter_op->source_span_count != 0) {
    for (iree_host_size_t i = 0; i < parameter_tensor->request_count; ++i) {
      archive_values[parameter_tensor->request_offset + i] =
          original_table->values[parameter_tensor->request_offset + i];
    }
    return iree_ok_status();
  }
  if (parameter_tensor->request_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "execution parameter %.*s must own one final-layout request",
        (int)parameter_tensor->layout.name.size,
        parameter_tensor->layout.name.data);
  }
  archive_values[parameter_tensor->request_offset] =
      id4_pipeline_parameter_request(
          parameter_tensor->layout.name,
          id4_pipeline_parameter_span(
              /*parameter_offset=*/0, parameter_tensor->offset,
              parameter_tensor->layout.byte_length));
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_layout_make_archive_requests(
    const id4_pipeline_plan_t* plan, iree_host_size_t slab_count,
    const id4_pipeline_parameter_slab_load_t* target_loads,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_request_table_t** out_request_tables,
    id4_pipeline_parameter_request_t** out_requests,
    id4_pipeline_parameter_slab_load_t** out_archive_loads) {
  *out_request_tables = NULL;
  *out_requests = NULL;
  *out_archive_loads = NULL;
  iree_host_size_t request_count = 0;
  for (iree_host_size_t i = 0; i < slab_count; ++i) {
    if (!iree_host_size_checked_add(request_count,
                                    target_loads[i].request_table->count,
                                    &request_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter layout request count overflows");
    }
  }

  id4_pipeline_parameter_request_table_t* request_tables = NULL;
  id4_pipeline_parameter_request_t* requests = NULL;
  id4_pipeline_parameter_slab_load_t* archive_loads = NULL;
  iree_status_t status = iree_allocator_malloc_array(host_allocator, slab_count,
                                                     sizeof(request_tables[0]),
                                                     (void**)&request_tables);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, request_count, sizeof(requests[0]), (void**)&requests);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, slab_count,
                                         sizeof(archive_loads[0]),
                                         (void**)&archive_loads);
  }
  if (iree_status_is_ok(status)) {
    memset(request_tables, 0, slab_count * sizeof(request_tables[0]));
    memset(requests, 0, request_count * sizeof(requests[0]));
    memcpy(archive_loads, target_loads, slab_count * sizeof(archive_loads[0]));
    iree_host_size_t request_offset = 0;
    for (iree_host_size_t i = 0; i < slab_count; ++i) {
      request_tables[i] = id4_pipeline_make_parameter_request_table(
          target_loads[i].request_table->count, &requests[request_offset]);
      archive_loads[i].request_table = &request_tables[i];
      request_offset += request_tables[i].count;
    }
  }

  const iree_host_size_t parameter_tensor_count =
      id4_pipeline_plan_parameter_tensor_count(plan);
  for (iree_host_size_t i = 0;
       i < parameter_tensor_count && iree_status_is_ok(status); ++i) {
    const id4_pipeline_parameter_tensor_plan_t* parameter_tensor =
        id4_pipeline_plan_parameter_tensor_at(plan, i);
    if (parameter_tensor->parameter_slab_index >= slab_count) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "parameter tensor %.*s slab index is outside layout",
                           (int)parameter_tensor->layout.name.size,
                           parameter_tensor->layout.name.data);
      break;
    }
    id4_pipeline_parameter_request_table_t* archive_table =
        &request_tables[parameter_tensor->parameter_slab_index];
    status = id4_pipeline_parameter_layout_populate_archive_tensor_requests(
        plan, parameter_tensor,
        &target_loads[parameter_tensor->parameter_slab_index], archive_table);
  }
  for (iree_host_size_t i = 0; i < request_count && iree_status_is_ok(status);
       ++i) {
    if (iree_string_view_is_empty(requests[i].key)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "parameter layout request %" PRIhsz " is not owned by a tensor", i);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_request_tables = request_tables;
    *out_requests = requests;
    *out_archive_loads = archive_loads;
  } else {
    iree_allocator_free(host_allocator, archive_loads);
    iree_allocator_free(host_allocator, requests);
    iree_allocator_free(host_allocator, request_tables);
  }
  return status;
}

static iree_status_t id4_pipeline_parameter_layout_submit_slab_gather(
    const id4_pipeline_parameter_layout_load_options_t* options,
    const id4_pipeline_parameter_slab_load_t* archive_load,
    iree_hal_buffer_t* target_buffer,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  id4_pipeline_parameter_slab_enumerator_state_t enumerator_state = {
      // Archive keys and final target spans for this slab.
      .request_table = archive_load->request_table,
      // First request in the slab table.
      .request_offset = 0,
      // All requests in the slab table.
      .request_count = archive_load->request_table->count,
      // Requests are contiguous in table order.
      .request_indices = NULL,
  };
  return iree_io_parameter_provider_gather(
      options->provider, archive_load->device, archive_load->queue_affinity,
      wait_semaphore_list, signal_semaphore_list, options->scope, target_buffer,
      archive_load->request_table->count,
      id4_pipeline_parameter_slab_enumerator(&enumerator_state));
}

iree_status_t id4_pipeline_parameter_layout_gather_slab(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_layout_load_options_t* options,
    iree_host_size_t target_slab_index, iree_hal_buffer_t* target_buffer,
    iree_allocator_t host_allocator) {
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_validate_plan(plan));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_layout_validate_load_options(options));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_layout_validate_index(plan, options->index));

  const iree_host_size_t slab_count =
      id4_pipeline_plan_parameter_slab_count(plan);
  if (target_slab_index >= slab_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter layout target slab %" PRIhsz
                            " is outside slab count %" PRIhsz,
                            target_slab_index, slab_count);
  }

  id4_pipeline_parameter_slab_load_t target_load;
  memset(&target_load, 0, sizeof(target_load));
  id4_pipeline_parameter_request_t* archive_requests = NULL;
  id4_pipeline_parameter_request_table_t archive_request_table;
  memset(&archive_request_table, 0, sizeof(archive_request_table));
  id4_pipeline_parameter_slab_load_t archive_load;
  memset(&archive_load, 0, sizeof(archive_load));
  iree_status_t status = id4_pipeline_parameter_layout_make_load(
      plan, target_slab_index, &target_load);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, target_load.request_table->count,
        sizeof(archive_requests[0]), (void**)&archive_requests);
  }
  if (iree_status_is_ok(status)) {
    memset(archive_requests, 0,
           target_load.request_table->count * sizeof(archive_requests[0]));
    archive_request_table = id4_pipeline_make_parameter_request_table(
        target_load.request_table->count, archive_requests);
    archive_load = target_load;
    archive_load.request_table = &archive_request_table;
    const iree_host_size_t parameter_tensor_count =
        id4_pipeline_plan_parameter_tensor_count(plan);
    for (iree_host_size_t i = 0;
         i < parameter_tensor_count && iree_status_is_ok(status); ++i) {
      const id4_pipeline_parameter_tensor_plan_t* parameter_tensor =
          id4_pipeline_plan_parameter_tensor_at(plan, i);
      if (parameter_tensor->parameter_slab_index != target_slab_index) continue;
      status = id4_pipeline_parameter_layout_populate_archive_tensor_requests(
          plan, parameter_tensor, &target_load, &archive_request_table);
    }
  }
  for (iree_host_size_t i = 0;
       i < archive_request_table.count && iree_status_is_ok(status); ++i) {
    if (iree_string_view_is_empty(archive_requests[i].key)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "parameter layout slab request %" PRIhsz
                                " is not owned by a tensor",
                                i);
    }
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_validate_resident_buffer(
        &target_load, target_buffer);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t placement =
        iree_hal_buffer_allocation_placement(target_buffer);
    if (iree_hal_buffer_placement_is_undefined(placement) ||
        iree_hal_queue_affinity_is_empty(placement.queue_affinity)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter layout target slab has no device placement");
    } else {
      archive_load.device = placement.device;
      archive_load.queue_affinity = placement.queue_affinity;
    }
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_layout_submit_slab_gather(
        options, &archive_load, target_buffer, options->wait_semaphore_list,
        options->signal_semaphore_list);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(options->signal_semaphore_list,
                                 iree_status_clone(status));
  }
  iree_allocator_free(host_allocator, archive_requests);
  return status;
}

iree_status_t id4_pipeline_parameter_layout_load(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_layout_load_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set) {
  IREE_ASSERT_ARGUMENT(out_slab_set);
  *out_slab_set = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_validate_plan(plan));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_layout_validate_load_options(options));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_layout_validate_index(plan, options->index));

  const iree_host_size_t slab_count =
      id4_pipeline_plan_parameter_slab_count(plan);
  id4_pipeline_parameter_slab_load_t* target_loads = NULL;
  id4_pipeline_parameter_request_table_t* archive_request_tables = NULL;
  id4_pipeline_parameter_request_t* archive_requests = NULL;
  id4_pipeline_parameter_slab_load_t* archive_loads = NULL;
  iree_hal_semaphore_t** completion_semaphores = NULL;
  uint64_t* completion_payload_values = NULL;
  id4_pipeline_parameter_slab_set_t* slab_set = NULL;

  iree_status_t status = id4_pipeline_parameter_layout_make_loads(
      plan, host_allocator, &target_loads);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_layout_make_archive_requests(
        plan, slab_count, target_loads, host_allocator, &archive_request_tables,
        &archive_requests, &archive_loads);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_set_create_uninitialized(
        slab_count, target_loads, id4_pipeline_plan_stage_name(plan),
        options->diagnostics_sink, host_allocator, &slab_set);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, slab_count,
                                         sizeof(completion_semaphores[0]),
                                         (void**)&completion_semaphores);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, slab_count,
                                         sizeof(completion_payload_values[0]),
                                         (void**)&completion_payload_values);
  }
  if (completion_semaphores) {
    memset(completion_semaphores, 0,
           slab_count * sizeof(completion_semaphores[0]));
  }
  for (iree_host_size_t i = 0; i < slab_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_semaphore_create(
        archive_loads[i].device, archive_loads[i].queue_affinity,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &completion_semaphores[i]);
    completion_payload_values[i] = 1;
    if (!iree_status_is_ok(status)) break;
    const iree_hal_semaphore_list_t signal_list = {
        // One completion semaphore for this slab gather.
        .count = 1,
        // Slab gather completion semaphore.
        .semaphores = &completion_semaphores[i],
        // Slab gather completion payload.
        .payload_values = &completion_payload_values[i],
    };
    status = id4_pipeline_parameter_layout_submit_slab_gather(
        options, &archive_loads[i],
        id4_pipeline_parameter_slab_set_buffer_at(slab_set, i),
        options->wait_semaphore_list, signal_list);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_semaphore_list_t completion_list = {
        // One completion edge per populated slab.
        .count = slab_count,
        // Per-slab gather completion semaphores.
        .semaphores = completion_semaphores,
        // Per-slab gather completion payload values.
        .payload_values = completion_payload_values,
    };
    status = iree_hal_device_queue_barrier(
        archive_loads[0].device, archive_loads[0].queue_affinity,
        completion_list, options->signal_semaphore_list,
        IREE_HAL_EXECUTE_FLAG_NONE);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(options->signal_semaphore_list,
                                 iree_status_clone(status));
  }
  for (iree_host_size_t i = 0; i < slab_count; ++i) {
    iree_hal_semaphore_release(completion_semaphores ? completion_semaphores[i]
                                                     : NULL);
  }
  iree_allocator_free(host_allocator, completion_payload_values);
  iree_allocator_free(host_allocator, completion_semaphores);
  iree_allocator_free(host_allocator, archive_loads);
  iree_allocator_free(host_allocator, archive_requests);
  iree_allocator_free(host_allocator, archive_request_tables);
  iree_allocator_free(host_allocator, target_loads);
  if (iree_status_is_ok(status)) {
    *out_slab_set = slab_set;
  } else {
    id4_pipeline_parameter_slab_set_release(slab_set);
  }
  return status;
}
