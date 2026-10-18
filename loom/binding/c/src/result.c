// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "result.h"

#include <string.h>

#include "diagnostic.h"
#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "iree/base/internal/atomics.h"
#include "loomc/iree.h"
#include "source.h"

typedef struct loomc_owned_diagnostic_t {
  // Public diagnostic view returned to callers.
  loomc_diagnostic_t value;
  // Packed related entries and strings, independent of result-array growth.
  void* storage;
} loomc_owned_diagnostic_t;

typedef struct loomc_owned_artifact_t {
  // Public artifact view returned to callers.
  loomc_artifact_t value;
  // Storage for value.format.
  loomc_string_view_t format_storage;
  // Storage for value.identifier.
  loomc_string_view_t identifier_storage;
  // Retained storage for value.contents.
  loomc_byte_sequence_t* contents_storage;
} loomc_owned_artifact_t;

struct loomc_result_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;
  // Terminal status code; the public result state is derived from this code.
  uint8_t status_code;
  // Source-content retention applied while diagnostics are copied.
  uint8_t source_retention;
  // Allocator used to release this result.
  loomc_allocator_t allocator;
  // Optional borrowed sink active only while the result is mutable.
  const loom_diagnostic_sink_t* loom_diagnostic_sink;
  // Growable diagnostic array.
  loomc_owned_diagnostic_t* diagnostics;
  // Number of live diagnostics.
  loomc_host_size_t diagnostic_count;
  // Allocated diagnostic capacity.
  loomc_host_size_t diagnostic_capacity;
  // Growable artifact array.
  loomc_owned_artifact_t* artifacts;
  // Number of live artifacts.
  loomc_host_size_t artifact_count;
  // Allocated artifact capacity.
  loomc_host_size_t artifact_capacity;
};

static loomc_status_t loomc_result_grow_array(loomc_allocator_t allocator,
                                              loomc_host_size_t element_size,
                                              loomc_host_size_t live_count,
                                              loomc_host_size_t required_count,
                                              loomc_host_size_t* capacity,
                                              void** data) {
  if (required_count <= *capacity) {
    return loomc_ok_status();
  }
  loomc_host_size_t new_capacity = *capacity == 0 ? 4 : *capacity * 2;
  while (new_capacity < required_count) {
    new_capacity *= 2;
  }
  void* new_data = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, new_capacity * element_size, &new_data));
  if (*data != NULL && live_count != 0) {
    memcpy(new_data, *data, live_count * element_size);
  }
  loomc_allocator_free(allocator, *data);
  *data = new_data;
  *capacity = new_capacity;
  return loomc_ok_status();
}

static void loomc_owned_diagnostic_deinitialize(
    loomc_allocator_t allocator, loomc_owned_diagnostic_t* diagnostic) {
  loomc_source_release((loomc_source_t*)diagnostic->value.range.source);
  for (loomc_host_size_t i = 0; i < diagnostic->value.related_location_count;
       ++i) {
    loomc_source_release(
        (loomc_source_t*)diagnostic->value.related_locations[i].range.source);
  }
  loomc_allocator_free(allocator, diagnostic->storage);
  *diagnostic = (loomc_owned_diagnostic_t){0};
}

static void loomc_owned_artifact_deinitialize(
    loomc_allocator_t allocator, loomc_owned_artifact_t* artifact) {
  loomc_allocator_free(allocator, (void*)artifact->format_storage.data);
  loomc_allocator_free(allocator, (void*)artifact->identifier_storage.data);
  loomc_byte_sequence_release(artifact->contents_storage);
  *artifact = (loomc_owned_artifact_t){0};
}

static void loomc_result_destroy(loomc_result_t* result) {
  IREE_ASSERT(result->loom_diagnostic_sink == NULL,
              "diagnostic sink must not outlive result construction");
  loomc_allocator_t allocator = result->allocator;
  for (loomc_host_size_t i = 0; i < result->diagnostic_count; ++i) {
    loomc_owned_diagnostic_deinitialize(allocator, &result->diagnostics[i]);
  }
  for (loomc_host_size_t i = 0; i < result->artifact_count; ++i) {
    loomc_owned_artifact_deinitialize(allocator, &result->artifacts[i]);
  }
  loomc_allocator_free(allocator, result->diagnostics);
  loomc_allocator_free(allocator, result->artifacts);
  loomc_allocator_free(allocator, result);
}

loomc_status_t loomc_result_create(loomc_status_code_t status_code,
                                   loomc_source_retention_t source_retention,
                                   loomc_allocator_t allocator,
                                   loomc_result_t** out_result) {
  if (out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_result must not be NULL");
  }
  *out_result = NULL;
  if (source_retention != LOOMC_SOURCE_RETENTION_EXACT &&
      source_retention != LOOMC_SOURCE_RETENTION_METADATA_ONLY) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result source retention is invalid");
  }
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*result), (void**)&result));
  memset(result, 0, sizeof(*result));
  iree_atomic_ref_count_init(&result->ref_count);
  result->allocator = allocator;
  result->status_code = (uint8_t)status_code;
  result->source_retention = (uint8_t)source_retention;
  *out_result = result;
  return loomc_ok_status();
}

loomc_allocator_t loomc_result_allocator(const loomc_result_t* result) {
  IREE_ASSERT_ARGUMENT(result);
  return result->allocator;
}

loomc_source_retention_t loomc_result_source_retention(
    const loomc_result_t* result) {
  IREE_ASSERT_ARGUMENT(result);
  return (loomc_source_retention_t)result->source_retention;
}

void loomc_result_set_status_code(loomc_result_t* result,
                                  loomc_status_code_t status_code) {
  result->status_code = (uint8_t)status_code;
}

void loomc_result_set_loom_diagnostic_sink(loomc_result_t* result,
                                           const loom_diagnostic_sink_t* sink) {
  IREE_ASSERT_ARGUMENT(result);
  result->loom_diagnostic_sink = sink;
}

const loom_diagnostic_sink_t* loomc_result_loom_diagnostic_sink(
    const loomc_result_t* result) {
  IREE_ASSERT_ARGUMENT(result);
  return result->loom_diagnostic_sink;
}

// Copies a string into the preallocated diagnostic payload.
static loomc_string_view_t loomc_diagnostic_copy_string(
    loomc_string_view_t value, char** cursor) {
  if (value.size == 0) {
    return loomc_string_view_empty();
  }
  memcpy(*cursor, value.data, value.size);
  loomc_string_view_t copied = loomc_make_string_view(*cursor, value.size);
  *cursor += value.size;
  return copied;
}

static loomc_status_t loomc_result_capture_diagnostic_source(
    const loomc_result_t* result, const loomc_source_t* source,
    loomc_source_t** out_source) {
  *out_source = NULL;
  if (source == NULL) {
    return loomc_ok_status();
  }
  if (result->source_retention == LOOMC_SOURCE_RETENTION_EXACT) {
    *out_source = (loomc_source_t*)source;
    loomc_source_retain(*out_source);
    return loomc_ok_status();
  }
  return loomc_source_clone_identity(source, result->allocator, out_source);
}

static loomc_status_t loomc_result_copy_diagnostic_range(
    const loomc_result_t* result, const loomc_source_range_t* range,
    loomc_source_t* shared_source, loomc_source_range_t* out_range) {
  *out_range = *range;
  out_range->source = NULL;
  if (shared_source != NULL) {
    loomc_source_retain(shared_source);
    out_range->source = shared_source;
  } else {
    loomc_source_t* captured_source = NULL;
    LOOMC_RETURN_IF_ERROR(loomc_result_capture_diagnostic_source(
        result, range->source, &captured_source));
    out_range->source = captured_source;
  }
  return loomc_ok_status();
}

loomc_status_t loomc_result_add_diagnostic(
    loomc_result_t* result, const loomc_diagnostic_t* diagnostic) {
  if (result == NULL || diagnostic == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result and diagnostic must not be NULL");
  }
  if (diagnostic->related_location_count != 0 &&
      diagnostic->related_locations == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "diagnostic related locations have a count but no storage");
  }
  if (diagnostic->parameter_count != 0 && diagnostic->parameters == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "diagnostic parameters have a count but no storage");
  }
  LOOMC_RETURN_IF_ERROR(loomc_result_grow_array(
      result->allocator, sizeof(result->diagnostics[0]),
      result->diagnostic_count, result->diagnostic_count + 1,
      &result->diagnostic_capacity, (void**)&result->diagnostics));
  loomc_host_size_t related_size = diagnostic->related_location_count *
                                   sizeof(loomc_diagnostic_related_location_t);
  loomc_host_size_t parameter_size =
      diagnostic->parameter_count * sizeof(loomc_diagnostic_parameter_t);
  loomc_host_size_t storage_size =
      related_size + parameter_size + diagnostic->code.size +
      diagnostic->message.size + diagnostic->formatted_text.size +
      diagnostic->formatted_json.size;
  for (loomc_host_size_t i = 0; i < diagnostic->related_location_count; ++i) {
    storage_size += diagnostic->related_locations[i].label.size;
  }
  for (loomc_host_size_t i = 0; i < diagnostic->parameter_count; ++i) {
    storage_size += diagnostic->parameters[i].name.size;
    storage_size += diagnostic->parameters[i].value.size;
  }
  void* storage = NULL;
  if (storage_size) {
    LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
        result->allocator, storage_size, &storage));
  }

  loomc_diagnostic_related_location_t* related_locations = NULL;
  char* cursor = storage;
  if (diagnostic->related_location_count) {
    related_locations = storage;
    cursor += related_size;
  }
  loomc_diagnostic_parameter_t* parameters = NULL;
  if (diagnostic->parameter_count) {
    parameters = (loomc_diagnostic_parameter_t*)cursor;
    cursor += parameter_size;
  }
  loomc_owned_diagnostic_t* target =
      &result->diagnostics[result->diagnostic_count];
  *target = (loomc_owned_diagnostic_t){
      .value = *diagnostic,
      .storage = storage,
  };
  target->value.range.source = NULL;
  target->value.code = loomc_diagnostic_copy_string(diagnostic->code, &cursor);
  target->value.message =
      loomc_diagnostic_copy_string(diagnostic->message, &cursor);
  target->value.formatted_text =
      loomc_diagnostic_copy_string(diagnostic->formatted_text, &cursor);
  target->value.formatted_json =
      loomc_diagnostic_copy_string(diagnostic->formatted_json, &cursor);
  target->value.related_locations = related_locations;
  target->value.parameters = parameters;
  for (loomc_host_size_t i = 0; i < diagnostic->parameter_count; ++i) {
    parameters[i] = diagnostic->parameters[i];
    parameters[i].name =
        loomc_diagnostic_copy_string(parameters[i].name, &cursor);
    parameters[i].value =
        loomc_diagnostic_copy_string(parameters[i].value, &cursor);
  }
  loomc_status_t status = loomc_result_copy_diagnostic_range(
      result, &diagnostic->range, /*shared_source=*/NULL, &target->value.range);
  for (loomc_host_size_t i = 0; i < diagnostic->related_location_count; ++i) {
    related_locations[i] = diagnostic->related_locations[i];
    related_locations[i].range.source = NULL;
    related_locations[i].label =
        loomc_diagnostic_copy_string(related_locations[i].label, &cursor);
    loomc_source_t* shared_source = NULL;
    if (diagnostic->related_locations[i].range.source ==
        diagnostic->range.source) {
      shared_source = (loomc_source_t*)target->value.range.source;
    }
    for (loomc_host_size_t j = 0; shared_source == NULL && j < i; ++j) {
      if (diagnostic->related_locations[i].range.source ==
          diagnostic->related_locations[j].range.source) {
        shared_source = (loomc_source_t*)related_locations[j].range.source;
      }
    }
    if (loomc_status_is_ok(status)) {
      status = loomc_result_copy_diagnostic_range(
          result, &diagnostic->related_locations[i].range, shared_source,
          &related_locations[i].range);
    }
  }
  if (!loomc_status_is_ok(status)) {
    loomc_owned_diagnostic_deinitialize(result->allocator, target);
    return status;
  }
  ++result->diagnostic_count;
  return loomc_ok_status();
}

static loomc_status_t loomc_result_prepare_artifact(
    loomc_result_t* result, loomc_string_view_t format,
    loomc_string_view_t identifier, loomc_owned_artifact_t** out_artifact) {
  *out_artifact = NULL;
  if (result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_result_grow_array(
      result->allocator, sizeof(result->artifacts[0]), result->artifact_count,
      result->artifact_count + 1, &result->artifact_capacity,
      (void**)&result->artifacts));
  loomc_owned_artifact_t* target = &result->artifacts[result->artifact_count];
  *target = (loomc_owned_artifact_t){0};
  loomc_status_t status = loomc_string_view_clone(format, result->allocator,
                                                  &target->format_storage);
  if (loomc_status_is_ok(status)) {
    status = loomc_string_view_clone(identifier, result->allocator,
                                     &target->identifier_storage);
  }
  if (loomc_status_is_ok(status)) {
    *out_artifact = target;
  } else {
    loomc_owned_artifact_deinitialize(result->allocator, target);
  }
  return status;
}

static void loomc_result_commit_artifact(loomc_result_t* result,
                                         loomc_artifact_kind_t kind,
                                         loomc_owned_artifact_t* artifact,
                                         loomc_byte_sequence_t* contents) {
  artifact->contents_storage = contents;
  artifact->value = (loomc_artifact_t){
      .kind = kind,
      .format = artifact->format_storage,
      .identifier = artifact->identifier_storage,
      .contents = contents,
  };
  ++result->artifact_count;
}

loomc_status_t loomc_result_add_artifact(loomc_result_t* result,
                                         const loomc_artifact_t* artifact) {
  if (result == NULL || artifact == NULL || artifact->contents == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "result, artifact, and artifact contents must not be NULL");
  }
  loomc_owned_artifact_t* target = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_result_prepare_artifact(
      result, artifact->format, artifact->identifier, &target));
  loomc_byte_sequence_retain(artifact->contents);
  loomc_result_commit_artifact(result, artifact->kind, target,
                               artifact->contents);
  return loomc_ok_status();
}

loomc_status_t loomc_result_add_artifact_take_contents(
    loomc_result_t* result, loomc_artifact_kind_t kind,
    loomc_string_view_t format, loomc_string_view_t identifier,
    loomc_byte_span_t contents) {
  if (contents.data == NULL && contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact contents have length but no data");
  }
  loomc_owned_artifact_t* target = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_prepare_artifact(result, format, identifier, &target));

  iree_byte_span_t storage =
      iree_make_byte_span((uint8_t*)contents.data, contents.data_length);
  iree_byte_sequence_t* sequence = NULL;
  loomc_status_t status =
      loomc_status_from_iree(iree_byte_sequence_create_from_span_move(
          &storage, iree_allocator_from_loomc(result->allocator), &sequence));
  if (loomc_status_is_ok(status)) {
    loomc_result_commit_artifact(result, kind, target,
                                 loomc_byte_sequence_from_iree(sequence));
  } else {
    loomc_owned_artifact_deinitialize(result->allocator, target);
  }
  return status;
}

void loomc_result_retain(loomc_result_t* result) {
  if (result == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&result->ref_count);
}

void loomc_result_release(loomc_result_t* result) {
  if (result == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&result->ref_count) == 1) {
    loomc_result_destroy(result);
  }
}

loomc_result_state_t loomc_result_state(const loomc_result_t* result) {
  switch (loomc_result_status_code(result)) {
    case LOOMC_STATUS_OK:
      return LOOMC_RESULT_STATE_SUCCEEDED;
    case LOOMC_STATUS_CANCELLED:
      return LOOMC_RESULT_STATE_CANCELLED;
    default:
      return LOOMC_RESULT_STATE_FAILED;
  }
}

loomc_status_code_t loomc_result_status_code(const loomc_result_t* result) {
  return result ? (loomc_status_code_t)result->status_code
                : LOOMC_STATUS_INVALID_ARGUMENT;
}

bool loomc_result_succeeded(const loomc_result_t* result) {
  return loomc_result_status_code(result) == LOOMC_STATUS_OK;
}

loomc_host_size_t loomc_result_diagnostic_count(const loomc_result_t* result) {
  return result ? result->diagnostic_count : 0;
}

const loomc_diagnostic_t* loomc_result_diagnostic_at(
    const loomc_result_t* result, loomc_host_size_t index) {
  if (result == NULL || index >= result->diagnostic_count) {
    return NULL;
  }
  return &result->diagnostics[index].value;
}

loomc_host_size_t loomc_result_artifact_count(const loomc_result_t* result) {
  return result ? result->artifact_count : 0;
}

const loomc_artifact_t* loomc_result_artifact_at(const loomc_result_t* result,
                                                 loomc_host_size_t index) {
  if (result == NULL || index >= result->artifact_count) {
    return NULL;
  }
  return &result->artifacts[index].value;
}
