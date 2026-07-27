// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "result.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "source.h"

typedef struct loomc_owned_diagnostic_t {
  // Public diagnostic view returned to callers.
  loomc_diagnostic_t value;
  // Storage for value.code.
  loomc_string_view_t code_storage;
  // Storage for value.message.
  loomc_string_view_t message_storage;
} loomc_owned_diagnostic_t;

typedef struct loomc_owned_artifact_t {
  // Public artifact view returned to callers.
  loomc_artifact_t value;
  // Storage for value.format.
  loomc_string_view_t format_storage;
  // Storage for value.identifier.
  loomc_string_view_t identifier_storage;
  // Storage for value.contents.
  loomc_byte_span_t contents_storage;
} loomc_owned_artifact_t;

typedef enum loomc_result_artifact_storage_mode_e {
  // Copy artifact contents into result-owned storage.
  LOOMC_RESULT_ARTIFACT_STORAGE_COPY = 0,

  // Take allocator-owned artifact contents on success.
  LOOMC_RESULT_ARTIFACT_STORAGE_TAKE = 1,
} loomc_result_artifact_storage_mode_t;

struct loomc_result_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used to release this result.
  loomc_allocator_t allocator;
  // Result state.
  loomc_result_state_t state;
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
  loomc_allocator_free(allocator, (void*)diagnostic->code_storage.data);
  loomc_allocator_free(allocator, (void*)diagnostic->message_storage.data);
  loomc_source_release((loomc_source_t*)diagnostic->value.range.source);
  *diagnostic = (loomc_owned_diagnostic_t){0};
}

static void loomc_owned_artifact_deinitialize(
    loomc_allocator_t allocator, loomc_owned_artifact_t* artifact) {
  loomc_allocator_free(allocator, (void*)artifact->format_storage.data);
  loomc_allocator_free(allocator, (void*)artifact->identifier_storage.data);
  loomc_allocator_free(allocator, (void*)artifact->contents_storage.data);
  *artifact = (loomc_owned_artifact_t){0};
}

static void loomc_result_destroy(loomc_result_t* result) {
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

loomc_status_t loomc_result_create(loomc_result_state_t state,
                                   loomc_allocator_t allocator,
                                   loomc_result_t** out_result) {
  if (out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_result must not be NULL");
  }
  *out_result = NULL;
  if (state > LOOMC_RESULT_STATE_CANCELLED) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result state is invalid");
  }
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*result), (void**)&result));
  memset(result, 0, sizeof(*result));
  iree_atomic_ref_count_init(&result->ref_count);
  result->allocator = allocator;
  result->state = state;
  *out_result = result;
  return loomc_ok_status();
}

loomc_allocator_t loomc_result_allocator(const loomc_result_t* result) {
  IREE_ASSERT_ARGUMENT(result);
  return result->allocator;
}

loomc_status_t loomc_result_set_state(loomc_result_t* result,
                                      loomc_result_state_t state) {
  if (result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result must not be NULL");
  }
  if (state > LOOMC_RESULT_STATE_CANCELLED) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result state is invalid");
  }
  result->state = state;
  return loomc_ok_status();
}

loomc_status_t loomc_result_add_diagnostic(
    loomc_result_t* result, const loomc_diagnostic_t* diagnostic) {
  if (result == NULL || diagnostic == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result and diagnostic must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_result_grow_array(
      result->allocator, sizeof(result->diagnostics[0]),
      result->diagnostic_count, result->diagnostic_count + 1,
      &result->diagnostic_capacity, (void**)&result->diagnostics));
  loomc_owned_diagnostic_t* target =
      &result->diagnostics[result->diagnostic_count];
  *target = (loomc_owned_diagnostic_t){0};
  loomc_status_t status = loomc_string_view_clone(
      diagnostic->code, result->allocator, &target->code_storage);
  if (loomc_status_is_ok(status)) {
    status = loomc_string_view_clone(diagnostic->message, result->allocator,
                                     &target->message_storage);
  }
  if (loomc_status_is_ok(status)) {
    target->value = *diagnostic;
    target->value.code = target->code_storage;
    target->value.message = target->message_storage;
    if (target->value.range.source) {
      loomc_source_retain((loomc_source_t*)target->value.range.source);
    }
    ++result->diagnostic_count;
  } else {
    loomc_owned_diagnostic_deinitialize(result->allocator, target);
  }
  return status;
}

static loomc_status_t loomc_result_add_artifact_storage(
    loomc_result_t* result, const loomc_artifact_t* artifact,
    loomc_result_artifact_storage_mode_t storage_mode) {
  if (result == NULL || artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "result and artifact must not be NULL");
  }
  if (artifact->contents.data == NULL && artifact->contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact contents have length but no data");
  }
  LOOMC_RETURN_IF_ERROR(loomc_result_grow_array(
      result->allocator, sizeof(result->artifacts[0]), result->artifact_count,
      result->artifact_count + 1, &result->artifact_capacity,
      (void**)&result->artifacts));
  loomc_owned_artifact_t* target = &result->artifacts[result->artifact_count];
  *target = (loomc_owned_artifact_t){0};
  loomc_status_t status = loomc_string_view_clone(
      artifact->format, result->allocator, &target->format_storage);
  if (loomc_status_is_ok(status)) {
    status = loomc_string_view_clone(artifact->identifier, result->allocator,
                                     &target->identifier_storage);
  }
  if (loomc_status_is_ok(status) &&
      storage_mode == LOOMC_RESULT_ARTIFACT_STORAGE_COPY &&
      artifact->contents.data_length != 0) {
    uint8_t* contents = NULL;
    status = loomc_allocator_malloc_uninitialized(
        result->allocator, artifact->contents.data_length, (void**)&contents);
    if (loomc_status_is_ok(status)) {
      memcpy(contents, artifact->contents.data, artifact->contents.data_length);
      target->contents_storage =
          loomc_make_byte_span(contents, artifact->contents.data_length);
    }
  }
  if (loomc_status_is_ok(status) &&
      storage_mode == LOOMC_RESULT_ARTIFACT_STORAGE_TAKE) {
    target->contents_storage = artifact->contents;
  }
  if (loomc_status_is_ok(status)) {
    target->value = *artifact;
    target->value.format = target->format_storage;
    target->value.identifier = target->identifier_storage;
    target->value.contents = target->contents_storage;
    ++result->artifact_count;
  } else {
    loomc_owned_artifact_deinitialize(result->allocator, target);
  }
  return status;
}

loomc_status_t loomc_result_add_artifact(loomc_result_t* result,
                                         const loomc_artifact_t* artifact) {
  return loomc_result_add_artifact_storage(result, artifact,
                                           LOOMC_RESULT_ARTIFACT_STORAGE_COPY);
}

loomc_status_t loomc_result_add_artifact_take_contents(
    loomc_result_t* result, loomc_artifact_kind_t kind,
    loomc_string_view_t format, loomc_string_view_t identifier,
    loomc_byte_span_t contents) {
  loomc_artifact_t artifact = {
      .kind = kind,
      .format = format,
      .identifier = identifier,
      .contents = contents,
  };
  return loomc_result_add_artifact_storage(result, &artifact,
                                           LOOMC_RESULT_ARTIFACT_STORAGE_TAKE);
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
  return result ? result->state : LOOMC_RESULT_STATE_FAILED;
}

bool loomc_result_succeeded(const loomc_result_t* result) {
  return result && result->state == LOOMC_RESULT_STATE_SUCCEEDED;
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
