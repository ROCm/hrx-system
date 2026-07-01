// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_SUPPORT_H_
#define EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_SUPPORT_H_

#include "experimental/id4/ideogram4/session_state.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Shared state for uploading one host tensor into a planned boundary binding.
typedef struct id4_ideogram4_boundary_upload_context_t {
  // Device receiving the queued upload operations.
  iree_hal_device_t* device;
  // Queue affinity used for every upload in this context.
  iree_hal_queue_affinity_t queue_affinity;
  // Plan whose boundary tensor table is searched by name.
  const id4_pipeline_plan_t* plan;
  // Binding set whose entries are updated by name.
  const id4_pipeline_buffer_binding_set_t* boundary_bindings;
  // Semaphore chaining the upload queue operations.
  iree_hal_semaphore_t* semaphore;
  // Payload value mutated after every queued update.
  uint64_t* payload_value;
} id4_ideogram4_boundary_upload_context_t;

// Validates that |actual_size| covers the expected options structure prefix.
iree_status_t id4_ideogram4_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name);

// Validates the shape of one caller-provided HAL semaphore list.
iree_status_t id4_ideogram4_validate_semaphore_list(
    iree_hal_semaphore_list_t semaphore_list, iree_string_view_t list_name);

// Validates a list of diagnostic tap names.
iree_status_t id4_ideogram4_validate_diagnostic_tap_names(
    iree_string_view_list_t names);

// Uploads one host tensor into a named planned boundary binding.
iree_status_t id4_ideogram4_upload_boundary_tensor(
    const id4_ideogram4_boundary_upload_context_t* context,
    iree_string_view_t binding_name, const void* source_data,
    iree_host_size_t source_length,
    iree_hal_semaphore_list_t initial_wait_semaphore_list);

// Creates a one-edge semaphore list backed by caller-owned stack storage.
iree_hal_semaphore_list_t id4_ideogram4_single_semaphore_list(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_hal_semaphore_t* semaphore, uint64_t payload_value);

// Creates a two-edge semaphore list backed by caller-owned stack storage.
iree_hal_semaphore_list_t id4_ideogram4_two_semaphore_list(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_hal_semaphore_t* first_semaphore, uint64_t first_payload_value,
    iree_hal_semaphore_t* second_semaphore, uint64_t second_payload_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_IDEOGRAM4_SESSION_SUPPORT_H_
