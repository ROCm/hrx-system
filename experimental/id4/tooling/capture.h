// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_TOOLING_CAPTURE_H_
#define EXPERIMENTAL_ID4_TOOLING_CAPTURE_H_

#include "experimental/id4/pipeline/plan.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Options for capturing executed stage tensors into fixture-comparison form.
typedef struct id4_tooling_capture_execution_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stable run identifier written into the actual manifest.
  iree_string_view_t run_id;
  // Directory that will receive manifest.json and tensor payload files.
  iree_string_view_t output_directory;
  // Executed plan whose exported boundary tensors should be captured.
  const id4_pipeline_plan_t* plan;
  // Device that can read the boundary bindings through queue copies.
  iree_hal_device_t* device;
  // Queue affinity used for readback queue copies.
  iree_hal_queue_affinity_t queue_affinity;
  // Number of boundary bindings supplied in plan boundary tensor order.
  iree_host_size_t boundary_binding_count;
  // Boundary bindings supplied in plan boundary tensor order.
  const iree_hal_buffer_binding_t* boundary_bindings;
  // Number of diagnostic tap bindings supplied in plan diagnostic tap order.
  iree_host_size_t diagnostic_tap_binding_count;
  // Diagnostic tap bindings supplied in plan diagnostic tap order.
  const iree_hal_buffer_binding_t* diagnostic_tap_bindings;
  // Semaphores the readback copies must wait on before reading captured
  // tensors.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Host allocator used for transient capture storage.
  iree_allocator_t host_allocator;
} id4_tooling_capture_execution_options_t;

// Captures all exported boundary tensors and planned diagnostic taps in
// |options->plan| into |options->output_directory| as an actual fixture
// manifest and exact tensor payloads.
iree_status_t id4_tooling_capture_execution(
    const id4_tooling_capture_execution_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_TOOLING_CAPTURE_H_
