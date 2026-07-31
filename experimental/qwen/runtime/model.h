// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_RUNTIME_MODEL_H_
#define EXPERIMENTAL_QWEN_RUNTIME_MODEL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loomc/sanitizer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_io_parameter_index_t iree_io_parameter_index_t;
typedef struct iree_io_parameter_provider_t iree_io_parameter_provider_t;
typedef struct qwen_loom_jit_t qwen_loom_jit_t;
typedef struct qwen_parameter_layout_t qwen_parameter_layout_t;

// Fixed Qwen3-30B-A3B model state.
//
// A model retains its single-device HAL group and parameter provider. Its raw
// GGUF payloads and immutable model auxiliaries occupy one resident device
// allocation. Model construction is asynchronous: the returned object may be
// used for host-side program preparation immediately, while issued device work
// waits on the model's internal readiness timeline.
typedef struct qwen_model_t qwen_model_t;

// Indexed parameter storage consumed by qwen_model_load.
typedef struct qwen_parameter_source_t {
  // Fixed parameter index borrowed for synchronous schema validation.
  iree_io_parameter_index_t* index;
  // Provider retained by the loaded model.
  iree_io_parameter_provider_t* provider;
  // Provider scope containing the indexed Qwen parameters.
  iree_string_view_t scope;
} qwen_parameter_source_t;

// Options controlling fixed model placement.
typedef struct qwen_model_options_t {
  // Size of this structure in bytes.
  iree_host_size_t structure_size;
  // Optional extension chain; must be NULL when no extensions are used.
  const void* next;
  // Single-device group retained by the model.
  iree_hal_device_group_t* device_group;
  // Device ordinal within |device_group|; must be zero.
  iree_host_size_t device_index;
  // Queue affinity used for allocation, gathering, and execution.
  iree_hal_queue_affinity_t queue_affinity;
  // Number of task workers used for independent Loom compiler jobs.
  iree_host_size_t jit_worker_count;
  // Loom sanitizer assertions compiled into prepared model executables.
  loomc_sanitizer_checks_t sanitizer_checks;
} qwen_model_options_t;

// Fixed model allocation statistics.
typedef struct qwen_model_statistics_t {
  // Number of original encoded GGUF parameter bytes.
  iree_device_size_t encoded_parameter_bytes;
  // Padding inserted between encoded parameter payloads.
  iree_device_size_t parameter_padding_bytes;
  // Immutable non-GGUF auxiliary bytes appended to the model allocation.
  iree_device_size_t immutable_auxiliary_bytes;
  // Complete resident allocation size including parameters and auxiliaries.
  iree_device_size_t allocation_bytes;
} qwen_model_statistics_t;

// Initializes |out_options| with single-device model defaults.
IREE_API_EXPORT void qwen_model_options_initialize(
    qwen_model_options_t* out_options);

// Validates and asynchronously loads the fixed Qwen parameter set.
//
// Schema validation and slab layout are completed before this call returns.
// The resident allocation waits on |wait_semaphore_list|, all 579 encoded
// parameters are gathered as one provider group, and immutable auxiliaries are
// uploaded before |signal_semaphore_list| is published.
//
// The returned model retains the provider and device group. |source->index| and
// |source->scope| are borrowed only for this call.
IREE_API_EXPORT iree_status_t qwen_model_load(
    const qwen_model_options_t* options, const qwen_parameter_source_t* source,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_allocator_t host_allocator, qwen_model_t** out_model);

// Retains |model| for the caller.
IREE_API_EXPORT void qwen_model_retain(qwen_model_t* model);

// Releases |model| and its resident allocation.
//
// Queue submissions retain their referenced storage until completion. The
// allocation uses indeterminate lifetime and is destroyed with its final
// retained buffer reference.
IREE_API_EXPORT void qwen_model_release(qwen_model_t* model);

// Returns the single-device group retained by |model|.
//
// The returned group is borrowed for the lifetime of |model|.
IREE_API_EXPORT iree_hal_device_group_t* qwen_model_device_group(
    const qwen_model_t* model);

// Returns the sole device owned by |model|'s group.
//
// The returned device is borrowed for the lifetime of |model|.
IREE_API_EXPORT iree_hal_device_t* qwen_model_device(const qwen_model_t* model);

// Returns the queue affinity selected when |model| was loaded.
IREE_API_EXPORT iree_hal_queue_affinity_t
qwen_model_queue_affinity(const qwen_model_t* model);

// Returns the model-owned readiness timepoint.
//
// The returned one-element list is borrowed for the lifetime of |model|.
// Runtime-issued work includes this wait automatically.
IREE_API_EXPORT iree_hal_semaphore_list_t
qwen_model_ready_semaphore_list(const qwen_model_t* model);

// Returns the shared device-specific Loom JIT owned by |model|.
//
// The returned JIT is borrowed for the lifetime of |model|. Every program
// prepared against the model uses this cache.
IREE_API_EXPORT qwen_loom_jit_t* qwen_model_loom_jit(const qwen_model_t* model);

// Returns the resident parameter allocation borrowed from |model|.
IREE_API_EXPORT iree_hal_buffer_t* qwen_model_parameter_buffer(
    const qwen_model_t* model);

// Returns the immutable typed parameter layout owned by |model|.
//
// Runtime implementation files include parameters.h to inspect the returned
// private layout.
IREE_API_EXPORT const qwen_parameter_layout_t* qwen_model_parameter_layout(
    const qwen_model_t* model);

// Returns resident allocation statistics.
IREE_API_EXPORT qwen_model_statistics_t
qwen_model_statistics(const qwen_model_t* model);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_MODEL_H_
