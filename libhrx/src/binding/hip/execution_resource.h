// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_EXECUTION_RESOURCE_H_
#define LIBHRX_SRC_BINDING_HIP_EXECUTION_RESOURCE_H_

#include "binding/hip/api.h"
#include "common/execution_resource.h"

typedef struct iree_hal_streaming_device_t iree_hal_streaming_device_t;

#ifdef __cplusplus
extern "C" {
#endif

// Creates a copyable HIP SM resource backed by an immutable device-owned set.
//
// |resources| uses family-local HAL execution-resource ordinals and is
// interned in |device|. The public HIP counts are derived from the raw
// execution units covered by the selected resources. |out_resource| is
// unchanged on failure.
iree_status_t iree_hip_execution_resource_create_sm(
    iree_hal_streaming_device_t* device,
    const iree_hal_queue_family_t* queue_family,
    iree_hal_queue_execution_resource_list_t resources, unsigned int flags,
    hipDevResource* out_resource);

// Interns the exact SM resource set encoded by a HIP CU mask.
//
// HIP mask bits address family-local raw execution units while HAL queues
// select indivisible canonical resources. Every selected resource must be
// selected in full and the resulting set must satisfy the queue family's
// resource-group constraints. Bits beyond the family's raw execution-unit
// count are ignored. An effectively empty mask selects the complete family.
// The returned set is borrowed from |device| and |out_set| is unchanged on
// failure.
iree_status_t iree_hip_execution_resource_intern_sm_cu_mask(
    iree_hal_streaming_device_t* device,
    const iree_hal_queue_family_t* queue_family,
    iree_host_size_t mask_word_count, const uint32_t* mask,
    const iree_hal_streaming_execution_resource_set_t** out_set);

// Writes the HIP CU mask for an exact SM resource set.
//
// |resources| contains family-local canonical resource ordinals and may use
// the empty-list shorthand for the complete family. |mask_word_count| must
// cover every raw execution unit in |queue_family|. Extra output words are
// cleared and |out_mask| is unchanged on failure.
iree_status_t iree_hip_execution_resource_write_sm_cu_mask(
    const iree_hal_queue_family_t* queue_family,
    iree_hal_queue_execution_resource_list_t resources,
    iree_host_size_t mask_word_count, uint32_t* out_mask);

// Resolves and validates |resource| against an explicit device incarnation.
// Returns the immutable borrowed set on success. |out_set| is unchanged on
// failure.
hipError_t iree_hip_execution_resource_resolve_sm_for_device(
    const hipDevResource* resource, iree_hal_streaming_device_t* device,
    const iree_hal_streaming_execution_resource_set_t** out_set);

// Resolves and validates |resource| against the live device registry. The
// returned device and set are borrowed and remain valid while HIP is
// initialized. Outputs are unchanged on failure.
hipError_t iree_hip_execution_resource_resolve_sm(
    const hipDevResource* resource, iree_hal_streaming_device_t** out_device,
    const iree_hal_streaming_execution_resource_set_t** out_set);

// Splits resolved |input_set| into equal-size disjoint exact SM resources.
//
// |input| must be the resource value already resolved to |device| and
// |input_set|. A NULL |out_resources| performs discovery and writes the maximum
// possible group count to |inout_group_count|. Otherwise |inout_group_count|
// supplies output capacity and receives the number produced. When
// |out_remainder| is non-NULL the partition count is reduced as needed so a
// nonempty remainder is also an exact usable set. All outputs are unchanged on
// failure.
hipError_t iree_hip_execution_resource_split_sm_by_count(
    iree_hal_streaming_device_t* device,
    const iree_hal_streaming_execution_resource_set_t* input_set,
    const hipDevResource* input, unsigned int flags, unsigned int minimum_count,
    unsigned int* inout_group_count, hipDevResource* out_resources,
    hipDevResource* out_remainder);

// Splits resolved |input_set| into structured disjoint exact SM resources.
//
// |group_parameters| supplies one requested partition shape per output. Zero
// counts request ordered automatic sizing and automatically-sized backfill
// groups may absorb otherwise unassigned resources. The parameter array is
// normalized in place only after the complete split succeeds. A NULL
// |out_resources| performs a dry run and permits automatically-sized groups to
// discover zero resources. |out_remainder| is still populated when requested.
// All outputs are unchanged on failure.
hipError_t iree_hip_execution_resource_split_sm(
    iree_hal_streaming_device_t* device,
    const iree_hal_streaming_execution_resource_set_t* input_set,
    const hipDevResource* input, unsigned int group_count, unsigned int flags,
    hipDevSmResourceGroupParams* group_parameters,
    hipDevResource* out_resources, hipDevResource* out_remainder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_HIP_EXECUTION_RESOURCE_H_
