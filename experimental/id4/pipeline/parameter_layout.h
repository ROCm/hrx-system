// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PARAMETER_LAYOUT_H_
#define EXPERIMENTAL_ID4_PIPELINE_PARAMETER_LAYOUT_H_

#include <stdint.h>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/program.h"
#include "iree/base/api.h"
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Storage role of one entry in a baked parameter layout.
typedef uint32_t id4_pipeline_parameter_layout_entry_kind_t;

// Baked parameter layout entry role values.
typedef enum id4_pipeline_parameter_layout_entry_kind_e {
  // Invalid entry role.
  ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_INVALID = 0,
  // Full logical source tensor retained for request-dependent slicing.
  ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_SOURCE = 1,
  // Immutable tensor stored in its final device execution layout.
  ID4_PIPELINE_PARAMETER_LAYOUT_ENTRY_KIND_EXECUTION = 2,
} id4_pipeline_parameter_layout_entry_kind_e;

// One durable parameter entry derived from a stage plan.
typedef struct id4_pipeline_parameter_layout_entry_t {
  // Storage role determining how this entry is populated and consumed.
  id4_pipeline_parameter_layout_entry_kind_t kind;
  // Archive key used to address the entry.
  iree_string_view_t key;
  // Physical transformation represented by the stored bytes.
  id4_pipeline_program_parameter_encoding_t encoding;
  // Scalar dtype of the stored bytes.
  id4_pipeline_tensor_dtype_t dtype;
  // Canonical physical shape of the stored tensor.
  id4_pipeline_tensor_shape_t shape;
  // Exact stored byte length.
  iree_device_size_t byte_length;
  // Required entry base alignment in bytes.
  iree_device_size_t alignment;
  // Stable fingerprint of source keys, scopes, dtypes, and shapes.
  uint64_t source_schema_fingerprint;
  // Checkpoint provider scope used to populate source-layout entries.
  iree_string_view_t source_scope;
  // Resident slab containing execution-layout bytes, or IREE_HOST_SIZE_MAX.
  iree_host_size_t parameter_slab_index;
  // Byte offset in parameter_slab_index for execution-layout entries.
  iree_device_size_t parameter_slab_offset;
} id4_pipeline_parameter_layout_entry_t;

// Aggregate storage requirements for one baked parameter layout.
typedef struct id4_pipeline_parameter_layout_statistics_t {
  // Number of unique full logical source entries retained for dynamic slicing.
  iree_host_size_t source_entry_count;
  // Sum of stored bytes across unique source entries.
  iree_device_size_t source_byte_length;
  // Number of unique immutable execution-layout entries.
  iree_host_size_t execution_entry_count;
  // Sum of stored bytes across unique execution-layout entries.
  iree_device_size_t execution_byte_length;
} id4_pipeline_parameter_layout_statistics_t;

// Options for loading resident slabs from a validated baked layout.
typedef struct id4_pipeline_parameter_layout_load_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Parameter index parsed from the same archive exposed by |provider|.
  iree_io_parameter_index_t* index;
  // Provider serving the validated archive index.
  iree_io_parameter_provider_t* provider;
  // Provider scope assigned to the baked archive.
  iree_string_view_t scope;
  // Semaphores that all direct archive gathers wait on.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled after every resident slab is populated.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Diagnostics sink for allocation and gather events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_parameter_layout_load_options_t;

// Options for populating a writable baked parameter archive.
typedef struct id4_pipeline_parameter_layout_populate_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Checkpoint provider supplying full request-dependent source entries.
  iree_io_parameter_provider_t* source_provider;
  // Writable archive index parsed from the target archive header.
  iree_io_parameter_index_t* target_index;
  // Writable provider exposing |target_index|.
  iree_io_parameter_provider_t* target_provider;
  // Provider scope assigned to the target archive.
  iree_string_view_t target_scope;
  // Prepared resident slabs containing final execution-layout entries.
  id4_pipeline_parameter_slab_set_t* parameter_slabs;
  // Maximum device-local bytes used to relay one source-layout chunk.
  iree_device_size_t staging_chunk_byte_capacity;
  // Semaphores that archive population waits on.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled after every archive entry is populated.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Diagnostics sink for archive population events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_parameter_layout_populate_options_t;

// Returns the number of unique durable entries required by |plan|.
iree_status_t id4_pipeline_parameter_layout_entry_count(
    const id4_pipeline_plan_t* plan, iree_host_size_t* out_count);

// Returns unique durable entry |index| with strings borrowed from |plan|.
iree_status_t id4_pipeline_parameter_layout_entry_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index,
    id4_pipeline_parameter_layout_entry_t* out_entry);

// Computes source and execution-layout archive storage independently.
iree_status_t id4_pipeline_parameter_layout_query_statistics(
    const id4_pipeline_plan_t* plan,
    id4_pipeline_parameter_layout_statistics_t* out_statistics);

// Adds all durable entries and their versioned physical-layout metadata to an
// IRPA archive builder. Entry payloads remain uninitialized.
iree_status_t id4_pipeline_parameter_layout_add_archive_entries(
    const id4_pipeline_plan_t* plan,
    iree_io_parameter_archive_builder_t* archive_builder);

// Verifies that |index| contains every durable entry required by |plan| with
// exactly matching physical-layout metadata and byte lengths.
iree_status_t id4_pipeline_parameter_layout_validate_index(
    const id4_pipeline_plan_t* plan, iree_io_parameter_index_t* index);

// Maps one planned parameter request to its baked archive entry while
// preserving |target_span|'s compact target range. The returned key is
// borrowed from |plan|.
iree_status_t id4_pipeline_parameter_layout_make_archive_request(
    const id4_pipeline_plan_t* plan, iree_host_size_t parameter_tensor_index,
    const id4_pipeline_parameter_request_t* planned_request,
    iree_io_parameter_span_t target_span,
    id4_pipeline_parameter_request_t* out_request);

// Asynchronously populates a writable archive from prepared execution-layout
// slabs and checkpoint source tensors. Source-layout entries are relayed in
// bounded chunks and never become model-scale resident duplicates.
iree_status_t id4_pipeline_parameter_layout_populate(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_layout_populate_options_t* options,
    iree_allocator_t host_allocator);

// Allocates final resident slabs and gathers a validated baked layout directly
// into them without running checkpoint encoders.
iree_status_t id4_pipeline_parameter_layout_load(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_layout_load_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PARAMETER_LAYOUT_H_
