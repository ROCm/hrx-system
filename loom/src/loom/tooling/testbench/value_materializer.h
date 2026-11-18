// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runtime value materialization for check testbench cases.
//
// This layer turns target-free check plans into typed runtime values that
// execution, oracle, comparison, and fixture-writing layers can consume. It is
// intentionally separate from testbench.h so pure discovery remains free of
// HAL and file-format dependencies.

#ifndef LOOM_TOOLING_TESTBENCH_VALUE_MATERIALIZER_H_
#define LOOM_TOOLING_TESTBENCH_VALUE_MATERIALIZER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/stream.h"
#include "iree/tooling/value_io.h"
#include "loom/tooling/testbench/scenario/entropy.h"
#include "loom/tooling/testbench/testbench.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_testbench_value_slot_flag_bits_e {
  // Slot currently owns a materialized value.
  LOOM_TESTBENCH_VALUE_SLOT_FLAG_ASSIGNED = 1u << 0,
} loom_testbench_value_slot_flag_bits_t;
typedef uint8_t loom_testbench_value_slot_flags_t;

typedef uint8_t loom_testbench_value_kind_t;
enum loom_testbench_value_kind_e {
  // No materialized value is present.
  LOOM_TESTBENCH_VALUE_KIND_NONE = 0,
  // Scalar value stored in |scalar|.
  LOOM_TESTBENCH_VALUE_KIND_SCALAR = 1,
  // HAL buffer binding stored in |buffer|.
  LOOM_TESTBENCH_VALUE_KIND_BUFFER = 2,
  // Immutable check.entropy identity stored in |entropy|.
  LOOM_TESTBENCH_VALUE_KIND_ENTROPY = 3,
};

// Canonical recipe-relative identity for a materialized buffer reference.
//
// Physical buffers differ between target and oracle realizations. A traceable
// reference instead names the authored allocation value and its accessible
// byte range within that allocation.
typedef struct loom_testbench_buffer_reference_t {
  // True when the remaining fields identify an authored allocation.
  bool is_traceable;
  // SSA value defining the root allocation in the scenario recipe.
  loom_value_id_t allocation_value_id;
  // Byte origin within the logical root allocation.
  iree_device_size_t byte_offset;
  // Accessible byte extent beginning at |byte_offset|.
  iree_device_size_t byte_length;
} loom_testbench_buffer_reference_t;

typedef struct loom_testbench_value_t {
  // Active payload discriminator.
  loom_testbench_value_kind_t kind;
  union {
    // Scalar payload used for check parameters and literals.
    iree_tooling_value_t scalar;
    // Retained HAL buffer payload used for shaped values.
    iree_tooling_buffer_binding_t buffer;
    // Immutable deterministic entropy identity.
    loom_testbench_entropy_t entropy;
  };
  // Recipe-relative identity when |kind| is BUFFER and the reference can be
  // traced to an authored allocation.
  loom_testbench_buffer_reference_t buffer_reference;
} loom_testbench_value_t;

typedef struct loom_testbench_value_slot_t {
  // Source Loom SSA value represented by this slot.
  loom_value_id_t value_id;
  // Slot state flags.
  loom_testbench_value_slot_flags_t flags;
  // Materialized value owned by this slot when ASSIGNED is set.
  loom_testbench_value_t value;
} loom_testbench_value_slot_t;

// Scope-local table of materialized runtime values.
typedef struct loom_testbench_value_table_t {
  // Module that owns the value IDs stored in |slots|.
  const loom_module_t* module;
  // Host allocator that owns |slots|.
  iree_allocator_t host_allocator;
  // Sorted slots keyed by source Loom SSA value ID.
  loom_testbench_value_slot_t* slots;
  // Allocated entry count in |slots|.
  iree_host_size_t slot_capacity;
  // Number of entries in |slots|.
  iree_host_size_t slot_count;
} loom_testbench_value_table_t;

// Logical buffer footprint reachable from a set of materialized values.
//
// Retained storage and reachable binding ranges are distinct facts. A small
// alias retains its complete authored root allocation but exposes only its
// referenced byte range to the invocation.
typedef struct loom_testbench_buffer_footprint_t {
  // Number of distinct authored root allocations retained by the values.
  iree_host_size_t storage_allocation_count;
  // Sum of complete retained root-allocation payload byte lengths.
  uint64_t storage_byte_length;
  // Number of disjoint reachable byte ranges after merging by root allocation.
  iree_host_size_t binding_range_count;
  // Sum of reachable byte ranges after merging aliases and repeated values.
  uint64_t binding_range_byte_length;
} loom_testbench_buffer_footprint_t;

// Returns true when both footprints describe the same storage and ranges.
bool loom_testbench_buffer_footprint_equal(
    const loom_testbench_buffer_footprint_t* lhs,
    const loom_testbench_buffer_footprint_t* rhs);

// Opens a stream for a check.file.* path.
typedef iree_status_t(IREE_API_PTR* loom_testbench_file_open_fn_t)(
    void* user_data, iree_string_view_t path, iree_io_stream_t** out_stream);

// File-open callback with caller-owned user data.
typedef struct loom_testbench_file_open_callback_t {
  // Callback function, or NULL when this file direction is unsupported.
  loom_testbench_file_open_fn_t fn;
  // Opaque caller-owned context passed to |fn|.
  void* user_data;
} loom_testbench_file_open_callback_t;

// Runtime resources used while materializing values.
typedef struct loom_testbench_value_materializer_options_t {
  // Borrowed pool owning final shaped generated and file-backed values.
  iree_hal_pool_t* target_pool;
  // Borrowed host-writable pool used when |target_pool| is not host-writable.
  iree_hal_pool_t* staging_pool;
  // Borrowed queue used with |staging_pool| for non-host-writable targets.
  iree_hal_queue_t* transfer_queue;
  // Minimum byte alignment of shaped value allocations.
  iree_device_size_t min_alignment;
  // Callback used to open check.file.read.* paths.
  loom_testbench_file_open_callback_t open_read_file;
  // Callback used to open check.file.write.* paths.
  loom_testbench_file_open_callback_t open_write_file;
  // Host allocator used for transient streams and diagnostics.
  iree_allocator_t host_allocator;
} loom_testbench_value_materializer_options_t;

// Initializes materializer options with no file callbacks and the system host
// allocator.
void loom_testbench_value_materializer_options_initialize(
    loom_testbench_value_materializer_options_t* out_options);

// Initializes an empty value table for the values reachable from |case_plan|.
iree_status_t loom_testbench_value_table_initialize_case(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_allocator_t host_allocator, loom_testbench_value_table_t* out_table);

// Initializes an empty table for one scenario's configuration recipe.
iree_status_t loom_testbench_value_table_initialize_scenario_configuration(
    const loom_module_t* module,
    const loom_testbench_scenario_plan_t* scenario_plan,
    iree_allocator_t host_allocator, loom_testbench_value_table_t* out_table);

// Initializes an empty table for one scenario trial domain, including values
// captured from its enclosing configuration recipe.
iree_status_t loom_testbench_value_table_initialize_scenario_trial(
    const loom_module_t* module,
    const loom_testbench_scenario_plan_t* scenario_plan,
    const loom_testbench_trial_plan_t* trial_plan,
    iree_allocator_t host_allocator, loom_testbench_value_table_t* out_table);

// Releases all values and storage owned by |table|.
void loom_testbench_value_table_deinitialize(
    loom_testbench_value_table_t* table);

// Clears all assigned values while retaining allocated table storage.
void loom_testbench_value_table_reset(loom_testbench_value_table_t* table);

// Returns true when |value_id| has a materialized value in |table|.
bool loom_testbench_value_table_contains(
    const loom_testbench_value_table_t* table, loom_value_id_t value_id);

// Releases any resources owned by |value| and resets it to empty.
void loom_testbench_value_deinitialize(loom_testbench_value_t* value);

// Returns true when |value| is a scalar.
bool loom_testbench_value_is_scalar(const loom_testbench_value_t* value);

// Returns true when |value| carries a HAL buffer.
bool loom_testbench_value_is_buffer(const loom_testbench_value_t* value);

// Returns true when |value| is an immutable entropy identity.
bool loom_testbench_value_is_entropy(const loom_testbench_value_t* value);

// Returns the HAL buffer view carried by |value|, or NULL when none exists.
iree_hal_buffer_view_t* loom_testbench_value_buffer_view(
    const loom_testbench_value_t* value);

// Assigns a canonical recipe-relative identity to a materialized buffer.
void loom_testbench_value_set_buffer_reference(
    loom_value_id_t allocation_value_id, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, loom_testbench_value_t* value);

// Retains |source| into |out_value|. The caller must deinitialize |out_value|.
void loom_testbench_value_retain(const loom_testbench_value_t* source,
                                 loom_testbench_value_t* out_value);

// Interprets |value| as an integer scalar and stores it as signed i64.
iree_status_t loom_testbench_value_as_i64(const loom_testbench_value_t* value,
                                          int64_t* out_value);

// Moves a freshly-created HAL buffer view into |out_value| as a buffer binding.
//
// On success |out_value| owns |buffer_view| plus a retain of its underlying HAL
// buffer and must be deinitialized by the caller. On failure |buffer_view|
// remains caller-owned.
iree_status_t loom_testbench_value_set_buffer_view_move(
    iree_hal_buffer_view_t* buffer_view, loom_testbench_value_t* out_value);

// Looks up |value_id| and returns a borrowed value valid until the table is
// reset or deinitialized.
iree_status_t loom_testbench_value_table_lookup_borrow(
    const loom_testbench_value_table_t* table, loom_value_id_t value_id,
    const loom_testbench_value_t** out_value);

// Looks up |value_id| and retains any contained resources for the caller. The
// caller must deinitialize |out_value|.
iree_status_t loom_testbench_value_table_lookup_retain(
    const loom_testbench_value_table_t* table, loom_value_id_t value_id,
    loom_testbench_value_t* out_value);

// Resolves the logical buffer footprint reachable from |value_ids|.
//
// Scalar and entropy values contribute no storage. Every buffer must retain
// its authored allocation reference, and that root allocation must still be
// materialized in |table|. Root storage is counted once, while overlapping and
// adjacent binding ranges are unioned within each root. This must be queried
// before the table is reset.
iree_status_t loom_testbench_value_table_resolve_buffer_footprint(
    const loom_testbench_value_table_t* table, iree_host_size_t value_id_count,
    const loom_value_id_t* value_ids,
    loom_testbench_buffer_footprint_t* out_footprint);

// Assigns |value| to |value_id|, replacing any previous value. On success,
// ownership moves into |table| and |value| is reset to empty.
iree_status_t loom_testbench_value_table_assign_move(
    loom_testbench_value_table_t* table, loom_value_id_t value_id,
    loom_testbench_value_t* value);

// Moves invocation results into |table| and restores shaped bindings from their
// source types. Scalar results are available before resolving dynamic
// dimensions. Providers may return raw storage spans; shaping retains their
// storage and logical allocation references without copying bytes. Results
// moved before a failure remain owned by |table|; the caller owns any remaining
// |results|.
iree_status_t loom_testbench_value_table_assign_results(
    loom_testbench_value_table_t* table, iree_host_size_t result_count,
    const loom_value_id_t* result_value_ids, loom_testbench_value_t* results);

// Materializes |source_count| source plans in source order into |table|.
iree_status_t loom_testbench_materialize_value_sources(
    const loom_testbench_value_materializer_options_t* options,
    const loom_testbench_value_source_plan_t* sources,
    iree_host_size_t source_count, loom_testbench_value_table_t* table);

// Materializes parameters and source values for one planned concrete sample.
iree_status_t loom_testbench_materialize_case_sample(
    const loom_testbench_value_materializer_options_t* options,
    const loom_testbench_case_plan_t* case_plan,
    loom_testbench_case_sample_t sample, loom_testbench_value_table_t* table);

// Writes one already-completed value to |path|, borrowing its storage only for
// this call. Applies no output policy; the caller selects the observation and
// destination before invoking this operation. Flush errors propagate to caller.
iree_status_t loom_testbench_write_file(
    const loom_testbench_value_materializer_options_t* options,
    const loom_testbench_file_write_plan_t* file_write,
    const loom_testbench_value_table_t* table, iree_string_view_t path);

// Writes planned case file outputs. ON_FAILURE outputs are written only when
// |case_failed| is true; ALWAYS outputs are written in both states.
iree_status_t loom_testbench_write_case_files(
    const loom_testbench_value_materializer_options_t* options,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_value_table_t* table, bool case_failed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_VALUE_MATERIALIZER_H_
