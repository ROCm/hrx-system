// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runtime value materialization for check testbench cases.
//
// This layer turns target-free check case plans into typed runtime values that
// execution, oracle, comparison, and fixture-writing layers can consume. It is
// intentionally separate from testbench.h so pure case discovery remains free
// of HAL and file-format dependencies.

#ifndef LOOM_TOOLING_TESTBENCH_VALUE_MATERIALIZER_H_
#define LOOM_TOOLING_TESTBENCH_VALUE_MATERIALIZER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/stream.h"
#include "iree/tooling/value_io.h"
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
};

typedef struct loom_testbench_value_t {
  // Active payload discriminator.
  loom_testbench_value_kind_t kind;
  union {
    // Scalar payload used for check parameters and literals.
    iree_tooling_value_t scalar;
    // Retained HAL buffer payload used for shaped values.
    iree_tooling_buffer_binding_t buffer;
  };
} loom_testbench_value_t;

typedef struct loom_testbench_value_slot_t {
  // Source Loom SSA value represented by this slot.
  loom_value_id_t value_id;
  // Slot state flags.
  loom_testbench_value_slot_flags_t flags;
  // Materialized value owned by this slot when ASSIGNED is set.
  loom_testbench_value_t value;
} loom_testbench_value_slot_t;

// Case-local table of materialized runtime values.
typedef struct loom_testbench_value_table_t {
  // Module that owns the value IDs stored in |slots|.
  const loom_module_t* module;
  // Case plan whose reachable values define |slots|.
  const loom_testbench_case_plan_t* case_plan;
  // Host allocator that owns |slots|.
  iree_allocator_t host_allocator;
  // Sorted slots keyed by source Loom SSA value ID.
  loom_testbench_value_slot_t* slots;
  // Allocated entry count in |slots|.
  iree_host_size_t slot_capacity;
  // Number of entries in |slots|.
  iree_host_size_t slot_count;
} loom_testbench_value_table_t;

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

// Runtime dependencies used while materializing values.
typedef struct loom_testbench_value_materializer_options_t {
  // Optional HAL device used when generated contents require a host-to-device
  // transfer. Heap allocators can materialize host-visible values with NULL.
  iree_hal_device_t* device;
  // HAL allocator used for shaped generated and file-backed values.
  iree_hal_allocator_t* device_allocator;
  // Optional buffer placement for generated and file-backed shaped values. When
  // zero-initialized, generated values use the materializer's default
  // device-local placement.
  iree_hal_buffer_params_t buffer_params;
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
iree_status_t loom_testbench_value_table_initialize(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
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

// Returns the HAL buffer view carried by |value|, or NULL when none exists.
iree_hal_buffer_view_t* loom_testbench_value_buffer_view(
    const loom_testbench_value_t* value);

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

// Assigns |value| to |value_id|, replacing any previous value. On success,
// ownership moves into |table| and |value| is reset to empty.
iree_status_t loom_testbench_value_table_assign_move(
    loom_testbench_value_table_t* table, loom_value_id_t value_id,
    loom_testbench_value_t* value);

// Materializes parameters and source values for one concrete case sample.
iree_status_t loom_testbench_materialize_case_sample(
    const loom_testbench_value_materializer_options_t* options,
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, loom_testbench_value_table_t* table);

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
