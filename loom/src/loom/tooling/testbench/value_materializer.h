// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runtime value materialization for check testbench cases.
//
// This layer turns target-free check case plans into typed VM/HAL values that
// execution, oracle, comparison, and fixture-writing layers can consume. It is
// intentionally separate from testbench.h so pure case discovery remains free
// of HAL, VM, and file-format dependencies.

#ifndef LOOM_TOOLING_TESTBENCH_VALUE_MATERIALIZER_H_
#define LOOM_TOOLING_TESTBENCH_VALUE_MATERIALIZER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/stream.h"
#include "iree/vm/api.h"
#include "loom/tooling/testbench/testbench.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_testbench_value_slot_flag_bits_e {
  // Slot currently owns a materialized variant.
  LOOM_TESTBENCH_VALUE_SLOT_FLAG_ASSIGNED = 1u << 0,
} loom_testbench_value_slot_flag_bits_t;
typedef uint8_t loom_testbench_value_slot_flags_t;

typedef struct loom_testbench_value_slot_t {
  // Source Loom SSA value represented by this slot.
  loom_value_id_t value_id;
  // Slot state flags.
  loom_testbench_value_slot_flags_t flags;
  // Materialized VM/HAL value owned by this slot when ASSIGNED is set.
  iree_vm_variant_t variant;
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

// Looks up |value_id| and returns a borrowed variant valid until the table is
// reset or deinitialized.
iree_status_t loom_testbench_value_table_lookup_borrow(
    const loom_testbench_value_table_t* table, loom_value_id_t value_id,
    const iree_vm_variant_t** out_variant);

// Looks up |value_id| and retains any contained ref for the caller. The caller
// must reset |out_variant| with iree_vm_variant_reset.
iree_status_t loom_testbench_value_table_lookup_retain(
    const loom_testbench_value_table_t* table, loom_value_id_t value_id,
    iree_vm_variant_t* out_variant);

// Assigns |variant| to |value_id|, replacing any previous value. On success,
// ownership moves into |table| and |variant| is reset to empty.
iree_status_t loom_testbench_value_table_assign_move(
    loom_testbench_value_table_t* table, loom_value_id_t value_id,
    iree_vm_variant_t* variant);

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
