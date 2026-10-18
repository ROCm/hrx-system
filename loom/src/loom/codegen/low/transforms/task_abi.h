// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Task ABI64 entry semantics with target-owned physical instruction emission.

#ifndef LOOM_CODEGEN_LOW_TRANSFORMS_TASK_ABI_H_
#define LOOM_CODEGEN_LOW_TRANSFORMS_TASK_ABI_H_

#include "loom/codegen/low/descriptors.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/types.h"
#include "loom/target/abi/task/state_layout.h"
#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared builder for one entry or query. The materializer owns insertion points
// and function lifecycle; target callbacks append physical Low instructions.
typedef struct loom_low_task_entry_builder_t {
  // Builder allocating persistent IR in the module's arena.
  loom_builder_t ir;
  // Target's unsigned 32-bit carrier for geometry, version, and success values.
  loom_type_t word_type;
  // Target's 64-bit carrier for state and binding pointers.
  loom_type_t pointer_type;
  // Source location retained by emitted entry instructions.
  loom_location_id_t location;
  // Borrowed target state, valid for the entire materializer call.
  void* target_data;
} loom_low_task_entry_builder_t;

// One admitted logical parameter's physical access. This record is constructed
// by the shared task ABI owner, not inferred by the target from IR or
// registers.
typedef struct loom_low_task_parameter_access_t {
  // Logical source type retained by the task parameter layout.
  loom_type_t source_type;
  // Admitted physical carrier of the original body parameter.
  loom_type_t carrier_type;
  // Byte offset of the constants/bindings pointer within dispatch state.
  uint32_t table_offset;
  // Byte offset of the value within its constants or bindings table.
  uint32_t value_offset;
  // Number of bytes read before any carrier widening.
  uint8_t byte_length;
  // Whether the original parameter is used. Unused slots receive a pure value
  // without reading dispatch memory until CFG cleanup removes their arguments.
  bool used;
} loom_low_task_parameter_access_t;

// Physical realization of task entry actions for one native architecture.
// Callbacks append any required instruction sequence; there is no assumption
// that a parameter load or builtin import is a single machine instruction.
typedef struct loom_low_task_entry_lowering_t {
  // Architecture identity whose concrete function versions this lowering owns.
  const loom_target_fact_type_t* fact_type;
  // Resolves target state and physical carrier types without mutating IR.
  // False means the selected representation cannot implement task entries.
  bool (*initialize)(const loom_low_descriptor_set_t* descriptors,
                     loom_low_task_entry_builder_t* builder);
  // Returns an empty string for an admitted source/carrier/width, or the
  // violated physical constraint. The shared owner emits its parameter
  // diagnostic. The initialized builder exposes concrete target vocabulary.
  iree_string_view_t (*parameter_constraint)(
      const loom_low_task_entry_builder_t* builder, loom_type_t source_type,
      loom_type_t carrier_type, uint8_t byte_length);
  // Imports one parameter or supplies its unused-slot value. table starts
  // invalid and caches this table's loaded pointer across subsequent
  // parameters.
  iree_status_t (*emit_parameter)(
      loom_low_task_entry_builder_t* builder,
      const loom_low_task_parameter_access_t* access, loom_value_id_t dispatch,
      loom_value_id_t* table, loom_value_id_t* out_value);
  // Loads the builtin from its selected state pointer and widens to the index
  // carrier. The schema row supplies the unsigned width and byte offset.
  iree_status_t (*emit_builtin)(loom_low_task_entry_builder_t* builder,
                                const loom_task_builtin_info_t* builtin,
                                loom_value_id_t state,
                                loom_value_id_t* out_value);
  // Emits the zero-valued 32-bit task dispatch success result.
  iree_status_t (*emit_success)(loom_low_task_entry_builder_t* builder,
                                loom_value_id_t* out_value);
  // Emits a query body with (maximum_version, environment) arguments. It
  // returns the library address when maximum_version >= required_version,
  // otherwise a null pointer. The shared owner creates and retains the
  // function.
  iree_status_t (*emit_query)(loom_low_task_entry_builder_t* builder,
                              loom_region_t* body,
                              loom_symbol_ref_t library_symbol,
                              uint32_t required_version);
} loom_low_task_entry_lowering_t;

// Replaces a logical kernel with the three-pointer ABI64 task entry. Parameter
// layout, predicates, body movement, return/live-in rewrites, symbol relinking,
// and concrete function-version updates have one shared owner. Other target
// families and ordinary functions are unchanged. target_data is caller-owned
// storage initialized by the target callback and borrowed only during this
// call.
iree_status_t loom_low_task_materialize_kernel(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    const loom_low_task_entry_lowering_t* lowering, void* target_data);

// Creates a version query and its concrete compiler version before source
// lowering when this target family has a task entry. Authored queries remain
// ordinary functions. Library symbols and query function contracts are shared;
// physical body emission uses the same target context as the dispatch entry.
iree_status_t loom_low_task_materialize_query(
    loom_pass_t* pass, loom_module_t* module,
    const loom_low_task_entry_lowering_t* lowering, void* target_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TRANSFORMS_TASK_ABI_H_
