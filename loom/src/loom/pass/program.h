// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiled pass pipeline instruction program.
//
// This is the cold translation boundary between pass.pipeline IR and the pass
// interpreter. Compilation verifies the selected pipeline, statically expands
// pass.call, resolves pass.run descriptors, decodes immutable pass options, and
// emits compact instructions whose nested bodies are represented by explicit
// instruction ranges. The interpreter can execute the program repeatedly
// without walking pipeline IR.

#ifndef LOOM_PASS_PROGRAM_H_
#define LOOM_PASS_PROGRAM_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/attribute.h"
#include "loom/ir/ir.h"
#include "loom/pass/invocation.h"
#include "loom/pass/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_pass_program_instruction_kind_e {
  LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE = 0,
  LOOM_PASS_PROGRAM_INSTRUCTION_FOR_EACH_SYMBOL = 1,
  LOOM_PASS_PROGRAM_INSTRUCTION_WHERE = 2,
  LOOM_PASS_PROGRAM_INSTRUCTION_REPEAT = 3,
  LOOM_PASS_PROGRAM_INSTRUCTION_FAIL = 4,
  LOOM_PASS_PROGRAM_INSTRUCTION_HALT = 5,
} loom_pass_program_instruction_kind_t;

typedef enum loom_pass_program_symbol_snapshot_kind_e {
  // Snapshot function-like symbols with bodies in module symbol table order.
  LOOM_PASS_PROGRAM_SYMBOL_SNAPSHOT_FUNCTIONS_BY_SYMBOL_ID = 0,
} loom_pass_program_symbol_snapshot_kind_t;

typedef struct loom_pass_program_attr_t loom_pass_program_attr_t;

typedef struct loom_pass_program_attr_value_t {
  // Attribute kind, using loom_attr_kind_t values.
  uint8_t kind;
  // Number of entries in array, dictionary, or predicate-list payloads.
  uint16_t count;
  // Attribute payload copied into program-owned storage where needed.
  union {
    // Signed integer payload for LOOM_ATTR_I64.
    int64_t i64_value;
    // Floating-point payload for LOOM_ATTR_F64.
    double f64_value;
    // Program-owned string payload for LOOM_ATTR_STRING.
    iree_string_view_t string_value;
    // Boolean payload for LOOM_ATTR_BOOL.
    bool bool_value;
    // Enum ordinal payload for LOOM_ATTR_ENUM.
    uint8_t enum_value;
    // Program-owned array payload for LOOM_ATTR_I64_ARRAY.
    const int64_t* i64_array;
    // Source-module symbol reference payload for LOOM_ATTR_SYMBOL.
    loom_symbol_ref_t symbol_value;
    // Source-module type table index for LOOM_ATTR_TYPE.
    loom_type_id_t type_id;
    // Source-module encoding table index for LOOM_ATTR_ENCODING.
    uint16_t encoding_id;
    // Program-owned predicate payload for LOOM_ATTR_PREDICATE_LIST.
    const loom_predicate_t* predicates;
    // Program-owned dictionary payload for LOOM_ATTR_DICT.
    const loom_pass_program_attr_t* dictionary_attrs;
  };
} loom_pass_program_attr_value_t;

struct loom_pass_program_attr_t {
  // Program-owned attribute name spelling.
  iree_string_view_t name;
  // Program-owned attribute value.
  loom_pass_program_attr_value_t value;
};

typedef struct loom_pass_program_attr_list_t {
  // Program-owned attribute entries in source dictionary order.
  const loom_pass_program_attr_t* attrs;
  // Number of entries in attrs.
  iree_host_size_t attr_count;
} loom_pass_program_attr_list_t;

typedef struct loom_pass_program_source_t {
  // Source op that produced this instruction.
  const loom_op_t* op;
  // pass.pipeline op whose body contributed this instruction.
  const loom_op_t* pipeline_op;
  // Arena-owned expansion stack of pass.call ops leading to this instruction.
  const loom_op_t* const* call_stack;
  // Number of entries in call_stack.
  iree_host_size_t call_stack_count;
} loom_pass_program_source_t;

typedef struct loom_pass_program_invoke_t {
  // Resolved pass descriptor from the active registry.
  const loom_pass_descriptor_t* descriptor;
  // Static pass metadata returned by descriptor->info().
  const loom_pass_info_t* info;
  // Arena-owned decoded immutable options for this invocation.
  const loom_pass_decoded_options_t* decoded_options;
  // Module-pass callback when info->kind is LOOM_PASS_MODULE.
  loom_module_pass_fn_t module_run;
  // Function-pass callback when info->kind is LOOM_PASS_FUNCTION.
  loom_function_pass_fn_t function_run;
} loom_pass_program_invoke_t;

typedef struct loom_pass_program_for_each_symbol_t {
  // Symbol kind this instruction snapshots and iterates.
  loom_pass_kind_t symbol_kind;
  // Deterministic snapshot plan used before entering the body.
  loom_pass_program_symbol_snapshot_kind_t snapshot_kind;
  // First instruction in the nested body.
  iree_host_size_t body_start;
  // One-past-last instruction in the nested body.
  iree_host_size_t body_end;
} loom_pass_program_for_each_symbol_t;

typedef struct loom_pass_program_where_t {
  // Program-owned predicate key.
  iree_string_view_t predicate;
  // Program-owned predicate attributes.
  loom_pass_program_attr_list_t attrs;
  // First instruction in the guarded body.
  iree_host_size_t body_start;
  // One-past-last instruction in the guarded body.
  iree_host_size_t body_end;
} loom_pass_program_where_t;

typedef struct loom_pass_program_repeat_t {
  // Repeat mode, using loom_pass_repeat_mode_t values.
  uint8_t mode;
  // Fixed iteration count when mode is LOOM_PASS_REPEAT_MODE_FIXED.
  int64_t count;
  // Upper bound when mode is LOOM_PASS_REPEAT_MODE_UNTIL_CONVERGED.
  int64_t max_iterations;
  // First instruction in the repeated body.
  iree_host_size_t body_start;
  // One-past-last instruction in the repeated body.
  iree_host_size_t body_end;
} loom_pass_program_repeat_t;

typedef struct loom_pass_program_message_t {
  // Program-owned message text.
  iree_string_view_t message;
} loom_pass_program_message_t;

typedef struct loom_pass_program_instruction_t {
  // Instruction opcode.
  loom_pass_program_instruction_kind_t kind;
  // Anchor kind active when this instruction executes.
  loom_pass_kind_t anchor_kind;
  // Source provenance for diagnostics and reproducers.
  loom_pass_program_source_t source;
  // Opcode-specific payload.
  union {
    // Payload for LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE.
    loom_pass_program_invoke_t invoke;
    // Payload for LOOM_PASS_PROGRAM_INSTRUCTION_FOR_EACH_SYMBOL.
    loom_pass_program_for_each_symbol_t for_each_symbol;
    // Payload for LOOM_PASS_PROGRAM_INSTRUCTION_WHERE.
    loom_pass_program_where_t where;
    // Payload for LOOM_PASS_PROGRAM_INSTRUCTION_REPEAT.
    loom_pass_program_repeat_t repeat;
    // Payload for LOOM_PASS_PROGRAM_INSTRUCTION_FAIL/HALT.
    loom_pass_program_message_t message;
  };
} loom_pass_program_instruction_t;

typedef struct loom_pass_program_compile_options_t {
  // Registry used to resolve pass.run keys. Required.
  const loom_pass_registry_t* registry;
  // Typed execution environment capabilities.
  loom_pass_environment_t environment;
  // Optional provider for pass.where predicates outside the core built-ins.
  loom_pass_predicate_provider_t predicate_provider;
} loom_pass_program_compile_options_t;

typedef struct loom_pass_program_t {
  // Source module containing the compiled pass.pipeline IR.
  const loom_module_t* source_module;
  // Source pass.pipeline op selected for this program.
  const loom_op_t* pipeline_op;
  // Root anchor kind selected by pipeline_op.
  loom_pass_kind_t root_kind;
  // Compiled instruction array.
  loom_pass_program_instruction_t* instructions;
  // Number of populated instructions.
  iree_host_size_t instruction_count;
  // Allocated instruction capacity.
  iree_host_size_t instruction_capacity;
  // Block pool that backs arena.
  iree_arena_block_pool_t* block_pool;
  // Arena owning instructions and decoded immutable payloads.
  iree_arena_allocator_t arena;
} loom_pass_program_t;

// Compiles one verified pass.pipeline op into an instruction program.
iree_status_t loom_pass_program_compile_pipeline(
    const loom_module_t* module, const loom_op_t* pipeline_op,
    const loom_pass_program_compile_options_t* options,
    iree_arena_block_pool_t* block_pool, loom_pass_program_t* out_program);

// Releases program-owned storage. Does not release the source module.
void loom_pass_program_deinitialize(loom_pass_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_PROGRAM_H_
