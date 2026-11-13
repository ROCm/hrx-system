// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_VM_OPS_H_
#define LOOM_OPS_VM_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_target_fact_type_t loom_vm_target_fact_type;

enum {
  LOOM_OP_VM_TARGET = LOOM_OP_KIND(LOOM_DIALECT_VM, 0),
  LOOM_OP_VM_COUNT_ = 1,
};

// Instruction set selected by vm.target.
typedef enum loom_vm_target_kind_e {
  LOOM_VM_TARGET_KIND_CORE = 1,
  LOOM_VM_TARGET_KIND_COUNT_ = 2,
} loom_vm_target_kind_t;

// LOOM_OP_VM_TARGET: Selects the portable VM instruction set and host function ABI.
// vm.target<core> @vm
LOOM_DEFINE_ISA(loom_vm_target_isa, LOOM_OP_VM_TARGET)
LOOM_DEFINE_ATTR_SYMBOL(loom_vm_target_symbol, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vm_target_kind, 1, loom_vm_target_kind_t)
LOOM_DEFINE_ATTR_I64(loom_vm_target_execution_unit_count, 2)
enum loom_vm_target_build_flag_bits_e {
  LOOM_VM_TARGET_BUILD_FLAG_HAS_EXECUTION_UNIT_COUNT = 1u << 0,
};
typedef uint32_t loom_vm_target_build_flags_t;
iree_status_t loom_vm_target_build(
    loom_builder_t* builder,
    loom_vm_target_build_flags_t build_flags,
    loom_vm_target_kind_t kind,
    loom_symbol_ref_t symbol,
    loom_optional int64_t execution_unit_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the vm dialect.
const loom_op_vtable_t* const* loom_vm_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the vm dialect.
const loom_op_semantics_t* loom_vm_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a vm op kind, or empty metadata.
loom_op_semantics_t loom_vm_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

// Additional named attribute helpers are generated with this dialect.
#include "loom/target/arch/vm/ops/ops.inc"

#endif  // LOOM_OPS_VM_OPS_H_
