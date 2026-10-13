// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_VM_FUNCTION_PLAN_H_
#define LOOM_TARGET_ARCH_VM_FUNCTION_PLAN_H_

#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "iree/vm/bytecode/wire/module.h"
#include "loom/codegen/low/allocation/call.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/vm/native_call_plan.h"
#include "loom/target/function_version.h"

#ifdef __cplusplus
extern "C" {
#endif

// Exact logical signature retained while building the program plan.
typedef struct loom_vm_function_signature_t {
  // Physical argument/result counts. Program planning assigns descriptor_base.
  iree_vm_bytecode_v0_signature_row_t row;
  // Module-owned argument descriptors followed by result descriptors, in
  // source order. Metadata planning finalizes them before function emission.
  iree_vm_bytecode_v0_signature_descriptor_row_t* fields;
  // Argument/result register positions finalized with the logical fields.
  loom_low_allocation_abi_location_t* registers;
} loom_vm_function_signature_t;

// Compiler-owned callable binding retained while building a VM program plan.
typedef struct loom_vm_program_callable_t {
  // Borrowed function definition or import declaration and signature values.
  loom_func_like_t function;
  // Concrete compiler products retained by the shared compilation pipeline.
  const loom_target_function_version_t* function_version;
  // Source-ordered entry argument IDs in the module.
  const loom_value_id_t* arguments;
  // Source-ordered signature result IDs in the module.
  loom_value_slice_t results;
  // Public name, or empty for an internal function.
  iree_string_view_t export_name;
  // Runtime import identity; empty for a local function.
  struct {
    // Module namespace owning the imported callable.
    iree_string_view_t module_name;
    // Export name within that module, independent of the local symbol name.
    iree_string_view_t symbol_name;
  } import;
  // Local-function or flat-import ordinal in the emitted image.
  uint16_t ordinal;
  // Source-ordered logical argument count.
  uint16_t argument_count;
  // Canonical callable ordinal assigned by signature sorting.
  uint16_t callable_ordinal;
  // Wire control.call target kind, shared by local and imported calls.
  uint8_t target_kind;
  // Exact logical fields and their physical argument/result bank counts.
  loom_vm_function_signature_t signature;
  // Shared allocation contract borrowing the classified signature registers.
  loom_low_call_contract_t call_contract;
  // Value and reference prefixes overwritten by argument/result transport.
  loom_low_call_clobber_t call_clobbers[2];
  // Structured synchronous native ABI, or NULL for ordinary VM callables.
  loom_vm_native_import_t* native;
} loom_vm_program_callable_t;

// Compiler-owned state shared across function planning. This representation
// never crosses into the target binary writer.
typedef struct loom_vm_program_build_t {
  // Arena-owned local and imported callable records in source symbol order.
  loom_vm_program_callable_t* values;
  // Direct symbol-indexed bindings for calls within the VM target contract.
  // Open declarations without an executable binding remain NULL until the
  // selected caller's schedule is validated while building the program plan.
  loom_vm_program_callable_t** bindings_by_symbol;
  // Number of records in |values|, bounded by the module symbol ID space.
  uint32_t count;
  // Number of local definitions, retaining their collected ordinal space.
  uint32_t definition_count;
  // Physical native callsites retained across per-function scratch resets.
  loom_vm_native_call_plan_t native_calls;
  // Read-only payloads retained for the program's data section.
  struct {
    // Symbol-indexed data ordinals; UINT16_MAX marks an unreferenced payload.
    uint16_t* ordinals_by_symbol;
    // Module symbol IDs in declaration order for numeric Low operands.
    const loom_symbol_id_t* symbols;
    // Number of entries in |symbols|.
    uint32_t symbol_count;
    // Borrowed definitions in first-use order, with symbol_count capacity.
    const loom_op_t** values;
    // Number of definitions in |values|.
    uint32_t count;
    // Maximum block alignment, at least the image's eight-byte alignment.
    uint32_t alignment;
  } rodata;
} loom_vm_program_build_t;

// Schedules and allocates one planned VM function with the common frame
// builder, then appends its final instruction stream to |stream|. |out_row|
// preserves its caller-initialized callable ordinal and bytecode offset and
// receives the final byte length and frame high waters. Branches target block
// markers using signed word offsets patched before this function returns.
// Constants retain canonical Low form until planning chooses an equivalent
// compact encoding of the complete value cell. Block offsets and byte lengths
// come from the produced stream, not nominal instruction sizes.
//
// The shared allocator owns edge and packet moves, including cycle
// temporaries. Common allocation repair materializes value and reference
// spills; the scheduler's storage layout owns their relative byte offsets.
// Value cells use stack storage and owned references use private storage.
// Outgoing overflow packets occupy the canonical offset-zero prefix; ordinary
// locals follow, with live caller values allocated outside the ABI prefixes.
// Overflow entry loads execute once before the branchable body; overflow
// returns store exact value cells before direct register permutations.
// Reference overflow uses an independent local-ref prefix, followed by private
// reference cells. Returning a ref through overflow preserves its source in one
// local-ref slot until all aliased results are published.
//
// |program| supplies callable signatures and symbol ordinals; data operands
// append their referenced payload once to its read-only data plan. All frame
// and instruction-planning scratch belongs to |arena| and can be reset when
// this call returns; only the stream bytes, scalar row fields, and
// program-owned rodata references and native callsites survive. Structured
// frame rejection returns OK with |out_accepted| false. Infrastructure and
// stream failures return a status and also leave |out_accepted| false.
iree_status_t loom_vm_function_plan_write(
    loom_module_t* module, loom_func_like_t function,
    const loom_target_function_version_t* function_version,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_vm_function_signature_t* signature, uint16_t function_ordinal,
    loom_vm_program_build_t* program, iree_arena_allocator_t* arena,
    iree_io_stream_t* stream, bool* out_accepted,
    iree_vm_bytecode_v0_function_row_t* out_row);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_FUNCTION_PLAN_H_
