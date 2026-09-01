// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_MODULE_H_
#define IREE_VM_MODULE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/function.h"
#include "iree/vm/ref.h"
#include "iree/vm/scalar.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_vm_frame_t iree_vm_frame_t;
typedef struct iree_vm_invocation_t iree_vm_invocation_t;
typedef struct iree_vm_linked_module_t iree_vm_linked_module_t;
typedef struct iree_vm_module_t iree_vm_module_t;
typedef struct iree_vm_module_vtable_t iree_vm_module_vtable_t;

//===----------------------------------------------------------------------===//
// Module Declarations
//===----------------------------------------------------------------------===//

// One four-byte machine signature leaf.
typedef uint16_t iree_vm_module_signature_type_kind_t;
enum iree_vm_module_signature_type_kind_e {
  IREE_VM_MODULE_SIGNATURE_TYPE_KIND_INVALID = 0x0000,
  // Values 0x0001 through 0x00FF match iree_vm_scalar_type_t.
  IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF = 0x0100,
  IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION = 0x0200,
};

typedef struct iree_vm_module_signature_type_t {
  // Scalar ID, REF, or FUNCTION.
  iree_vm_module_signature_type_kind_t kind;
  // Module-local ref/callable ordinal, or zero for a scalar.
  uint16_t type_ordinal;
} iree_vm_module_signature_type_t;

static_assert(sizeof(iree_vm_module_signature_type_t) == 4,
              "module signature types must remain four bytes");

typedef struct iree_vm_module_signature_type_span_t {
  // Stable source-ordered machine signature types.
  const iree_vm_module_signature_type_t* data;
  // Number of signature types in |data|.
  iree_host_size_t count;
} iree_vm_module_signature_type_span_t;

typedef struct iree_vm_module_signature_t {
  // Source-ordered machine arguments.
  iree_vm_module_signature_type_span_t arguments;
  // Source-ordered machine results.
  iree_vm_module_signature_type_span_t results;
} iree_vm_module_signature_t;

enum iree_vm_module_import_flag_bits_e {
  IREE_VM_MODULE_IMPORT_FLAG_NONE = 0u,
  // Absence of the target is permitted during program linking.
  IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL = 1u << 0,
};
typedef uint32_t iree_vm_module_import_flags_t;

enum iree_vm_module_function_flag_bits_e {
  IREE_VM_MODULE_FUNCTION_FLAG_NONE = 0u,
  // The function may suspend before completing.
  IREE_VM_MODULE_FUNCTION_FLAG_MAY_YIELD = 1u << 0,
};
typedef uint32_t iree_vm_module_function_flags_t;

enum iree_vm_callable_type_flag_bits_e {
  IREE_VM_CALLABLE_TYPE_FLAG_NONE = 0u,
  // Callables of this type are permitted to suspend.
  IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD = 1u << 0,
};
typedef uint32_t iree_vm_callable_type_flags_t;

// Transient implementation query result for one import group.
typedef struct iree_vm_module_import_group_t {
  // Exact target module name.
  iree_string_view_t target_module_name;
  // First flat import ordinal in the group.
  iree_host_size_t first_import_ordinal;
  // Nonzero number of imports in the group.
  iree_host_size_t import_count;
} iree_vm_module_import_group_t;

// Transient implementation query result for one import declaration.
typedef struct iree_vm_module_import_declaration_t {
  // Exact target module name.
  iree_string_view_t target_module_name;
  // Exact target export name.
  iree_string_view_t target_export_name;
  // Module-local callable type defining the expected contract.
  iree_host_size_t callable_type_ordinal;
  // Import behavior flags.
  iree_vm_module_import_flags_t flags;
  // Stable metadata entry count for this declaration.
  iree_host_size_t metadata_count;
} iree_vm_module_import_declaration_t;

// Transient implementation query result for one export declaration.
typedef struct iree_vm_module_export_declaration_t {
  // Exact public export name.
  iree_string_view_t export_name;
  // Module-local callable type defining the public contract.
  iree_host_size_t callable_type_ordinal;
  // Private module-local execution target.
  iree_host_size_t function_ordinal;
  // Stable metadata entry count for this declaration.
  iree_host_size_t metadata_count;
} iree_vm_module_export_declaration_t;

// Transient implementation query result for one structural callable type.
// Module callable-type tables are unique and strictly ordered by nesting depth,
// structural signature, then flags. Structural signatures compare argument
// count and source-ordered argument types followed by result count and
// source-ordered result types. Types compare by kind, then ref namespace and
// local name or earlier callable ordinal. Nested function types always name
// earlier rows.
typedef struct iree_vm_module_callable_type_declaration_t {
  // Exact structural signature.
  iree_vm_module_signature_t signature;
  // Permitted callable behavior.
  iree_vm_callable_type_flags_t flags;
  // Maximum nested callable depth, with zero identifying a leaf signature.
  uint16_t nesting_depth;
  // Reserved zero bits for future callable attributes.
  uint16_t reserved;
} iree_vm_module_callable_type_declaration_t;

//===----------------------------------------------------------------------===//
// Typed Metadata
//===----------------------------------------------------------------------===//

// Stable metadata value types. IDs are append-only. Zero is invalid; unknown
// nonzero IDs remain valid opaque byte spans. BOOL is one canonical 0/1 byte.
// I64, U64, and F64 are exactly eight little-endian bytes and may be unaligned.
// UTF8 is validated UTF-8 and may contain embedded NUL bytes. BYTES and unknown
// types are opaque.
enum iree_vm_metadata_value_type_e {
  IREE_VM_METADATA_VALUE_TYPE_INVALID = 0u,
  IREE_VM_METADATA_VALUE_TYPE_BOOL = 1u,
  IREE_VM_METADATA_VALUE_TYPE_I64 = 2u,
  IREE_VM_METADATA_VALUE_TYPE_U64 = 3u,
  IREE_VM_METADATA_VALUE_TYPE_F64 = 4u,
  IREE_VM_METADATA_VALUE_TYPE_UTF8 = 5u,
  IREE_VM_METADATA_VALUE_TYPE_BYTES = 6u,
};
typedef uint32_t iree_vm_metadata_value_type_t;

typedef struct iree_vm_metadata_value_t {
  // Stable value encoding.
  iree_vm_metadata_value_type_t type;
  // Stable module-lifetime bytes.
  iree_const_byte_span_t data;
} iree_vm_metadata_value_t;

typedef struct iree_vm_metadata_entry_t {
  // Stable nonempty UTF-8 key without embedded NUL bytes. Entries in each
  // module/import/export scope are strictly increasing by exact key bytes and
  // therefore unique.
  iree_string_view_t key;
  // Stable typed value.
  iree_vm_metadata_value_t value;
} iree_vm_metadata_entry_t;

//===----------------------------------------------------------------------===//
// Public Module Reflection
//===----------------------------------------------------------------------===//

// Borrowed identity of one public import declaration.
typedef struct iree_vm_import_t {
  // Module owning the declaration.
  const iree_vm_module_t* module;
  // Host-side flat import ordinal.
  iree_host_size_t ordinal;
} iree_vm_import_t;

// Borrowed identity of one public export declaration.
typedef struct iree_vm_export_t {
  // Module owning the declaration.
  const iree_vm_module_t* module;
  // Host-side export ordinal.
  iree_host_size_t ordinal;
} iree_vm_export_t;

// Borrowed identity of one structural callable type.
typedef struct iree_vm_callable_type_t {
  // Module owning the structural type.
  const iree_vm_module_t* module;
  // Host-side callable-type ordinal.
  iree_host_size_t ordinal;
} iree_vm_callable_type_t;

enum iree_vm_signature_type_kind_e {
  IREE_VM_SIGNATURE_TYPE_KIND_INVALID = 0u,
  IREE_VM_SIGNATURE_TYPE_KIND_SCALAR = 1u,
  IREE_VM_SIGNATURE_TYPE_KIND_REF = 2u,
  IREE_VM_SIGNATURE_TYPE_KIND_FUNCTION = 3u,
};
typedef uint32_t iree_vm_signature_type_kind_t;

// Resolved public reflection type.
typedef struct iree_vm_signature_type_t {
  // Active union member.
  iree_vm_signature_type_kind_t kind;
  // Resolved type payload selected by |kind|.
  union {
    // Exact scalar type.
    iree_vm_scalar_type_t scalar;
    // Canonical exact ref descriptor.
    iree_vm_ref_type_t ref;
    // Reachable module-bound callable type.
    iree_vm_callable_type_t callable;
  } value;
} iree_vm_signature_type_t;

typedef struct iree_vm_signature_type_span_t {
  // Contiguous resolved types in caller-owned query storage.
  const iree_vm_signature_type_t* data;
  // Number of resolved types in |data|.
  iree_host_size_t count;
} iree_vm_signature_type_span_t;

typedef struct iree_vm_signature_field_t {
  // Required resolved VM type.
  iree_vm_signature_type_t type;
  // Source-level field name anchored here, or an empty view when unavailable.
  iree_string_view_t name;
  // Source-level authored type anchored here, or empty when unavailable.
  // Multi-field aggregates anchor both strings at their first machine field.
  iree_string_view_t authored_type;
} iree_vm_signature_field_t;

typedef struct iree_vm_signature_field_span_t {
  // Contiguous fields in caller-owned query storage.
  const iree_vm_signature_field_t* data;
  // Number of fields in |data|.
  iree_host_size_t count;
} iree_vm_signature_field_span_t;

typedef struct iree_vm_signature_field_storage_t {
  // Writable fields in caller-owned query storage.
  iree_vm_signature_field_t* data;
  // Actual available field count.
  iree_host_size_t count;
} iree_vm_signature_field_storage_t;

typedef struct iree_vm_import_target_t {
  // Exact target module name.
  iree_string_view_t module_name;
  // Exact target export name.
  iree_string_view_t export_name;
} iree_vm_import_target_t;

typedef struct iree_vm_import_description_t {
  // Exact import target.
  iree_vm_import_target_t target;
  // Import behavior flags.
  iree_vm_module_import_flags_t flags;
  // Expected callable behavior.
  iree_vm_callable_type_flags_t callable_flags;
  // Source-ordered machine argument fields.
  iree_vm_signature_field_span_t arguments;
  // Source-ordered machine result fields.
  iree_vm_signature_field_span_t results;
  // Declaration documentation, or an empty view when unavailable.
  iree_string_view_t documentation;
  // Complete authored function type, or an empty view when unavailable.
  iree_string_view_t authored_type;
} iree_vm_import_description_t;

typedef struct iree_vm_export_description_t {
  // Exact public alias name.
  iree_string_view_t name;
  // Actual public callable behavior.
  iree_vm_callable_type_flags_t callable_flags;
  // Source-ordered machine argument fields.
  iree_vm_signature_field_span_t arguments;
  // Source-ordered machine result fields.
  iree_vm_signature_field_span_t results;
  // Declaration documentation, or an empty view when unavailable.
  iree_string_view_t documentation;
  // Complete authored function type, or an empty view when unavailable.
  iree_string_view_t authored_type;
} iree_vm_export_description_t;

typedef struct iree_vm_callable_type_description_t {
  // Permitted callable behavior.
  iree_vm_callable_type_flags_t flags;
  // Source-ordered resolved argument types.
  iree_vm_signature_type_span_t arguments;
  // Source-ordered resolved result types.
  iree_vm_signature_type_span_t results;
} iree_vm_callable_type_description_t;

enum iree_vm_module_declaration_kind_e {
  IREE_VM_MODULE_DECLARATION_KIND_INVALID = 0u,
  IREE_VM_MODULE_DECLARATION_KIND_IMPORT = 1u,
  IREE_VM_MODULE_DECLARATION_KIND_EXPORT = 2u,
};
typedef uint32_t iree_vm_module_declaration_kind_t;

typedef struct iree_vm_module_declaration_t {
  // Import or export.
  iree_vm_module_declaration_kind_t kind;
  // Flat declaration ordinal.
  iree_host_size_t ordinal;
} iree_vm_module_declaration_t;

typedef struct iree_vm_module_presentation_query_t {
  // Declaration whose optional presentation is requested.
  iree_vm_module_declaration_t declaration;
  // Complete machine argument-then-result fields, or empty for a size probe.
  iree_vm_signature_field_storage_t fields;
  // Max-aligned transient tail, or empty for a size probe.
  iree_byte_span_t transient_storage;
} iree_vm_module_presentation_query_t;

typedef struct iree_vm_module_presentation_t {
  // Exact transient-tail bytes required after max-alignment padding.
  iree_host_size_t required_transient_storage_size;
  // Declaration documentation, or an empty view when unavailable.
  iree_string_view_t documentation;
  // Complete authored function type, or an empty view when unavailable.
  iree_string_view_t authored_type;
} iree_vm_module_presentation_t;

enum iree_vm_module_metadata_scope_kind_e {
  IREE_VM_MODULE_METADATA_SCOPE_KIND_INVALID = 0u,
  IREE_VM_MODULE_METADATA_SCOPE_KIND_MODULE = 1u,
  IREE_VM_MODULE_METADATA_SCOPE_KIND_IMPORT = 2u,
  IREE_VM_MODULE_METADATA_SCOPE_KIND_EXPORT = 3u,
};
typedef uint32_t iree_vm_module_metadata_scope_kind_t;

typedef struct iree_vm_module_metadata_scope_t {
  // Module, import, or export.
  iree_vm_module_metadata_scope_kind_t kind;
  // Import/export ordinal, or zero for the module scope.
  iree_host_size_t ordinal;
} iree_vm_module_metadata_scope_t;

typedef struct iree_vm_module_metadata_query_t {
  // Selected public metadata scope.
  iree_vm_module_metadata_scope_t scope;
  // Valid entry ordinal within that scope.
  iree_host_size_t ordinal;
} iree_vm_module_metadata_query_t;

//===----------------------------------------------------------------------===//
// Fixed Module Descriptor
//===----------------------------------------------------------------------===//

typedef struct iree_vm_module_counts_t {
  // Private executable-function domain.
  iree_host_size_t function_count;
  // Structural callable-type domain.
  iree_host_size_t callable_type_count;
  // Sorted import-group domain.
  iree_host_size_t import_group_count;
  // Flat import declaration domain.
  iree_host_size_t import_count;
  // Public export declaration domain.
  iree_host_size_t export_count;
  // Module-scope metadata entry count.
  iree_host_size_t metadata_count;
} iree_vm_module_counts_t;

enum iree_vm_module_flag_bits_e {
  IREE_VM_MODULE_FLAG_NONE = 0u,
  // The module has executable canonical ref types and may join a program.
  IREE_VM_MODULE_FLAG_LINKABLE = 1u << 0,
};
typedef uint32_t iree_vm_module_flags_t;

// Fixed immutable facts read directly by common VM code. The module
// implementation owns and keeps this descriptor live and immutable.
typedef struct iree_vm_module_descriptor_t {
  // Nonempty exact module link name.
  iree_string_view_t name;
  // Generic module capabilities.
  iree_vm_module_flags_t flags;
  // Flat canonical type handles in module-local ordinal order.
  iree_vm_ref_type_span_t ref_types;
  // Fixed declaration counts.
  iree_vm_module_counts_t counts;
  // Max-aligned opaque bytes required in every process. Common code assigns
  // one stable slice and never interprets its contents. Lifecycle callbacks
  // are still visited when this size is zero.
  iree_host_size_t process_storage_size;
} iree_vm_module_descriptor_t;

//===----------------------------------------------------------------------===//
// Module Execution ABI
//===----------------------------------------------------------------------===//

enum iree_vm_execution_outcome_e {
  IREE_VM_EXECUTION_OUTCOME_COMPLETED = 0u,
  IREE_VM_EXECUTION_OUTCOME_SUSPENDED = 1u,
};
typedef uint32_t iree_vm_execution_outcome_t;

enum iree_vm_cancel_reason_e {
  IREE_VM_CANCEL_REASON_NONE = 0u,
  IREE_VM_CANCEL_REASON_CANCELLED = 1u,
  IREE_VM_CANCEL_REASON_DEADLINE_EXCEEDED = 2u,
};
typedef uint32_t iree_vm_cancel_reason_t;

// No-fail level-triggered host wake callback.
typedef void(IREE_API_PTR* iree_vm_invocation_wake_fn_t)(void* user_data);

// Host wake callback copied for one active invocation operation.
typedef struct iree_vm_invocation_wake_callback_t {
  // Optional callback that makes the owning host request runnable.
  iree_vm_invocation_wake_fn_t function;
  // Caller-owned context satisfying the invocation retirement contract.
  void* user_data;
} iree_vm_invocation_wake_callback_t;

// Transient execution view supplied to one module callback.
typedef struct iree_vm_module_execution_t {
  // Active invocation and composite frame stack.
  iree_vm_invocation_t* invocation;
  // Immutable program-specific linkage for this module slot.
  const iree_vm_linked_module_t* linked_module;
  // Stable opaque process slice, null exactly when its declared size is zero.
  void* process_storage;
} iree_vm_module_execution_t;

// Correlated private facts for materializing one local function value.
typedef struct iree_vm_module_local_function_t {
  // Module-local executable function ordinal.
  uint16_t function_ordinal;
  // Module-local callable type defining the structural contract.
  uint16_t callable_type_ordinal;
  // Actual function behavior flags.
  iree_vm_module_function_flags_t flags;
} iree_vm_module_local_function_t;

static_assert(sizeof(iree_vm_module_local_function_t) == 8,
              "local function descriptors must remain eight bytes");

// Number of directly addressed registers in each physical call-packet bank.
// Higher logical ordinals use the corresponding overflow bank.
enum { IREE_VM_CALL_DIRECT_REGISTER_COUNT = 16 };

typedef struct iree_vm_call_value_arguments_t {
  // Read-only value arguments below IREE_VM_CALL_DIRECT_REGISTER_COUNT.
  const uint64_t* direct;
  // Read-only value arguments at or above the direct-register count.
  const uint64_t* overflow;
} iree_vm_call_value_arguments_t;

typedef struct iree_vm_call_ref_arguments_t {
  // Ref arguments below the direct-register count, mutable for explicit moves.
  iree_vm_ref_t* direct;
  // Remaining ref arguments, mutable only for explicit moves.
  iree_vm_ref_t* overflow;
} iree_vm_call_ref_arguments_t;

typedef struct iree_vm_call_value_results_t {
  // Write-only value results below the direct-register count.
  uint64_t* direct;
  // Write-only value results at or above the direct-register count.
  uint64_t* overflow;
} iree_vm_call_value_results_t;

typedef struct iree_vm_call_ref_results_t {
  // Ref results below the direct-register count.
  iree_vm_ref_t* direct;
  // Ref results at or above the direct-register count.
  iree_vm_ref_t* overflow;
} iree_vm_call_ref_results_t;

typedef struct iree_vm_call_function_arguments_t {
  // Read-only function arguments below the direct-register count.
  const iree_vm_function_ref_t* direct;
  // Read-only function arguments at or above the direct-register count.
  const iree_vm_function_ref_t* overflow;
} iree_vm_call_function_arguments_t;

typedef struct iree_vm_call_function_results_t {
  // Write-only function results below the direct-register count.
  iree_vm_function_ref_t* direct;
  // Write-only function results at or above the direct-register count.
  iree_vm_function_ref_t* overflow;
} iree_vm_call_function_results_t;

// Canonical implementer-only physical function call packet.
//
// The exact function signature derives every direct and overflow count. Host
// invocation uses source-ordered variants and never constructs this packet.
// The packet object itself is callback-local, but every nonnull bank base
// remains stable and accessible until this logical function completes or is
// unwound. A yielding implementation copies the bases it needs into its frame.
// Root result banks are invocation-owned staging and never alias the host's
// per-drive public result span.
typedef struct iree_vm_call_packet_t {
  // Physical value argument banks.
  iree_vm_call_value_arguments_t value_arguments;
  // Physical ref argument banks.
  iree_vm_call_ref_arguments_t ref_arguments;
  // Physical value result banks.
  iree_vm_call_value_results_t value_results;
  // Physical ref result banks.
  iree_vm_call_ref_results_t ref_results;
  // Physical function argument banks.
  iree_vm_call_function_arguments_t function_arguments;
  // Physical function result banks.
  iree_vm_call_function_results_t function_results;
} iree_vm_call_packet_t;

// Loads exact bits from one valid physical value-argument ordinal.
static inline uint64_t iree_vm_call_value_argument_load(
    const iree_vm_call_packet_t* call, uint16_t value_ordinal) {
  return value_ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
             ? call->value_arguments.direct[value_ordinal]
             : call->value_arguments
                   .overflow[value_ordinal -
                             IREE_VM_CALL_DIRECT_REGISTER_COUNT];
}

// Stores exact bits to one valid physical value-result ordinal.
static inline void iree_vm_call_value_result_store(
    const iree_vm_call_packet_t* call, uint16_t value_ordinal, uint64_t value) {
  uint64_t* slot =
      value_ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
          ? &call->value_results.direct[value_ordinal]
          : &call->value_results
                 .overflow[value_ordinal - IREE_VM_CALL_DIRECT_REGISTER_COUNT];
  *slot = value;
}

// Loads one complete valid function argument.
static inline iree_vm_function_ref_t iree_vm_call_function_argument_load(
    const iree_vm_call_packet_t* call, uint16_t function_ordinal) {
  return function_ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
             ? call->function_arguments.direct[function_ordinal]
             : call->function_arguments
                   .overflow[function_ordinal -
                             IREE_VM_CALL_DIRECT_REGISTER_COUNT];
}

// Stores one complete valid function result.
static inline void iree_vm_call_function_result_store(
    const iree_vm_call_packet_t* call, uint16_t function_ordinal,
    iree_vm_function_ref_t function_ref) {
  iree_vm_function_ref_t* slot =
      function_ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
          ? &call->function_results.direct[function_ordinal]
          : &call->function_results
                 .overflow[function_ordinal -
                           IREE_VM_CALL_DIRECT_REGISTER_COUNT];
  *slot = function_ref;
}

// Loads one valid ref argument as an internal borrow, replacing |inout_ref|
// without changing the packet source. The two locations must be disjoint.
IREE_API_EXPORT void iree_vm_call_ref_argument_load_borrow(
    const iree_vm_call_packet_t* call, uint16_t ref_ordinal,
    iree_vm_ref_t* inout_ref);

// Moves one valid ref argument while preserving its complete internal
// ownership state. The packet source is cleared and must be disjoint from
// |inout_ref|.
IREE_API_EXPORT void iree_vm_call_ref_argument_load_move(
    const iree_vm_call_packet_t* call, uint16_t ref_ordinal,
    iree_vm_ref_t* inout_ref);

// Moves one ref into a valid result ordinal, replacing its prior contents and
// promoting a borrowed source before publication. The two locations must be
// disjoint.
IREE_API_EXPORT void iree_vm_call_ref_result_store_move(
    const iree_vm_call_packet_t* call, uint16_t ref_ordinal,
    iree_vm_ref_t* inout_ref);

typedef struct iree_vm_module_function_start_params_t {
  // Active invocation, linked module, and opaque process state.
  iree_vm_module_execution_t execution;
  // Module-local function ordinal.
  uint16_t function_ordinal;
  // Canonical physical argument/result packet.
  iree_vm_call_packet_t call;
} iree_vm_module_function_start_params_t;

typedef struct iree_vm_module_function_resume_params_t {
  // Active invocation, linked module, and opaque process state.
  iree_vm_module_execution_t execution;
  // Exact top frame owned by the executing module.
  iree_vm_frame_t* frame;
} iree_vm_module_function_resume_params_t;

// Execution callbacks publish |out_outcome| only on OK. OK+COMPLETED means the
// logical function has no remaining frame and every exact result is
// initialized. OK+SUSPENDED means either that all state needed by a later
// resume is in durable frames or provider operations, or that the callback
// requested a child call and returned control to the invocation driver. The
// driver drains requested calls before reporting suspension to the host. A
// non-OK return leaves |out_outcome| untouched, quiesces callback sources owned
// by the module, and may leave frames for the invocation core's exact terminal
// unwind. Status is never a suspension or control-flow token. The invocation
// core enters callbacks with masked floating-point traps, nearest-even
// rounding, and gradual underflow. Implementations preserve those control
// modes across every callback; sticky floating-point exception flags are
// outside the module ABI contract.

// Starts one valid module-local function.
typedef iree_status_t(IREE_API_PTR* iree_vm_module_function_start_fn_t)(
    iree_vm_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome);

// Resumes the module-owned top frame.
typedef iree_status_t(IREE_API_PTR* iree_vm_module_function_resume_fn_t)(
    iree_vm_module_t* module,
    const iree_vm_module_function_resume_params_t* params,
    iree_vm_execution_outcome_t* out_outcome);

//===----------------------------------------------------------------------===//
// Generic Module Provider ABI
//===----------------------------------------------------------------------===//

// Generic ref-counted module implementation base.
//
// Implementations embed this at offset zero and keep the vtable and descriptor
// immutable for its lifetime. One module may be shared by multiple programs
// and receive concurrent callbacks for independent processes. Mutable process
// state lives only in the opaque process slice.
struct iree_vm_module_t {
  // Intrusive module owner count, published last by module initialization.
  iree_atomic_ref_count_t ref_count;
  // Immutable generic implementation vtable.
  const iree_vm_module_vtable_t* vtable;
  // Immutable implementation-owned descriptor.
  const iree_vm_module_descriptor_t* descriptor;
};

static_assert(offsetof(iree_vm_module_t, ref_count) == 0,
              "module ref count must remain at offset zero");
static_assert(sizeof(void*) != 8 || sizeof(iree_vm_module_t) == 24,
              "64-bit module bases must remain 24 bytes");
static_assert(sizeof(void*) != 4 || sizeof(iree_vm_module_t) == 12,
              "32-bit module bases must remain 12 bytes");

// Physically constructs one module's exact zeroed process-storage span.
//
// The callback is synchronous and cannot execute guest code, call another
// module, or yield. It may retain process-lifetime resources using
// |host_allocator|. Failure releases its own partial work before returning;
// the core detaches only the previously attached module prefix.
typedef iree_status_t(IREE_API_PTR* iree_vm_module_attach_state_fn_t)(
    iree_vm_module_t* module, iree_byte_span_t zeroed_storage,
    iree_allocator_t host_allocator);

// Validates that one attached process-storage span is publishable. The callback
// is synchronous and cannot execute guest code, call another module, or yield.
// Failure leaves a representation accepted by |detach_state|.
typedef iree_status_t(IREE_API_PTR* iree_vm_module_seal_state_fn_t)(
    iree_vm_module_t* module, iree_byte_span_t storage);

// Releases one successfully attached process-storage span without failing. The
// callback accepts sealed and unsealed storage and never frees the containing
// process allocation.
typedef void(IREE_API_PTR* iree_vm_module_detach_state_fn_t)(
    iree_vm_module_t* module, iree_byte_span_t storage);

// Populates one complete import group for an already validated ordinal.
typedef void(IREE_API_PTR* iree_vm_module_query_import_group_fn_t)(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_group);

// Populates one complete import declaration for an already validated ordinal.
typedef void(IREE_API_PTR* iree_vm_module_query_import_fn_t)(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_import);

// Populates one complete export declaration for an already validated ordinal.
typedef void(IREE_API_PTR* iree_vm_module_query_export_fn_t)(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_export);

// Populates one callable declaration for an already validated ordinal.
typedef void(IREE_API_PTR* iree_vm_module_query_callable_type_fn_t)(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type);

// Queries optional authored presentation without allocating.
//
// The callback always overwrites |out_presentation|. A complete fill requires
// the exact field count and at least the reported transient-tail capacity. If
// either is insufficient, it reports the requirement, returns empty function
// views, and touches neither field nor transient storage. A complete fill
// writes every field name and authored type but never reads or writes its VM
// type; common code fills exact types after the callback. Views may borrow
// immutable module storage or the supplied transient tail.
typedef void(IREE_API_PTR* iree_vm_module_query_presentation_fn_t)(
    const iree_vm_module_t* module,
    const iree_vm_module_presentation_query_t* query,
    iree_vm_module_presentation_t* out_presentation);

// Populates one complete stable metadata entry for an already validated query.
// The returned key and value bytes remain valid for the module lifetime.
typedef void(IREE_API_PTR* iree_vm_module_metadata_by_ordinal_fn_t)(
    const iree_vm_module_t* module,
    const iree_vm_module_metadata_query_t* query,
    iree_vm_metadata_entry_t* out_entry);

enum {
  // Initial incompatible generic module provider ABI version.
  IREE_VM_MODULE_ABI_VERSION_0 = 0,
};

// Versioned generic module implementation interface. State lifecycle callbacks
// are independently nullable identity operations. All other callbacks are
// required in version zero.
struct iree_vm_module_vtable_t {
  // Accessible bytes in this vtable.
  uint32_t structure_size;
  // Incompatible module ABI version.
  uint32_t abi_version;
  // Private final-refcount callback.
  void(IREE_API_PTR* destroy)(iree_vm_module_t* module);
  // Starts one module-local function.
  iree_vm_module_function_start_fn_t function_start;
  // Resumes one module-owned top frame.
  iree_vm_module_function_resume_fn_t function_resume;
  // Constructs one zeroed opaque process slice.
  iree_vm_module_attach_state_fn_t attach_state;
  // Validates one attached slice for process publication.
  iree_vm_module_seal_state_fn_t seal_state;
  // Releases one attached or sealed slice without failing.
  iree_vm_module_detach_state_fn_t detach_state;
  // Queries one import group by valid ordinal.
  iree_vm_module_query_import_group_fn_t query_import_group;
  // Queries one import declaration by valid ordinal.
  iree_vm_module_query_import_fn_t query_import;
  // Queries one export declaration by valid ordinal.
  iree_vm_module_query_export_fn_t query_export;
  // Queries one structural callable type by valid ordinal.
  iree_vm_module_query_callable_type_fn_t query_callable_type;
  // Queries optional authored presentation.
  iree_vm_module_query_presentation_fn_t query_presentation;
  // Queries stable typed metadata by valid scope and ordinal.
  iree_vm_module_metadata_by_ordinal_fn_t metadata_by_ordinal;
};

// Accessible bytes required by the complete version-zero vtable.
#define IREE_VM_MODULE_VTABLE_V0_REQUIRED_SIZE              \
  (offsetof(iree_vm_module_vtable_t, metadata_by_ordinal) + \
   sizeof(iree_vm_module_metadata_by_ordinal_fn_t))

static_assert(sizeof(void*) != 8 ||
                  offsetof(iree_vm_module_vtable_t, attach_state) == 32,
              "64-bit module vtable hot prefix must remain 32 bytes");

// Failure-atomically validates and publishes the first module owner. One
// allocation-free common walk validates all implementation-independent
// callable, import-group, import, export, and metadata contracts. Failure
// clears the temporary base pointers and leaves implementation cleanup with the
// factory; it never calls the destroy callback.
IREE_API_EXPORT iree_status_t
iree_vm_module_initialize(const iree_vm_module_vtable_t* vtable,
                          const iree_vm_module_descriptor_t* descriptor,
                          iree_vm_module_t* out_module);

// Common resume callback for modules whose functions cannot yield. Reaching
// this callback is a trusted module/core ABI violation and returns a terminal
// failure without mutating |out_outcome|.
IREE_API_EXPORT iree_status_t iree_vm_module_function_resume_unreachable(
    iree_vm_module_t* module,
    const iree_vm_module_function_resume_params_t* params,
    iree_vm_execution_outcome_t* out_outcome);

// Retains |module| for the caller. A null module is ignored.
IREE_API_EXPORT void iree_vm_module_retain(iree_vm_module_t* module);

// Releases |module| from the caller. A null module is ignored.
IREE_API_EXPORT void iree_vm_module_release(iree_vm_module_t* module);

//===----------------------------------------------------------------------===//
// Module Execution Helpers
//===----------------------------------------------------------------------===//

// Returns the active operation's level-triggered host wake callback.
IREE_API_EXPORT iree_vm_invocation_wake_callback_t
iree_vm_invocation_wake_callback(iree_vm_invocation_t* invocation);

// Returns the current sticky cancellation reason. Idle invocations return
// IREE_VM_CANCEL_REASON_NONE.
IREE_API_EXPORT iree_vm_cancel_reason_t
iree_vm_invocation_cancel_reason(const iree_vm_invocation_t* invocation);

// Requests one function in the current linked module. The correlated local
// descriptor is trusted provider data validated before target entry.
//
// This never enters the target inline. On success it publishes SUSPENDED and
// the current callback must return that outcome. The invocation driver starts
// the child and immediately resumes any preserved caller frame when the child
// completes without yielding to the host.
IREE_API_EXPORT iree_status_t
iree_vm_invocation_call_local(const iree_vm_module_execution_t* execution,
                              iree_vm_module_local_function_t local_function,
                              const iree_vm_call_packet_t* call,
                              iree_vm_execution_outcome_t* out_outcome);

// Requests one exact target in the current linked module's resolved import
// table with the same deferred-entry contract as
// |iree_vm_invocation_call_local|.
IREE_API_EXPORT iree_status_t iree_vm_invocation_call_import(
    const iree_vm_module_execution_t* execution, uint16_t import_ordinal,
    const iree_vm_call_packet_t* call,
    iree_vm_execution_outcome_t* out_outcome);

// Requests one program-bound function through an expected local callable type
// with the same deferred-entry contract as |iree_vm_invocation_call_local|.
IREE_API_EXPORT iree_status_t iree_vm_invocation_call_function_ref(
    const iree_vm_module_execution_t* execution,
    iree_vm_function_ref_t function_ref,
    uint16_t expected_callable_type_ordinal, const iree_vm_call_packet_t* call,
    iree_vm_execution_outcome_t* out_outcome);

// Materializes one module-local function as a borrowed program-bound value.
// Failure leaves |out_function_ref| untouched.
IREE_API_EXPORT iree_status_t iree_vm_function_ref_from_local_function(
    const iree_vm_module_execution_t* execution,
    iree_vm_module_local_function_t local_function,
    iree_vm_function_ref_t* out_function_ref);

// Materializes one resolved import as a borrowed program-bound value. An
// unresolved optional import succeeds with canonical null. Failure leaves
// |out_function_ref| untouched.
IREE_API_EXPORT iree_status_t iree_vm_function_ref_from_import(
    const iree_vm_module_execution_t* execution, uint16_t import_ordinal,
    iree_vm_function_ref_t* out_function_ref);

// Module-owned durable frame payload layout.
typedef struct iree_vm_frame_layout_t {
  // Requested module payload size in bytes.
  iree_host_size_t storage_size;
  // Requested power-of-two payload alignment.
  iree_host_size_t storage_alignment;
} iree_vm_frame_layout_t;

// No-fail module payload cleanup invoked exactly once before a frame is
// removed normally or during terminal unwind.
typedef void(IREE_API_PTR* iree_vm_frame_cleanup_fn_t)(iree_vm_frame_t* frame);

// Pushes one complete durable frame failure-atomically. The module must fully
// initialize every payload field its cleanup may inspect before any fallible
// operation or externally visible side effect.
IREE_API_EXPORT iree_status_t iree_vm_invocation_push_frame(
    const iree_vm_module_function_start_params_t* start_params,
    iree_vm_frame_layout_t layout, iree_vm_frame_cleanup_fn_t cleanup,
    iree_vm_frame_t** out_frame);

// Pops the exact top |frame|, runs cleanup, and rewinds the invocation stack.
IREE_API_EXPORT void iree_vm_invocation_pop_frame(
    iree_vm_invocation_t* invocation, iree_vm_frame_t* frame);

// Returns the aligned module-owned payload of |frame|.
IREE_API_EXPORT void* iree_vm_frame_storage(iree_vm_frame_t* frame);

// Returns the module-local function ordinal represented by |frame|.
IREE_API_EXPORT uint16_t
iree_vm_frame_function_ordinal(const iree_vm_frame_t* frame);

//===----------------------------------------------------------------------===//
// Generic Module Queries
//===----------------------------------------------------------------------===//

// Returns the validated module link name. |module| must be nonnull.
IREE_API_EXPORT iree_string_view_t
iree_vm_module_name(const iree_vm_module_t* module);

// Returns the validated flat import count. |module| must be nonnull.
IREE_API_EXPORT iree_host_size_t
iree_vm_module_import_count(const iree_vm_module_t* module);

// Returns the validated public export count. |module| must be nonnull.
IREE_API_EXPORT iree_host_size_t
iree_vm_module_export_count(const iree_vm_module_t* module);

// Returns the validated private function count. |module| must be nonnull.
IREE_API_EXPORT iree_host_size_t
iree_vm_module_function_count(const iree_vm_module_t* module);

// Returns the validated canonical ref-type count. |module| must be nonnull.
IREE_API_EXPORT iree_host_size_t
iree_vm_module_ref_type_count(const iree_vm_module_t* module);

// Returns one canonical ref type by module-local ordinal. Failure leaves
// |out_type| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_ref_type_by_ordinal(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_ref_type_t* out_type);

// Returns one borrowed public import identity by flat ordinal. Failure leaves
// |out_import| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_import_by_ordinal(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_import_t* out_import);

// Returns one borrowed public export identity by ordinal. Failure leaves
// |out_export| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_export_by_ordinal(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_export_t* out_export);

// Looks up one exact name in the module's sorted export directory. Failure
// leaves |out_export| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_lookup_export(
    const iree_vm_module_t* module, iree_string_view_t name,
    iree_vm_export_t* out_export);

// Returns one complete semantic import group. Failure leaves |out_group|
// untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_query_import_group(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_group);

// Returns one complete semantic import declaration. Failure leaves
// |out_import| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_query_import(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_import);

// Returns one complete semantic export declaration. Failure leaves
// |out_export| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_query_export(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_export);

// Returns one complete semantic callable declaration. Failure leaves
// |out_callable_type| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_query_callable_type(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type);

// Returns the stable target of one valid borrowed import identity.
IREE_API_EXPORT iree_vm_import_target_t
iree_vm_import_target(iree_vm_import_t import_value);

// Returns the stable name of one valid borrowed export identity.
IREE_API_EXPORT iree_string_view_t
iree_vm_export_name(iree_vm_export_t export_value);

// Description queries always compute an exact caller-storage requirement.
// |out_required_storage_size| is required. A null |out_description| or
// insufficient storage returns OK with the exact requirement and publishes no
// description. The requirement includes the fixed field array, padding needed
// to max-align the provider's transient tail, and that tail. Nonempty storage
// must be max-aligned; a zero-byte requirement accepts an empty span while
// still publishing |out_description|. Each call dispatches exactly one
// presentation query. Invalid identity, storage, or arithmetic leaves both
// outputs untouched.

// Computes one complete immutable import description.
IREE_API_EXPORT iree_status_t iree_vm_import_query_description(
    iree_vm_import_t import_value, iree_byte_span_t storage,
    iree_host_size_t* out_required_storage_size,
    iree_vm_import_description_t* out_description);

// Computes one complete immutable export description.
IREE_API_EXPORT iree_status_t iree_vm_export_query_description(
    iree_vm_export_t export_value, iree_byte_span_t storage,
    iree_host_size_t* out_required_storage_size,
    iree_vm_export_description_t* out_description);

// Computes one level of a structural callable-type description.
IREE_API_EXPORT iree_status_t iree_vm_callable_type_query_description(
    iree_vm_callable_type_t callable_type, iree_byte_span_t storage,
    iree_host_size_t* out_required_storage_size,
    iree_vm_callable_type_description_t* out_description);

// Returns the stable module-scope metadata count. |module| must be nonnull.
IREE_API_EXPORT iree_host_size_t
iree_vm_module_metadata_count(const iree_vm_module_t* module);

// Returns one stable module-scope metadata entry. Failure leaves |out_entry|
// untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_metadata_by_ordinal(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_metadata_entry_t* out_entry);

// Looks up one exact module-scope metadata key. Absence returns OK, sets
// |out_found| false, and leaves |out_value| untouched. Failure leaves both
// outputs untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_try_lookup_metadata(
    const iree_vm_module_t* module, iree_string_view_t key, bool* out_found,
    iree_vm_metadata_value_t* out_value);

// Returns the stable metadata count for one valid borrowed import identity.
IREE_API_EXPORT iree_host_size_t
iree_vm_import_metadata_count(iree_vm_import_t import_value);

// Returns one stable import-scope metadata entry. Failure leaves |out_entry|
// untouched.
IREE_API_EXPORT iree_status_t iree_vm_import_metadata_by_ordinal(
    iree_vm_import_t import_value, iree_host_size_t ordinal,
    iree_vm_metadata_entry_t* out_entry);

// Looks up one exact import-scope metadata key. Absence returns OK, sets
// |out_found| false, and leaves |out_value| untouched. Failure leaves both
// outputs untouched.
IREE_API_EXPORT iree_status_t iree_vm_import_try_lookup_metadata(
    iree_vm_import_t import_value, iree_string_view_t key, bool* out_found,
    iree_vm_metadata_value_t* out_value);

// Returns the stable metadata count for one valid borrowed export identity.
IREE_API_EXPORT iree_host_size_t
iree_vm_export_metadata_count(iree_vm_export_t export_value);

// Returns one stable export-scope metadata entry. Failure leaves |out_entry|
// untouched.
IREE_API_EXPORT iree_status_t iree_vm_export_metadata_by_ordinal(
    iree_vm_export_t export_value, iree_host_size_t ordinal,
    iree_vm_metadata_entry_t* out_entry);

// Looks up one exact export-scope metadata key. Absence returns OK, sets
// |out_found| false, and leaves |out_value| untouched. Failure leaves both
// outputs untouched.
IREE_API_EXPORT iree_status_t iree_vm_export_try_lookup_metadata(
    iree_vm_export_t export_value, iree_string_view_t key, bool* out_found,
    iree_vm_metadata_value_t* out_value);

// Extracts one canonical BOOL metadata value. Failure leaves |out_value|
// untouched.
IREE_API_EXPORT iree_status_t iree_vm_bool_from_metadata_value(
    iree_vm_metadata_value_t value, bool* out_value);

// Extracts one little-endian I64 metadata value. Failure leaves |out_value|
// untouched.
IREE_API_EXPORT iree_status_t iree_vm_i64_from_metadata_value(
    iree_vm_metadata_value_t value, int64_t* out_value);

// Extracts one little-endian U64 metadata value. Failure leaves |out_value|
// untouched.
IREE_API_EXPORT iree_status_t iree_vm_u64_from_metadata_value(
    iree_vm_metadata_value_t value, uint64_t* out_value);

// Extracts one little-endian binary64 metadata value. Failure leaves
// |out_value| untouched.
IREE_API_EXPORT iree_status_t iree_vm_f64_from_metadata_value(
    iree_vm_metadata_value_t value, double* out_value);

// Extracts one validated UTF-8 view. The result borrows the metadata storage;
// failure leaves |out_value| untouched.
IREE_API_EXPORT iree_status_t iree_vm_string_view_from_metadata_value(
    iree_vm_metadata_value_t value, iree_string_view_t* out_value);

// Extracts one opaque byte span. The result borrows the metadata storage;
// failure leaves |out_value| untouched.
IREE_API_EXPORT iree_status_t iree_vm_const_byte_span_from_metadata_value(
    iree_vm_metadata_value_t value, iree_const_byte_span_t* out_value);

// Common provider callback for modules with no declaration presentation.
IREE_API_EXPORT void iree_vm_module_query_presentation_none(
    const iree_vm_module_t* module,
    const iree_vm_module_presentation_query_t* query,
    iree_vm_module_presentation_t* out_presentation);

// Common unreachable provider callback for modules with no metadata.
IREE_API_EXPORT void iree_vm_module_metadata_by_ordinal_none(
    const iree_vm_module_t* module,
    const iree_vm_module_metadata_query_t* query,
    iree_vm_metadata_entry_t* out_entry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_MODULE_H_
