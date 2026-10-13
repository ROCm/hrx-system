// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_NATIVE_CALL_H_
#define IREE_VM_NATIVE_CALL_H_

#include <string.h>

#include "iree/base/api.h"
#include "iree/vm/ref.h"
#include "iree/vm/scalar.h"

#ifdef __cplusplus
extern "C" {
#endif

// Native calls synchronously borrow a caller's physical storage. They cannot
// suspend, call back into the VM, or retain the view or scratch after return.
// Native APIs may accept asynchronous work under their own capture/borrow
// contracts; the VM adds no completion or retirement handling.
//
// The native entry's immutable schema is used only in admission and linking.
// Its typed entry reads verified selectors directly while constructing native
// arguments. It admits dynamic ref types/nulls and native argument values.
// All fallible preparation precedes an accepting native operation. Results own
// their resources before publication, and publication occurs after all input
// use, allowing a result to replace an input slot. Failure leaves results
// untouched; input owners remain the caller's responsibility on every path.

// A scalar selector is a register index, or a local byte offset tagged with
// bit 31. Local scalar cells occupy eight bytes and may be unaligned.
#define IREE_VM_NATIVE_CALL_LOCAL_VALUE UINT32_C(0x80000000)

typedef struct iree_vm_native_call_range_t {
  // Number of fixed-size child records.
  uint32_t count;
  // Word offset from the root record, including for an empty range.
  uint32_t offset;
} iree_vm_native_call_range_t;

// Scalar field kinds use the exact iree_vm_scalar_type_t encoding. REF names a
// module-local ref type; RANGE names an earlier record layout. RESULT applies
// only to scalar/ref leaves, including leaves inside ranges.
typedef uint16_t iree_vm_native_call_field_kind_t;
enum iree_vm_native_call_field_kind_e {
  IREE_VM_NATIVE_CALL_FIELD_REF = 0x0100,
  IREE_VM_NATIVE_CALL_FIELD_RANGE = 0x0200,
};
#define IREE_VM_NATIVE_CALL_FIELD_RESULT UINT16_C(0x8000)

typedef struct iree_vm_native_call_field_t {
  // Exact scalar kind, REF, or RANGE, optionally combined with RESULT.
  iree_vm_native_call_field_kind_t kind;
  // Module-local ref/layout ordinal, or zero for a scalar.
  uint16_t type_ordinal;
} iree_vm_native_call_field_t;
static_assert(sizeof(iree_vm_native_call_field_t) == 4,
              "native fields must remain four bytes");

// A fixed record of ordered fields. A selector occupies one word and a range
// header occupies two; field offsets are derived, never authored. Layouts form
// an acyclic DAG with child ordinals strictly less than their parent. Each
// range stores its complete fixed-record array followed by descendants in
// field-order depth-first traversal. Empty ranges consume no words. This
// canonical layout proves complete, nonoverlapping record coverage with one
// cold traversal.
typedef struct iree_vm_native_call_layout_t {
  // First field in the schema's concatenated field table.
  uint32_t field_base;
  // Word length of the fixed record, including range headers.
  uint32_t word_count;
  // Number of fields in this fixed record.
  uint16_t field_count;
  // Maximum child nesting depth, with zero for a record without ranges.
  uint16_t depth;
} iree_vm_native_call_layout_t;
static_assert(sizeof(iree_vm_native_call_layout_t) == 12,
              "native layouts must remain twelve bytes");

// Immutable indexed declaration shared by native implementations and image
// admission. Layout field slices exactly partition |fields| in layout order.
// Scratch requirements are native implementation details, not layout identity.
typedef struct iree_vm_native_call_schema_t {
  // Immutable layouts in child-before-parent order.
  const iree_vm_native_call_layout_t* layouts;
  // Immutable fields partitioned by |layouts|.
  const iree_vm_native_call_field_t* fields;
  // Number of layouts, at most 65536.
  uint32_t layout_count;
  // Number of fields in the complete table.
  uint32_t field_count;
  // Ordinal of the root record for this entry.
  uint16_t root_ordinal;
} iree_vm_native_call_schema_t;

// Validates an external declaration without traversing or expanding its DAG.
// Ref ordinals name entries in a module's ref-type table of |ref_type_count|.
IREE_API_EXPORT iree_status_t iree_vm_native_call_schema_validate(
    const iree_vm_native_call_schema_t* schema, uint32_t ref_type_count);

typedef struct iree_vm_native_call_t {
  // Borrowed immutable verified callsite words.
  const uint32_t* record;
  // Caller scalar registers; outputs may replace inputs after native use.
  uint64_t* values;
  // Caller-owned ref registers followed by local ref slots.
  iree_vm_ref_t* refs;
  // Caller local bytes addressed by tagged scalar selectors.
  uint8_t* local_bytes;
  // Exclusive max-aligned native scratch of the linked extent.
  uint8_t* scratch;
} iree_vm_native_call_t;

static inline uint64_t iree_vm_native_call_load_value(
    const iree_vm_native_call_t* call, uint32_t selector) {
  if (!(selector & IREE_VM_NATIVE_CALL_LOCAL_VALUE)) {
    return call->values[selector];
  }
  uint64_t value;
  memcpy(&value,
         call->local_bytes + (selector & ~IREE_VM_NATIVE_CALL_LOCAL_VALUE),
         sizeof(value));
  return value;
}

static inline void iree_vm_native_call_store_value(
    const iree_vm_native_call_t* call, uint32_t selector, uint64_t value) {
  if (!(selector & IREE_VM_NATIVE_CALL_LOCAL_VALUE)) {
    call->values[selector] = value;
  } else {
    memcpy(call->local_bytes + (selector & ~IREE_VM_NATIVE_CALL_LOCAL_VALUE),
           &value, sizeof(value));
  }
}

typedef iree_status_t(IREE_API_PTR* iree_vm_native_call_fn_t)(
    const void* user_data, const iree_vm_native_call_t* call);

// One module-owned synchronous native entry, independent of cardinalities.
// Names are strictly sorted within the native directory. This directory is
// distinct from ordinary exports and cannot produce VM function references.
typedef struct iree_vm_native_export_t {
  // Exact nonempty link name within the containing module.
  iree_string_view_t name;
  // Exact schema/semantics version accepted by this entry.
  uint32_t version;
  // Trusted typed record declaration.
  const iree_vm_native_call_schema_t* schema;
  // Native scratch bytes per record, indexed by layout ordinal. Each entry is
  // a multiple of eight; NULL means no scratch. Not part of the guest ABI.
  const uint32_t* scratch_sizes;
  // Optional cold provider check for cross-field relations, such as matching
  // request/result cardinalities. Generic schema checks have already passed.
  iree_status_t(IREE_API_PTR* verify)(const uint32_t* record);
  // Synchronous typed implementation.
  iree_vm_native_call_fn_t entry;
  // Immutable provider context, retained by the containing module.
  const void* user_data;
} iree_vm_native_export_t;

typedef struct iree_vm_native_call_storage_t {
  // Number of accessible scalar register cells.
  uint32_t value_count;
  // Number of accessible ref registers plus local ref slots.
  uint32_t ref_count;
  // Number of accessible local bytes.
  uint32_t local_byte_length;
} iree_vm_native_call_storage_t;

// Immutable module-owned callsite declaration. Its provider guarantees that
// storage extents describe the named function. Linking verifies the untrusted
// record against the resolved native export before any invocation can use it.
typedef struct iree_vm_native_callsite_t {
  // Exact required target module name.
  iree_string_view_t module_name;
  // Exact native export name.
  iree_string_view_t export_name;
  // Exact export schema/semantics version.
  uint32_t version;
  // Caller function ordinal owning the physical storage.
  uint32_t function_ordinal;
  // Borrowed immutable word-aligned callsite record.
  const uint32_t* record;
  // Number of words in the complete record including all descendants.
  uint32_t word_count;
  // Physical extents established by the caller's module implementation.
  iree_vm_native_call_storage_t storage;
} iree_vm_native_callsite_t;

typedef struct iree_vm_native_module_t {
  // Sorted native export directory owned by the module.
  const iree_vm_native_export_t* exports;
  // Number of native exports.
  iree_host_size_t export_count;
  // Native callsite declarations owned by the module.
  const iree_vm_native_callsite_t* callsites;
  // Number of callsite declarations.
  iree_host_size_t callsite_count;
} iree_vm_native_module_t;

// Validates an external callsite record against trusted provider metadata and
// returns its exact scratch requirement. Scalar/ref result destinations must
// be distinct, including overlapping local byte ranges. Failure leaves the
// output untouched. Allocations occur only during this cold binding step.
IREE_API_EXPORT iree_status_t iree_vm_native_call_verify(
    const iree_vm_native_export_t* target,
    const iree_vm_native_callsite_t* callsite, iree_allocator_t host_allocator,
    iree_host_size_t* out_scratch_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_VM_NATIVE_CALL_H_
