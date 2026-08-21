// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_REF_H_
#define IREE_VM_REF_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_vm_ref_type_descriptor_t iree_vm_ref_type_descriptor_t;
typedef struct iree_vm_ref_type_table_t iree_vm_ref_type_table_t;

//===----------------------------------------------------------------------===//
// Reference Types
//===----------------------------------------------------------------------===//

// State-free canonical ref-type handle. The table, descriptor, strings,
// destroy callback, and containing code are borrowed from a host-managed
// provider scope and must remain live while any ref, variant, or module can
// name the handle.
typedef const iree_vm_ref_type_descriptor_t* iree_vm_ref_type_t;

// Generic view of dense append-order family handle storage. |data| points to a
// provider-owned family-specific structure containing only append-order
// iree_vm_ref_type_t fields. Field ordinal N has byte offset
// `N * sizeof(iree_vm_ref_type_t)` with no intervening padding.
typedef struct iree_vm_ref_type_storage_t {
  // Provider-owned family-specific typed table.
  const void* data;
  // Number of append-order handles in |data|.
  iree_host_size_t count;
} iree_vm_ref_type_storage_t;

// Immutable span of canonical ref-type handles.
typedef struct iree_vm_ref_type_span_t {
  // Contiguous type handles.
  const iree_vm_ref_type_t* data;
  // Number of type handles in |data|.
  iree_host_size_t count;
} iree_vm_ref_type_span_t;

// Loads one valid append-order handle from |storage|. Byte copying lets generic
// code read a named-field structure without violating C/C++ array bounds or
// effective-type rules. |ordinal| must be less than |storage.count|.
static inline iree_vm_ref_type_t iree_vm_ref_type_storage_at(
    iree_vm_ref_type_storage_t storage, iree_host_size_t ordinal) {
  iree_vm_ref_type_t type = NULL;
  memcpy(&type,
         (const uint8_t*)storage.data + ordinal * sizeof(iree_vm_ref_type_t),
         sizeof(type));
  return type;
}

// Required ref-type table feature bits.
enum iree_vm_ref_type_table_flag_bits_e {
  IREE_VM_REF_TYPE_TABLE_FLAG_NONE = 0u,
  // The descriptors exist only for provider-independent reflection. Such a
  // table cannot be registered for executable module construction.
  IREE_VM_REF_TYPE_TABLE_FLAG_REFLECTION_ONLY = 1u << 0,
};
typedef uint32_t iree_vm_ref_type_table_flags_t;

// Immutable provider-owned table defining one complete type namespace.
// Concrete family types are append-only. A consumer validates the count and
// expected name at every ordinal it knows and copies that prefix into its own
// named structure. Provider storage is never overlaid with a separately
// compiled consumer structure type.
struct iree_vm_ref_type_table_t {
  // Accessible bytes in this generic table descriptor.
  uint32_t structure_size;
  // Required generic table features.
  iree_vm_ref_type_table_flags_t flags;
  // Nonempty namespace exclusively owned by this table.
  iree_string_view_t namespace_name;
  // Complete dense append-order type-handle storage.
  iree_vm_ref_type_storage_t types;
};

// Accessible bytes required by the version-zero generic table prefix.
#define IREE_VM_REF_TYPE_TABLE_V0_REQUIRED_SIZE \
  (offsetof(iree_vm_ref_type_table_t, types) +  \
   sizeof(iree_vm_ref_type_storage_t))

// Releases the complete provider-defined object after its intrusive reference
// count reaches zero. The callback cannot fail.
typedef void(IREE_API_PTR* iree_vm_ref_destroy_fn_t)(void* object);

// Immutable descriptor for one exact ref type. Version zero has no object size,
// counter offset, custom retain/release hook, subtype relation, allocator, or
// formatting callback.
struct iree_vm_ref_type_descriptor_t {
  // Optional no-fail final-release callback.
  iree_vm_ref_destroy_fn_t destroy;
  // Provider family containing this descriptor.
  const iree_vm_ref_type_table_t* table;
  // Nonempty local name within |table->namespace_name|.
  iree_string_view_t type_name;
};

// Structured identity of one exact ref type.
typedef struct iree_vm_ref_type_key_t {
  // Provider namespace.
  iree_string_view_t namespace_name;
  // Local type name within the provider namespace.
  iree_string_view_t type_name;
} iree_vm_ref_type_key_t;

// Returns the structured identity of |type|. A null type returns two null
// string views.
static inline iree_vm_ref_type_key_t iree_vm_ref_type_key(
    iree_vm_ref_type_t type) {
  iree_vm_ref_type_key_t key = {
      iree_string_view_empty(),
      iree_string_view_empty(),
  };
  if (type) {
    key.namespace_name = type->table->namespace_name;
    key.type_name = type->type_name;
  }
  return key;
}

//===----------------------------------------------------------------------===//
// Reference-counted Objects
//===----------------------------------------------------------------------===//

// Fixed intrusive prefix of every VM-visible ref object. Providers initialize
// this prefix once and never query, copy, move, or directly modify it. Every
// VM-visible object pointer addresses this prefix at offset zero.
typedef struct iree_vm_ref_object_t {
  // Opaque intrusive owner-count storage.
  iree_atomic_ref_count_t ref_count;
} iree_vm_ref_object_t;

#define IREE_VM_REF_OBJECT_SIZE 4
#define IREE_VM_REF_OBJECT_ALIGNMENT 4

static_assert(sizeof(iree_vm_ref_object_t) == IREE_VM_REF_OBJECT_SIZE,
              "VM ref-object prefix must remain one 32-bit word");
static_assert(iree_alignof(iree_vm_ref_object_t) ==
                  IREE_VM_REF_OBJECT_ALIGNMENT,
              "VM ref-object prefix must remain four-byte aligned");
static_assert(iree_alignof(iree_vm_ref_type_descriptor_t) >= 4,
              "VM ref descriptors require two zero low bits");

// Initializes |out_object| with one real owner.
IREE_API_EXPORT void iree_vm_ref_object_initialize(
    iree_vm_ref_object_t* out_object);

// Adds one owner through the non-null object's offset-zero prefix. Retaining is
// always valid while the caller has live access to |object|. The owner count is
// never queried.
IREE_API_EXPORT void iree_vm_ref_object_retain(void* object);

// Releases one raw object owner through its matching trusted descriptor. Final
// release invokes |type->destroy| when present. A null |object| is ignored; a
// non-null object requires its exact canonical |type|.
IREE_API_EXPORT void iree_vm_ref_object_release(void* object,
                                                iree_vm_ref_type_t type);

//===----------------------------------------------------------------------===//
// Reference Values
//===----------------------------------------------------------------------===//

// Complete typed ref state. |type_and_state| contains the descriptor pointer in
// its upper bits and one ownership bit in bit zero. Canonical null is all zero.
// This is an in-process native ABI and is never serialized or copied across
// bitness.
typedef struct iree_vm_ref_t {
  // VM-visible object pointer or null.
  void* object;
  // Exact descriptor pointer plus owned/borrowed state.
  uintptr_t type_and_state;
} iree_vm_ref_t;

// Low descriptor-pointer bits used by the public carrier representation.
enum iree_vm_ref_state_bits_e {
  IREE_VM_REF_STATE_OWNED = 0u,
  IREE_VM_REF_STATE_BORROWED = 1u,
  IREE_VM_REF_STATE_MASK = 3u,
};

static_assert(sizeof(iree_vm_ref_t) == 2 * sizeof(void*),
              "VM refs must remain two native words");
static_assert(iree_alignof(iree_vm_ref_t) == iree_alignof(void*),
              "VM refs must retain native pointer alignment");
static_assert(offsetof(iree_vm_ref_t, object) == 0,
              "VM ref object pointer must be first");
static_assert(offsetof(iree_vm_ref_t, type_and_state) == sizeof(void*),
              "VM ref type and state must be second");

// Returns canonical null.
static inline iree_vm_ref_t iree_vm_ref_null(void) {
  iree_vm_ref_t ref = {NULL, 0};
  return ref;
}

// Returns true when |ref| is canonical null.
static inline bool iree_vm_ref_is_null(iree_vm_ref_t ref) {
  return ref.object == NULL && ref.type_and_state == 0;
}

// Returns the exact type of |ref| or null for canonical null.
static inline iree_vm_ref_type_t iree_vm_ref_type(iree_vm_ref_t ref) {
  return ref.object ? (iree_vm_ref_type_t)(ref.type_and_state &
                                           ~(uintptr_t)IREE_VM_REF_STATE_MASK)
                    : NULL;
}

// Returns true when |ref| contains a non-null object with exact |type|.
IREE_API_EXPORT bool iree_vm_ref_isa(iree_vm_ref_t ref,
                                     iree_vm_ref_type_t type);

// Creates a borrowed ref from a trusted pointer/type pair. A non-null |ptr|
// requires its exact non-null canonical |type|. A null pointer produces
// canonical null regardless of |type|.
IREE_API_EXPORT iree_vm_ref_t
iree_vm_ref_from_ptr_borrowed(void* ptr, iree_vm_ref_type_t type);

// Retains |ptr| and returns one owned ref. A non-null |ptr| requires its exact
// non-null canonical |type|. A null pointer produces canonical null regardless
// of |type|.
IREE_API_EXPORT iree_vm_ref_t
iree_vm_ref_from_ptr_retained(void* ptr, iree_vm_ref_type_t type);

// Moves one pointer owner into a ref and clears |inout_ptr|. A non-null pointer
// requires its exact non-null canonical |type|. The |inout_ptr| argument must
// address an actual void* object; typed pointer-to-pointer values use a stamped
// adapter and are never cast to void**.
IREE_API_EXPORT iree_vm_ref_t
iree_vm_ref_from_ptr_move(void** inout_ptr, iree_vm_ref_type_t type);

// Returns an owned copy of |ref|, promoting a borrow when required.
IREE_API_EXPORT iree_vm_ref_t iree_vm_ref_retain(iree_vm_ref_t ref);

// Moves |ref| and leaves its source canonical null. A borrowed source is
// promoted before clearing so the result cannot escape its borrow anchor.
IREE_API_EXPORT iree_vm_ref_t iree_vm_ref_move(iree_vm_ref_t* ref);

// Releases an owned ref when present and writes canonical null.
IREE_API_EXPORT void iree_vm_ref_reset(iree_vm_ref_t* ref);

// Validates |ref| against |expected_type| and borrows its object pointer.
// Canonical null succeeds and produces null. Failure leaves |out_ptr|
// untouched.
IREE_API_EXPORT iree_status_t iree_vm_ptr_from_ref_borrowed(
    iree_vm_ref_t ref, iree_vm_ref_type_t expected_type, void** out_ptr);

// Validates and retains |ref| into an owned object pointer. Canonical null
// succeeds and produces null. Failure leaves |out_ptr| untouched.
IREE_API_EXPORT iree_status_t iree_vm_ptr_from_ref_retained(
    iree_vm_ref_t ref, iree_vm_ref_type_t expected_type, void** out_ptr);

// Validates and moves |ref| into an owned object pointer. Success clears |ref|.
// Failure leaves both |ref| and |out_ptr| untouched.
IREE_API_EXPORT iree_status_t iree_vm_ptr_from_ref_move(
    iree_vm_ref_t* ref, iree_vm_ref_type_t expected_type, void** out_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_REF_H_
