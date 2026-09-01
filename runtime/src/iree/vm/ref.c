// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/ref.h"

static inline iree_vm_ref_object_t* iree_vm_ref_object_cast(void* object) {
  return (iree_vm_ref_object_t*)object;
}

static inline bool iree_vm_ref_is_borrowed(iree_vm_ref_t ref) {
  return (ref.type_and_state & IREE_VM_REF_STATE_MASK) ==
         IREE_VM_REF_STATE_BORROWED;
}

static inline iree_vm_ref_t iree_vm_ref_make(void* object,
                                             iree_vm_ref_type_t type,
                                             uintptr_t state) {
  if (!object) return iree_vm_ref_null();
  iree_vm_ref_t ref = {
      object,
      (uintptr_t)type | state,
  };
  return ref;
}

static iree_status_t iree_vm_ref_check_type(iree_vm_ref_t ref,
                                            iree_vm_ref_type_t expected_type) {
  if (!expected_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected ref type is null");
  }
  if (ref.object && iree_vm_ref_type(ref) != expected_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "ref type mismatch");
  }
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_ref_object_initialize(
    iree_vm_ref_object_t* out_object) {
  iree_atomic_ref_count_init(&out_object->ref_count);
}

IREE_API_EXPORT void iree_vm_ref_object_retain(void* object) {
  if (object) {
    iree_atomic_ref_count_inc(&iree_vm_ref_object_cast(object)->ref_count);
  }
}

IREE_API_EXPORT void iree_vm_ref_object_release(void* object,
                                                iree_vm_ref_type_t type) {
  if (!object) return;
  iree_vm_ref_object_t* ref_object = iree_vm_ref_object_cast(object);
  if (iree_atomic_ref_count_dec(&ref_object->ref_count) == 1 && type->destroy) {
    type->destroy(object);
  }
}

IREE_API_EXPORT bool iree_vm_ref_isa(iree_vm_ref_t ref,
                                     iree_vm_ref_type_t type) {
  return ref.object && iree_vm_ref_type(ref) == type;
}

IREE_API_EXPORT iree_vm_ref_t
iree_vm_ref_from_ptr_borrowed(void* ptr, iree_vm_ref_type_t type) {
  return iree_vm_ref_make(ptr, type, IREE_VM_REF_STATE_BORROWED);
}

IREE_API_EXPORT iree_vm_ref_t
iree_vm_ref_from_ptr_retained(void* ptr, iree_vm_ref_type_t type) {
  if (ptr) iree_vm_ref_object_retain(ptr);
  return iree_vm_ref_make(ptr, type, IREE_VM_REF_STATE_OWNED);
}

IREE_API_EXPORT iree_vm_ref_t
iree_vm_ref_from_ptr_move(void** inout_ptr, iree_vm_ref_type_t type) {
  void* ptr = *inout_ptr;
  *inout_ptr = NULL;
  return iree_vm_ref_make(ptr, type, IREE_VM_REF_STATE_OWNED);
}

IREE_API_EXPORT iree_vm_ref_t iree_vm_ref_retain(iree_vm_ref_t ref) {
  if (!ref.object) return iree_vm_ref_null();
  iree_vm_ref_object_retain(ref.object);
  return iree_vm_ref_make(ref.object, iree_vm_ref_type(ref),
                          IREE_VM_REF_STATE_OWNED);
}

IREE_API_EXPORT iree_vm_ref_t iree_vm_ref_move(iree_vm_ref_t* ref) {
  iree_vm_ref_t result = *ref;
  if (result.object && iree_vm_ref_is_borrowed(result)) {
    iree_vm_ref_object_retain(result.object);
    result.type_and_state &= ~(uintptr_t)IREE_VM_REF_STATE_MASK;
  }
  *ref = iree_vm_ref_null();
  return result;
}

IREE_API_EXPORT void iree_vm_ref_reset(iree_vm_ref_t* ref) {
  iree_vm_ref_t old_ref = *ref;
  *ref = iree_vm_ref_null();
  if (old_ref.object && !iree_vm_ref_is_borrowed(old_ref)) {
    iree_vm_ref_object_release(old_ref.object, iree_vm_ref_type(old_ref));
  }
}

IREE_API_EXPORT iree_status_t iree_vm_ptr_from_ref_borrowed(
    iree_vm_ref_t ref, iree_vm_ref_type_t expected_type, void** out_ptr) {
  if (!out_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_ptr is required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_ref_check_type(ref, expected_type));
  *out_ptr = ref.object;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_ptr_from_ref_retained(
    iree_vm_ref_t ref, iree_vm_ref_type_t expected_type, void** out_ptr) {
  if (!out_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_ptr is required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_ref_check_type(ref, expected_type));
  if (ref.object) iree_vm_ref_object_retain(ref.object);
  *out_ptr = ref.object;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_ptr_from_ref_move(
    iree_vm_ref_t* ref, iree_vm_ref_type_t expected_type, void** out_ptr) {
  if (!ref || !out_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ref and out_ptr are required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_ref_check_type(*ref, expected_type));
  iree_vm_ref_t old_ref = *ref;
  if (old_ref.object && iree_vm_ref_is_borrowed(old_ref)) {
    iree_vm_ref_object_retain(old_ref.object);
  }
  *ref = iree_vm_ref_null();
  *out_ptr = old_ref.object;
  return iree_ok_status();
}
