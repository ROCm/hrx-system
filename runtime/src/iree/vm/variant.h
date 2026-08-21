// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_VARIANT_H_
#define IREE_VM_VARIANT_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/vm/function.h"
#include "iree/vm/ref.h"
#include "iree/vm/scalar.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Variant Values
//===----------------------------------------------------------------------===//

// Fixed process-local logical call value. The raw lanes define the native ABI
// layout used by inline construction and calling-compiler optimization. They
// are not a wire format or an independent construction API. An owning carrier
// must be moved or reset exactly once.
typedef struct iree_vm_variant_t {
  // Canonical scalar bits or integer representation of an object pointer.
  uint64_t payload;
  // Scalar tag or descriptor/target bits plus the carrier kind.
  uint64_t metadata;
} iree_vm_variant_t;

static_assert(sizeof(iree_vm_variant_t) == 16,
              "VM variants must remain 16 bytes");
static_assert(iree_alignof(iree_vm_variant_t) == iree_alignof(uint64_t),
              "VM variants must retain native 64-bit alignment");
static_assert(offsetof(iree_vm_variant_t, payload) == 0,
              "VM variant payload must be first");
static_assert(offsetof(iree_vm_variant_t, metadata) == 8,
              "VM variant metadata must be second");
static_assert(sizeof(uintptr_t) <= sizeof(uint64_t),
              "VM variants require pointers no wider than 64 bits");

// Low metadata bits selecting the complete carrier kind.
enum iree_vm_variant_tag_bits_e {
  IREE_VM_VARIANT_TAG_OWNED_REF = 0u,
  IREE_VM_VARIANT_TAG_BORROWED_REF = 1u,
  IREE_VM_VARIANT_TAG_FUNCTION_REF = 2u,
  IREE_VM_VARIANT_TAG_SCALAR = 3u,
  IREE_VM_VARIANT_TAG_MASK = 3u,
};

#define IREE_VM_VARIANT_SCALAR_METADATA_(scalar_type) \
  (((uint64_t)(scalar_type) << 2) | IREE_VM_VARIANT_TAG_SCALAR)

// Returns the canonical moved-from/unused carrier.
static inline iree_vm_variant_t iree_vm_variant_empty(void) {
  iree_vm_variant_t variant = {
      0, IREE_VM_VARIANT_SCALAR_METADATA_(IREE_VM_SCALAR_TYPE_INVALID)};
  return variant;
}

// Returns canonical null, valid at every ref position.
static inline iree_vm_variant_t iree_vm_variant_null(void) {
  iree_vm_variant_t variant = {0, IREE_VM_VARIANT_TAG_OWNED_REF};
  return variant;
}

// Returns true when |variant| is canonical empty.
static inline bool iree_vm_variant_is_empty(iree_vm_variant_t variant) {
  return variant.payload == 0 &&
         variant.metadata ==
             IREE_VM_VARIANT_SCALAR_METADATA_(IREE_VM_SCALAR_TYPE_INVALID);
}

// Returns true when |variant| is a canonical null ref or function ref.
static inline bool iree_vm_variant_is_null(iree_vm_variant_t variant) {
  return variant.payload == 0 &&
         (variant.metadata == IREE_VM_VARIANT_TAG_OWNED_REF ||
          variant.metadata == IREE_VM_VARIANT_TAG_FUNCTION_REF);
}

// Returns true when |variant| carries a non-INVALID scalar.
static inline bool iree_vm_variant_is_scalar(iree_vm_variant_t variant) {
  return (variant.metadata & IREE_VM_VARIANT_TAG_MASK) ==
             IREE_VM_VARIANT_TAG_SCALAR &&
         (variant.metadata >> 2) != IREE_VM_SCALAR_TYPE_INVALID;
}

// Returns true when |variant| carries a function reference.
static inline bool iree_vm_variant_is_function_ref(iree_vm_variant_t variant) {
  return (variant.metadata & IREE_VM_VARIANT_TAG_MASK) ==
         IREE_VM_VARIANT_TAG_FUNCTION_REF;
}

// Returns true when |variant| carries a ref or canonical null.
static inline bool iree_vm_variant_is_ref(iree_vm_variant_t variant) {
  return (variant.metadata & IREE_VM_VARIANT_TAG_MASK) <=
         IREE_VM_VARIANT_TAG_BORROWED_REF;
}

// Returns the scalar type or INVALID when |variant| is not scalar.
static inline iree_vm_scalar_type_t iree_vm_variant_scalar_type(
    iree_vm_variant_t variant) {
  return iree_vm_variant_is_scalar(variant)
             ? (iree_vm_scalar_type_t)(variant.metadata >> 2)
             : (iree_vm_scalar_type_t)IREE_VM_SCALAR_TYPE_INVALID;
}

// Returns the exact ref type or null for canonical null/non-ref values.
static inline iree_vm_ref_type_t iree_vm_variant_ref_type(
    iree_vm_variant_t variant) {
  return iree_vm_variant_is_ref(variant) && variant.payload
             ? (iree_vm_ref_type_t)(uintptr_t)(variant.metadata &
                                               ~(uint64_t)
                                                   IREE_VM_VARIANT_TAG_MASK)
             : NULL;
}

//===----------------------------------------------------------------------===//
// Function Reference Values
//===----------------------------------------------------------------------===//

// Returns a non-owning function-reference variant. A non-null |function_ref|
// must be a trusted VM-produced value whose target low bits are zero.
static inline iree_vm_variant_t iree_vm_variant_from_function_ref(
    iree_vm_function_ref_t function_ref) {
  iree_vm_variant_t variant = {
      function_ref.program_bits,
      function_ref.target_bits | IREE_VM_VARIANT_TAG_FUNCTION_REF,
  };
  return variant;
}

// Extracts one non-owning function reference that borrows its program. Failure
// leaves |out_function_ref| untouched.
IREE_API_EXPORT iree_status_t iree_vm_function_ref_from_variant(
    iree_vm_variant_t variant, iree_vm_function_ref_t* out_function_ref);

//===----------------------------------------------------------------------===//
// Scalar Values
//===----------------------------------------------------------------------===//

// Returns one i8 variant.
static inline iree_vm_variant_t iree_vm_variant_from_i8(int8_t value) {
  iree_vm_variant_t variant = {
      (uint8_t)value,
      IREE_VM_VARIANT_SCALAR_METADATA_(IREE_VM_SCALAR_TYPE_I8),
  };
  return variant;
}

// Returns one i16 variant.
static inline iree_vm_variant_t iree_vm_variant_from_i16(int16_t value) {
  iree_vm_variant_t variant = {
      (uint16_t)value,
      IREE_VM_VARIANT_SCALAR_METADATA_(IREE_VM_SCALAR_TYPE_I16),
  };
  return variant;
}

// Returns one i32 variant.
static inline iree_vm_variant_t iree_vm_variant_from_i32(int32_t value) {
  iree_vm_variant_t variant = {
      (uint32_t)value,
      IREE_VM_VARIANT_SCALAR_METADATA_(IREE_VM_SCALAR_TYPE_I32),
  };
  return variant;
}

// Returns one i64 variant preserving its complete bits.
static inline iree_vm_variant_t iree_vm_variant_from_i64(int64_t value) {
  uint64_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  iree_vm_variant_t variant = {
      bits,
      IREE_VM_VARIANT_SCALAR_METADATA_(IREE_VM_SCALAR_TYPE_I64),
  };
  return variant;
}

// Returns one f8E4M3FN-bits variant without numeric conversion.
static inline iree_vm_variant_t iree_vm_variant_from_f8e4m3fn_bits(
    uint8_t bits) {
  iree_vm_variant_t variant = {
      bits,
      IREE_VM_VARIANT_SCALAR_METADATA_(IREE_VM_SCALAR_TYPE_F8E4M3FN),
  };
  return variant;
}

// Returns one f8E5M2-bits variant without numeric conversion.
static inline iree_vm_variant_t iree_vm_variant_from_f8e5m2_bits(uint8_t bits) {
  iree_vm_variant_t variant = {
      bits,
      IREE_VM_VARIANT_SCALAR_METADATA_(IREE_VM_SCALAR_TYPE_F8E5M2),
  };
  return variant;
}

// Returns one f16-bits variant without numeric conversion.
static inline iree_vm_variant_t iree_vm_variant_from_f16_bits(uint16_t bits) {
  iree_vm_variant_t variant = {
      bits,
      IREE_VM_VARIANT_SCALAR_METADATA_(IREE_VM_SCALAR_TYPE_F16),
  };
  return variant;
}

// Returns one bf16-bits variant without numeric conversion.
static inline iree_vm_variant_t iree_vm_variant_from_bf16_bits(uint16_t bits) {
  iree_vm_variant_t variant = {
      bits,
      IREE_VM_VARIANT_SCALAR_METADATA_(IREE_VM_SCALAR_TYPE_BF16),
  };
  return variant;
}

// Returns one f32 variant preserving its complete object-representation bits.
static inline iree_vm_variant_t iree_vm_variant_from_f32(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  iree_vm_variant_t variant = {
      bits,
      IREE_VM_VARIANT_SCALAR_METADATA_(IREE_VM_SCALAR_TYPE_F32),
  };
  return variant;
}

// Returns one f64 variant preserving its complete object-representation bits.
static inline iree_vm_variant_t iree_vm_variant_from_f64(double value) {
  uint64_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  iree_vm_variant_t variant = {
      bits,
      IREE_VM_VARIANT_SCALAR_METADATA_(IREE_VM_SCALAR_TYPE_F64),
  };
  return variant;
}

#undef IREE_VM_VARIANT_SCALAR_METADATA_

// Constructs a scalar from dynamically selected bits. Narrow scalar types
// canonicalize unused high bits. INVALID and unknown scalar types fail with
// INVALID_ARGUMENT and leave |out_variant| untouched.
IREE_API_EXPORT iree_status_t
iree_vm_variant_from_scalar_bits(iree_vm_scalar_type_t scalar_type,
                                 uint64_t bits, iree_vm_variant_t* out_variant);

// Extracts exact canonical scalar bits from |variant|. Type mismatch leaves
// |out_bits| untouched.
IREE_API_EXPORT iree_status_t iree_vm_scalar_bits_from_variant(
    iree_vm_variant_t variant, iree_vm_scalar_type_t expected_type,
    uint64_t* out_bits);

// Extracts one exact i8 value, leaving |out_value| untouched on failure.
IREE_API_EXPORT iree_status_t iree_vm_i8_from_variant(iree_vm_variant_t variant,
                                                      int8_t* out_value);

// Extracts one exact i16 value, leaving |out_value| untouched on failure.
IREE_API_EXPORT iree_status_t
iree_vm_i16_from_variant(iree_vm_variant_t variant, int16_t* out_value);

// Extracts one exact i32 value, leaving |out_value| untouched on failure.
IREE_API_EXPORT iree_status_t
iree_vm_i32_from_variant(iree_vm_variant_t variant, int32_t* out_value);

// Extracts one exact i64 value, leaving |out_value| untouched on failure.
IREE_API_EXPORT iree_status_t
iree_vm_i64_from_variant(iree_vm_variant_t variant, int64_t* out_value);

// Extracts exact f8E4M3FN bits, leaving |out_bits| untouched on failure.
IREE_API_EXPORT iree_status_t iree_vm_f8e4m3fn_bits_from_variant(
    iree_vm_variant_t variant, uint8_t* out_bits);

// Extracts exact f8E5M2 bits, leaving |out_bits| untouched on failure.
IREE_API_EXPORT iree_status_t
iree_vm_f8e5m2_bits_from_variant(iree_vm_variant_t variant, uint8_t* out_bits);

// Extracts exact f16 bits, leaving |out_bits| untouched on failure.
IREE_API_EXPORT iree_status_t
iree_vm_f16_bits_from_variant(iree_vm_variant_t variant, uint16_t* out_bits);

// Extracts exact bf16 bits, leaving |out_bits| untouched on failure.
IREE_API_EXPORT iree_status_t
iree_vm_bf16_bits_from_variant(iree_vm_variant_t variant, uint16_t* out_bits);

// Extracts one exact f32 value, leaving |out_value| untouched on failure.
IREE_API_EXPORT iree_status_t
iree_vm_f32_from_variant(iree_vm_variant_t variant, float* out_value);

// Extracts one exact f64 value, leaving |out_value| untouched on failure.
IREE_API_EXPORT iree_status_t
iree_vm_f64_from_variant(iree_vm_variant_t variant, double* out_value);

//===----------------------------------------------------------------------===//
// Reference Values
//===----------------------------------------------------------------------===//

// Creates a borrowed ref variant from a trusted pointer/type pair.
IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_from_ptr_borrowed(void* ptr, iree_vm_ref_type_t type);

// Retains a trusted pointer/type pair into an owned ref variant.
IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_from_ptr_retained(void* ptr, iree_vm_ref_type_t type);

// Moves a trusted pointer owner into a ref variant and clears |inout_ptr|.
IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_from_ptr_move(void** inout_ptr, iree_vm_ref_type_t type);

// Creates a borrowed ref variant from |ref| without changing |ref|.
IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_from_ref_borrowed(iree_vm_ref_t ref);

// Retains |ref| into an owned ref variant without changing |ref|.
IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_from_ref_retained(iree_vm_ref_t ref);

// Moves |ref| into an owned ref variant and leaves |ref| canonical null.
IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_from_ref_move(iree_vm_ref_t* ref);

// Extracts a borrowed complete ref without changing |variant|. Non-ref input
// leaves |out_ref| untouched.
IREE_API_EXPORT iree_status_t iree_vm_ref_from_variant_borrowed(
    iree_vm_variant_t variant, iree_vm_ref_t* out_ref);

// Extracts an owned complete ref without changing |variant|. Non-ref input
// leaves |out_ref| untouched.
IREE_API_EXPORT iree_status_t iree_vm_ref_from_variant_retained(
    iree_vm_variant_t variant, iree_vm_ref_t* out_ref);

// Moves an owned complete ref out of |variant|. Success leaves |variant| empty.
// Failure leaves source and output untouched.
IREE_API_EXPORT iree_status_t iree_vm_ref_from_variant_move(
    iree_vm_variant_t* variant, iree_vm_ref_t* out_ref);

// Extracts an exactly typed borrowed pointer without changing |variant|.
// Canonical null succeeds and produces null. Failure leaves |out_ptr|
// untouched.
IREE_API_EXPORT iree_status_t iree_vm_ptr_from_variant_borrowed(
    iree_vm_variant_t variant, iree_vm_ref_type_t expected_type,
    void** out_ptr);

// Extracts and retains an exactly typed pointer without changing |variant|.
// Canonical null succeeds and produces null. Failure leaves |out_ptr|
// untouched.
IREE_API_EXPORT iree_status_t iree_vm_ptr_from_variant_retained(
    iree_vm_variant_t variant, iree_vm_ref_type_t expected_type,
    void** out_ptr);

// Moves an exactly typed pointer owner out of |variant|. Success leaves
// |variant| empty. Failure leaves source and output untouched.
IREE_API_EXPORT iree_status_t
iree_vm_ptr_from_variant_move(iree_vm_variant_t* variant,
                              iree_vm_ref_type_t expected_type, void** out_ptr);

// Returns true when |variant| contains a non-null ref with exact |type|.
IREE_API_EXPORT bool iree_vm_variant_ref_isa(iree_vm_variant_t variant,
                                             iree_vm_ref_type_t type);

// Returns an owned copy of |variant|. Scalars, function refs, empty, and null
// carriers copy directly. A borrowed ref is promoted.
IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_retain(iree_vm_variant_t variant);

// Moves |variant| and leaves its source empty. A borrowed ref is promoted
// before clearing.
IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_move(iree_vm_variant_t* variant);

// Releases an owned ref when present and writes canonical empty.
IREE_API_EXPORT void iree_vm_variant_reset(iree_vm_variant_t* variant);

//===----------------------------------------------------------------------===//
// Variant Storage
//===----------------------------------------------------------------------===//

// Mutable count-bearing variant storage. A nonzero |count| requires non-null
// |data|.
typedef struct iree_vm_variant_span_t {
  // Contiguous variant storage.
  iree_vm_variant_t* data;
  // Number of variants in |data|.
  iree_host_size_t count;
} iree_vm_variant_span_t;

// Returns a variant span over |data|.
static inline iree_vm_variant_span_t iree_vm_variant_span_from_ptr(
    iree_vm_variant_t* data, iree_host_size_t count) {
  iree_vm_variant_span_t span = {data, count};
  return span;
}

// Returns an empty variant span.
static inline iree_vm_variant_span_t iree_vm_variant_span_empty(void) {
  iree_vm_variant_span_t span = {NULL, 0};
  return span;
}

// Returns true when |span| contains no variants.
static inline bool iree_vm_variant_span_is_empty(iree_vm_variant_span_t span) {
  return span.count == 0;
}

// Resets every variant without freeing backing storage.
IREE_API_EXPORT void iree_vm_variant_span_reset(iree_vm_variant_span_t span);

#define iree_vm_variant_span_from_array(array) \
  iree_vm_variant_span_from_ptr((array), IREE_ARRAYSIZE(array))

//===----------------------------------------------------------------------===//
// Typed Reference Adapters
//===----------------------------------------------------------------------===//

// Defines the complete typed adapter family for one provider-table field.
// |prefix| is the function prefix, |table_type| is the provider family's
// append-only named table, |table_field| selects the exact descriptor, and
// |object_type| is the complete C object type. |types| must be a matching
// resolved family prefix. Typed move helpers stage through a local void* and
// never alias an object_type** as void**.
#define IREE_VM_DEFINE_TYPE_ADAPTERS(prefix, table_type, table_field,         \
                                     object_type)                             \
  IREE_ATTRIBUTE_UNUSED static inline iree_vm_ref_t                           \
  prefix##_ref_from_ptr_borrowed(const table_type* types, object_type* ptr) { \
    return iree_vm_ref_from_ptr_borrowed((void*)ptr, types->table_field);     \
  }                                                                           \
  IREE_ATTRIBUTE_UNUSED static inline iree_vm_ref_t                           \
  prefix##_ref_from_ptr_retained(const table_type* types, object_type* ptr) { \
    return iree_vm_ref_from_ptr_retained((void*)ptr, types->table_field);     \
  }                                                                           \
  IREE_ATTRIBUTE_UNUSED static inline iree_vm_ref_t                           \
  prefix##_ref_from_ptr_move(const table_type* types,                         \
                             object_type** inout_ptr) {                       \
    void* ptr = (void*)*inout_ptr;                                            \
    iree_vm_ref_t ref = iree_vm_ref_from_ptr_move(&ptr, types->table_field);  \
    *inout_ptr = (object_type*)ptr;                                           \
    return ref;                                                               \
  }                                                                           \
  IREE_ATTRIBUTE_UNUSED static inline iree_vm_variant_t                       \
  prefix##_variant_from_ptr_borrowed(const table_type* types,                 \
                                     object_type* ptr) {                      \
    return iree_vm_variant_from_ptr_borrowed((void*)ptr, types->table_field); \
  }                                                                           \
  IREE_ATTRIBUTE_UNUSED static inline iree_vm_variant_t                       \
  prefix##_variant_from_ptr_retained(const table_type* types,                 \
                                     object_type* ptr) {                      \
    return iree_vm_variant_from_ptr_retained((void*)ptr, types->table_field); \
  }                                                                           \
  IREE_ATTRIBUTE_UNUSED static inline iree_vm_variant_t                       \
  prefix##_variant_from_ptr_move(const table_type* types,                     \
                                 object_type** inout_ptr) {                   \
    void* ptr = (void*)*inout_ptr;                                            \
    iree_vm_variant_t variant =                                               \
        iree_vm_variant_from_ptr_move(&ptr, types->table_field);              \
    *inout_ptr = (object_type*)ptr;                                           \
    return variant;                                                           \
  }                                                                           \
  IREE_ATTRIBUTE_UNUSED static inline iree_status_t                           \
  prefix##_ptr_from_ref_borrowed(const table_type* types, iree_vm_ref_t ref,  \
                                 object_type** out_ptr) {                     \
    void* ptr = NULL;                                                         \
    iree_status_t status = iree_vm_ptr_from_ref_borrowed(                     \
        ref, types->table_field, out_ptr ? &ptr : NULL);                      \
    if (iree_status_is_ok(status)) {                                          \
      *out_ptr = (object_type*)ptr;                                           \
    }                                                                         \
    return status;                                                            \
  }                                                                           \
  IREE_ATTRIBUTE_UNUSED static inline iree_status_t                           \
  prefix##_ptr_from_ref_retained(const table_type* types, iree_vm_ref_t ref,  \
                                 object_type** out_ptr) {                     \
    void* ptr = NULL;                                                         \
    iree_status_t status = iree_vm_ptr_from_ref_retained(                     \
        ref, types->table_field, out_ptr ? &ptr : NULL);                      \
    if (iree_status_is_ok(status)) {                                          \
      *out_ptr = (object_type*)ptr;                                           \
    }                                                                         \
    return status;                                                            \
  }                                                                           \
  IREE_ATTRIBUTE_UNUSED static inline iree_status_t                           \
  prefix##_ptr_from_ref_move(const table_type* types, iree_vm_ref_t* ref,     \
                             object_type** out_ptr) {                         \
    void* ptr = NULL;                                                         \
    iree_status_t status = iree_vm_ptr_from_ref_move(ref, types->table_field, \
                                                     out_ptr ? &ptr : NULL);  \
    if (iree_status_is_ok(status)) {                                          \
      *out_ptr = (object_type*)ptr;                                           \
    }                                                                         \
    return status;                                                            \
  }                                                                           \
  IREE_ATTRIBUTE_UNUSED static inline iree_status_t                           \
  prefix##_ptr_from_variant_borrowed(const table_type* types,                 \
                                     iree_vm_variant_t variant,               \
                                     object_type** out_ptr) {                 \
    void* ptr = NULL;                                                         \
    iree_status_t status = iree_vm_ptr_from_variant_borrowed(                 \
        variant, types->table_field, out_ptr ? &ptr : NULL);                  \
    if (iree_status_is_ok(status)) {                                          \
      *out_ptr = (object_type*)ptr;                                           \
    }                                                                         \
    return status;                                                            \
  }                                                                           \
  IREE_ATTRIBUTE_UNUSED static inline iree_status_t                           \
  prefix##_ptr_from_variant_retained(const table_type* types,                 \
                                     iree_vm_variant_t variant,               \
                                     object_type** out_ptr) {                 \
    void* ptr = NULL;                                                         \
    iree_status_t status = iree_vm_ptr_from_variant_retained(                 \
        variant, types->table_field, out_ptr ? &ptr : NULL);                  \
    if (iree_status_is_ok(status)) {                                          \
      *out_ptr = (object_type*)ptr;                                           \
    }                                                                         \
    return status;                                                            \
  }                                                                           \
  IREE_ATTRIBUTE_UNUSED static inline iree_status_t                           \
  prefix##_ptr_from_variant_move(const table_type* types,                     \
                                 iree_vm_variant_t* variant,                  \
                                 object_type** out_ptr) {                     \
    void* ptr = NULL;                                                         \
    iree_status_t status = iree_vm_ptr_from_variant_move(                     \
        variant, types->table_field, out_ptr ? &ptr : NULL);                  \
    if (iree_status_is_ok(status)) {                                          \
      *out_ptr = (object_type*)ptr;                                           \
    }                                                                         \
    return status;                                                            \
  }                                                                           \
  IREE_ATTRIBUTE_UNUSED static inline bool prefix##_ref_isa(                  \
      const table_type* types, iree_vm_ref_t ref) {                           \
    return iree_vm_ref_isa(ref, types->table_field);                          \
  }                                                                           \
  IREE_ATTRIBUTE_UNUSED static inline bool prefix##_variant_isa(              \
      const table_type* types, iree_vm_variant_t variant) {                   \
    return iree_vm_variant_ref_isa(variant, types->table_field);              \
  }

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_VARIANT_H_
